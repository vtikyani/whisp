import logging
from enum import Enum, auto

from pynput import keyboard as pynput_kb
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal

from whisp.widget import WaveformWidget
from whisp.audio import AudioCapture
from whisp.vad import VADThread
from whisp.transcriber import TranscriberThread
from whisp.typer import Typer
from whisp.hotkey import HotkeyListener

log = logging.getLogger("whisp")


class _EscSignal(QObject):
    pressed = pyqtSignal()


class AppState(Enum):
    IDLE = auto()
    LISTENING = auto()
    DEACTIVATING = auto()


class WhispApp:
    def __init__(self, config: dict):
        self.config = config
        self.state = AppState.IDLE

        self.qapp = QApplication([])
        self.qapp.setQuitOnLastWindowClosed(False)

        # Components
        num_bars = config.get("waveform_bars", 7)
        self.widget = WaveformWidget(num_bars=num_bars)
        self.audio = AudioCapture(num_bars=num_bars)
        self.typer = Typer(delay_ms=config.get("typing_delay_ms", 10))
        self.hotkey = HotkeyListener(hotkey_str=config.get("hotkey", "ctrl+shift+space"))

        # These are created/destroyed per activation cycle
        self.vad_thread = None
        self.transcriber_thread = None
        self._esc_listener = None
        self._esc_signal = _EscSignal()
        self._esc_signal.pressed.connect(self._deactivate)

        # Wire signals
        self.audio.fft_ready.connect(self.widget.update_levels)
        self.hotkey.toggled.connect(self._on_hotkey_toggle)

    def run(self):
        self.hotkey.start()
        log.info("Whisp running. Press %s to activate.", self.config.get("hotkey", "ctrl+shift+space"))
        return self.qapp.exec()

    # -- State transitions --

    def _on_hotkey_toggle(self):
        if self.state == AppState.IDLE:
            self._activate()
        elif self.state == AppState.LISTENING:
            self._deactivate()

    def _activate(self):
        log.info("Activating...")
        self.state = AppState.LISTENING

        # Show widget (green immediately — audio capture is live)
        self.widget.set_listening(True)
        self.widget.show_widget()

        # Play activation chime
        if self.config.get("audio_feedback", True):
            try:
                from whisp.chime import play_activate

                play_activate()
            except Exception as e:
                log.debug("Chime error: %s", e)

        # Start audio capture
        try:
            self.audio.start()
        except Exception:
            log.error("No microphone detected")
            self.widget.hide_widget()
            self.state = AppState.IDLE
            return

        # Start VAD thread
        self.vad_thread = VADThread(
            audio_queue=self.audio.audio_queue,
            silence_ms=self.config.get("vad_silence_ms", 300),
            threshold=self.config.get("vad_threshold", 0.5),
            max_phrase_ms=self.config.get("vad_max_phrase_ms", 3000),
        )

        # Start transcriber thread
        self.transcriber_thread = TranscriberThread(
            model_name=self.config.get("whisper_model", "large-v3"),
            language=self.config.get("language", "en"),
            typing_delay_ms=self.config.get("typing_delay_ms", 10),
            hallucination_filter=self.config.get("hallucination_filter", True),
            min_audio_rms=self.config.get("min_audio_rms", 0.005),
        )

        # Wire VAD -> transcriber
        self.vad_thread.phrase_ready.connect(self.transcriber_thread.submit_phrase)
        self.vad_thread.speech_started.connect(self._on_speech_started)

        # Wire transcriber signals
        self.transcriber_thread.model_ready.connect(self._on_model_ready)
        self.transcriber_thread.text_ready.connect(self._on_text_ready)

        # Start threads
        self.vad_thread.start()
        self.transcriber_thread.start()

        # Register Esc to deactivate
        def _on_esc(key):
            if key == pynput_kb.Key.esc:
                self._esc_signal.pressed.emit()
                return False
        self._esc_listener = pynput_kb.Listener(on_press=_on_esc)
        self._esc_listener.start()

    def _deactivate(self):
        if self.state == AppState.DEACTIVATING:
            return
        log.info("Deactivating...")
        self.state = AppState.DEACTIVATING

        # Unregister Esc listener
        if self._esc_listener is not None:
            self._esc_listener.stop()
            self._esc_listener = None

        # Play deactivation chime
        if self.config.get("audio_feedback", True):
            try:
                from whisp.chime import play_deactivate

                play_deactivate()
            except Exception as e:
                log.debug("Chime error: %s", e)

        # Stop audio capture
        self.audio.stop()

        # Stop VAD thread
        if self.vad_thread is not None:
            self.vad_thread.stop()
            self.vad_thread.wait(2000)
            self.vad_thread = None

        # Stop transcriber (will drain queue first)
        if self.transcriber_thread is not None:
            self.transcriber_thread.stop()
            self.transcriber_thread.wait(10000)  # up to 10s to drain
            self.transcriber_thread = None

        # Hide widget
        self.widget.hide_widget()
        self.state = AppState.IDLE
        log.info("Deactivated. State: IDLE")

    def _on_model_ready(self):
        log.info("Whisper model loaded")

    def _on_speech_started(self):
        """Called when VAD detects speech."""
        pass

    def _on_text_ready(self, text: str):
        """Type transcribed text into the focused window."""
        if self.state in (AppState.LISTENING, AppState.DEACTIVATING):
            self.typer.type_text(text)
