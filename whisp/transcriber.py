import gc
import logging
import queue
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger("whisp")

_HALLUCINATIONS = frozenset({
    "thank you.", "thank you", "thanks.", "thanks for watching.",
    "you", "bye.", "bye-bye.", ".", "...", "[silence]",
    "[ silence ]", "♪", "subtitles by",
})


class TranscriberThread(QThread):
    """Loads Whisper model, transcribes phrase audio, emits text. Types via typer."""

    model_ready = pyqtSignal()
    text_ready = pyqtSignal(str)  # transcribed text for a phrase

    def __init__(self, model_name="large-v3", language="en", typing_delay_ms=10,
                 hallucination_filter=True, min_audio_rms=0.005):
        super().__init__()
        self.model_name = model_name
        self.language = language
        self.typing_delay_ms = typing_delay_ms
        self.hallucination_filter = hallucination_filter
        self.min_audio_rms = min_audio_rms
        self.phrase_queue = queue.Queue()
        self._running = False
        self._model = None
        self._is_first_phrase = True
        self._prev_text = ""

    def submit_phrase(self, audio: np.ndarray):
        self.phrase_queue.put(audio)

    def run(self):
        self._running = True
        self._is_first_phrase = True
        self._prev_text = ""
        log.info("Transcriber thread started, loading model '%s'...", self.model_name)

        # Load model
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device="cuda",
                compute_type="float16",
                local_files_only=True,
            )
            log.info("Whisper model loaded (CUDA)")
        except Exception:
            log.warning("CUDA unavailable, falling back to CPU")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
            log.info("Whisper model loaded (CPU)")

        self.model_ready.emit()

        # Process phrases
        while self._running or not self.phrase_queue.empty():
            try:
                audio = self.phrase_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text = self._transcribe(audio)
            if text:
                # Prepend space between phrases
                if not self._is_first_phrase:
                    text = " " + text
                self._is_first_phrase = False
                log.info("Transcribed: %s", text)
                self.text_ready.emit(text)

        # Unload model
        self._unload_model()
        log.info("Transcriber thread stopped")

    def _build_prompt(self) -> str:
        if self._prev_text:
            return self._prev_text[-150:]
        return "Transcribe with proper punctuation and capitalization."

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            return ""

        # RMS energy gate — skip near-silent chunks before hitting Whisper
        if self.hallucination_filter:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < self.min_audio_rms:
                log.debug("Skipping low-energy audio (rms=%.5f)", rms)
                return ""

        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                without_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=self._build_prompt(),
            )

            parts = []
            for seg in segments:
                if self.hallucination_filter and seg.no_speech_prob > 0.6:
                    log.debug("Skipping segment (no_speech_prob=%.3f): %s", seg.no_speech_prob, seg.text)
                    continue
                parts.append(seg.text)

            text = " ".join(parts).strip()

            if self.hallucination_filter and text.lower() in _HALLUCINATIONS:
                log.debug("Filtered hallucination: %s", text)
                return ""

            if text:
                self._prev_text = (self._prev_text + " " + text).strip()

            return text
        except Exception as e:
            log.error("Transcription error: %s", e)
            return ""

    def _unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log.info("VRAM freed")
            except ImportError:
                pass

    def stop(self):
        self._running = False
