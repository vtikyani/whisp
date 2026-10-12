"""AppController: the dictation state machine.

Hotkeys, the overlay's buttons and the tray menu come in; recordings,
overlay states and sounds go out. Everything here runs on the GUI thread and
never blocks: work that could wait (reading the selected text for command
mode) runs on a short-lived worker thread and reports back through a signal.

The recording state machine is the table in docs/ARCHITECTURE.md. In short:
pressing dictate (or command) starts a recording; releasing it within
``hotkeys.tap_to_lock_ms`` locks hands-free mode, releasing later stops. A
hands-free recording stops on the next press of any recording shortcut, the
overlay's stop button, the time limit, or the PC being locked (what was said
until then is kept). Esc and the overlay's ✕ cancel. A
chord interrupted by another key in its first second was meant as a different
shortcut, so the recording is cancelled without a sound.

Starting a session never waits for earlier ones. Stopped sessions wait here
until their transcript (and, for commands, the selected text) is in, then go
to delivery strictly in the order they were started.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from whisp.core.types import (
    AppCategory,
    DeliveryResult,
    EngineState,
    EngineStatus,
    Outcome,
    RecordingMode,
    SessionInfo,
    SessionKind,
    TargetWindow,
    Transcript,
)

if TYPE_CHECKING:
    from whisp.app.delivery import DeliveryService
    from whisp.app.sounds import SoundPlayer
    from whisp.core.asr.manager import ModelManager
    from whisp.core.asr.service import AsrService
    from whisp.core.session import SessionPipeline
    from whisp.core.text.polish import Polisher
    from whisp.platform.win32.audio_mute import MuteService
    from whisp.platform.win32.hotkeys import HotkeyService
    from whisp.platform.win32.insert import TextInserter
    from whisp.storage.settings import SettingsStore
    from whisp.ui.overlay import Overlay

log = logging.getLogger(__name__)

INTERRUPT_WINDOW_S = 1.0
"""A chord interrupted by another key this soon after it started was a different shortcut, not dictation."""
INTERRUPT_GRACE_MS = 150
"""After an interruption, how long to wait for the longer Whisp shortcut it may have been part of
(Ctrl+Win → Ctrl+Win+Space) before discarding the recording. If it arrives, the recording is kept."""
MUTE_DELAY_MS = 250
"""Other apps are muted only once a recording has lasted this long, so a shortcut pressed by accident
(or the start of Ctrl+Win+D) doesn't make the user's music skip."""
STOP_GESTURE_S = 0.6
"""After a press stops hands-free, a longer shortcut that completes this soon is the same keypress
(Ctrl+Win+Space starts with Ctrl+Win), so it must not start a new recording."""
MINUTE_MS = 60_000
RESULT_WAIT_MS = 15_000
"""How long finished sessions wait behind an earlier one that hasn't reported, before going ahead without it."""
MAX_ABANDONED = 32

BINDABLE_ACTIONS = ("dictate", "command", "paste_last", "copy_last", "hands_free")
ACTION_LABELS = {
    "dictate": "Dictation",
    "command": "Command mode",
    "paste_last": "Paste last transcript",
    "copy_last": "Copy last transcript",
    "hands_free": "Hands-free",
}

MSG_ONE_MINUTE_LEFT = "1 minute left. Recording stops at the limit, and everything you said is kept."
MSG_NO_MODEL = "Download a speech model to start dictating."
MSG_COMMAND_NEEDS_POLISH = "Command mode needs AI polish. Turn it on in Styles."
MSG_LOADING_MODEL = "Loading speech model…"
MSG_START_FAILED = "Couldn't start recording. Please try again."
MSG_SUBMIT_FAILED = "Something went wrong with that dictation. Please try again."
MSG_COPIED = "Copied — press Ctrl + V to paste."


class AppState(enum.Enum):
    """What the tray icon shows."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass(eq=False)
class _Session:
    id: str
    kind: SessionKind
    mode: RecordingMode
    trigger: str
    """The action whose release or interruption ends this session (``"dictate"``, ``"command"``, ``"hands_free"``)."""
    target: TargetWindow | None
    started_at: float
    started_monotonic: float
    target_at_stop: bool = False
    """Started from the tray menu: the window focused when recording stops is where the text goes."""
    stopped_at: float | None = None
    stopped_monotonic: float | None = None
    transcript: Transcript | None = None
    selection: str | None = None
    selection_pending: bool = False

    @property
    def ready(self) -> bool:
        return self.stopped_monotonic is not None and self.transcript is not None and not self.selection_pending

    def info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.id,
            kind=self.kind,
            mode=self.mode,
            target=self.target,
            started_at=self.started_at,
            stopped_at=self.stopped_at if self.stopped_at is not None else self.started_at,
            stopped_monotonic=self.stopped_monotonic if self.stopped_monotonic is not None else self.started_monotonic,
            selection=self.selection,
        )


class AppController(QObject):
    """Runs the recording state machine on the GUI thread. See the module docstring."""

    state_changed = Signal()
    """Something the tray or UI shows changed: ``state``, ``status_message()``, ``binding_errors``, paused."""
    notification = Signal(str, str, str)
    """title, message, kind (``"info"`` | ``"warning"`` | ``"error"``). Only emitted when notifications are on."""
    open_window_requested = Signal(str, str)
    """page, section (``""`` for none): the user needs the main window, e.g. to download a speech model."""

    _selection_captured = Signal(str, object)
    """session_id, selected text or None — from the selection worker thread."""

    def __init__(
        self,
        settings: SettingsStore,
        *,
        models: ModelManager,
        asr: AsrService,
        pipeline: SessionPipeline,
        polisher: Polisher,
        inserter: TextInserter,
        hotkeys: HotkeyService,
        overlay: Overlay,
        sounds: SoundPlayer,
        delivery: DeliveryService,
        mute: MuteService | None = None,
        get_target: Callable[[], TargetWindow | None] | None = None,
        categorize: Callable[[str, str, Mapping[str, str]], AppCategory] | None = None,
        display_chord: Callable[[str], list[str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        # The defaults are the real platform and text functions; tests pass their own.
        if get_target is None:
            from whisp.platform.win32.foreground import get_foreground_target as get_target
        if categorize is None:
            from whisp.core.text.context import categorize
        if display_chord is None:
            from whisp.platform.win32.keys import display_chord
        self._settings = settings
        self._models = models
        self._asr = asr
        self._pipeline = pipeline
        self._polisher = polisher
        self._inserter = inserter
        self._hotkeys = hotkeys
        self._overlay = overlay
        self._sounds = sounds
        self._delivery = delivery
        self._mute = mute
        self._get_target = get_target
        self._categorize = categorize
        self._display_chord = display_chord
        self._clock = clock
        self._wall_clock = wall_clock

        self._recording: _Session | None = None
        self._sessions: dict[str, _Session] = {}
        """Sessions not yet handed to delivery, in start order (the recording one is always last)."""
        self._in_delivery: set[str] = set()
        self._abandoned: dict[str, _Session] = {}
        """Sessions that newer ones stopped waiting for; delivered if their result still turns up."""
        self._paused = False
        self._closed = False
        self._binding_errors: dict[str, str] = {}
        self._notified_binding_errors: set[tuple[str, str]] = set()
        self._notified_cancel_errors: set[str] = set()
        self._note_shown = False
        self._last_result: DeliveryResult | None = None
        self._capture_lock = threading.Lock()

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._hold_timer.timeout.connect(self._on_hold_timer)
        self._max_timer = QTimer(self)
        self._max_timer.setSingleShot(True)
        self._max_timer.timeout.connect(self._on_max_length)
        self._length_warning_timer = QTimer(self)
        self._length_warning_timer.setSingleShot(True)
        self._length_warning_timer.timeout.connect(self._on_length_warning)
        self._mute_timer = QTimer(self)
        self._mute_timer.setSingleShot(True)
        self._mute_timer.setInterval(MUTE_DELAY_MS)
        self._mute_timer.timeout.connect(self._on_mute_timer)
        self._interrupted: _Session | None = None
        """A just-started recording whose shortcut was interrupted; cancelled when the grace timer fires."""
        self._stop_press: tuple[str, float] | None = None
        """(action, time) of the last press that stopped a hands-free recording."""
        self._interrupt_timer = QTimer(self)
        self._interrupt_timer.setSingleShot(True)
        self._interrupt_timer.setInterval(INTERRUPT_GRACE_MS)
        self._interrupt_timer.timeout.connect(self._on_interrupt_grace_expired)
        self._wait_timer = QTimer(self)
        self._wait_timer.setSingleShot(True)
        self._wait_timer.setInterval(RESULT_WAIT_MS)
        self._wait_timer.timeout.connect(self._on_result_wait_expired)

        hotkeys.pressed.connect(self._on_hotkey_pressed)
        hotkeys.released.connect(self._on_hotkey_released)
        hotkeys.interrupted.connect(self._on_hotkey_interrupted)
        hotkeys.binding_error.connect(self._on_binding_error)
        hotkeys.session_locked.connect(self._on_session_locked)
        pipeline.level.connect(self._on_level)
        pipeline.preview.connect(self._on_preview)
        pipeline.finished.connect(self._on_finished)
        pipeline.no_speech.connect(self._on_no_speech)
        pipeline.failed.connect(self._on_failed)
        overlay.stop_clicked.connect(self._on_overlay_stop)
        overlay.cancel_clicked.connect(self._on_overlay_cancel)
        overlay.idle_clicked.connect(self._on_overlay_idle)
        delivery.delivered.connect(self._on_delivered)
        asr.status_changed.connect(self._on_asr_status)
        settings.changed.connect(self._on_settings_changed)
        self._selection_captured.connect(self._on_selection_captured)

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        """Apply the keyboard shortcuts. Call after ``HotkeyService.start()``."""
        self._apply_bindings()
        self._update_hotkey_hint()
        self.state_changed.emit()

    def shutdown(self) -> None:
        """Cancel any recording and stop reacting to events. Safe to call more than once."""
        if self._closed:
            return
        session = self._recording
        self._recording = None
        self._closed = True
        for timer in (self._hold_timer, self._max_timer, self._wait_timer, self._mute_timer):
            timer.stop()
        if session is not None:
            log.info("Session %s cancelled because Whisp is quitting", session.id)
            self._call("cancel the recording", self._pipeline.cancel, session.id)
            self._call("turn off Esc to cancel", self._hotkeys.set_cancel_active, False)
            self._restore_other_apps()
        self._sessions.clear()
        self._abandoned.clear()

    # -- public API (GUI thread) -------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording is not None

    @property
    def recording_mode(self) -> RecordingMode | None:
        return self._recording.mode if self._recording is not None else None

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def binding_errors(self) -> dict[str, str]:
        """Shortcuts that couldn't be registered: ``{action: user-facing message}``."""
        return dict(self._binding_errors)

    @property
    def last_result(self) -> DeliveryResult | None:
        return self._last_result

    @property
    def state(self) -> AppState:
        if self._paused:
            return AppState.PAUSED
        if self._recording is not None:
            return AppState.RECORDING
        if self._sessions or self._in_delivery:
            return AppState.PROCESSING
        if self.problem() is not None:
            return AppState.ERROR
        return AppState.IDLE

    def problem(self) -> str | None:
        """A plain-language description of something stopping dictation from working, or None."""
        if self._binding_errors:
            action = next((a for a in BINDABLE_ACTIONS if a in self._binding_errors), next(iter(self._binding_errors)))
            return f"{ACTION_LABELS.get(action, action)} shortcut: {self._binding_errors[action]}"
        status = self._asr.status
        if status.state is EngineState.NOT_INSTALLED:
            return MSG_NO_MODEL
        if status.state is EngineState.ERROR:
            return status.message or "The speech model couldn't load. Open Whisp to fix it."
        return None

    def status_message(self) -> str:
        """One or two short lines for the tray tooltip."""
        state = self.state
        if state is AppState.PAUSED:
            return "Paused. Shortcuts are off until you resume."
        if state is AppState.RECORDING:
            return "Listening…"
        if state is AppState.PROCESSING:
            return "Working on your text…"
        problem = self.problem()
        if problem is not None:
            return problem
        if self._asr.status.state is EngineState.LOADING:
            return MSG_LOADING_MODEL
        keys = self.hotkey_keys()
        ready = f"Ready. Hold {' + '.join(keys)} to dictate." if keys else "Ready."
        last = self._last_result_line()
        return f"{ready}\n{last}" if last else ready

    def hotkey_keys(self) -> list[str]:
        """The dictation shortcut as key-cap labels, e.g. ``["Ctrl", "Win"]``; empty if none is set."""
        chord = self._settings.current.hotkeys.dictate.strip()
        if not chord:
            return []
        try:
            return list(self._display_chord(chord))
        except ValueError:
            return []
        except Exception:
            log.exception("Couldn't format the dictation shortcut")
            return []

    def start_hands_free(self, *, from_menu: bool = False) -> None:
        """Same as the hands-free shortcut: start a hands-free dictation, or stop the one in progress.

        ``from_menu`` is for the tray menu, which takes focus away from the user's
        app: the text then goes to whichever window is focused when recording stops.
        """
        if self._closed or self._paused:
            return
        session = self._recording
        if session is None:
            self._start(SessionKind.DICTATION, RecordingMode.HANDS_FREE, trigger="hands_free", target_at_stop=from_menu)
        elif session.mode is RecordingMode.HANDS_FREE:
            self._stop(session)

    def toggle_paused(self) -> None:
        """Pause or resume Whisp. Pausing finishes (doesn't discard) a recording in progress."""
        if self._closed:
            return
        self._paused = not self._paused
        if self._paused and self._recording is not None:
            self._stop(self._recording)
        self._call("update the shortcuts", self._hotkeys.set_paused, self._paused)
        log.info("Whisp %s", "paused" if self._paused else "resumed")
        self.state_changed.emit()

    def paste_last(self) -> None:
        """Insert the most recent text again into the focused window."""
        if not self._closed:
            self._delivery.paste_last()

    def copy_last(self) -> None:
        """Put the most recent text on the clipboard."""
        if not self._closed:
            self._delivery.copy_last()

    @Slot(object)
    def on_foreground_changed(self, target: TargetWindow | None) -> None:
        """Keep the idle overlay on the active monitor (connect to ``ForegroundWatcher.changed``)."""
        if not self._closed and self._recording is None:
            self._overlay.set_target(target)

    # -- hotkeys -------------------------------------------------------------------------

    @Slot(str)
    def _on_hotkey_pressed(self, action: str) -> None:
        if self._closed or self._paused:
            return
        if self._completes_stop_press(action):
            return
        if action in ("hands_free", "command") and self._adopt_recording(action):
            return
        if action in ("dictate", "command"):
            session = self._recording
            if session is None:
                kind = SessionKind.DICTATION if action == "dictate" else SessionKind.COMMAND
                self._start(kind, RecordingMode.HOLD_PENDING, trigger=action)
            elif session.mode is RecordingMode.HANDS_FREE:
                self._stop(session)
                self._stop_press = (action, self._clock())
        elif action == "hands_free":
            stopping = self._recording is not None and self._recording.mode is RecordingMode.HANDS_FREE
            self.start_hands_free()
            if stopping:
                self._stop_press = (action, self._clock())
        elif action == "cancel":
            if self._recording is not None:
                self._cancel(self._recording)
        elif action == "paste_last":
            self.paste_last()
        elif action == "copy_last":
            self.copy_last()
        else:
            log.warning("Ignoring unknown shortcut action %r", action)

    @Slot(str)
    def _on_hotkey_released(self, action: str) -> None:
        session = self._recording
        if self._closed or session is None or session.trigger != action or session.mode is RecordingMode.HANDS_FREE:
            return
        if session is self._interrupted:
            # The shortcut was part of another key combination (Ctrl+Win+D let go at once). Letting go
            # is neither a tap nor the end of a hold: the grace timer, or the longer Whisp shortcut if
            # it follows, decides what happens to this recording.
            return
        held_ms = (self._clock() - session.started_monotonic) * 1000.0
        if held_ms < self._settings.current.hotkeys.tap_to_lock_ms:
            self._lock_hands_free(session)
        else:
            self._stop(session)

    @Slot(str)
    def _on_hotkey_interrupted(self, action: str) -> None:
        session = self._recording
        if self._closed or session is None or session.trigger != action or session.mode is RecordingMode.HANDS_FREE:
            return
        if self._clock() - session.started_monotonic < INTERRUPT_WINDOW_S:
            # Don't discard yet: the extra key may complete a longer Whisp shortcut, whose press can
            # arrive just after this. See _adopt_recording.
            self._interrupted = session
            self._interrupt_timer.start()

    @Slot()
    def _on_interrupt_grace_expired(self) -> None:
        session, self._interrupted = self._interrupted, None
        if self._closed or session is None or session is not self._recording:
            return
        log.info("Session %s cancelled: the shortcut was part of another key combination", session.id)
        self._cancel(session, silent=True)

    def _completes_stop_press(self, action: str) -> bool:
        """True if ``action`` is the rest of the keypress that just stopped hands-free (see STOP_GESTURE_S)."""
        if self._stop_press is None or self._recording is not None:
            return False
        stopped_by, at = self._stop_press
        if self._clock() - at >= STOP_GESTURE_S or not self._chord_extends(action, stopped_by):
            return False
        self._stop_press = None
        log.debug("Ignoring %s: it completes the keypress that stopped hands-free", action)
        return True

    def _chord_extends(self, longer: str, shorter: str) -> bool:
        """Whether ``longer``'s shortcut is ``shorter``'s plus more keys (Ctrl+Win+Space vs Ctrl+Win)."""
        hotkeys = self._settings.current.hotkeys
        long_chord, short_chord = getattr(hotkeys, longer, ""), getattr(hotkeys, shorter, "")
        if not long_chord or not short_chord:
            return False
        return set(short_chord.split("+")) < set(long_chord.split("+"))

    def _adopt_recording(self, action: str) -> bool:
        """Turn a recording that just started into ``action``'s kind of recording, keeping its audio.

        Pressing Ctrl+Win+Space starts with Ctrl+Win, so the dictate shortcut fires first. When the
        longer hands-free or command shortcut follows within a moment, that recording becomes
        hands-free (or a command) instead of being discarded and restarted.
        """
        session = self._recording
        # Any recording that an interruption could still cancel (held for a tap or a little longer, under
        # INTERRUPT_WINDOW_S) can be taken over; otherwise a slow Ctrl+Win → Space would be cancelled
        # by the interruption while the hands-free press was ignored, and nothing would record.
        if session is None or session.trigger == action or session.mode is RecordingMode.HANDS_FREE:
            return False
        if self._clock() - session.started_monotonic >= INTERRUPT_WINDOW_S:
            return False
        self._interrupt_timer.stop()
        self._interrupted = None
        if action == "hands_free":
            session.trigger = "hands_free"
            self._lock_hands_free(session)
        else:
            if not self._polish_ready():
                self._cancel(session, silent=True)
                self._overlay.show_notice(MSG_COMMAND_NEEDS_POLISH, "warning")
                return True
            session.kind = SessionKind.COMMAND
            session.trigger = "command"
            session.selection_pending = True
            self._overlay.show_recording(SessionKind.COMMAND, session.mode)
        log.info("Session %s continues as %s", session.id, action)
        self.state_changed.emit()
        return True

    @Slot(str, str)
    def _on_binding_error(self, action: str, message: str) -> None:
        if self._closed:
            return
        log.warning("Shortcut for %s isn't available: %s", action, message)
        if action == "cancel":
            # Esc is taken only while recording, so it isn't a standing problem (nothing would ever clear it):
            # say so once, without marking Whisp as broken.
            if message not in self._notified_cancel_errors:
                self._notified_cancel_errors.add(message)
                self._notify("Esc can't cancel", message, "warning")
            return
        self._binding_errors[action] = message
        self._report_binding_errors()
        self.state_changed.emit()

    @Slot()
    def _on_session_locked(self) -> None:
        """The PC was locked or the session disconnected: stop listening to the room, keeping what was said."""
        session = self._recording
        if self._closed or session is None:
            return
        log.info("Session %s stopped because the PC was locked", session.id)
        self._stop(session)

    # -- overlay -------------------------------------------------------------------------

    @Slot()
    def _on_overlay_stop(self) -> None:
        session = self._recording
        if not self._closed and session is not None and session.mode is RecordingMode.HANDS_FREE:
            self._stop(session)

    @Slot()
    def _on_overlay_cancel(self) -> None:
        if not self._closed and self._recording is not None:
            self._cancel(self._recording)

    @Slot()
    def _on_overlay_idle(self) -> None:
        if self._recording is None:
            self.start_hands_free()

    # -- recording -----------------------------------------------------------------------

    def _start(self, kind: SessionKind, mode: RecordingMode, *, trigger: str, target_at_stop: bool = False) -> None:
        target = self._capture_target()
        if not self._speech_model_available():
            log.info("Not recording: no speech model is installed")
            self._overlay.set_target(target)
            self._overlay.show_notice(MSG_NO_MODEL, "warning")
            if self._settings.current.general.onboarding_completed:
                self.open_window_requested.emit("settings", "model")
            else:
                self.open_window_requested.emit("onboarding", "")
            return
        if kind is SessionKind.COMMAND and not self._polish_ready():
            log.info("Not recording a command: AI polish isn't ready")
            self._overlay.set_target(target)
            self._overlay.show_notice(MSG_COMMAND_NEEDS_POLISH, "warning")
            return

        session = _Session(
            id=uuid.uuid4().hex,
            kind=kind,
            mode=mode,
            trigger=trigger,
            target=target,
            started_at=self._wall_clock(),
            started_monotonic=self._clock(),
            target_at_stop=target_at_stop,
        )
        if kind is SessionKind.COMMAND:
            # The selection is read when recording stops, not now: the command shortcut is
            # still held, and Ctrl+C sent while Alt/Win are down isn't a copy.
            session.selection_pending = True
        self._sessions[session.id] = session
        self._recording = session
        try:
            self._pipeline.start(session.id)
        except Exception:
            log.exception("Couldn't start recording session %s", session.id)
            self._recording = None
            del self._sessions[session.id]
            self._overlay.show_outcome(Outcome.FAILED, MSG_START_FAILED)
            self._sounds.play("error")
            return
        self._hotkeys.set_cancel_active(True)
        self._overlay.set_target(target)
        self._overlay.show_recording(kind, mode)
        self._sounds.play("start")
        if self._mute is not None and self._settings.current.audio.mute_other_apps:
            self._mute_timer.start()
        if mode is RecordingMode.HOLD_PENDING:
            self._hold_timer.start(self._settings.current.hotkeys.tap_to_lock_ms)
        max_minutes = self._settings.current.audio.max_recording_minutes
        self._max_timer.start(max_minutes * MINUTE_MS)
        if max_minutes > 1:
            self._length_warning_timer.start((max_minutes - 1) * MINUTE_MS)
        self._warm_up_engine()
        log.info(
            "Session %s started (%s, %s, %s)",
            session.id,
            kind.value,
            mode.value,
            target.process_name if target is not None else "no target",
        )
        self.state_changed.emit()

    def _stop(self, session: _Session) -> None:
        if session.mode is RecordingMode.HOLD_PENDING:
            session.mode = RecordingMode.HOLD
        session.stopped_at = self._wall_clock()
        session.stopped_monotonic = self._clock()
        if session.target_at_stop:
            session.target = self._capture_target()
        self._leave_recording(session)
        self._read_selection_if_needed(session)
        self._pipeline.stop(session.id)
        self._hotkeys.set_cancel_active(False)
        self._overlay.show_processing()
        self._sounds.play("stop")
        log.info(
            "Session %s stopped after %.1f s (%s)",
            session.id,
            session.stopped_monotonic - session.started_monotonic,
            session.mode.value,
        )
        self.state_changed.emit()

    def _cancel(self, session: _Session, *, silent: bool = False) -> None:
        self._leave_recording(session)
        self._sessions.pop(session.id, None)
        self._pipeline.cancel(session.id)
        self._hotkeys.set_cancel_active(False)
        self._overlay.show_outcome(Outcome.CANCELLED)
        if not silent:
            self._sounds.play("cancel")
        log.info("Session %s cancelled", session.id)
        self._flush()
        self.state_changed.emit()

    def _lock_hands_free(self, session: _Session) -> None:
        session.mode = RecordingMode.HANDS_FREE
        self._hold_timer.stop()
        self._overlay.set_mode(RecordingMode.HANDS_FREE)
        self._sounds.play("lock")
        log.info("Session %s locked hands-free", session.id)
        self.state_changed.emit()

    def _leave_recording(self, session: _Session) -> None:
        """The controller is free for the next recording from here on."""
        if self._recording is session:
            self._recording = None
        if self._interrupted is session:
            self._interrupted = None
            self._interrupt_timer.stop()
        self._hold_timer.stop()
        self._max_timer.stop()
        self._length_warning_timer.stop()
        self._mute_timer.stop()
        self._restore_other_apps()
        if self._note_shown:
            self._note_shown = False
            self._overlay.set_note(None)

    @Slot()
    def _on_mute_timer(self) -> None:
        session = self._recording
        if self._closed or session is None or self._mute is None:
            return
        if session is self._interrupted:
            # Probably the start of another shortcut (Ctrl+Win+D): don't make the music skip while
            # that's being decided. Cancelling stops this timer; keeping the recording lets it fire.
            self._mute_timer.start()
            return
        self._call("mute other apps", self._mute.mute)

    def _restore_other_apps(self) -> None:
        if self._mute is not None:
            self._call("restore other apps' sound", self._mute.restore)

    @Slot()
    def _on_hold_timer(self) -> None:
        session = self._recording
        if session is None or session.mode is not RecordingMode.HOLD_PENDING:
            return
        held_ms = (self._clock() - session.started_monotonic) * 1000
        remaining_ms = self._settings.current.hotkeys.tap_to_lock_ms - held_ms
        if remaining_ms > 0:
            # Timers can fire a little early; this is still a possible tap.
            self._hold_timer.start(max(1, int(remaining_ms) + 1))
            return
        session.mode = RecordingMode.HOLD
        self._overlay.set_mode(RecordingMode.HOLD)

    @Slot()
    def _on_length_warning(self) -> None:
        if self._closed or self._recording is None:
            return
        self._overlay.show_notice(MSG_ONE_MINUTE_LEFT, "info")
        self._sounds.play("lock")

    @Slot()
    def _on_max_length(self) -> None:
        session = self._recording
        if self._closed or session is None:
            return
        minutes = self._settings.current.audio.max_recording_minutes
        log.info("Session %s reached the %d-minute limit", session.id, minutes)
        self._stop(session)
        message = f"Recording stopped at the {minutes}-minute limit. You can change it in Settings."
        self._overlay.show_notice(message, "info")
        self._notify("Recording stopped", message, "info")

    # -- command mode selection --------------------------------------------------------

    def _read_selection_if_needed(self, session: _Session) -> None:
        if session.kind is SessionKind.COMMAND and session.selection_pending:
            self._capture_selection(session.id)

    def _capture_selection(self, session_id: str) -> None:
        """Read the selected text on a worker thread; the result comes back through ``_selection_captured``.

        The inserter waits for the user to let go of the shortcut's modifier keys before copying.
        """

        def work() -> None:
            text: str | None = None
            try:
                # One clipboard round-trip at a time.
                with self._capture_lock:
                    text = self._inserter.capture_selection()
            except Exception:
                log.exception("Couldn't read the selected text for session %s", session_id)
            log.info("Session %s: selection %s", session_id, f"{len(text)} chars" if text else "empty")
            try:
                self._selection_captured.emit(session_id, text)
            except RuntimeError:
                log.debug("Selection for session %s arrived after the controller was gone", session_id)

        threading.Thread(target=work, name="whisp-selection", daemon=True).start()

    @Slot(str, object)
    def _on_selection_captured(self, session_id: str, text: object) -> None:
        session = self._sessions.get(session_id) or self._abandoned.get(session_id)
        if self._closed or session is None or not session.selection_pending:
            return
        session.selection = text if isinstance(text, str) and text else None
        session.selection_pending = False
        if session.id in self._abandoned and session.ready:
            self._submit(self._abandoned.pop(session.id))
        self._flush()

    # -- pipeline ------------------------------------------------------------------------

    @Slot(float)
    def _on_level(self, level: float) -> None:
        if not self._closed and self._recording is not None:
            self._overlay.set_level(level)

    @Slot(str, str)
    def _on_preview(self, session_id: str, text: str) -> None:
        session = self._recording
        if not self._closed and session is not None and session.id == session_id:
            self._overlay.set_preview(text)

    @Slot(object)
    def _on_finished(self, transcript: Transcript) -> None:
        if self._closed:
            return
        session = self._sessions.get(transcript.session_id)
        if session is None:
            late = self._abandoned.get(transcript.session_id)
            if late is None:
                log.debug("Ignoring a transcript for unknown session %s", transcript.session_id)
                return
            log.warning("Session %s finished late; delivering it now", late.id)
            late.transcript = transcript
            if late.ready:
                self._submit(self._abandoned.pop(late.id))
            return
        if session is self._recording:
            # The pipeline ends a recording itself when the microphone disappears or stalls.
            log.info("Session %s: recording ended by the pipeline", session.id)
            self._end_recording_early(session)
            self._read_selection_if_needed(session)
            self._overlay.show_processing()
        session.transcript = transcript
        self._flush()
        self.state_changed.emit()

    @Slot(str)
    def _on_no_speech(self, session_id: str) -> None:
        if self._closed:
            return
        session = self._sessions.pop(session_id, None) or self._abandoned.pop(session_id, None)
        if session is None:
            return
        log.info("Session %s: no speech", session_id)
        if session is self._recording:
            self._end_recording_early(session)
            self._overlay.show_outcome(Outcome.NO_SPEECH)  # ends the recording view
        else:
            self._show_outcome(Outcome.NO_SPEECH, None)
        self._flush()
        self.state_changed.emit()

    @Slot(str, str)
    def _on_failed(self, session_id: str, message: str) -> None:
        if self._closed:
            return
        session = self._sessions.pop(session_id, None) or self._abandoned.pop(session_id, None)
        if session is None:
            return
        log.warning("Session %s failed: %s", session_id, message)
        if session is self._recording:
            # The microphone couldn't open, went silent or disappeared. Make sure nothing is left
            # running, and end the recording view.
            self._end_recording_early(session)
            self._call("cancel the failed recording", self._pipeline.cancel, session_id)
            self._overlay.show_outcome(Outcome.FAILED, message)
        else:
            self._show_outcome(Outcome.FAILED, message)
        if self._recording is None:
            self._sounds.play("error")
        self._notify("Dictation didn't work", message, "error")
        self._flush()
        self.state_changed.emit()

    def _end_recording_early(self, session: _Session) -> None:
        """The pipeline ended a session that was still recording (no ``stop()`` from us); leave recording cleanly."""
        session.stopped_at = self._wall_clock()
        session.stopped_monotonic = self._clock()
        if session.mode is RecordingMode.HOLD_PENDING:
            session.mode = RecordingMode.HOLD
        if session.target_at_stop:
            session.target = self._capture_target()
        self._leave_recording(session)
        self._hotkeys.set_cancel_active(False)

    # -- ordering and delivery -----------------------------------------------------------

    def _flush(self) -> None:
        """Hand every session at the front of the line that's ready to delivery, in start order."""
        while self._sessions:
            session = next(iter(self._sessions.values()))
            if not session.ready:
                break
            del self._sessions[session.id]
            self._submit(session)
        self._update_wait_timer()

    def _submit(self, session: _Session) -> None:
        assert session.transcript is not None
        self._in_delivery.add(session.id)
        try:
            self._delivery.submit(session.info(), session.transcript)
        except Exception:
            log.exception("Couldn't hand session %s to delivery", session.id)
            self._in_delivery.discard(session.id)
            self._show_outcome(Outcome.FAILED, MSG_SUBMIT_FAILED)
        self.state_changed.emit()

    def _blocked_head(self) -> _Session | None:
        """The first session, if it's holding up delivery: waiting for its selection, or behind finished ones."""
        sessions = list(self._sessions.values())
        if not sessions or sessions[0].ready:
            return None
        head = sessions[0]
        waiting_for_selection = head.transcript is not None and head.selection_pending
        if waiting_for_selection or any(s.ready for s in sessions[1:]):
            return head
        return None

    def _update_wait_timer(self) -> None:
        if self._blocked_head() is None:
            self._wait_timer.stop()
        elif not self._wait_timer.isActive():
            self._wait_timer.start()

    @Slot()
    def _on_result_wait_expired(self) -> None:
        head = self._blocked_head()
        if self._closed or head is None:
            return
        if head.transcript is not None and head.selection_pending:
            log.warning("Session %s: the selected text didn't arrive in time", head.id)
            head.selection_pending = False
        else:
            log.error(
                "Session %s has had no result for %d s; delivering newer dictations first",
                head.id,
                RESULT_WAIT_MS // 1000,
            )
            del self._sessions[head.id]
            self._abandoned[head.id] = head
            while len(self._abandoned) > MAX_ABANDONED:
                self._abandoned.pop(next(iter(self._abandoned)))
        self._flush()

    @Slot(object)
    def _on_delivered(self, result: DeliveryResult) -> None:
        if self._closed:
            return
        self._in_delivery.discard(result.session_id)
        if result.outcome in (Outcome.INSERTED, Outcome.COPIED, Outcome.FAILED):
            self._last_result = result
        self._show_outcome(result.outcome, result.message)
        if result.outcome is Outcome.FAILED:
            if self._recording is None:
                self._sounds.play("error")
            self._notify("Dictation didn't work", result.message or MSG_SUBMIT_FAILED, "error")
        self.state_changed.emit()

    def _show_outcome(self, outcome: Outcome, message: str | None) -> None:
        """Show how a session ended without disturbing a recording or hiding work that's still in progress.

        While a recording runs, or other dictations are still being processed,
        only what the user must know is mentioned, on the pill's note line:
        text left on the clipboard, and failures.
        """
        if self._recording is None and not self._sessions and not self._in_delivery:
            self._overlay.show_outcome(outcome, message)
        elif outcome is Outcome.COPIED:
            self._overlay.show_notice(message or MSG_COPIED, "info")
        elif outcome is Outcome.FAILED:
            self._overlay.show_notice(message or MSG_SUBMIT_FAILED, "error")

    # -- engine, settings, shortcuts ---------------------------------------------------

    @Slot(object)
    def _on_asr_status(self, status: EngineStatus) -> None:
        if self._closed:
            return
        self._update_note(status)
        self.state_changed.emit()

    def _warm_up_engine(self) -> None:
        """Start loading the speech model the moment someone starts talking, if it isn't loaded."""
        status = self._asr.status
        if status.state in (EngineState.UNLOADED, EngineState.ERROR, EngineState.NOT_INSTALLED):
            self._call("load the speech model", self._asr.ensure_loaded)
        self._update_note(self._asr.status)

    def _update_note(self, status: EngineStatus) -> None:
        loading = status.state in (EngineState.LOADING, EngineState.UNLOADED, EngineState.NOT_INSTALLED)
        want = self._recording is not None and loading
        if want and not self._note_shown:
            self._note_shown = True
            self._overlay.set_note(MSG_LOADING_MODEL)
        elif not want and self._note_shown:
            self._note_shown = False
            self._overlay.set_note(None)

    def _speech_model_available(self) -> bool:
        if self._asr.status.state is not EngineState.NOT_INSTALLED:
            return True
        # The engine's status can lag right after a download finishes; ask the model manager too.
        model_id = self._settings.current.transcription.model_id
        for variant in ("gpu", "cpu"):
            try:
                if self._models.is_installed(model_id, variant):
                    return True
            except KeyError:
                continue  # this model has no such variant, or isn't in the catalog
            except Exception:
                log.exception("Couldn't check whether %s is installed", model_id)
        return False

    def _polish_ready(self) -> bool:
        try:
            return bool(self._polisher.is_ready())
        except Exception:
            log.exception("Couldn't check whether AI polish is ready")
            return False

    @Slot(object)
    def _on_settings_changed(self, sections: set[str]) -> None:
        if not self._closed and "hotkeys" in sections:
            self._apply_bindings()
            self._update_hotkey_hint()

    def _apply_bindings(self) -> None:
        hotkeys = self._settings.current.hotkeys
        bindings = {action: chord for action in BINDABLE_ACTIONS if (chord := getattr(hotkeys, action).strip())}
        try:
            errors = dict(self._hotkeys.set_bindings(bindings) or {})
        except Exception:
            log.exception("Couldn't apply the keyboard shortcuts")
            errors = {"dictate": "Whisp couldn't set up its keyboard shortcuts. Restart Whisp to try again."}
        for action, message in errors.items():
            log.warning("Shortcut for %s isn't available: %s", action, message)
        self._binding_errors = {str(action): str(message) for action, message in errors.items() if message}
        self._report_binding_errors()
        self.state_changed.emit()

    def _report_binding_errors(self) -> None:
        current = set(self._binding_errors.items())
        for action, message in sorted(current - self._notified_binding_errors):
            self._notify("Shortcut not available", f"{ACTION_LABELS.get(action, action)}: {message}", "warning")
        self._notified_binding_errors = current

    def _update_hotkey_hint(self) -> None:
        self._overlay.set_hotkey_hint(self.hotkey_keys())

    # -- helpers -------------------------------------------------------------------------

    def _capture_target(self) -> TargetWindow | None:
        try:
            target = self._get_target()
        except Exception:
            log.exception("Couldn't read the focused window")
            return None
        return self._categorized(target) if target is not None else None

    def _categorized(self, target: TargetWindow) -> TargetWindow:
        overrides = self._settings.current.text.app_categories
        try:
            category = self._categorize(target.process_name, target.title, overrides)
        except Exception:
            log.exception("Couldn't categorise %s", target.process_name)
            return target
        # Keep a category the platform layer already knew unless the user chose one for this app.
        if (
            category is AppCategory.OTHER
            and target.category is not AppCategory.OTHER
            and target.process_name.lower() not in overrides
        ):
            return target
        return target if category is target.category else dataclasses.replace(target, category=category)

    def _notify(self, title: str, message: str, kind: str) -> None:
        if self._settings.current.general.show_notifications:
            self.notification.emit(title, message, kind)

    def _last_result_line(self) -> str:
        result = self._last_result
        if result is None:
            return ""
        if result.outcome is Outcome.INSERTED:
            words = "1 word" if result.word_count == 1 else f"{result.word_count:,} words"
            app = result.target.app_name if result.target is not None and result.target.app_name else ""
            return f"Last: {words} into {app}." if app else f"Last: {words}."
        if result.outcome is Outcome.COPIED:
            return "Last: copied to the clipboard."
        return "Last: didn't work."

    @staticmethod
    def _call(what: str, fn: Callable[..., object], *args: object) -> None:
        try:
            fn(*args)
        except Exception:
            log.exception("Couldn't %s", what)
