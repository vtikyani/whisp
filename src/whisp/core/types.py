"""Plain data types shared by every Whisp subsystem.

Nothing here has behaviour or heavy imports, so audio, speech recognition,
text processing, the Windows layer and the UI can all depend on this module
without depending on each other.

Dictated text, selections, audio and window titles are left out of ``repr()``,
so a stray log line can't print them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
"""All audio inside Whisp is mono float32 at 16 kHz."""

FRAME_SAMPLES = 512
"""One VAD frame: 512 samples = 32 ms at 16 kHz."""

FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


class SessionKind(enum.Enum):
    DICTATION = "dictation"
    """Speech becomes text that is inserted at the cursor."""
    COMMAND = "command"
    """Speech is an instruction that rewrites the currently selected text."""


class RecordingMode(enum.Enum):
    HOLD_PENDING = "hold_pending"
    """Key is down; not yet known whether this is a hold or a quick tap."""
    HOLD = "hold"
    """Push-to-talk: recording stops when the hotkey is released."""
    HANDS_FREE = "hands_free"
    """Locked: recording stops on the next press, the stop button, or Esc (cancel)."""


class Outcome(enum.Enum):
    INSERTED = "inserted"
    COPIED = "copied"
    """Text was left on the clipboard instead (focus changed, or insertion failed)."""
    NO_SPEECH = "no_speech"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AppCategory(enum.Enum):
    MESSAGING = "messaging"
    EMAIL = "email"
    CODE = "code"
    DOCUMENT = "document"
    OTHER = "other"


@dataclass(frozen=True)
class TargetWindow:
    """The window that had keyboard focus when a dictation started."""

    hwnd: int
    pid: int
    process_name: str
    """Executable file name, lower-case, e.g. ``"slack.exe"``."""
    app_name: str
    """Friendly name, e.g. ``"Slack"`` (from the executable's version info when available)."""
    title: str = field(repr=False)
    """Window title. Used only for app categorisation; never stored in history or logged."""
    category: AppCategory = AppCategory.OTHER
    is_fullscreen: bool = False
    monitor_rect: tuple[int, int, int, int] | None = None
    """(left, top, right, bottom) in physical pixels of the monitor holding the window."""


class KeyKind(enum.Enum):
    """The kind of key the user last typed — never which key or character it was."""

    SPACE = "space"
    """Space or Tab."""
    NEWLINE = "newline"
    """Enter."""
    TEXT = "text"
    """A printable character: letter, digit or punctuation."""
    EDIT = "edit"
    """Backspace, Delete, or a key combined with Ctrl/Alt/Win (paste, undo…): the text changed unpredictably."""
    MOVE = "move"
    """Arrows, Home/End, Page Up/Down: the cursor moved."""


@dataclass(frozen=True)
class KeyActivity:
    """The user's most recent keystroke, as far as smart spacing needs to know."""

    serial: int
    """Increases with every key the user types. Whisp's own keystrokes don't count."""
    kind: KeyKind
    hwnd: int
    """The window that had focus when it was typed."""
    at: float
    """``time.monotonic()`` when it was typed."""


@dataclass(frozen=True)
class AudioDevice:
    id: str
    """Stable key used in settings: the device name as reported by Windows."""
    name: str
    host_api: str
    is_default: bool
    max_input_channels: int
    default_samplerate: float


@dataclass(frozen=True)
class Segment:
    """A slice of one session's audio, ready for speech recognition."""

    session_id: str
    index: int
    """0-based order within the session. Results are joined in this order."""
    samples: np.ndarray = field(repr=False)
    """Mono float32 at 16 kHz, already trimmed and padded."""
    speech_ms: int
    """How much of the segment the VAD judged to be speech."""
    is_final: bool
    """True for the last segment of a session (the tail flushed on stop)."""


@dataclass(frozen=True)
class AsrResult:
    session_id: str
    index: int
    text: str = field(repr=False)
    audio_ms: int
    inference_ms: int
    language: str | None = None
    error: str | None = None
    """Set when recognition failed; ``text`` is then empty."""


class EngineState(enum.Enum):
    NOT_INSTALLED = "not_installed"
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True)
class EngineStatus:
    state: EngineState
    model_id: str | None = None
    device: str = "cpu"
    """``"cuda"`` or ``"cpu"`` — what the model is actually running on."""
    device_name: str = "CPU"
    """Human-readable, e.g. ``"NVIDIA GeForce RTX 4070"``."""
    message: str | None = None
    """Short explanation for ERROR / NOT_INSTALLED, or progress text while LOADING."""


@dataclass(frozen=True)
class Transcript:
    """Everything recognised in one session, before text processing."""

    session_id: str
    text: str = field(repr=False)
    """Segment texts joined with correct spacing."""
    audio_ms: int
    speech_ms: int
    inference_ms: int
    segments: tuple[AsrResult, ...] = ()
    language: str | None = None


@dataclass(frozen=True)
class SessionInfo:
    """What the controller knows about a session when handing it to delivery."""

    session_id: str
    kind: SessionKind
    mode: RecordingMode
    target: TargetWindow | None
    started_at: float
    """``time.time()`` when recording started."""
    stopped_at: float
    """``time.time()`` when the user stopped recording."""
    stopped_monotonic: float
    """``time.monotonic()`` at stop — used to measure stop-to-insert latency."""
    selection: str | None = field(default=None, repr=False)
    """Command mode only: the text that was selected when the session started."""


@dataclass(frozen=True)
class InsertionResult:
    ok: bool
    method: str
    """``"paste"``, ``"type"`` or ``"clipboard"`` (left on clipboard for the user)."""
    focus_changed: bool = False
    error: str | None = None


@dataclass(frozen=True)
class DeliveryResult:
    session_id: str
    kind: SessionKind
    outcome: Outcome
    final_text: str = field(repr=False)
    word_count: int
    latency_ms: int
    """From the user stopping recording to the text being inserted."""
    polished: bool = False
    target: TargetWindow | None = None
    message: str | None = None
    """Short user-facing explanation for COPIED / FAILED outcomes."""
