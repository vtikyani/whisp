"""SessionPipeline: microphone -> VAD -> segments -> speech recognition -> Transcript.

One long-lived thread, ``whisp-pipeline``, owns the microphone stream, the
VAD and every session's segmenter. Public methods only queue a command and
return, so the GUI thread never waits. Speech-recognition callbacks arrive on
``whisp-asr`` and are handed back through the same queue, so session state is
only ever touched by the pipeline thread. Signals are emitted from it; Qt
delivers them on each receiver's own thread.

A session's life:

1. ``start`` opens the microphone. Every frame goes through Silero VAD and the
   session's ``Segmenter``; each segment that ends at a natural pause goes to
   ``AsrService.submit()`` straight away, so on stop only the tail is left.
2. With live preview on, the speech still in progress is sent to
   ``AsrService.submit_preview()`` about every 800 ms of audio, and ``preview``
   carries the text so far. Preview text never reaches the transcript.
3. ``stop`` stops the microphone, processes every frame it had already
   captured, and sends what the segmenter still holds as the final segment.
4. When every segment's result is back, exactly one of ``finished``,
   ``no_speech`` or ``failed`` is emitted. ``cancel`` drops the session without
   any of them.

A new session can start while earlier ones still wait for results. ``finished``
signals always come in the order the sessions were started, so text is
inserted in the order it was spoken.

If the microphone disappears mid-recording, the session ends on its own with
whatever was captured; ``failed`` is used only when nothing usable was.
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PySide6.QtCore import QObject, Signal

from whisp.core.audio import AudioError, MicrophoneStream
from whisp.core.segmenter import SegmentAudio, Segmenter, SegmenterConfig
from whisp.core.text.scripts import join_spoken
from whisp.core.types import SAMPLE_RATE, AsrResult, Segment, Transcript
from whisp.core.vad import SileroVad

if TYPE_CHECKING:
    from whisp.storage.settings import SettingsStore

log = logging.getLogger(__name__)

MSG_MIC_FAILED = "Couldn't start the microphone. Check Settings → Microphone."
MSG_MIC_SILENT = (
    "The microphone is sending only silence. Check that it's switched on and not muted, and that Windows "
    "lets apps use it (Settings → Privacy & security → Microphone)."
)
MSG_VAD_FAILED = "Speech detection couldn't start. Reinstalling Whisp should fix this."
MSG_ASR_UNAVAILABLE = "Speech recognition isn't available right now. Please try again."
MSG_INTERNAL = "Something went wrong while recording. Please try again."

_READ_TIMEOUT_S = 0.05
"""Longest a queued command waits while the microphone has no new frame (frames normally come every 32 ms)."""
_PREVIEW_EVERY_SAMPLES = SAMPLE_RATE * 800 // 1000
"""Live preview cadence, in captured audio rather than wall time."""
_SILENT_MIC_SAMPLES = SAMPLE_RATE
"""At least this much digitally silent audio means a muted, switched-off or blocked microphone, not a quiet room."""
_DIGITAL_SILENCE_PEAK = 1.0 / 32768
"""Loudest sample such a microphone delivers: exact zeros through WASAPI, one 16-bit step through MME.

Any working microphone's noise floor is well above it.
"""
_SHUTDOWN_WAIT_S = 2.0


class AudioStream(Protocol):
    """What the pipeline needs from a microphone stream. ``MicrophoneStream`` is the real one."""

    def start(self) -> None: ...

    def read(self, timeout: float) -> np.ndarray | None: ...

    def stop(self) -> None: ...

    @property
    def device_name(self) -> str: ...

    @property
    def error(self) -> str | None: ...


StreamFactory = Callable[[str | None, Callable[[float], None] | None], AudioStream]
"""Called as ``factory(device, on_level)`` on the pipeline thread for every session."""


class AsrBackend(Protocol):
    """The parts of ``AsrService`` the pipeline uses."""

    def submit(self, segment: Segment, callback: Callable[[AsrResult], None]) -> None: ...

    def submit_preview(self, session_id: str, samples: np.ndarray, callback: Callable[[str], None]) -> None: ...

    def discard(self, session_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class _Start:
    session_id: str


@dataclass(frozen=True, slots=True)
class _Stop:
    session_id: str


@dataclass(frozen=True, slots=True)
class _Cancel:
    session_id: str


@dataclass(frozen=True, slots=True)
class _SegmentDone:
    result: AsrResult


@dataclass(frozen=True, slots=True)
class _PreviewDone:
    session_id: str
    index: int
    """The segment index the previewed audio was going to become."""
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Shutdown:
    """Stop the pipeline thread."""


_Command = _Start | _Stop | _Cancel | _SegmentDone | _PreviewDone | _Shutdown


class _Phase(enum.Enum):
    RECORDING = "recording"
    FINISHING = "finishing"
    """Microphone closed; waiting for recognition results."""


@dataclass(eq=False)
class _Session:
    id: str
    segmenter: Segmenter
    stream: AudioStream | None = None
    phase: _Phase = _Phase.RECORDING
    samples: int = 0
    """Audio captured so far, in samples."""
    peak: float = 0.0
    """Largest absolute sample seen; only tracked while the audio still looks digitally silent."""
    next_index: int = 0
    outstanding: set[int] = field(default_factory=set)
    results: dict[int, AsrResult] = field(default_factory=dict)
    device_error: str | None = None
    since_preview: int = 0
    voiced_since_preview: bool = False
    preview_text: str = field(default="", repr=False)
    """Preview of the speech still in progress."""
    stand_ins: dict[int, str] = field(default_factory=dict, repr=False)
    """Preview text shown for committed segments until their real result arrives. Display only."""
    last_preview: str = field(default="", repr=False)


class SessionPipeline(QObject):
    """Records sessions and turns them into transcripts. See the module docstring."""

    level = Signal(float)
    """Microphone level (0-1) while recording, at most ~30 times a second."""
    preview = Signal(str, str)
    """session_id, text so far (live preview only; never the final transcript)."""
    finished = Signal(object)
    """Transcript. Emitted in session start order."""
    no_speech = Signal(str)
    """session_id: nothing was said."""
    failed = Signal(str, str)
    """session_id, user-facing message."""

    def __init__(
        self,
        settings: SettingsStore,
        asr: AsrBackend,
        *,
        stream_factory: StreamFactory = MicrophoneStream,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._asr = asr
        self._stream_factory = stream_factory
        self._config = SegmenterConfig()
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._closed = False
        self._level_value: float | None = None
        """Latest level from the audio thread, emitted by the pipeline thread."""
        # Everything below is owned by the pipeline thread.
        self._vad: SileroVad | None = None
        self._running = True
        self._recording: _Session | None = None
        self._sessions: dict[str, _Session] = {}
        self._order: list[str] = []
        """Sessions that haven't reported yet, in start order."""
        self._ready: dict[str, Transcript] = {}
        """Finished transcripts waiting for an earlier session to report first."""
        self._thread = threading.Thread(target=self._run, name="whisp-pipeline", daemon=True)
        self._thread.start()

    # -- public API (any thread) ---------------------------------------------------

    def start(self, session_id: str) -> None:
        """Start recording a new session. Returns at once; the microphone opens on the pipeline thread."""
        self._post(_Start(session_id))

    def stop(self, session_id: str) -> None:
        """Stop recording and finish the session; everything captured so far is transcribed."""
        self._post(_Stop(session_id))

    def cancel(self, session_id: str) -> None:
        """Stop recording and discard the session. No result signal follows."""
        self._post(_Cancel(session_id))

    def shutdown(self) -> None:
        """Close the microphone, drop pending sessions and stop the pipeline thread.

        Waits up to two seconds for the thread, so the microphone is released
        before PortAudio is torn down at exit; it normally takes milliseconds.
        """
        if self._closed:
            return
        self._closed = True
        self._commands.put(_Shutdown())
        if threading.current_thread() is self._thread:
            return
        self._thread.join(_SHUTDOWN_WAIT_S)
        if self._thread.is_alive():
            log.warning("Pipeline thread didn't stop within %.1f s", _SHUTDOWN_WAIT_S)

    def _post(self, command: _Command) -> None:
        if self._closed:
            log.debug("Pipeline is shut down; ignoring %s", type(command).__name__)
            return
        self._commands.put(command)

    # -- callbacks from other threads (only queue or store, never touch session state) --

    def _on_level(self, level: float) -> None:
        self._level_value = level

    def _on_segment_result(self, result: AsrResult) -> None:
        self._commands.put(_SegmentDone(result))

    # -- pipeline thread -------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._vad = SileroVad()
        except Exception:
            log.exception("Couldn't load the speech detector")
        while self._running:
            try:
                self._step()
            except Exception:
                log.exception("Unexpected error in the pipeline loop")
        self._teardown()

    def _step(self) -> None:
        if self._recording is None:
            self._handle(self._commands.get())
            return
        while self._running and not self._commands.empty():
            self._handle(self._commands.get_nowait())
        session = self._recording
        if self._running and session is not None:
            self._pump(session)

    def _handle(self, command: _Command) -> None:
        try:
            match command:
                case _Start(session_id=sid):
                    self._handle_start(sid)
                case _Stop(session_id=sid):
                    self._handle_stop(sid)
                case _Cancel(session_id=sid):
                    self._handle_cancel(sid)
                case _SegmentDone(result=result):
                    self._handle_segment_done(result)
                case _PreviewDone():
                    self._handle_preview_done(command)
                case _Shutdown():
                    self._running = False
        except Exception:
            log.exception("Pipeline couldn't handle %s", type(command).__name__)
            # A failed preview only costs the preview; anything else leaves the session in an unknown state.
            if isinstance(command, _SegmentDone):
                self._abandon(command.result.session_id, MSG_INTERNAL)
            elif isinstance(command, _Start | _Stop):
                self._abandon(command.session_id, MSG_INTERNAL)

    def _handle_start(self, sid: str) -> None:
        if sid in self._sessions or sid in self._ready:
            log.warning("Session %s already exists; ignoring start", sid)
            return
        session = _Session(sid, Segmenter(self._config))
        self._sessions[sid] = session
        self._order.append(sid)
        previous = self._recording
        if previous is not None:
            log.warning("Session %s started while %s was still recording; finishing that one first", sid, previous.id)
            self._end_recording(previous)
        if self._vad is None:
            self._fail(sid, MSG_VAD_FAILED)
            return
        self._vad.reset()
        self._level_value = None
        stream: AudioStream | None = None
        try:
            stream = self._stream_factory(self._settings.current.audio.input_device, self._on_level)
            stream.start()
        except AudioError as exc:
            log.warning("Session %s: microphone didn't start: %s", sid, exc)
            self._close_quietly(stream)
            self._fail(sid, str(exc))
            return
        except Exception:
            log.exception("Session %s: microphone didn't start", sid)
            self._close_quietly(stream)
            self._fail(sid, MSG_MIC_FAILED)
            return
        session.stream = stream
        self._recording = session
        log.info("Session %s: recording from %r", sid, stream.device_name)

    def _handle_stop(self, sid: str) -> None:
        session = self._sessions.get(sid)
        if session is None or session.phase is not _Phase.RECORDING:
            log.debug("Session %s isn't recording; ignoring stop", sid)
            return
        self._end_recording(session)

    def _handle_cancel(self, sid: str) -> None:
        session = self._sessions.pop(sid, None)
        had_transcript = self._ready.pop(sid, None) is not None
        if session is None and not had_transcript:
            log.debug("Session %s is unknown or already reported; ignoring cancel", sid)
            return
        if sid in self._order:
            self._order.remove(sid)
        if session is not None:
            self._release_stream(session)
        self._discard_asr_work(sid)
        log.info("Session %s: cancelled", sid)
        self._release_finished()

    def _handle_segment_done(self, result: AsrResult) -> None:
        session = self._sessions.get(result.session_id)
        if session is None or result.index not in session.outstanding:
            log.debug("Ignoring result %d for session %s (cancelled or duplicate)", result.index, result.session_id)
            return
        session.outstanding.discard(result.index)
        session.results[result.index] = result
        session.stand_ins.pop(result.index, None)
        if result.error:
            log.warning("Session %s: segment %d wasn't recognised: %s", session.id, result.index, result.error)
        if self._recording is session:
            self._emit_preview(session)
        self._check_complete(session)

    def _handle_preview_done(self, done: _PreviewDone) -> None:
        session = self._sessions.get(done.session_id)
        if session is None or self._recording is not session:
            return
        text = done.text.strip() if isinstance(done.text, str) else ""
        if done.index == session.next_index:
            session.preview_text = text
        elif done.index in session.outstanding:
            # The previewed speech was committed meanwhile; show it until the real result arrives.
            session.stand_ins[done.index] = text
        else:
            return
        self._emit_preview(session)

    def _pump(self, session: _Session) -> None:
        stream = session.stream
        if stream is None:
            return
        try:
            frame = stream.read(_READ_TIMEOUT_S)
            if frame is None:
                error = stream.error
                if error is not None:
                    log.warning("Session %s: microphone lost: %s", session.id, error)
                    session.device_error = error
                    self._end_recording(session)
                return
            self._process_frame(session, frame)
        except Exception:
            log.exception("Session %s: processing audio failed; finishing with what was captured", session.id)
            self._salvage(session)
            return
        self._emit_level()

    def _process_frame(self, session: _Session, frame: np.ndarray) -> None:
        if self._vad is None:
            raise RuntimeError("speech detector isn't loaded")
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        session.samples += samples.size
        if session.peak <= _DIGITAL_SILENCE_PEAK and samples.size:
            session.peak = max(session.peak, float(np.max(np.abs(samples))))
        prob = self._vad(samples)
        for audio in session.segmenter.push(samples, prob):
            self._submit(session, audio, is_final=False)
        session.since_preview += samples.size
        if prob >= self._config.end_threshold:
            session.voiced_since_preview = True
        if self._recording is session:
            self._maybe_preview(session)

    def _end_recording(self, session: _Session, *, drain: bool = True) -> None:
        """Stop the microphone, process what it captured, flush the segmenter and submit the tail."""
        if self._recording is session:
            self._recording = None
        stream, session.stream = session.stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                log.exception("Session %s: stopping the microphone failed", session.id)
            if drain:
                try:
                    while (frame := stream.read(0.0)) is not None:
                        self._process_frame(session, frame)
                except Exception:
                    log.exception("Session %s: processing the last captured audio failed", session.id)
        try:
            tail = session.segmenter.flush()
        except Exception:
            log.exception("Session %s: flushing the segmenter failed", session.id)
            tail = []
        for i, audio in enumerate(tail):
            self._submit(session, audio, is_final=i == len(tail) - 1)
        session.phase = _Phase.FINISHING
        log.info(
            "Session %s: recorded %d ms; %d segment(s), %d ms of speech",
            session.id,
            session.samples * 1000 // SAMPLE_RATE,
            session.next_index,
            session.segmenter.total_speech_ms,
        )
        self._check_complete(session)

    def _salvage(self, session: _Session) -> None:
        try:
            self._end_recording(session, drain=False)
        except Exception:
            log.exception("Session %s: couldn't finish after an error", session.id)
            self._abandon(session.id, MSG_INTERNAL)

    def _submit(self, session: _Session, audio: SegmentAudio, *, is_final: bool) -> None:
        index = session.next_index
        session.next_index += 1
        if session.preview_text:
            session.stand_ins[index] = session.preview_text
            session.preview_text = ""
        session.since_preview = 0
        audio_ms = audio.samples.size * 1000 // SAMPLE_RATE
        segment = Segment(
            session_id=session.id,
            index=index,
            samples=audio.samples,
            speech_ms=audio.speech_ms,
            is_final=is_final,
        )
        session.outstanding.add(index)
        log.debug(
            "Session %s: segment %d to recognition (%d ms audio, %d ms speech%s)",
            session.id,
            index,
            audio_ms,
            audio.speech_ms,
            ", final" if is_final else "",
        )
        try:
            self._asr.submit(segment, self._on_segment_result)
        except Exception:
            log.exception("Session %s: couldn't queue segment %d for recognition", session.id, index)
            session.outstanding.discard(index)
            session.results[index] = AsrResult(session.id, index, "", audio_ms, 0, error=MSG_ASR_UNAVAILABLE)

    def _maybe_preview(self, session: _Session) -> None:
        if session.since_preview < _PREVIEW_EVERY_SAMPLES or not session.voiced_since_preview:
            return
        if not self._settings.current.transcription.live_preview:
            return
        samples = session.segmenter.pending()
        if samples is None:
            return
        session.since_preview = 0
        session.voiced_since_preview = False
        sid, index = session.id, session.next_index

        def deliver(text: str) -> None:
            self._commands.put(_PreviewDone(sid, index, text))

        try:
            self._asr.submit_preview(sid, samples, deliver)
        except Exception:
            log.exception("Session %s: couldn't request a live preview", sid)

    def _emit_preview(self, session: _Session) -> None:
        if not self._settings.current.transcription.live_preview:
            return
        parts = []
        for index in range(session.next_index):
            result = session.results.get(index)
            if result is None:
                text = session.stand_ins.get(index, "")
            else:
                text = result.text.strip() if result.error is None else ""
            if text:
                parts.append(text)
        if session.preview_text:
            parts.append(session.preview_text)
        text = join_spoken(parts)
        if text == session.last_preview:
            return
        session.last_preview = text
        self.preview.emit(session.id, text)

    def _emit_level(self) -> None:
        value = self._level_value
        if value is None:
            return
        self._level_value = None
        self.level.emit(value)

    def _check_complete(self, session: _Session) -> None:
        if session.phase is not _Phase.FINISHING or session.outstanding:
            return
        results = tuple(session.results[i] for i in sorted(session.results))
        text = join_spoken(r.text for r in results if r.error is None)
        errors = [r.error for r in results if r.error]
        audio_ms = session.samples * 1000 // SAMPLE_RATE
        speech_ms = session.segmenter.total_speech_ms
        if text:
            if errors:
                log.warning("Session %s: %d of %d segment(s) failed recognition", session.id, len(errors), len(results))
            if session.device_error:
                log.warning("Session %s: delivering what was captured before the microphone was lost", session.id)
            transcript = Transcript(
                session_id=session.id,
                text=text,
                audio_ms=audio_ms,
                speech_ms=speech_ms,
                inference_ms=sum(r.inference_ms for r in results),
                segments=results,
                language=_main_language(results),
            )
            log.info(
                "Session %s: transcript ready: %d word(s), %d segment(s), %d ms audio, %d ms speech, %d ms recognition",
                session.id,
                len(text.split()),
                len(results),
                audio_ms,
                speech_ms,
                transcript.inference_ms,
            )
            self._sessions.pop(session.id, None)
            self._ready[session.id] = transcript
            self._release_finished()
        elif errors:
            self._fail(session.id, errors[0])
        elif session.device_error:
            self._fail(session.id, session.device_error)
        elif session.peak <= _DIGITAL_SILENCE_PEAK and session.samples >= _SILENT_MIC_SAMPLES:
            self._fail(session.id, MSG_MIC_SILENT)
        else:
            log.info("Session %s: no speech (%d ms audio, %d ms speech)", session.id, audio_ms, speech_ms)
            self._sessions.pop(session.id, None)
            if session.id in self._order:
                self._order.remove(session.id)
            self.no_speech.emit(session.id)
            self._release_finished()

    def _fail(self, sid: str, message: str) -> None:
        self._sessions.pop(sid, None)
        self._ready.pop(sid, None)
        if sid in self._order:
            self._order.remove(sid)
        log.info("Session %s failed: %s", sid, message)
        self.failed.emit(sid, message)
        self._release_finished()

    def _abandon(self, sid: str, message: str) -> None:
        """Give up on a session after an internal error: close its microphone, drop its work, report."""
        session = self._sessions.get(sid)
        if session is None and sid not in self._ready:
            return
        if session is not None:
            self._release_stream(session)
        self._discard_asr_work(sid)
        self._fail(sid, message)

    def _release_finished(self) -> None:
        while self._order and self._order[0] in self._ready:
            self.finished.emit(self._ready.pop(self._order.pop(0)))

    def _release_stream(self, session: _Session) -> None:
        if self._recording is session:
            self._recording = None
        stream, session.stream = session.stream, None
        self._close_quietly(stream)

    def _close_quietly(self, stream: AudioStream | None) -> None:
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            log.exception("Closing the microphone failed")

    def _discard_asr_work(self, sid: str) -> None:
        try:
            self._asr.discard(sid)
        except Exception:
            log.exception("Session %s: couldn't discard queued recognition work", sid)

    def _teardown(self) -> None:
        if self._recording is not None:
            self._release_stream(self._recording)
        for sid in list(self._sessions):
            self._discard_asr_work(sid)
        self._sessions.clear()
        self._ready.clear()
        self._order.clear()
        log.info("Pipeline stopped")


def _main_language(results: tuple[AsrResult, ...]) -> str | None:
    """The language that covers the most recognised audio."""
    weights: Counter[str] = Counter()
    for result in results:
        if result.language and result.error is None and result.text.strip():
            weights[result.language] += max(1, result.audio_ms)
    return weights.most_common(1)[0][0] if weights else None
