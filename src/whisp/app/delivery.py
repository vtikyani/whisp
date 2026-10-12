"""DeliveryService: finished transcripts in, text at the cursor out.

Everything runs on one worker thread, ``whisp-delivery``, strictly in the
order work was submitted, so dictations are inserted in the order they were
spoken. For each session:

    dictation:  context -> text rules -> optional AI polish -> smart spacing
                -> insert -> history -> ``delivered``
    command:    AI rewrite of the selected text -> paste over it -> history
                -> ``delivered``

Words are never thrown away. If a text step fails, the text from the step
before is used. If insertion fails, the text goes to the clipboard. Any other
unexpected error ends that one session as FAILED; the thread keeps running.

Never log transcript text, selections or clipboard contents.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from PySide6.QtCore import QObject, Signal, Slot

from whisp.core.types import (
    AppCategory,
    DeliveryResult,
    InsertionResult,
    KeyActivity,
    Outcome,
    SessionInfo,
    SessionKind,
    TargetWindow,
    Transcript,
)
from whisp.storage.history import HistoryEntry, HistoryStore, count_words
from whisp.storage.settings import Settings, SettingsStore

if TYPE_CHECKING:
    from whisp.core.text.context import TextContext
    from whisp.core.text.polish import Polisher
    from whisp.platform.win32.insert import TextInserter

log = logging.getLogger(__name__)

MSG_COPIED = "Copied — press Ctrl + V to paste."
MSG_COPIED_FOCUS = "The window changed. Copied — press Ctrl + V to paste."
MSG_COPIED_FALLBACK = "Couldn't type here. Copied — press Ctrl + V to paste."
MSG_INSERT_FAILED = "Couldn't insert your text. You'll find it in History."
MSG_INSERT_FAILED_NO_HISTORY = "Couldn't insert your text. Please try again."
MSG_NO_SELECTION = "Select some text first, then hold the command shortcut and say what to change."
MSG_POLISH_NOT_READY = "Command mode needs AI polish. Turn it on in Styles."
MSG_REWRITE_FAILED = "Couldn't change the selected text, so it was left as it was."
MSG_NOTHING_YET = "Nothing to paste yet. Dictate something first."
MSG_COPY_FAILED = "Couldn't copy to the clipboard. Please try again."
MSG_UNEXPECTED = "Something went wrong with that dictation. Please try again."

DEFAULT_SHUTDOWN_TIMEOUT_S = 1.5


class SpacingMemoryLike(Protocol):
    """The parts of ``core.text.spacing.SpacingMemory`` delivery uses."""

    def remember(self, target: TargetWindow | None, text: str) -> None: ...

    def prefix_for(self, target: TargetWindow | None, text: str, mode: str) -> str: ...


@dataclass(frozen=True)
class TextTools:
    """The text-processing functions delivery runs. ``load()`` gives the real ones from ``core.text``."""

    build_context: Callable[[Settings, TargetWindow | None, str | None], TextContext]
    categorize: Callable[[str, str, Mapping[str, str]], AppCategory]
    apply_rules: Callable[[str, TextContext], str]
    spacing: SpacingMemoryLike

    @classmethod
    def load(cls, *, typing_activity: Callable[[int], KeyActivity | None] | None = None) -> TextTools:
        """``typing_activity`` (``HotkeyService.typing_activity``) lets smart spacing see whether the
        user typed in a window between dictations."""
        from whisp.core.text.context import build_context, categorize
        from whisp.core.text.rules import apply_rules
        from whisp.core.text.spacing import SpacingMemory

        return cls(
            build_context=build_context,
            categorize=categorize,
            apply_rules=apply_rules,
            spacing=SpacingMemory(activity=typing_activity),
        )


@dataclass(frozen=True)
class _SessionJob:
    info: SessionInfo
    transcript: Transcript


@dataclass(frozen=True)
class _LastTextJob:
    action: str
    """``"paste"`` into the focused window, or ``"copy"`` to the clipboard."""
    requested_monotonic: float


@dataclass(frozen=True)
class _ForgetLastJob:
    """History was cleared: paste-last and copy-last mustn't bring back text from before."""


_Job = _SessionJob | _LastTextJob | _ForgetLastJob


class DeliveryService(QObject):
    """Processes and inserts text on the ``whisp-delivery`` thread, one job at a time, in submission order."""

    delivered = Signal(object)
    """DeliveryResult, emitted from the delivery thread (Qt queues it to receivers on the GUI thread)."""

    def __init__(
        self,
        settings: SettingsStore,
        history: HistoryStore,
        polisher: Polisher,
        inserter: TextInserter,
        *,
        text_tools: TextTools | None = None,
        get_target: Callable[[], TargetWindow | None] | None = None,
        typing_activity: Callable[[int], KeyActivity | None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if text_tools is None:
            text_tools = TextTools.load(typing_activity=typing_activity)
        if get_target is None:
            from whisp.platform.win32.foreground import get_foreground_target as get_target
        self._settings = settings
        self._history = history
        self._polisher = polisher
        self._inserter = inserter
        self._text = text_tools
        self._get_target = get_target
        self._queue: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._submit_lock = threading.Lock()
        self._closed = False
        self._last_text: str | None = None
        """The most recent text delivered; only touched on the delivery thread."""
        history.cleared.connect(self._on_history_cleared)
        self._thread = threading.Thread(target=self._run, name="whisp-delivery", daemon=True)
        self._thread.start()

    # -- public API (any thread) ---------------------------------------------------

    def submit(self, info: SessionInfo, transcript: Transcript) -> None:
        """Queue a finished session. Sessions are delivered in the order they are submitted."""
        self._enqueue(_SessionJob(info, transcript))

    def paste_last(self) -> None:
        """Insert the most recent text again, into whichever window has focus now."""
        self._enqueue(_LastTextJob("paste", time.monotonic()))

    def copy_last(self) -> None:
        """Put the most recent text on the clipboard."""
        self._enqueue(_LastTextJob("copy", time.monotonic()))

    def shutdown(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_S) -> None:
        """Finish queued work, waiting at most ``timeout`` seconds. Later submissions are ignored."""
        with self._submit_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        if threading.current_thread() is self._thread:
            return
        self._thread.join(max(0.0, timeout))
        if self._thread.is_alive():
            log.warning("Delivery was still busy after %.1f s; leaving it to finish in the background", timeout)

    @Slot()
    def _on_history_cleared(self) -> None:
        # Queued behind the work already submitted, so text dictated before the clear is forgotten too.
        self._enqueue(_ForgetLastJob())

    # -- worker thread ---------------------------------------------------------------

    def _enqueue(self, job: _Job) -> None:
        with self._submit_lock:
            if self._closed:
                log.warning("Delivery is shut down; dropping a %s request", type(job).__name__)
                return
            self._queue.put(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            if isinstance(job, _ForgetLastJob):
                if self._last_text is not None:
                    log.info("History was cleared, so the last text is forgotten too")
                self._last_text = None
                continue
            result = self._process(job)
            try:
                self.delivered.emit(result)
            except RuntimeError:
                # The service object was deleted during shutdown; nobody is listening any more.
                log.debug("Delivery result dropped: the service is gone")

    def _process(self, job: _Job) -> DeliveryResult:
        try:
            if isinstance(job, _SessionJob):
                if job.info.kind is SessionKind.COMMAND:
                    return self._deliver_command(job.info, job.transcript)
                return self._deliver_dictation(job.info, job.transcript)
            return self._deliver_last(job)
        except Exception:
            if isinstance(job, _SessionJob):
                log.exception("Delivery failed for session %s", job.info.session_id)
                return DeliveryResult(
                    session_id=job.info.session_id,
                    kind=job.info.kind,
                    outcome=Outcome.FAILED,
                    final_text="",
                    word_count=0,
                    latency_ms=_elapsed_ms(job.info.stopped_monotonic),
                    target=job.info.target,
                    message=MSG_UNEXPECTED,
                )
            log.exception("Couldn't %s the last text", job.action)
            return DeliveryResult(
                session_id=uuid.uuid4().hex,
                kind=SessionKind.DICTATION,
                outcome=Outcome.FAILED,
                final_text="",
                word_count=0,
                latency_ms=_elapsed_ms(job.requested_monotonic),
                message=MSG_UNEXPECTED,
            )

    # -- dictation -----------------------------------------------------------------------

    def _deliver_dictation(self, info: SessionInfo, transcript: Transcript) -> DeliveryResult:
        settings = self._settings.current
        target = self._categorized(info.target, settings)
        if not transcript.text.strip():
            log.info("Session %s: empty transcript", info.session_id)
            return self._result(info, Outcome.NO_SPEECH, "", 0, target=target)

        ctx = self._context(settings, target, transcript.language)
        text = self._rules(transcript.text, ctx)
        polished = False
        if ctx is not None and text.strip() and self._should_polish(settings, ctx):
            text, polished = self._polish(text, ctx)
        if not text.strip():
            log.info("Session %s: nothing left to insert after the text rules", info.session_id)
            return self._result(info, Outcome.NO_SPEECH, "", 0, target=target)

        prefix = self._spacing_prefix(target, text, settings)
        method = self._insert_method(target, settings)
        insertion = self._insert(prefix + text, target, settings, method)
        outcome, message = self._outcome(insertion, text, prefix)
        latency_ms = _elapsed_ms(info.stopped_monotonic)
        if outcome is Outcome.INSERTED:
            self._remember_spacing(target, prefix + text)
        self._last_text = text
        words = count_words(text)
        self._record(info, transcript, target, text, words, latency_ms, polished, outcome, settings)
        log.info(
            "Session %s: %s (%d words, %d ms, method %s%s)",
            info.session_id,
            outcome.value,
            words,
            latency_ms,
            insertion.method,
            ", polished" if polished else "",
        )
        return self._result(
            info, outcome, text, words, target=target, latency_ms=latency_ms, polished=polished, message=message
        )

    def _should_polish(self, settings: Settings, ctx: TextContext) -> bool:
        if not settings.polish.enabled or ctx.style == "verbatim":
            return False
        return self._polisher_ready()

    def _polish(self, text: str, ctx: TextContext) -> tuple[str, bool]:
        try:
            result = self._polisher.polish(text, ctx)
        except Exception:
            log.exception("AI polish raised; inserting the text without it")
            return text, False
        if result.used and result.text.strip():
            return result.text, True
        if result.error:
            log.warning("AI polish not applied after %d ms: %s", result.elapsed_ms, result.error)
        return text, False

    # -- command mode ----------------------------------------------------------------------

    def _deliver_command(self, info: SessionInfo, transcript: Transcript) -> DeliveryResult:
        settings = self._settings.current
        target = self._categorized(info.target, settings)
        instruction = transcript.text.strip()
        if not instruction:
            log.info("Session %s: empty command", info.session_id)
            return self._result(info, Outcome.NO_SPEECH, "", 0, target=target)
        words = count_words(instruction)
        selection = info.selection or ""
        if not selection.strip():
            log.info("Session %s: command without selected text", info.session_id)
            return self._result(info, Outcome.FAILED, "", words, target=target, message=MSG_NO_SELECTION)
        if not self._polisher_ready():
            log.info("Session %s: command mode needs AI polish, which isn't ready", info.session_id)
            return self._result(info, Outcome.FAILED, "", words, target=target, message=MSG_POLISH_NOT_READY)

        ctx = self._context(settings, target, transcript.language)
        rewritten = self._rewrite(info.session_id, selection, instruction, ctx) if ctx is not None else None
        if rewritten is None:
            return self._result(info, Outcome.FAILED, "", words, target=target, message=MSG_REWRITE_FAILED)

        method = self._insert_method(target, settings)
        insertion = self._insert(rewritten, target, settings, method)
        outcome, message = self._outcome(insertion, rewritten, "")
        latency_ms = _elapsed_ms(info.stopped_monotonic)
        if outcome is Outcome.INSERTED:
            self._remember_spacing(target, rewritten)
        self._last_text = rewritten
        self._record(
            info, transcript, target, rewritten, words, latency_ms, True, outcome, settings, raw_text=instruction
        )
        log.info(
            "Session %s: command %s (selection %d chars -> %d chars, %d ms)",
            info.session_id,
            outcome.value,
            len(selection),
            len(rewritten),
            latency_ms,
        )
        return self._result(
            info, outcome, rewritten, words, target=target, latency_ms=latency_ms, polished=True, message=message
        )

    def _rewrite(self, session_id: str, selection: str, instruction: str, ctx: TextContext) -> str | None:
        try:
            result = self._polisher.rewrite(selection, instruction, ctx)
        except Exception:
            log.exception("Session %s: AI rewrite raised", session_id)
            return None
        if result.used and result.text.strip():
            return result.text
        log.warning(
            "Session %s: AI rewrite not applied after %d ms: %s",
            session_id,
            result.elapsed_ms,
            result.error or "no change returned",
        )
        return None

    # -- paste / copy the last text ----------------------------------------------------

    def _deliver_last(self, job: _LastTextJob) -> DeliveryResult:
        text = self._last_text or self._history_last_text()
        session_id = uuid.uuid4().hex
        if not text:
            return DeliveryResult(
                session_id=session_id,
                kind=SessionKind.DICTATION,
                outcome=Outcome.FAILED,
                final_text="",
                word_count=0,
                latency_ms=_elapsed_ms(job.requested_monotonic),
                message=MSG_NOTHING_YET,
            )
        words = count_words(text)
        if job.action == "copy":
            copied = self._copy(text)
            log.info("Copied the last text to the clipboard: %s", "ok" if copied else "failed")
            return DeliveryResult(
                session_id=session_id,
                kind=SessionKind.DICTATION,
                outcome=Outcome.COPIED if copied else Outcome.FAILED,
                final_text=text,
                word_count=words,
                latency_ms=_elapsed_ms(job.requested_monotonic),
                message=MSG_COPIED if copied else MSG_COPY_FAILED,
            )

        settings = self._settings.current
        target = self._categorized(self._current_target(), settings)
        prefix = self._spacing_prefix(target, text, settings)
        method = self._insert_method(target, settings)
        insertion = self._insert(prefix + text, target, settings, method)
        outcome, message = self._outcome(insertion, text, prefix)
        if outcome is Outcome.INSERTED:
            self._remember_spacing(target, prefix + text)
        latency_ms = _elapsed_ms(job.requested_monotonic)
        log.info("Pasted the last text again: %s (%d words, %d ms)", outcome.value, words, latency_ms)
        return DeliveryResult(
            session_id=session_id,
            kind=SessionKind.DICTATION,
            outcome=outcome,
            final_text=text,
            word_count=words,
            latency_ms=latency_ms,
            target=target,
            message=message,
        )

    # -- steps (each one degrades instead of failing the session) ---------------------

    def _categorized(self, target: TargetWindow | None, settings: Settings) -> TargetWindow | None:
        if target is None or target.category is not AppCategory.OTHER:
            return target
        try:
            category = self._text.categorize(target.process_name, target.title, settings.text.app_categories)
        except Exception:
            log.exception("Couldn't categorise %s", target.process_name)
            return target
        return target if category is target.category else dataclasses.replace(target, category=category)

    def _context(self, settings: Settings, target: TargetWindow | None, language: str | None) -> TextContext | None:
        try:
            return self._text.build_context(settings, target, language)
        except Exception:
            log.exception("Couldn't build the text context; inserting the transcript as heard")
            return None

    def _rules(self, raw: str, ctx: TextContext | None) -> str:
        if ctx is None:
            return raw.strip()
        try:
            return self._text.apply_rules(raw, ctx)
        except Exception:
            log.exception("Text rules failed; inserting the transcript as heard")
            return raw.strip()

    def _polisher_ready(self) -> bool:
        try:
            return bool(self._polisher.is_ready())
        except Exception:
            log.exception("Couldn't check whether AI polish is ready")
            return False

    def _spacing_prefix(self, target: TargetWindow | None, text: str, settings: Settings) -> str:
        try:
            return self._text.spacing.prefix_for(target, text, settings.text.smart_spacing)
        except Exception:
            log.exception("Smart spacing failed; inserting without a leading space")
            return ""

    def _remember_spacing(self, target: TargetWindow | None, inserted: str) -> None:
        try:
            self._text.spacing.remember(target, inserted)
        except Exception:
            log.exception("Couldn't update smart spacing")

    def _insert_method(self, target: TargetWindow | None, settings: Settings) -> str:
        """The user's choice for this app, else what works best for it, else the user's general choice."""
        process = target.process_name.strip().lower() if target is not None else ""
        if process:
            override = settings.insertion.app_methods.get(process)
            if override:
                return override
            try:
                preferred = self._inserter.default_method_for(process)
            except Exception:
                log.exception("Couldn't look up the insertion method for %s", process)
                preferred = None
            if preferred:
                return str(preferred)
        return settings.insertion.method

    def _insert(self, text: str, target: TargetWindow | None, settings: Settings, method: str) -> InsertionResult:
        try:
            return self._inserter.insert(
                text,
                target=target,
                method=method,
                restore_clipboard=settings.insertion.restore_clipboard,
                on_focus_change=settings.insertion.on_focus_change,
            )
        except Exception:
            log.exception("Text insertion raised")
            return InsertionResult(ok=False, method=method, error="insertion raised an exception")

    def _copy(self, text: str) -> bool:
        try:
            return bool(self._inserter.copy_to_clipboard(text))
        except Exception:
            log.exception("Copying to the clipboard raised")
            return False

    def _outcome(self, insertion: InsertionResult, text: str, prefix: str) -> tuple[Outcome, str | None]:
        """Map what the inserter did to an outcome, falling back to the clipboard so words are never lost."""
        if insertion.ok and insertion.method != "clipboard":
            return Outcome.INSERTED, None
        if insertion.ok:
            # Left on the clipboard on purpose (the window changed). Drop our smart-spacing
            # space so it doesn't end up wherever the user pastes it.
            if prefix:
                self._copy(text)
            return Outcome.COPIED, MSG_COPIED_FOCUS if insertion.focus_changed else MSG_COPIED
        log.warning("Insertion failed (method %s): %s", insertion.method, insertion.error or "no reason given")
        if insertion.method != "clipboard" and self._copy(text):
            return Outcome.COPIED, MSG_COPIED_FALLBACK
        history_on = self._settings.current.privacy.history_enabled and self._history.persistent
        return Outcome.FAILED, MSG_INSERT_FAILED if history_on else MSG_INSERT_FAILED_NO_HISTORY

    def _record(
        self,
        info: SessionInfo,
        transcript: Transcript,
        target: TargetWindow | None,
        final_text: str,
        words: int,
        latency_ms: int,
        polished: bool,
        outcome: Outcome,
        settings: Settings,
        *,
        raw_text: str | None = None,
    ) -> None:
        entry = HistoryEntry(
            id=0,
            created_at=info.started_at,
            kind=info.kind.value,
            app_name=target.app_name if target is not None else "",
            process_name=target.process_name if target is not None else "",
            category=target.category.value if target is not None else AppCategory.OTHER.value,
            raw_text=transcript.text if raw_text is None else raw_text,
            final_text=final_text,
            word_count=words,
            audio_ms=transcript.audio_ms,
            speech_ms=transcript.speech_ms,
            latency_ms=latency_ms,
            model_id=settings.transcription.model_id,
            polished=polished,
            outcome=outcome.value,
        )
        try:
            self._history.add(entry)
        except Exception:
            log.exception("Couldn't save session %s to history", info.session_id)

    def _history_last_text(self) -> str | None:
        try:
            return self._history.last_text()
        except Exception:
            log.exception("Couldn't read the last text from history")
            return None

    def _current_target(self) -> TargetWindow | None:
        try:
            return self._get_target()
        except Exception:
            log.exception("Couldn't read the focused window")
            return None

    @staticmethod
    def _result(
        info: SessionInfo,
        outcome: Outcome,
        text: str,
        words: int,
        *,
        target: TargetWindow | None,
        latency_ms: int | None = None,
        polished: bool = False,
        message: str | None = None,
    ) -> DeliveryResult:
        return DeliveryResult(
            session_id=info.session_id,
            kind=info.kind,
            outcome=outcome,
            final_text=text,
            word_count=words,
            latency_ms=_elapsed_ms(info.stopped_monotonic) if latency_ms is None else latency_ms,
            polished=polished,
            target=target,
            message=message,
        )


def _elapsed_ms(since_monotonic: float) -> int:
    return max(0, round((time.monotonic() - since_monotonic) * 1000))
