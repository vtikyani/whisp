"""AsrService: the speech recognition worker.

One long-lived thread, ``whisp-asr``, owns the loaded model and runs every
recognition, one at a time. Public methods only queue work and return, so the
GUI thread never waits (``shutdown`` waits a bounded time).

Work is taken in this order:

1. answers for jobs dropped by ``discard`` (cheap, so callers hear back at once);
2. commands: start, settings or installed models changed, ``ensure_loaded``;
3. segments from ``submit``, first in, first out;
4. live previews from ``submit_preview``: only the newest per session is kept;
   a session's queued preview is dropped when one of its segments is submitted,
   and later previews are ignored once its final segment has been.

Every job given to ``submit`` gets exactly one callback: the recognised text, or
an ``AsrResult`` with ``error`` set when no model can be loaded, the session was
discarded, recognition failed, or Whisp is shutting down. Callbacks run on
``whisp-asr``, except for jobs still queued when ``shutdown`` is called, which
are answered on the calling thread right away. Preview callbacks are best
effort: at most one each, and only with real text (a dropped or failed preview
just keeps the previous one on screen).

The model loads on demand: ``start`` loads the configured model if it's
installed, and jobs wait while it loads. Changing ``transcription.model_id`` or
``transcription.device`` reloads it; ``transcription.language`` is read for every
job, which has the same effect as reloading without the cost. After
``transcription.unload_after_minutes`` without work the model is unloaded
(0 = never) and loads again with the next job.

Which files and device: with device ``auto``, the GPU (fp32) files on CUDA if
``detect_gpu().cuda_ready`` and they're installed; otherwise the int8 ``cpu``
variant if installed; otherwise the fp32 files on the CPU. ``gpu`` and ``cpu``
force the device; the engine falls back to the CPU with an explanation if the GPU
can't be used. If the GPU fails in the middle of a recognition, the model is
reloaded on the CPU for the rest of the run, since a CUDA error usually leaves
the GPU unusable for the process, and that segment is recognised again there.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, Signal

from whisp.core.asr.downloads import DownloadCancelled, DownloadError, ProgressCallback
from whisp.core.asr.engine import CPU, CUDA, EngineError, EngineFactory, EngineSpec, load_engine
from whisp.core.asr.gpu import GpuInfo, detect_gpu, install_cuda_runtime, refresh_gpu
from whisp.core.types import SAMPLE_RATE, AsrResult, EngineState, EngineStatus, Segment

if TYPE_CHECKING:
    from whisp.core.asr.catalog import ModelInfo
    from whisp.core.asr.engine import AsrEngine
    from whisp.core.asr.manager import ModelManager
    from whisp.storage.settings import SettingsStore

log = logging.getLogger(__name__)

SHUTDOWN_WAIT_S = 2.0
"""Longest ``shutdown`` waits for the worker (it may be in the middle of a long recognition or a load)."""
FINISHED_SESSIONS_KEPT = 256
"""Recently finished or discarded sessions remembered so their late previews are ignored."""
IDLE_CHECK_S = 5.0
"""While waiting to unload an idle model, the deadline is checked again at least this often."""

MSG_NOT_DOWNLOADED = "{name} isn't downloaded yet. Download it in Settings to start dictating."
MSG_UNKNOWN_MODEL = "The chosen speech model isn't available in this version of Whisp. Pick one in Settings."
MSG_LOADING = "Loading {name}…"
MSG_LOAD_FAILED = "The speech model couldn't be loaded. Delete it and download it again in Settings."
MSG_RECOGNITION_FAILED = "Speech recognition failed for part of this dictation."
MSG_CANCELLED = "Cancelled."
MSG_CLOSING = "Whisp is closing."
MSG_GPU_LOST = "The graphics card stopped working with Whisp, so it uses the CPU until you restart Whisp."
MSG_GPU_PACK_FAILED = "Couldn't set up GPU acceleration. Whisp keeps using the CPU. Details are in the log."

GpuInstaller = Callable[[ProgressCallback, threading.Event], None]
"""Downloads and installs the GPU runtime pack (``gpu.install_cuda_runtime``); blocking."""


@dataclass(eq=False)
class _SegmentJob:
    segment: Segment
    callback: Callable[[AsrResult], None]
    _claimed: threading.Lock = field(default_factory=threading.Lock)

    @property
    def audio_ms(self) -> int:
        return int(np.asarray(self.segment.samples).size * 1000 // SAMPLE_RATE)

    def answer(self, result: AsrResult) -> None:
        """Call the callback, once. A second answer is a bug; it's logged and ignored."""
        if not self._claimed.acquire(blocking=False):
            log.error("Segment %d of session %s was answered twice", self.segment.index, self.segment.session_id)
            return
        try:
            self.callback(result)
        except Exception:
            log.exception("A speech recognition callback failed")

    def fail(self, message: str) -> None:
        segment = self.segment
        self.answer(AsrResult(segment.session_id, segment.index, "", self.audio_ms, 0, error=message))


@dataclass(eq=False)
class _PreviewJob:
    session_id: str
    samples: np.ndarray
    callback: Callable[[str], None]


class _Reason(enum.Enum):
    START = "start"
    ENSURE = "ensure"
    SETTINGS = "settings"
    MODELS = "models"
    RELOAD = "reload"


@dataclass(frozen=True)
class _Refresh:
    reason: _Reason


@dataclass(frozen=True)
class _Shutdown:
    pass


@dataclass(frozen=True)
class _Plan:
    """What should be loaded for the current settings and installed files."""

    info: ModelInfo
    folders: dict[str, Path]
    use_cuda: bool

    @property
    def key(self) -> tuple[str, tuple[str, ...], bool]:
        return self.info.id, tuple(sorted(self.folders)), self.use_cuda


class AsrService(QObject):
    """Runs speech recognition on its own thread. See the module docstring."""

    status_changed = Signal(object)
    """EngineStatus, on every transition (emitted from ``whisp-asr``)."""
    gpu_runtime_progress = Signal(object, object)
    """bytes_done, bytes_total while the GPU runtime pack downloads (emitted from its thread)."""
    gpu_runtime_finished = Signal(object)
    """GpuInfo once the GPU runtime pack is installed; the model then reloads onto the GPU."""
    gpu_runtime_failed = Signal(str)
    """A short message for the user, or "" when the user cancelled."""

    def __init__(
        self,
        settings: SettingsStore,
        models: ModelManager,
        *,
        engine_factory: EngineFactory = load_engine,
        gpu_info: Callable[[], GpuInfo] = detect_gpu,
        gpu_installer: GpuInstaller = install_cuda_runtime,
        gpu_refresh: Callable[[], GpuInfo] = refresh_gpu,
        clock: Callable[[], float] = time.monotonic,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._models = models
        self._factory = engine_factory
        self._gpu_info = gpu_info
        self._gpu_installer = gpu_installer
        self._gpu_refresh = gpu_refresh
        self._gpu_cancel: threading.Event | None = None
        """Set while the GPU runtime pack downloads (guarded by _cond)."""
        self._clock = clock
        self._cond = threading.Condition(threading.Lock())
        # Guarded by _cond:
        self._segments: deque[_SegmentJob] = deque()
        self._previews: OrderedDict[str, _PreviewJob] = OrderedDict()
        self._finished: OrderedDict[str, None] = OrderedDict()
        self._discarded: list[_SegmentJob] = []
        self._commands: deque[_Refresh | _Shutdown] = deque()
        self._closed = False
        self._status = self._initial_status()
        self._thread: threading.Thread | None = None
        # Owned by the worker thread:
        self._engine: AsrEngine | None = None
        self._loaded_key: tuple[str, tuple[str, ...], bool] | None = None
        self._failed_key: tuple[str, tuple[str, ...], bool] | None = None
        self._retained: str | None = None
        self._last_used = clock()
        self._gpu_lost = False
        settings.changed.connect(self._on_settings_changed)
        models.changed.connect(self._on_models_changed)

    # -- public API (any thread) ---------------------------------------------------------------

    def start(self) -> None:
        """Start the worker; it loads the configured model if that's installed. Safe to call twice."""
        with self._cond:
            if self._closed or self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="whisp-asr", daemon=True)
            self._commands.append(_Refresh(_Reason.START))
        self._thread.start()

    def shutdown(self) -> None:
        """Answer queued work, unload the model and stop the worker. Waits up to two seconds."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            if self._gpu_cancel is not None:
                self._gpu_cancel.set()  # an interrupted download resumes next time
            pending = [*self._discarded, *self._segments]
            self._discarded.clear()
            self._segments.clear()
            self._previews.clear()
            self._commands.clear()
            self._commands.append(_Shutdown())
            self._cond.notify_all()
            thread = self._thread
        for job in pending:
            job.fail(MSG_CLOSING)
        if thread is None or thread is threading.current_thread():
            return
        thread.join(SHUTDOWN_WAIT_S)
        if thread.is_alive():
            log.warning("Speech recognition didn't stop within %.1f s", SHUTDOWN_WAIT_S)

    @property
    def status(self) -> EngineStatus:
        with self._cond:
            return self._status

    def ensure_loaded(self) -> None:
        """Load the configured model in the background, checking again what is installed. Returns at once.

        Does nothing while the model is ready or loading. Retries a model that failed to load.
        """
        with self._cond:
            if self._closed or self._status.state in (EngineState.READY, EngineState.LOADING):
                return
            self._queue_command(_Refresh(_Reason.ENSURE))

    def reload(self) -> None:
        """Choose the model, variant and device again now, e.g. after the GPU runtime was installed."""
        with self._cond:
            if not self._closed:
                self._queue_command(_Refresh(_Reason.RELOAD))

    @property
    def gpu_runtime_installing(self) -> bool:
        with self._cond:
            return self._gpu_cancel is not None

    def install_gpu_runtime(self) -> bool:
        """Download the GPU runtime pack in the background, then move the model onto the GPU.

        Progress and the outcome arrive as ``gpu_runtime_*`` signals. Returns False if a
        download is already running or Whisp is closing. Any thread.
        """
        with self._cond:
            if self._closed or self._gpu_cancel is not None:
                return False
            cancel = threading.Event()
            self._gpu_cancel = cancel
        thread = threading.Thread(target=self._install_gpu_runtime, args=(cancel,), name="whisp-gpu-pack", daemon=True)
        thread.start()
        return True

    def cancel_gpu_runtime(self) -> None:
        """Stop the GPU runtime download; what's done so far is kept for next time. Any thread."""
        with self._cond:
            if self._gpu_cancel is not None:
                self._gpu_cancel.set()

    def _install_gpu_runtime(self, cancel: threading.Event) -> None:
        try:
            self._gpu_installer(lambda done, total: self.gpu_runtime_progress.emit(done, total), cancel)
            info = self._gpu_refresh()
        except DownloadCancelled:
            log.info("GPU runtime download cancelled")
            self.gpu_runtime_failed.emit("")
            return
        except DownloadError as exc:
            log.warning("GPU runtime download failed: %s", exc)
            self.gpu_runtime_failed.emit(str(exc))
            return
        except Exception:
            log.exception("Installing the GPU runtime pack failed")
            self.gpu_runtime_failed.emit(MSG_GPU_PACK_FAILED)
            return
        finally:
            with self._cond:
                if self._gpu_cancel is cancel:
                    self._gpu_cancel = None
        log.info("GPU runtime pack ready (%s)", "CUDA works" if info.cuda_ready else info.detail)
        self.reload()
        self.gpu_runtime_finished.emit(info)

    def submit(self, segment: Segment, callback: Callable[[AsrResult], None]) -> None:
        """Queue a segment. ``callback`` gets exactly one ``AsrResult`` (see the module docstring)."""
        job = _SegmentJob(segment, callback)
        with self._cond:
            closed = self._closed
            if not closed:
                self._segments.append(job)
                # A queued preview of this session shows speech this segment now carries. Segments run
                # first, so its text would arrive too late to be shown: don't spend recognition time on it.
                self._previews.pop(segment.session_id, None)
                if segment.is_final:
                    self._remember_finished(segment.session_id)
                self._cond.notify()
        if closed:
            job.fail(MSG_CLOSING)

    def submit_preview(self, session_id: str, samples: np.ndarray, callback: Callable[[str], None]) -> None:
        """Queue a live preview at the lowest priority, replacing any queued one for the session."""
        with self._cond:
            if self._closed or session_id in self._finished:
                return
            self._previews.pop(session_id, None)
            self._previews[session_id] = _PreviewJob(session_id, samples, callback)
            self._cond.notify()

    def discard(self, session_id: str) -> None:
        """Drop a cancelled session's queued work. Its segments are answered with an error, previews not at all."""
        with self._cond:
            if self._closed:
                return
            kept: deque[_SegmentJob] = deque()
            for job in self._segments:
                (self._discarded if job.segment.session_id == session_id else kept).append(job)
            self._segments = kept
            self._previews.pop(session_id, None)
            self._remember_finished(session_id)
            self._cond.notify()

    def _initial_status(self) -> EngineStatus:
        """Whether the configured model is installed, known before the worker runs (a cheap file check)."""
        model_id = self._settings.current.transcription.model_id
        try:
            info = self._models.model_info(model_id)
            installed = self._models.installed_variants(model_id)
        except KeyError:
            return EngineStatus(EngineState.NOT_INSTALLED, model_id, message=MSG_UNKNOWN_MODEL)
        if not installed:
            return EngineStatus(EngineState.NOT_INSTALLED, model_id, message=MSG_NOT_DOWNLOADED.format(name=info.name))
        return EngineStatus(EngineState.UNLOADED, model_id)

    # -- signal handlers (GUI thread) --------------------------------------------------------------

    def _on_settings_changed(self, sections: object) -> None:
        if isinstance(sections, set | frozenset | list | tuple) and "transcription" not in sections:
            return
        with self._cond:
            if not self._closed and self._thread is not None:
                self._queue_command(_Refresh(_Reason.SETTINGS))

    def _on_models_changed(self) -> None:
        with self._cond:
            if not self._closed and self._thread is not None:
                self._queue_command(_Refresh(_Reason.MODELS))

    # -- queue helpers (caller holds _cond) ---------------------------------------------------------

    def _queue_command(self, command: _Refresh) -> None:
        if command not in self._commands:
            self._commands.append(command)
            self._cond.notify()

    def _remember_finished(self, session_id: str) -> None:
        self._finished[session_id] = None
        self._finished.move_to_end(session_id)
        while len(self._finished) > FINISHED_SESSIONS_KEPT:
            self._finished.popitem(last=False)

    # -- worker thread ----------------------------------------------------------------------------------

    def _run(self) -> None:
        log.info("Speech recognition worker started")
        try:
            while True:
                kind, item = self._next()
                try:
                    if kind == "discarded":
                        for job in item:
                            job.fail(MSG_CANCELLED)
                    elif kind == "command":
                        if isinstance(item, _Shutdown):
                            break
                        self._refresh(item.reason)
                    elif kind == "segment":
                        self._recognize(item)
                    elif kind == "preview":
                        self._preview(item)
                    elif kind == "idle":
                        self._unload_idle()
                except Exception:
                    log.exception("Unexpected error in the speech recognition worker")
                    if kind == "segment":
                        item.fail(MSG_RECOGNITION_FAILED)
        finally:
            self._unload()
            log.info("Speech recognition worker stopped")

    def _next(self) -> tuple[str, object]:
        with self._cond:
            while True:
                if self._discarded:
                    jobs, self._discarded = self._discarded, []
                    return "discarded", jobs
                if self._commands:
                    return "command", self._commands.popleft()
                if self._segments:
                    return "segment", self._segments.popleft()
                if self._previews:
                    return "preview", self._previews.popitem(last=False)[1]
                timeout = self._idle_timeout()
                if timeout is not None and timeout <= 0:
                    return "idle", None
                self._cond.wait(None if timeout is None else min(timeout, IDLE_CHECK_S))

    def _idle_timeout(self) -> float | None:
        minutes = self._settings.current.transcription.unload_after_minutes
        if self._engine is None or minutes <= 0:
            return None
        return self._last_used + minutes * 60.0 - self._clock()

    def _refresh(self, reason: _Reason) -> None:
        if reason in (_Reason.ENSURE, _Reason.SETTINGS, _Reason.MODELS, _Reason.RELOAD):
            self._failed_key = None
        plan = self._plan()
        if plan is None:
            self._unload()
            return
        if self._engine is not None:
            if plan.key != self._loaded_key or reason is _Reason.RELOAD:
                log.info("Speech model settings changed (%s); reloading", reason.value)
                self._unload()
                self._load(plan)
            return
        state = self.status.state
        eager = reason in (_Reason.START, _Reason.ENSURE, _Reason.RELOAD) or (
            reason is _Reason.MODELS and state in (EngineState.NOT_INSTALLED, EngineState.ERROR)
        )
        if eager:
            self._load(plan)
        elif state is not EngineState.LOADING:
            self._set_status(EngineStatus(EngineState.UNLOADED, plan.info.id))

    def _plan(self) -> _Plan | None:
        """What to load now. None (with the status set to say why) if nothing can be."""
        transcription = self._settings.current.transcription
        model_id = transcription.model_id
        try:
            info = self._models.model_info(model_id)
            installed = self._models.installed_variants(model_id)
        except KeyError:
            log.warning("Speech model %r isn't in the catalog", model_id)
            self._set_status(EngineStatus(EngineState.NOT_INSTALLED, model_id, message=MSG_UNKNOWN_MODEL))
            return None
        if not installed:
            message = MSG_NOT_DOWNLOADED.format(name=info.name)
            self._set_status(EngineStatus(EngineState.NOT_INSTALLED, model_id, message=message))
            return None
        device = transcription.device
        use_cuda = False
        if device == "gpu" and not self._gpu_lost:
            use_cuda = True
        elif device == "auto" and "gpu" in installed and not self._gpu_lost:
            use_cuda = self._gpu_info().cuda_ready
        folders = {variant: self._models.model_path(model_id, variant) for variant in installed}
        return _Plan(info, folders, use_cuda)

    def _ensure_engine(self) -> AsrEngine | None:
        plan = self._plan()
        if plan is None:
            self._unload()
            return None
        if self._engine is not None and plan.key == self._loaded_key:
            return self._engine
        if self._engine is not None:
            self._unload()
        if plan.key == self._failed_key:
            return None
        return self._load(plan)

    def _load(self, plan: _Plan) -> AsrEngine | None:
        info = plan.info
        device, device_name = (CUDA, self._gpu_info().name or "NVIDIA GPU") if plan.use_cuda else (CPU, "CPU")
        self._set_status(
            EngineStatus(EngineState.LOADING, info.id, device, device_name, MSG_LOADING.format(name=info.name))
        )
        if not self._models.retain(info.id, next(iter(plan.folders))):
            message = MSG_NOT_DOWNLOADED.format(name=info.name)
            self._set_status(EngineStatus(EngineState.NOT_INSTALLED, info.id, message=message))
            return None
        started = time.perf_counter()
        spec = EngineSpec(model=info, folders=plan.folders, use_cuda=plan.use_cuda)
        try:
            engine = self._factory(spec)
        except EngineError as exc:
            self._models.release(info.id)
            self._failed_key = plan.key
            log.warning("Couldn't load %s: %s", info.id, exc)
            self._set_status(EngineStatus(EngineState.ERROR, info.id, message=str(exc)))
            return None
        except Exception:
            self._models.release(info.id)
            self._failed_key = plan.key
            log.exception("Couldn't load %s", info.id)
            self._set_status(EngineStatus(EngineState.ERROR, info.id, message=MSG_LOAD_FAILED))
            return None
        self._engine = engine
        self._loaded_key = plan.key
        self._retained = info.id
        self._last_used = self._clock()
        message = engine.notice or (MSG_GPU_LOST if self._gpu_lost and self._settings_device() != CPU else None)
        log.info(
            "Speech model %s ready on %s (%s variant) after %.1f s",
            info.id,
            engine.device_name,
            engine.variant,
            time.perf_counter() - started,
        )
        self._set_status(EngineStatus(EngineState.READY, info.id, engine.device, engine.device_name, message))
        return engine

    def _unload(self) -> None:
        engine, self._engine = self._engine, None
        self._loaded_key = None
        if engine is not None:
            try:
                engine.close()
            except Exception:
                log.exception("Closing the speech model failed")
        if self._retained is not None:
            self._models.release(self._retained)
            self._retained = None

    def _unload_idle(self) -> None:
        if self._engine is None:
            return
        model_id = self._retained
        minutes = self._settings.current.transcription.unload_after_minutes
        log.info("Unloading the speech model after %d idle minute(s)", minutes)
        self._unload()
        self._set_status(EngineStatus(EngineState.UNLOADED, model_id))

    def _recognize(self, job: _SegmentJob) -> None:
        segment = job.segment
        while True:
            engine = self._ensure_engine()
            if engine is None:
                job.fail(self.status.message or MSG_LOAD_FAILED)
                return
            language = self._settings.current.transcription.language
            started = time.perf_counter()
            try:
                text, detected = engine.transcribe(np.asarray(segment.samples, dtype=np.float32), language)
            except Exception:
                log.exception("Recognising segment %d of session %s failed", segment.index, segment.session_id)
                if not self._after_failure(engine):
                    job.fail(MSG_RECOGNITION_FAILED)
                    return
                # The GPU was given up (at most once per run): these words get a second try on the CPU.
                log.info("Recognising segment %d of session %s again on the CPU", segment.index, segment.session_id)
                continue
            finally:
                self._last_used = self._clock()
            break
        took_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "Session %s segment %d: %d ms of audio recognised in %d ms on %s (%d words)",
            segment.session_id,
            segment.index,
            job.audio_ms,
            took_ms,
            engine.device,
            len(text.split()),
        )
        job.answer(AsrResult(segment.session_id, segment.index, text, job.audio_ms, took_ms, language=detected))

    def _preview(self, job: _PreviewJob) -> None:
        engine = self._ensure_engine()
        if engine is None:
            return
        language = self._settings.current.transcription.language
        started = time.perf_counter()
        try:
            text, _ = engine.transcribe(np.asarray(job.samples, dtype=np.float32), language)
        except Exception:
            log.exception("Live preview for session %s failed", job.session_id)
            self._after_failure(engine)
            return
        finally:
            self._last_used = self._clock()
        log.debug("Preview for session %s in %d ms", job.session_id, int((time.perf_counter() - started) * 1000))
        if not text:
            return
        try:
            job.callback(text)
        except Exception:
            log.exception("A live preview callback failed")

    def _after_failure(self, engine: AsrEngine) -> bool:
        """After a failed recognition: give up the GPU if that's where it failed. True if it was given up now."""
        if engine.device != CUDA or self._gpu_lost:
            return False
        log.error("Speech recognition failed on the GPU; switching to the CPU until Whisp restarts")
        self._gpu_lost = True
        self._unload()
        self._set_status(EngineStatus(EngineState.UNLOADED, self._settings.current.transcription.model_id))
        return True

    def _settings_device(self) -> str:
        return self._settings.current.transcription.device

    def _set_status(self, status: EngineStatus) -> None:
        with self._cond:
            if status == self._status:
                return
            self._status = status
        log.info(
            "Speech engine: %s%s%s",
            status.state.value,
            f" ({status.model_id} on {status.device_name})" if status.state is EngineState.READY else "",
            f": {status.message}" if status.message else "",
        )
        try:
            self.status_changed.emit(status)
        except RuntimeError:
            log.debug("Speech service is gone; dropping a status update")
