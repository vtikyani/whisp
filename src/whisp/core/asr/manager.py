"""ModelManager: install, check, locate and delete speech models.

Layout on disk::

    <models_dir>/<model_id>/<variant>/<repo files...>
    <models_dir>/<model_id>/<variant>/whisp-manifest.json

The manifest is written last, after every file has been downloaded and
verified, so a folder without one (an interrupted or cancelled download) is
never treated as installed. That matters because onnx-asr assumes any folder
it is given is complete. ``is_installed`` checks the manifest against the
catalog and the file sizes on disk; it is cheap enough to call from any
thread, including the GUI thread.

Each download runs on its own short-lived daemon thread and reports through
``progress``, ``changed`` and ``download_failed``. Cancelling a download
discards its partial files; a download interrupted by quitting Whisp or a
network failure keeps them, and the next attempt resumes.

The speech engine marks the model it loads with ``retain`` and ``release``.
``delete`` refuses to remove a model that is in use, and ``retain`` refuses a
model that is being deleted, so the two can never race.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from PySide6.QtCore import QObject, Signal

from whisp.core.asr.catalog import CATALOG, ModelInfo, ModelVariant
from whisp.core.asr.downloads import (
    HF_BASE_URL,
    DownloadCancelled,
    DownloadError,
    DownloadItem,
    download_files,
    hf_resolve_url,
    make_client,
)

log = logging.getLogger(__name__)

MANIFEST_NAME = "whisp-manifest.json"
MANIFEST_FORMAT = 1
DELETE_WAIT_S = 5.0
"""How long ``delete`` waits for a cancelled download of the same model to stop."""

MSG_DOWNLOAD_FAILED = "The download didn't work. Please try again."
MSG_IN_USE = "Whisp is using this model right now. Choose a different model first, then try again."
MSG_BUSY = "This model is still being downloaded. Try again in a moment."


class ModelInUseError(PermissionError):
    """The model can't be deleted right now. ``str(error)`` is a user-facing message.

    A ``PermissionError`` (an ``OSError``), because to callers it is the same
    situation as Windows refusing to delete a file that is open.
    """


@dataclass(eq=False)
class _Job:
    model_id: str
    variant: str
    cancel: threading.Event = field(default_factory=threading.Event)
    discard: bool = False
    """Cancelled by the user: remove the partial files rather than keep them for resuming."""
    done: int = 0
    total: int = 0
    thread: threading.Thread | None = None


class ModelManager(QObject):
    """Installs and locates speech models. See the module docstring."""

    progress = Signal(str, object, object)
    """model_id, bytes_done, bytes_total (Python ints)."""
    changed = Signal()
    """Something was installed, removed, cancelled or failed."""
    download_failed = Signal(str, str)
    """model_id, user-facing message."""

    def __init__(
        self,
        models_dir: Path,
        *,
        catalog: Sequence[ModelInfo] = CATALOG,
        base_url: str = HF_BASE_URL,
        client_factory: Callable[[], httpx.Client] = make_client,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._models_dir = Path(models_dir)
        self._catalog = {info.id: info for info in catalog}
        self._base_url = base_url
        self._client_factory = client_factory
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._in_use: dict[str, int] = {}
        self._deleting: set[str] = set()
        self._warned: set[str] = set()

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    # -- queries (any thread) ---------------------------------------------------------

    def model_info(self, model_id: str) -> ModelInfo:
        """The catalog entry this manager uses for ``model_id``. Raises ``KeyError`` for unknown ids."""
        return self._model(model_id)

    def is_installed(self, model_id: str, variant: str) -> bool:
        """True if every file of this variant is in place, as recorded by a complete download.

        Raises ``KeyError`` for a model that isn't in the catalog or a variant it doesn't have.
        """
        info, spec = self._lookup(model_id, variant)
        return self._check_installed(info, spec)

    def model_path(self, model_id: str, variant: str) -> Path:
        """Folder holding (or that will hold) this variant's files. Raises ``KeyError`` like ``is_installed``."""
        info, spec = self._lookup(model_id, variant)
        return self._models_dir / info.id / spec.key

    def installed_variants(self, model_id: str) -> list[str]:
        """Keys of the installed variants, in catalog order. Raises ``KeyError`` for unknown models."""
        info = self._model(model_id)
        return [v.key for v in info.variants if self._check_installed(info, v)]

    def install_state(self) -> dict[str, dict[str, Any]]:
        """Per model: ``installed`` (variant keys), ``downloading``, ``variant`` being downloaded,
        ``progress`` (``{"done", "total"}`` or None) and ``in_use``."""
        with self._lock:
            jobs = {model_id: (job.variant, job.done, job.total) for model_id, job in self._jobs.items()}
            in_use = {model_id for model_id, count in self._in_use.items() if count > 0}
        state: dict[str, dict[str, Any]] = {}
        for info in self._catalog.values():
            job = jobs.get(info.id)
            state[info.id] = {
                "installed": [v.key for v in info.variants if self._check_installed(info, v)],
                "downloading": job is not None,
                "variant": job[0] if job else None,
                "progress": {"done": job[1], "total": job[2]} if job else None,
                "in_use": info.id in in_use,
            }
        return state

    def is_downloading(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._jobs

    # -- downloads ------------------------------------------------------------------------

    def download(self, model_id: str, variant: str) -> None:
        """Start downloading a variant in the background. Returns at once.

        Raises ``KeyError`` for an unknown model or variant. Does nothing if the
        variant is already installed or this model is already downloading.
        """
        info, spec = self._lookup(model_id, variant)
        if self._check_installed(info, spec):
            log.info("%s (%s) is already installed", model_id, variant)
            self.changed.emit()
            return
        with self._lock:
            if model_id in self._deleting:
                log.warning("Not downloading %s: it is being deleted", model_id)
                return
            if model_id in self._jobs:
                log.info("%s is already downloading", model_id)
                return
            job = _Job(model_id, variant, total=spec.size)
            job.thread = threading.Thread(
                target=self._run_download, args=(info, spec, job), name=f"whisp-download-{model_id}", daemon=True
            )
            self._jobs[model_id] = job
        log.info("Downloading %s (%s, %d bytes)", model_id, variant, spec.size)
        job.thread.start()
        self.changed.emit()

    def cancel(self, model_id: str) -> None:
        """Stop a download and discard what it fetched. Returns at once; ``changed`` follows."""
        with self._lock:
            job = self._jobs.get(model_id)
            if job is None:
                return
            job.discard = True
            job.cancel.set()
        log.info("Cancelling the download of %s", model_id)

    # -- deletion -----------------------------------------------------------------------------

    def delete(self, model_id: str) -> None:
        """Remove every installed variant of a model, and any download of it in progress. Blocking.

        Raises ``KeyError`` for unknown models, ``ModelInUseError`` (an ``OSError``) while the
        speech engine uses the model, and ``OSError`` if Windows won't let the files go.
        """
        info = self._model(model_id)
        with self._lock:
            if self._in_use.get(model_id, 0) > 0:
                raise ModelInUseError(MSG_IN_USE)
            if model_id in self._deleting:
                raise ModelInUseError(MSG_BUSY)
            self._deleting.add(model_id)
            job = self._jobs.get(model_id)
            if job is not None:
                job.discard = True
                job.cancel.set()
        try:
            if job is not None and job.thread is not None and job.thread is not threading.current_thread():
                job.thread.join(DELETE_WAIT_S)
                if job.thread.is_alive():
                    raise ModelInUseError(MSG_BUSY)
            folder = self._models_dir / info.id
            # Manifests go first: a model that is only partly deleted must never look installed.
            for variant in info.variants:
                (folder / variant.key / MANIFEST_NAME).unlink(missing_ok=True)
            if folder.exists():
                _rmtree(folder)
            log.info("Deleted speech model %s", model_id)
        finally:
            with self._lock:
                self._deleting.discard(model_id)
        self.changed.emit()

    # -- in-use tracking (the speech engine) --------------------------------------------------

    def retain(self, model_id: str, variant: str) -> bool:
        """Mark a variant as in use before loading it. False if it isn't installed or is being deleted.

        Every successful ``retain`` must be paired with a ``release``.
        """
        info, spec = self._lookup(model_id, variant)
        with self._lock:
            if model_id in self._deleting:
                return False
            if not self._check_installed(info, spec):
                return False
            self._in_use[model_id] = self._in_use.get(model_id, 0) + 1
        return True

    def release(self, model_id: str) -> None:
        """The engine no longer uses this model (after ``retain``)."""
        with self._lock:
            count = self._in_use.get(model_id, 0)
            if count <= 1:
                self._in_use.pop(model_id, None)
            else:
                self._in_use[model_id] = count - 1
            if count == 0:
                log.warning("release(%s) without a matching retain", model_id)

    def in_use(self, model_id: str) -> bool:
        with self._lock:
            return self._in_use.get(model_id, 0) > 0

    # -- internals -------------------------------------------------------------------------------

    def _model(self, model_id: str) -> ModelInfo:
        try:
            return self._catalog[model_id]
        except KeyError:
            raise KeyError(f"unknown speech model: {model_id!r}") from None

    def _lookup(self, model_id: str, variant: str) -> tuple[ModelInfo, ModelVariant]:
        info = self._model(model_id)
        return info, info.variant(variant)

    def _check_installed(self, info: ModelInfo, spec: ModelVariant) -> bool:
        folder = self._models_dir / info.id / spec.key
        try:
            manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
            self._warn_once(f"{info.id} ({spec.key}): unreadable manifest ({exc})")
            return False
        if manifest != _manifest_for(info, spec):
            self._warn_once(f"{info.id} ({spec.key}) on disk is a different version; it needs downloading again")
            return False
        for item in spec.files:
            try:
                if (folder / item.path).stat().st_size != item.size:
                    self._warn_once(f"{info.id} ({spec.key}): {item.path} has the wrong size")
                    return False
            except OSError:
                self._warn_once(f"{info.id} ({spec.key}): {item.path} is missing")
                return False
        return True

    def _warn_once(self, problem: str) -> None:
        """Log a problem with installed files once; ``is_installed`` runs too often to repeat it every time."""
        with self._lock:
            if problem in self._warned:
                return
            self._warned.add(problem)
        log.warning("Speech model files: %s", problem)

    def _run_download(self, info: ModelInfo, spec: ModelVariant, job: _Job) -> None:
        folder = self._models_dir / info.id / spec.key
        started = time.monotonic()
        try:
            (folder / MANIFEST_NAME).unlink(missing_ok=True)  # not installed until every file checks out
            items = [
                DownloadItem(
                    url=hf_resolve_url(info.repo_id, info.revision, f.path, base_url=self._base_url),
                    dest=_inside(folder, f.path),
                    size=f.size,
                    sha256=f.sha256,
                )
                for f in spec.files
            ]
            with self._client_factory() as client:

                def report(done: int, total: int) -> None:
                    self._on_progress(job, done, total)

                download_files(items, progress=report, cancel=job.cancel, client=client)
            _write_manifest(folder / MANIFEST_NAME, _manifest_for(info, spec))
            with self._lock:
                self._warned = {problem for problem in self._warned if not problem.startswith(f"{info.id} ")}
            log.info("Installed %s (%s) in %.0f s", info.id, spec.key, time.monotonic() - started)
        except DownloadCancelled:
            if job.discard:
                log.info("Download of %s (%s) cancelled; removing its partial files", info.id, spec.key)
                self._discard(folder)
            else:
                log.info("Download of %s (%s) stopped; it will resume next time", info.id, spec.key)
        except DownloadError as exc:
            log.warning("Download of %s (%s) failed: %s", info.id, spec.key, exc)
            self._emit(self.download_failed, info.id, str(exc))
        except Exception:
            log.exception("Download of %s (%s) failed", info.id, spec.key)
            self._emit(self.download_failed, info.id, MSG_DOWNLOAD_FAILED)
        finally:
            with self._lock:
                if self._jobs.get(info.id) is job:
                    del self._jobs[info.id]
            self._emit(self.changed)

    def _on_progress(self, job: _Job, done: int, total: int) -> None:
        job.done, job.total = done, total
        if not job.cancel.is_set():
            self._emit(self.progress, job.model_id, done, total)

    @staticmethod
    def _emit(signal: Any, *args: object) -> None:
        """Emit from a download thread, which can outlive the manager when Whisp quits mid-download."""
        try:
            signal.emit(*args)
        except RuntimeError:
            log.debug("Speech model manager is gone; dropping a download notification")

    def _discard(self, folder: Path) -> None:
        try:
            if folder.exists():
                _rmtree(folder)
            parent = folder.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            log.exception("Couldn't remove the partial download in %s", folder)


def _manifest_for(info: ModelInfo, spec: ModelVariant) -> dict[str, Any]:
    return {
        "format": MANIFEST_FORMAT,
        "model_id": info.id,
        "variant": spec.key,
        "repo_id": info.repo_id,
        "revision": info.revision,
        "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in spec.files],
    }


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _inside(folder: Path, relative: str) -> Path:
    """``folder / relative``, refusing anything that would land outside ``folder``."""
    dest = (folder / relative).resolve()
    if not dest.is_relative_to(folder.resolve()):
        raise ValueError(f"model file path escapes its folder: {relative!r}")
    return dest


def _rmtree(folder: Path) -> None:
    def clear_readonly(func: Callable[..., Any], path: str, _exc: BaseException) -> None:
        # Files copied from read-only media keep their read-only flag; clear it and try once more.
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(folder, onexc=clear_readonly)
