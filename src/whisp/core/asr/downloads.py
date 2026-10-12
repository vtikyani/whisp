"""Resumable, verified HTTP downloads with progress.

Every file is streamed to ``<name>.part`` next to its destination. An interrupted
download resumes where it stopped with an HTTP ``Range`` request (the bytes
already on disk are hashed first, so the final checksum covers the whole file).
When a file is complete its size and SHA-256 are checked, it is flushed to disk
and renamed into place atomically: a destination file either doesn't exist or is
exactly the expected file.

Transient problems (dropped connections, timeouts, server errors, rate limits)
are retried with exponential backoff. A file that keeps making progress is
retried for as long as it does; one that fails ``MAX_ATTEMPTS`` times in a row
without receiving a byte gives up. Failures raise ``DownloadError`` with a short,
plain message for the user; setting the caller's cancel event raises
``DownloadCancelled`` (partial files are kept so a later download resumes).

Hugging Face files come from ``https://huggingface.co/<repo>/resolve/<revision>/<path>``
with a pinned revision; no Hugging Face client library is involved.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import random
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import httpx

from whisp import APP_NAME, __version__
from whisp.core.net import is_certificate_error, tls_context

log = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co"
PROGRESS_INTERVAL_S = 0.1
"""Progress callbacks are throttled to about ten a second; the final one always comes."""
DISK_MARGIN_BYTES = 512 * 1024 * 1024
"""Free space that must remain after a download, so finishing one never fills the disk to the brim."""
MAX_ATTEMPTS = 5
"""Consecutive failed attempts (without receiving any data) before a file gives up."""
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 30.0
RETRY_AFTER_MAX_S = 60.0
CONNECT_TIMEOUT_S = 15.0
READ_TIMEOUT_S = 60.0

MSG_OFFLINE = "Couldn't reach the download server. Check your internet connection and try again."
MSG_CERTIFICATE = (
    "The download server's security certificate couldn't be verified, so nothing was downloaded. "
    "On a work network, ask IT to add the network's certificate to Windows."
)
MSG_STALLED = "The download stopped responding. Check your internet connection and try again."
MSG_DROPPED = "The connection dropped during the download. Check your internet connection and try again."
MSG_SERVER = "The download server isn't working right now (error {status}). Please try again later."
MSG_SERVER_GENERIC = "The download server isn't working right now. Please try again later."
MSG_UNAVAILABLE = "The file isn't available on the download server (error {status}). Please try again later."
MSG_CHECKSUM = "A downloaded file was damaged on the way. Please try again."
MSG_NO_SPACE = "There isn't enough free disk space. Free up at least {amount} on {drive} and try again."
MSG_DISK_FILLED = "The disk filled up during the download. Free up some space and try again."
MSG_WRITE = "Whisp couldn't save the download to {folder}. Check that the drive is available and try again."

_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_DISK_FULL_WINERRORS = frozenset({39, 112})  # ERROR_HANDLE_DISK_FULL, ERROR_DISK_FULL
_PART_SUFFIX = ".part"
_HASH_BLOCK = 4 << 20


class DownloadError(Exception):
    """A download failed for good. ``str(error)`` is a short message for the user."""


class DownloadCancelled(Exception):
    """The caller's cancel event was set."""


@dataclass(frozen=True)
class DownloadItem:
    """One file to fetch."""

    url: str
    dest: Path
    size: int
    """Expected size in bytes."""
    sha256: str
    """Expected SHA-256, lower-case hex."""


ProgressCallback = Callable[[int, int], None]
"""Called as ``progress(bytes_done, bytes_total)``, counted across all files of one download."""


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


def hf_resolve_url(repo_id: str, revision: str, path: str, *, base_url: str = HF_BASE_URL) -> str:
    """Direct download URL of ``path`` in a Hugging Face model repo at a pinned revision."""
    return f"{base_url.rstrip('/')}/{repo_id}/resolve/{revision}/{quote(path)}"


def make_client() -> httpx.Client:
    """An HTTP client suited to large downloads: follows redirects, no compression, sane timeouts."""
    return httpx.Client(
        verify=tls_context(),
        follow_redirects=True,
        timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        # Byte ranges must refer to the file itself, never to a compressed rendition of it.
        headers={"User-Agent": f"{APP_NAME}/{__version__}", "Accept-Encoding": "identity"},
    )


def part_path(dest: Path) -> Path:
    """Where a file is assembled while it downloads."""
    return dest.with_name(dest.name + _PART_SUFFIX)


def format_size(num_bytes: int) -> str:
    """``1395864371`` -> ``"1.3 GB"`` (1024-based, as Windows Explorer shows sizes)."""
    value = float(max(0, num_bytes))
    if value < 1024:
        return f"{int(value)} bytes"
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if value < 100 else f"{value:.0f} {unit}"
    raise AssertionError("unreachable")


def remaining_bytes(items: Sequence[DownloadItem]) -> int:
    """Bytes still to fetch. Complete files and the partial data already on disk don't count."""
    total = 0
    for item in items:
        if _file_size(item.dest) == item.size:
            continue
        have = _file_size(part_path(item.dest))
        total += item.size - have if 0 <= have <= item.size else item.size
    return total


def check_free_space(items: Sequence[DownloadItem], *, margin: int = DISK_MARGIN_BYTES) -> None:
    """Raise ``DownloadError`` if a drive can't hold what's left to download, plus a safety margin."""
    needed: dict[str, int] = {}
    probe: dict[str, Path] = {}
    for item in items:
        folder = _existing_ancestor(item.dest.parent)
        volume = folder.anchor or str(folder)
        needed[volume] = needed.get(volume, 0) + remaining_bytes([item])
        probe.setdefault(volume, folder)
    for volume, amount in needed.items():
        if amount == 0:
            continue
        try:
            free = shutil.disk_usage(probe[volume]).free
        except OSError:
            log.warning("Couldn't read the free space on %s; downloading anyway", volume)
            continue
        if free < amount + margin:
            log.warning("Not enough disk space on %s: %d bytes free, %d needed", volume, free, amount + margin)
            drive = Path(volume).drive or volume
            raise DownloadError(MSG_NO_SPACE.format(amount=format_size(amount + margin - free), drive=drive))


def download_files(
    items: Sequence[DownloadItem],
    *,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    client: httpx.Client | None = None,
    check_space: bool = True,
) -> None:
    """Fetch ``items`` in order, resuming and verifying each. Blocking: call it from a worker thread.

    Files already in place with the right size and checksum are kept. Raises
    ``DownloadError`` (with a user-facing message) or ``DownloadCancelled``.
    """
    cancel = cancel if cancel is not None else threading.Event()
    if cancel.is_set():
        raise DownloadCancelled
    if check_space:
        check_free_space(items)
    reporter = _Progress(progress, sum(item.size for item in items))
    http = client if client is not None else make_client()
    started = time.monotonic()
    try:
        for item in items:
            _download_one(http, item, reporter, cancel)
            reporter.complete_file(item.size)
    finally:
        if client is None:
            http.close()
    reporter.finish()
    took = time.monotonic() - started
    log.info("Downloaded %d file(s), %s, in %.1f s", len(items), format_size(reporter.total), took)


def sha256_file(path: Path, cancel: threading.Event | None = None) -> str:
    """SHA-256 of a file (lower-case hex). Checks ``cancel`` between blocks."""
    digest = hashlib.sha256()
    _hash_into(digest, path, cancel)
    return digest.hexdigest()


# -- internals -----------------------------------------------------------------------


class _Progress:
    """Turns per-file byte counts into throttled ``progress(done, total)`` calls."""

    def __init__(self, callback: ProgressCallback | None, total: int) -> None:
        self._callback = callback
        self.total = total
        self._finished = 0
        self._current = 0
        self._last_emit = 0.0
        self._last_done = -1

    def file_bytes(self, done_in_file: int) -> None:
        self._current = done_in_file
        self._emit(force=False)

    def complete_file(self, size: int) -> None:
        self._finished += size
        self._current = 0
        self._emit(force=False)

    def finish(self) -> None:
        self._emit(force=True)

    def _emit(self, *, force: bool) -> None:
        if self._callback is None:
            return
        done = min(self.total, self._finished + self._current)
        now = time.monotonic()
        if not force and (done == self._last_done or now - self._last_emit < PROGRESS_INTERVAL_S):
            return
        self._last_emit = now
        self._last_done = done
        self._callback(done, self.total)


class _Retry(Exception):
    """A transient failure: worth trying again after a pause."""

    def __init__(self, message: str, *, progressed: bool, delay: float | None = None) -> None:
        super().__init__(message)
        self.progressed = progressed
        self.delay = delay


class _Restart(Exception):
    """The partial file can't be continued; this file starts again from the beginning."""


def _download_one(client: httpx.Client, item: DownloadItem, reporter: _Progress, cancel: threading.Event) -> None:
    dest = item.dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if _file_size(dest) == item.size and sha256_file(dest, cancel) == item.sha256:
            log.debug("%s is already in place", dest.name)
            return
    except OSError as exc:
        raise _write_error(exc, dest.parent) from exc
    _remove(dest)
    part = part_path(dest)
    attempts = 0
    restarts = 0
    checksum_failures = 0
    while True:
        if cancel.is_set():
            raise DownloadCancelled
        start, digest = _resume_point(part, item, cancel)
        reporter.file_bytes(start)
        try:
            digest = _fetch(client, item, part, start, digest, reporter, cancel)
        except _Restart:
            _remove(part)
            restarts += 1
            if restarts > 2:
                log.error("The server keeps sending unexpected data for %s", dest.name)
                raise DownloadError(MSG_CHECKSUM) from None
            log.info("Restarting the download of %s from the beginning", dest.name)
            continue
        except _Retry as exc:
            attempts = 1 if exc.progressed else attempts + 1
            if attempts >= MAX_ATTEMPTS:
                log.error("Giving up on %s after %d attempts: %s", dest.name, attempts, exc.__cause__ or exc)
                raise DownloadError(str(exc)) from exc
            delay = exc.delay if exc.delay is not None else _backoff(attempts)
            log.warning("Download of %s interrupted (%s); retrying in %.1f s", dest.name, exc.__cause__ or exc, delay)
            if cancel.wait(delay):
                raise DownloadCancelled from None
            continue
        if digest.hexdigest() != item.sha256:
            _remove(part)
            checksum_failures += 1
            log.warning("%s failed its checksum (%d time(s))", dest.name, checksum_failures)
            if checksum_failures >= 2:
                raise DownloadError(MSG_CHECKSUM)
            continue
        try:
            os.replace(part, dest)
        except OSError as exc:
            raise _write_error(exc, dest.parent) from exc
        return


def _resume_point(part: Path, item: DownloadItem, cancel: threading.Event) -> tuple[int, _Digest]:
    """How many bytes of ``part`` can be kept, and the running hash over them."""
    digest = hashlib.sha256()
    have = _file_size(part)
    if have <= 0 or have > item.size:
        _remove(part)
        return 0, digest
    try:
        _hash_into(digest, part, cancel)
    except OSError as exc:
        raise _write_error(exc, part.parent) from exc
    log.info("Resuming %s at %s of %s", item.dest.name, format_size(have), format_size(item.size))
    return have, digest


def _fetch(
    client: httpx.Client,
    item: DownloadItem,
    part: Path,
    start: int,
    digest: _Digest,
    reporter: _Progress,
    cancel: threading.Event,
) -> _Digest:
    """Stream the rest of ``item`` into ``part``. Returns the hash over the whole file once every byte is written."""
    if start == item.size:
        return digest  # complete already; the caller checks the hash
    headers = {"Range": f"bytes={start}-"} if start else {}
    received = 0
    try:
        with client.stream("GET", item.url, headers=headers) as response:
            status = response.status_code
            if status == 416:
                raise _Restart
            if status in _RETRY_STATUSES:
                raise _Retry(MSG_SERVER.format(status=status), progressed=False, delay=_retry_after(response))
            if status >= 400:
                log.error("Download of %s failed: HTTP %d", item.dest.name, status)
                message = MSG_SERVER if status >= 500 else MSG_UNAVAILABLE
                raise DownloadError(message.format(status=status))
            if start and status == 206:
                if not _content_range_starts_at(response, start):
                    raise _Restart
                mode = "ab"
            else:
                # 200: the whole file is coming, even if we asked for a range.
                start, digest, mode = 0, hashlib.sha256(), "wb"
            try:
                fh = open(part, mode)  # noqa: SIM115 - closed in the finally below
            except OSError as exc:
                raise _write_error(exc, part.parent) from exc
            try:
                # Data is written as it arrives (tens of KB at a time), so a dropped connection loses
                # almost nothing and a cancel takes effect at once.
                for chunk in response.iter_raw():
                    if cancel.is_set():
                        raise DownloadCancelled
                    if start + received + len(chunk) > item.size:
                        # More data than the file should have: the server is sending something else.
                        raise _Restart
                    try:
                        fh.write(chunk)
                    except OSError as exc:
                        raise _write_error(exc, part.parent) from exc
                    digest.update(chunk)
                    received += len(chunk)
                    reporter.file_bytes(start + received)
                try:
                    fh.flush()
                    os.fsync(fh.fileno())
                except OSError as exc:
                    raise _write_error(exc, part.parent) from exc
            finally:
                fh.close()
    except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ProxyError) as exc:
        if is_certificate_error(exc):
            log.error("Download of %s refused: the server's certificate couldn't be verified", item.dest.name)
            raise DownloadError(MSG_CERTIFICATE) from exc
        raise _Retry(MSG_OFFLINE, progressed=False) from exc
    except httpx.TimeoutException as exc:
        raise _Retry(MSG_STALLED, progressed=received > 0) from exc
    except httpx.TooManyRedirects as exc:
        log.error("Download of %s failed: too many redirects", item.dest.name)
        raise DownloadError(MSG_SERVER_GENERIC) from exc
    except httpx.TransportError as exc:
        raise _Retry(MSG_DROPPED, progressed=received > 0) from exc
    if start + received != item.size:
        raise _Retry(MSG_DROPPED, progressed=received > 0)
    return digest


def _hash_into(digest: _Digest, path: Path, cancel: threading.Event | None) -> None:
    with open(path, "rb") as fh:
        while block := fh.read(_HASH_BLOCK):
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled
            digest.update(block)


def _content_range_starts_at(response: httpx.Response, start: int) -> bool:
    """True if a 206 response's ``Content-Range: bytes <first>-<last>/<size>`` begins at ``start``."""
    unit, _, rest = response.headers.get("Content-Range", "").partition(" ")
    try:
        first = int(rest.split("-", 1)[0])
    except ValueError:
        return False
    return unit.strip().lower() == "bytes" and first == start


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return min(RETRY_AFTER_MAX_S, max(0.0, float(response.headers["Retry-After"])))
    except (KeyError, ValueError):
        return None


def _backoff(attempt: int) -> float:
    base = min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** max(0, attempt - 1)))
    return base * random.uniform(0.75, 1.25)


def _write_error(exc: OSError, folder: Path) -> DownloadError:
    if exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) in _DISK_FULL_WINERRORS:
        log.error("The disk filled up while downloading to %s", folder)
        return DownloadError(MSG_DISK_FILLED)
    log.error("Couldn't write a download to %s: %s", folder, exc)
    return DownloadError(MSG_WRITE.format(folder=folder))


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise _write_error(exc, path.parent) from exc


def _existing_ancestor(folder: Path) -> Path:
    """The nearest existing folder at or above ``folder`` (``disk_usage`` needs a path that exists)."""
    current = folder
    while not current.exists() and current.parent != current:
        current = current.parent
    return current
