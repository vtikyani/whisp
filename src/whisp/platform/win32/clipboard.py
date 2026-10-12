"""The Windows clipboard: text in and out, history-excluded writes, and full snapshot/restore.

Whisp pastes dictations through the clipboard, so it must leave the user's
clipboard as it found it and keep dictated text out of Win+V history, the
cloud clipboard and third-party clipboard managers.

Every function opens the clipboard itself (retrying for a moment if another
app holds it), and raises ``ClipboardError`` with a plain-language message if
it can't. Nothing here logs clipboard contents — only format ids and sizes.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_OpenClipboard = _user32.OpenClipboard
_OpenClipboard.argtypes = [wintypes.HWND]
_OpenClipboard.restype = wintypes.BOOL

_CloseClipboard = _user32.CloseClipboard
_CloseClipboard.argtypes = []
_CloseClipboard.restype = wintypes.BOOL

_EmptyClipboard = _user32.EmptyClipboard
_EmptyClipboard.argtypes = []
_EmptyClipboard.restype = wintypes.BOOL

_GetClipboardData = _user32.GetClipboardData
_GetClipboardData.argtypes = [wintypes.UINT]
_GetClipboardData.restype = wintypes.HANDLE

_SetClipboardData = _user32.SetClipboardData
_SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
_SetClipboardData.restype = wintypes.HANDLE

_EnumClipboardFormats = _user32.EnumClipboardFormats
_EnumClipboardFormats.argtypes = [wintypes.UINT]
_EnumClipboardFormats.restype = wintypes.UINT

_IsClipboardFormatAvailable = _user32.IsClipboardFormatAvailable
_IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
_IsClipboardFormatAvailable.restype = wintypes.BOOL

_GetClipboardSequenceNumber = _user32.GetClipboardSequenceNumber
_GetClipboardSequenceNumber.argtypes = []
_GetClipboardSequenceNumber.restype = wintypes.DWORD

_RegisterClipboardFormatW = _user32.RegisterClipboardFormatW
_RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
_RegisterClipboardFormatW.restype = wintypes.UINT

_GetClipboardFormatNameW = _user32.GetClipboardFormatNameW
_GetClipboardFormatNameW.argtypes = [wintypes.UINT, wintypes.LPWSTR, ctypes.c_int]
_GetClipboardFormatNameW.restype = ctypes.c_int

_CreateWindowExW = _user32.CreateWindowExW
_CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]
_CreateWindowExW.restype = wintypes.HWND

_DestroyWindow = _user32.DestroyWindow
_DestroyWindow.argtypes = [wintypes.HWND]
_DestroyWindow.restype = wintypes.BOOL

_GlobalAlloc = _kernel32.GlobalAlloc
_GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
_GlobalAlloc.restype = wintypes.HGLOBAL

_GlobalLock = _kernel32.GlobalLock
_GlobalLock.argtypes = [wintypes.HGLOBAL]
_GlobalLock.restype = wintypes.LPVOID

_GlobalUnlock = _kernel32.GlobalUnlock
_GlobalUnlock.argtypes = [wintypes.HGLOBAL]
_GlobalUnlock.restype = wintypes.BOOL

_GlobalSize = _kernel32.GlobalSize
_GlobalSize.argtypes = [wintypes.HGLOBAL]
_GlobalSize.restype = ctypes.c_size_t

_GlobalFree = _kernel32.GlobalFree
_GlobalFree.argtypes = [wintypes.HGLOBAL]
_GlobalFree.restype = wintypes.HGLOBAL

GMEM_MOVEABLE = 0x0002
HWND_MESSAGE = wintypes.HWND(-3)

CF_TEXT = 1
CF_BITMAP = 2
CF_METAFILEPICT = 3
CF_OEMTEXT = 7
CF_DIB = 8
CF_PALETTE = 9
CF_UNICODETEXT = 13
CF_ENHMETAFILE = 14
CF_HDROP = 15
CF_LOCALE = 16
CF_DIBV5 = 17
CF_OWNERDISPLAY = 0x0080
CF_DSPBITMAP = 0x0082
CF_DSPMETAFILEPICT = 0x0083
CF_DSPENHMETAFILE = 0x008E
CF_PRIVATEFIRST = 0x0200
CF_PRIVATELAST = 0x02FF
CF_GDIOBJFIRST = 0x0300
CF_GDIOBJLAST = 0x03FF

_GDI_HANDLE_FORMATS = frozenset(
    {
        CF_BITMAP,
        CF_METAFILEPICT,
        CF_PALETTE,
        CF_ENHMETAFILE,
        CF_OWNERDISPLAY,
        CF_DSPBITMAP,
        CF_DSPMETAFILEPICT,
        CF_DSPENHMETAFILE,
    }
)
"""Formats whose data is a GDI handle, not memory. Windows re-creates bitmaps from CF_DIB/CF_DIBV5."""

_STANDARD_FORMAT_NAMES = {
    CF_TEXT: "CF_TEXT",
    CF_BITMAP: "CF_BITMAP",
    CF_METAFILEPICT: "CF_METAFILEPICT",
    CF_OEMTEXT: "CF_OEMTEXT",
    CF_DIB: "CF_DIB",
    CF_PALETTE: "CF_PALETTE",
    CF_UNICODETEXT: "CF_UNICODETEXT",
    CF_ENHMETAFILE: "CF_ENHMETAFILE",
    CF_HDROP: "CF_HDROP",
    CF_LOCALE: "CF_LOCALE",
    CF_DIBV5: "CF_DIBV5",
}

_OLE_PRIVATE_DATA = "Ole Private Data"
"""OLE's pointer to the source app's live data object: meaningless once that app no longer owns the clipboard."""

EXCLUDE_FROM_MONITORS = "ExcludeClipboardContentFromMonitorProcessing"
CAN_INCLUDE_IN_HISTORY = "CanIncludeInClipboardHistory"
CAN_UPLOAD_TO_CLOUD = "CanUploadToCloudClipboard"
_EXCLUSION_FORMAT_NAMES = (EXCLUDE_FROM_MONITORS, CAN_INCLUDE_IN_HISTORY, CAN_UPLOAD_TO_CLOUD)

OPEN_TIMEOUT_S = 0.5
"""How long to keep retrying when another app has the clipboard open."""

MAX_FORMAT_BYTES = 64 * 1024 * 1024
"""Formats larger than this are left out of a snapshot (and logged)."""

SNAPSHOT_TIME_BUDGET_S = 2.0
"""Apps that render clipboard formats on demand can be slow; stop collecting more formats after this long."""


class ClipboardError(OSError):
    """The clipboard couldn't be read or written. ``str(error)`` is a plain-language message."""


@dataclass(frozen=True)
class ClipboardSnapshot:
    """Everything on the clipboard that can be copied byte for byte, captured at one moment."""

    formats: tuple[tuple[int, bytes], ...] = field(repr=False)
    """(format id, data) pairs in the order the source app offered them."""
    sequence: int = 0
    """The clipboard sequence number when the snapshot was taken."""

    @property
    def is_empty(self) -> bool:
        return not self.formats

    @property
    def size_bytes(self) -> int:
        return sum(len(data) for _, data in self.formats)

    @property
    def format_ids(self) -> tuple[int, ...]:
        return tuple(fmt for fmt, _ in self.formats)


# -- format helpers ------------------------------------------------------------------

_format_lock = threading.Lock()
_registered_formats: dict[str, int] = {}
_format_names: dict[int, str] = {}


def registered_format(name: str) -> int:
    """Id of a named clipboard format, registering it the first time."""
    with _format_lock:
        fmt = _registered_formats.get(name)
        if fmt is None:
            fmt = int(_RegisterClipboardFormatW(name))
            if not fmt:
                raise ClipboardError(f"Windows couldn't register the clipboard format {name!r}.")
            _registered_formats[name] = fmt
        return fmt


def format_name(fmt: int) -> str:
    """A readable name for a clipboard format id (for logs)."""
    if fmt in _STANDARD_FORMAT_NAMES:
        return _STANDARD_FORMAT_NAMES[fmt]
    with _format_lock:
        cached = _format_names.get(fmt)
    if cached is not None:
        return cached
    name = f"#{fmt}"
    if fmt >= 0xC000:
        buf = ctypes.create_unicode_buffer(256)
        length = _GetClipboardFormatNameW(fmt, buf, len(buf))
        if length > 0:
            name = buf.value
    with _format_lock:
        _format_names[fmt] = name
    return name


def is_memory_format(fmt: int) -> bool:
    """True for formats whose data is a movable memory block (HGLOBAL) that can be copied as bytes."""
    if fmt in _GDI_HANDLE_FORMATS:
        return False
    # Private formats may hold any kind of handle; GDI-object formats hold GDI handles.
    return not (CF_PRIVATEFIRST <= fmt <= CF_PRIVATELAST or CF_GDIOBJFIRST <= fmt <= CF_GDIOBJLAST)


def _should_capture(fmt: int) -> bool:
    if not is_memory_format(fmt):
        return False
    if fmt >= 0xC000:
        name = format_name(fmt)
        # The exclusion markers are re-added on every restore; OLE's private pointer is stale by then.
        if name == _OLE_PRIVATE_DATA or name in _EXCLUSION_FORMAT_NAMES:
            return False
    return True


# -- open/close ----------------------------------------------------------------------


@contextmanager
def _owner_window() -> Iterator[wintypes.HWND]:
    """A short-lived message-only window to own the clipboard while we write to it.

    Windows requires an owner window for writes (with no owner, ``EmptyClipboard``
    leaves the clipboard ownerless and writes can fail). The window is destroyed
    right after the clipboard is closed, so no thread has to pump messages for it.
    """
    hwnd = _CreateWindowExW(0, "STATIC", None, 0, 0, 0, 0, 0, HWND_MESSAGE, None, None, None)
    if not hwnd:
        raise ClipboardError(f"Windows couldn't create a clipboard window (error {ctypes.get_last_error()}).")
    try:
        yield hwnd
    finally:
        _DestroyWindow(hwnd)


def _open(owner: wintypes.HWND | None) -> None:
    deadline = time.monotonic() + OPEN_TIMEOUT_S
    delay = 0.005
    while True:
        if _OpenClipboard(owner):
            return
        error = ctypes.get_last_error()
        if time.monotonic() >= deadline:
            log.warning("Couldn't open the clipboard within %.0f ms (error %d)", OPEN_TIMEOUT_S * 1000, error)
            raise ClipboardError("The clipboard is busy — another app is using it.")
        time.sleep(delay)
        delay = min(delay * 2, 0.05)


@contextmanager
def _session(*, write: bool) -> Iterator[None]:
    if write:
        with _owner_window() as hwnd:
            _open(hwnd)
            try:
                yield
            finally:
                _CloseClipboard()
    else:
        _open(None)
        try:
            yield
        finally:
            _CloseClipboard()


# -- memory helpers ------------------------------------------------------------------


def _read_handle(handle: int) -> bytes | None:
    size = _GlobalSize(handle)
    if not size:
        return None
    pointer = _GlobalLock(handle)
    if not pointer:
        return None
    try:
        return ctypes.string_at(pointer, size)
    finally:
        _GlobalUnlock(handle)


def _set_data(fmt: int, data: bytes) -> None:
    """Copy ``data`` into a new memory block and hand it to the clipboard (which then owns it)."""
    handle = _GlobalAlloc(GMEM_MOVEABLE, max(len(data), 1))
    if not handle:
        raise ClipboardError("There wasn't enough memory to copy to the clipboard.")
    pointer = _GlobalLock(handle)
    if not pointer:
        _GlobalFree(handle)
        raise ClipboardError("There wasn't enough memory to copy to the clipboard.")
    try:
        ctypes.memmove(pointer, data, len(data))
    finally:
        _GlobalUnlock(handle)
    if not _SetClipboardData(fmt, handle):
        error = ctypes.get_last_error()
        # Ownership only passes to the system on success; otherwise the block is still ours to free.
        _GlobalFree(handle)
        raise ClipboardError(f"Windows refused the clipboard data ({format_name(fmt)}, error {error}).")


def _set_history_exclusions() -> None:
    """Mark the current clipboard content as private: no Win+V history, no cloud sync, no clipboard managers."""
    zero = (0).to_bytes(4, "little")
    for name in _EXCLUSION_FORMAT_NAMES:
        try:
            _set_data(registered_format(name), zero)
        except ClipboardError:
            # The content itself is already on the clipboard; a missing marker only means it may show in history.
            log.warning("Couldn't mark clipboard content as private (%s)", name, exc_info=True)


def _decode_text(raw: bytes) -> str:
    text = raw[: len(raw) - len(raw) % 2].decode("utf-16-le", errors="replace")
    end = text.find("\x00")
    return text if end < 0 else text[:end]


# -- public API ----------------------------------------------------------------------


def get_sequence_number() -> int:
    """Windows' clipboard change counter. It changes whenever anyone writes to the clipboard."""
    return int(_GetClipboardSequenceNumber())


def has_text() -> bool:
    """True if text can be read from the clipboard (doesn't need to open it)."""
    return bool(_IsClipboardFormatAvailable(CF_UNICODETEXT))


def get_text() -> str | None:
    """The clipboard's text, or None if it holds no text."""
    with _session(write=False):
        handle = _GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        raw = _read_handle(handle)
    return None if raw is None else _decode_text(raw)


def set_text(text: str, *, exclude_from_history: bool = True) -> None:
    """Replace the clipboard content with ``text``.

    With ``exclude_from_history`` (the default) the content is also marked so
    it stays out of Win+V history, the cloud clipboard and clipboard managers.
    """
    data = (text + "\x00").encode("utf-16-le")
    with _session(write=True):
        if not _EmptyClipboard():
            raise ClipboardError(f"Windows couldn't clear the clipboard (error {ctypes.get_last_error()}).")
        _set_data(CF_UNICODETEXT, data)
        if exclude_from_history:
            _set_history_exclusions()


def snapshot() -> ClipboardSnapshot:
    """Capture every memory-based clipboard format so it can be put back later with ``restore``.

    GDI-handle formats (bitmaps, metafiles, palettes) are skipped — Windows
    rebuilds bitmaps from CF_DIB/CF_DIBV5 — as are formats over
    ``MAX_FORMAT_BYTES``.
    """
    started = time.monotonic()
    items: list[tuple[int, bytes]] = []
    with _session(write=False):
        sequence = get_sequence_number()
        fmt = 0
        while True:
            fmt = int(_EnumClipboardFormats(fmt))
            if not fmt:
                break
            if not _should_capture(fmt):
                continue
            if time.monotonic() - started > SNAPSHOT_TIME_BUDGET_S:
                log.warning(
                    "Clipboard snapshot stopped after %.1f s; later formats were left out", SNAPSHOT_TIME_BUDGET_S
                )
                break
            handle = _GetClipboardData(fmt)
            if not handle:
                continue
            size = _GlobalSize(handle)
            if size > MAX_FORMAT_BYTES:
                log.info("Clipboard format %s is %d MB; not saving it", format_name(fmt), size // (1024 * 1024))
                continue
            data = _read_handle(handle)
            if data is not None:
                items.append((fmt, data))
    snap = ClipboardSnapshot(tuple(items), sequence)
    log.debug(
        "Clipboard snapshot: %d formats, %d bytes in %.0f ms",
        len(snap.formats),
        snap.size_bytes,
        (time.monotonic() - started) * 1000,
    )
    return snap


def restore(snap: ClipboardSnapshot) -> None:
    """Put a snapshot back on the clipboard, marked so clipboard history doesn't record it a second time.

    An empty snapshot leaves the clipboard empty.
    """
    restored = 0
    with _session(write=True):
        if not _EmptyClipboard():
            raise ClipboardError(f"Windows couldn't clear the clipboard (error {ctypes.get_last_error()}).")
        for fmt, data in snap.formats:
            try:
                _set_data(fmt, data)
                restored += 1
            except ClipboardError:
                log.warning("Couldn't restore clipboard format %s", format_name(fmt), exc_info=True)
        if restored:
            _set_history_exclusions()
    if restored < len(snap.formats):
        log.warning("Restored %d of %d clipboard formats", restored, len(snap.formats))


def wait_for_change(since: int, timeout_s: float) -> bool:
    """Wait until the clipboard sequence number differs from ``since``. Returns False on timeout."""
    deadline = time.monotonic() + timeout_s
    while get_sequence_number() == since:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True
