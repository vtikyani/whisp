"""The focused window: who owns it, what it's called, whether it's fullscreen — and when it changes.

``get_foreground_target()`` is cheap (well under 5 ms; version-info lookups
are cached per executable) and safe on any thread: it never sends window
messages, so it can't hang on a busy app.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import threading
from ctypes import wintypes
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from whisp.core.types import AppCategory, TargetWindow

log = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
_version = ctypes.WinDLL("version", use_last_error=True)


class RECT(ctypes.Structure):
    _fields_ = (("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG))


class MONITORINFO(ctypes.Structure):
    _fields_ = (("cbSize", wintypes.DWORD), ("rcMonitor", RECT), ("rcWork", RECT), ("dwFlags", wintypes.DWORD))


WINEVENTPROC = ctypes.WINFUNCTYPE(
    None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND, wintypes.LONG, wintypes.LONG, wintypes.DWORD, wintypes.DWORD
)

_GetForegroundWindow = _user32.GetForegroundWindow
_GetForegroundWindow.argtypes = []
_GetForegroundWindow.restype = wintypes.HWND

_GetWindowThreadProcessId = _user32.GetWindowThreadProcessId
_GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_GetWindowThreadProcessId.restype = wintypes.DWORD

_InternalGetWindowText = _user32.InternalGetWindowText
_InternalGetWindowText.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_InternalGetWindowText.restype = ctypes.c_int

_GetClassNameW = _user32.GetClassNameW
_GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_GetClassNameW.restype = ctypes.c_int

_GetWindowRect = _user32.GetWindowRect
_GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
_GetWindowRect.restype = wintypes.BOOL

_MonitorFromWindow = _user32.MonitorFromWindow
_MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
_MonitorFromWindow.restype = wintypes.HMONITOR

_GetMonitorInfoW = _user32.GetMonitorInfoW
_GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
_GetMonitorInfoW.restype = wintypes.BOOL

_GetShellWindow = _user32.GetShellWindow
_GetShellWindow.argtypes = []
_GetShellWindow.restype = wintypes.HWND

_GetDesktopWindow = _user32.GetDesktopWindow
_GetDesktopWindow.argtypes = []
_GetDesktopWindow.restype = wintypes.HWND

_IsWindowVisible = _user32.IsWindowVisible
_IsWindowVisible.argtypes = [wintypes.HWND]
_IsWindowVisible.restype = wintypes.BOOL

_IsZoomed = _user32.IsZoomed
_IsZoomed.argtypes = [wintypes.HWND]
_IsZoomed.restype = wintypes.BOOL

_IsWindow = _user32.IsWindow
_IsWindow.argtypes = [wintypes.HWND]
_IsWindow.restype = wintypes.BOOL


class _GUITHREADINFO(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    )


_GetGUIThreadInfo = _user32.GetGUIThreadInfo
_GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(_GUITHREADINFO)]
_GetGUIThreadInfo.restype = wintypes.BOOL

_GetWindowLongPtrW = _user32.GetWindowLongPtrW
_GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
_GetWindowLongPtrW.restype = ctypes.c_ssize_t

_FindWindowExW = _user32.FindWindowExW
_FindWindowExW.argtypes = [wintypes.HWND, wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR]
_FindWindowExW.restype = wintypes.HWND

_SetWinEventHook = _user32.SetWinEventHook
_SetWinEventHook.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HMODULE,
    WINEVENTPROC,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
]
_SetWinEventHook.restype = wintypes.HANDLE

_UnhookWinEvent = _user32.UnhookWinEvent
_UnhookWinEvent.argtypes = [wintypes.HANDLE]
_UnhookWinEvent.restype = wintypes.BOOL

_GetMessageW = _user32.GetMessageW
_GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
_GetMessageW.restype = wintypes.BOOL

_PeekMessageW = _user32.PeekMessageW
_PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
_PeekMessageW.restype = wintypes.BOOL

_TranslateMessage = _user32.TranslateMessage
_TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
_TranslateMessage.restype = wintypes.BOOL

_DispatchMessageW = _user32.DispatchMessageW
_DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
_DispatchMessageW.restype = ctypes.c_ssize_t

_PostThreadMessageW = _user32.PostThreadMessageW
_PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_PostThreadMessageW.restype = wintypes.BOOL

_OpenProcess = _kernel32.OpenProcess
_OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_OpenProcess.restype = wintypes.HANDLE

_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL

_QueryFullProcessImageNameW = _kernel32.QueryFullProcessImageNameW
_QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    ctypes.POINTER(wintypes.DWORD),
]
_QueryFullProcessImageNameW.restype = wintypes.BOOL

_GetCurrentProcess = _kernel32.GetCurrentProcess
_GetCurrentProcess.argtypes = []
_GetCurrentProcess.restype = wintypes.HANDLE

_GetCurrentProcessId = _kernel32.GetCurrentProcessId
_GetCurrentProcessId.argtypes = []
_GetCurrentProcessId.restype = wintypes.DWORD

_GetCurrentThreadId = _kernel32.GetCurrentThreadId
_GetCurrentThreadId.argtypes = []
_GetCurrentThreadId.restype = wintypes.DWORD

_OpenProcessToken = _advapi32.OpenProcessToken
_OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
_OpenProcessToken.restype = wintypes.BOOL

_GetTokenInformation = _advapi32.GetTokenInformation
_GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_GetTokenInformation.restype = wintypes.BOOL

_GetFileVersionInfoSizeW = _version.GetFileVersionInfoSizeW
_GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
_GetFileVersionInfoSizeW.restype = wintypes.DWORD

_GetFileVersionInfoW = _version.GetFileVersionInfoW
_GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID]
_GetFileVersionInfoW.restype = wintypes.BOOL

_VerQueryValueW = _version.VerQueryValueW
_VerQueryValueW.argtypes = [
    wintypes.LPCVOID,
    wintypes.LPCWSTR,
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.UINT),
]
_VerQueryValueW.restype = wintypes.BOOL

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_ELEVATION_CLASS = 20  # TokenElevation
MONITOR_DEFAULTTONEAREST = 0x00000002
GWL_STYLE = -16
WS_CAPTION = 0x00C00000
EVENT_SYSTEM_FOREGROUND = 0x0003
WINEVENT_OUTOFCONTEXT = 0x0000
WM_APP = 0x8000
_WM_STOP = WM_APP + 1
PM_NOREMOVE = 0x0000

_SHELL_CLASSES = frozenset({"Progman", "WorkerW"})
_UWP_FRAME_HOST = "applicationframehost.exe"
_UWP_CORE_WINDOW = "Windows.UI.Core.CoreWindow"


# -- small window/process queries ------------------------------------------------------


def foreground_window() -> int:
    """Handle of the window that has the focus, or 0."""
    return int(_GetForegroundWindow() or 0)


def window_process_id(hwnd: int) -> int:
    pid = wintypes.DWORD()
    _GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def window_thread_id(hwnd: int) -> int:
    return int(_GetWindowThreadProcessId(hwnd, None))


def focused_control(hwnd: int) -> int:
    """The window with the keyboard focus in ``hwnd``'s thread, such as a text box inside it; 0 if unknown."""
    thread_id = window_thread_id(hwnd)
    if not thread_id:
        return 0
    info = _GUITHREADINFO(cbSize=ctypes.sizeof(_GUITHREADINFO))
    if not _GetGUIThreadInfo(thread_id, ctypes.byref(info)):
        return 0
    return int(info.hwndFocus or 0)


def is_window(hwnd: int) -> bool:
    return bool(hwnd) and bool(_IsWindow(hwnd))


def current_process_id() -> int:
    return int(_GetCurrentProcessId())


def is_own_hidden_window(hwnd: int) -> bool:
    """True for an invisible window of Whisp itself, such as the one that owns the tray menu.

    Such a window can be left with the focus after using the tray menu; text "typed" into it is lost.
    Whisp's visible main window is a fine place to dictate into (the onboarding practice box).
    """
    return bool(hwnd) and window_process_id(hwnd) == current_process_id() and not _IsWindowVisible(hwnd)


def _window_text(hwnd: int) -> str:
    # InternalGetWindowText reads the stored title without sending WM_GETTEXT, so a hung or busy
    # window (including one of ours while its thread is blocked) can't stall the caller.
    buf = ctypes.create_unicode_buffer(1024)
    length = _InternalGetWindowText(hwnd, buf, len(buf))
    return buf.value[:length] if length > 0 else ""


def window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    length = _GetClassNameW(hwnd, buf, len(buf))
    return buf.value[:length] if length > 0 else ""


def process_image_path(pid: int) -> str | None:
    """Full path of a process's executable, or None if Windows won't say (e.g. protected processes)."""
    if pid <= 0:
        return None
    handle = _OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if not _QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return None
        return buf.value[: size.value]
    finally:
        _CloseHandle(handle)


# -- friendly app names ------------------------------------------------------------------


def _query_version_string(block: ctypes.Array, sub_block: str) -> str | None:
    pointer = wintypes.LPVOID()
    length = wintypes.UINT()
    if not _VerQueryValueW(block, sub_block, ctypes.byref(pointer), ctypes.byref(length)) or not pointer.value:
        return None
    if length.value == 0:
        return None
    value = ctypes.wstring_at(pointer.value, length.value).split("\x00", 1)[0]
    return value.strip() or None


def _version_translations(block: ctypes.Array) -> list[tuple[int, int]]:
    pointer = wintypes.LPVOID()
    length = wintypes.UINT()
    pairs: list[tuple[int, int]] = []
    if _VerQueryValueW(block, "\\VarFileInfo\\Translation", ctypes.byref(pointer), ctypes.byref(length)):
        count = length.value // 4
        words = ctypes.cast(pointer, ctypes.POINTER(wintypes.WORD * (count * 2))).contents if count else []
        pairs = [(words[i * 2], words[i * 2 + 1]) for i in range(count)]
    # Common fallbacks: US English with Unicode, Windows-1252 and neutral code pages.
    for fallback in ((0x0409, 0x04B0), (0x0409, 0x04E4), (0x0409, 0x0000)):
        if fallback not in pairs:
            pairs.append(fallback)
    return pairs


def _clean_name(value: str) -> str:
    return " ".join(value.replace("®", "").replace("™", "").split())


def _stem_name(path: str) -> str:
    """Title-case an all-lower or all-upper stem ("slack" -> "Slack"); keep mixed case ("WindowsTerminal")."""
    stem = Path(path).stem
    return stem.title() if stem.islower() or stem.isupper() else stem


@functools.lru_cache(maxsize=512)
def app_name_for(path: str) -> str:
    """Friendly name from the executable's version resource ("Google Chrome"), else its title-cased stem."""
    fallback = _stem_name(path)
    try:
        size = _GetFileVersionInfoSizeW(path, None)
        if not size:
            return fallback
        block = ctypes.create_string_buffer(size)
        if not _GetFileVersionInfoW(path, 0, size, block):
            return fallback
        translations = _version_translations(block)
        for key in ("FileDescription", "ProductName"):
            for lang, codepage in translations:
                value = _query_version_string(block, f"\\StringFileInfo\\{lang:04x}{codepage:04x}\\{key}")
                if value:
                    cleaned = _clean_name(value)
                    if cleaned:
                        return cleaned
    except OSError:
        log.debug("Couldn't read version info for %s", Path(path).name, exc_info=True)
    return fallback


# -- fullscreen ------------------------------------------------------------------------


def covers_monitor(window: tuple[int, int, int, int], monitor: tuple[int, int, int, int]) -> bool:
    """True if a window rectangle (left, top, right, bottom) covers the whole monitor rectangle."""
    left, top, right, bottom = window
    m_left, m_top, m_right, m_bottom = monitor
    return left <= m_left and top <= m_top and right >= m_right and bottom >= m_bottom


def _monitor_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    monitor = _MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    if not monitor:
        return None
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    if not _GetMonitorInfoW(monitor, ctypes.byref(info)):
        return None
    r = info.rcMonitor
    return (r.left, r.top, r.right, r.bottom)


def _is_fullscreen(hwnd: int, class_name: str, monitor: tuple[int, int, int, int] | None) -> bool:
    if monitor is None or class_name in _SHELL_CLASSES:
        return False
    if hwnd in (int(_GetShellWindow() or 0), int(_GetDesktopWindow() or 0)):
        return False
    rect = RECT()
    if not _GetWindowRect(hwnd, ctypes.byref(rect)):
        return False
    if not covers_monitor((rect.left, rect.top, rect.right, rect.bottom), monitor):
        return False
    # A maximised window with a normal title bar also covers the monitor when the taskbar auto-hides;
    # that's not fullscreen.
    style = _GetWindowLongPtrW(hwnd, GWL_STYLE)
    return not (_IsZoomed(hwnd) and (style & WS_CAPTION) == WS_CAPTION)


# -- targets ---------------------------------------------------------------------------


def target_for_window(hwnd: int) -> TargetWindow | None:
    """Describe a top-level window, or None if it no longer exists."""
    if not is_window(hwnd):
        return None
    pid = window_process_id(hwnd)
    path = process_image_path(pid)
    process_name = Path(path).name.lower() if path else ""
    if process_name == _UWP_FRAME_HOST:
        # Store apps run inside a frame window owned by ApplicationFrameHost; the app itself
        # owns the CoreWindow inside it.
        core = _FindWindowExW(hwnd, None, _UWP_CORE_WINDOW, None)
        if core:
            app_pid = window_process_id(core)
            app_path = process_image_path(app_pid) if app_pid and app_pid != pid else None
            if app_path:
                pid, path, process_name = app_pid, app_path, Path(app_path).name.lower()
    class_name = window_class(hwnd)
    monitor = _monitor_rect(hwnd)
    return TargetWindow(
        hwnd=int(hwnd),
        pid=pid,
        process_name=process_name,
        app_name=app_name_for(path) if path else "",
        title=_window_text(hwnd),
        category=AppCategory.OTHER,
        is_fullscreen=_is_fullscreen(hwnd, class_name, monitor),
        monitor_rect=monitor,
    )


def get_foreground_target() -> TargetWindow | None:
    """The window that currently has keyboard focus, or None. Any thread."""
    try:
        hwnd = foreground_window()
        return target_for_window(hwnd) if hwnd else None
    except Exception:
        log.exception("Couldn't read the focused window")
        return None


# -- elevation -------------------------------------------------------------------------


def _token_elevated(process: int) -> bool | None:
    token = wintypes.HANDLE()
    if not _OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        elevation = wintypes.DWORD()
        returned = wintypes.DWORD()
        if not _GetTokenInformation(
            token, TOKEN_ELEVATION_CLASS, ctypes.byref(elevation), ctypes.sizeof(elevation), ctypes.byref(returned)
        ):
            return None
        return bool(elevation.value)
    finally:
        _CloseHandle(token)


def is_process_elevated(pid: int) -> bool | None:
    """True if the process runs as administrator, False if not, None if Windows won't tell us."""
    if pid <= 0:
        return None
    handle = _OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        return _token_elevated(handle)
    finally:
        _CloseHandle(handle)


@functools.cache
def current_process_elevated() -> bool:
    """True if Whisp itself runs as administrator."""
    return bool(_token_elevated(_GetCurrentProcess()))


# -- watcher ---------------------------------------------------------------------------


class ForegroundWatcher(QObject):
    """Emits ``changed(TargetWindow | None)`` whenever another window takes the focus.

    Runs a Win32 message loop on its own thread (``whisp-foreground``) with an
    out-of-context WinEvent hook; the signal is emitted from that thread.
    """

    changed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._stop = threading.Event()
        self._watching = threading.Event()
        # Keep the ctypes callback alive for as long as the hook may call it.
        self._callback = WINEVENTPROC(self._on_event)

    @property
    def is_watching(self) -> bool:
        """True once focus changes are being reported (the hook is set up on the watcher thread after ``start``)."""
        return self._watching.is_set()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread_id = 0
            self._thread = threading.Thread(target=self._run, name="whisp-foreground", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Unhook and end the thread. Safe to call more than once."""
        with self._lock:
            thread = self._thread
            self._thread = None
            self._stop.set()
            if self._thread_id:
                _PostThreadMessageW(self._thread_id, _WM_STOP, 0, 0)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                log.error("The foreground watcher thread didn't stop within 2 s")

    def _run(self) -> None:
        hook = None
        try:
            msg = wintypes.MSG()
            # Create this thread's message queue before publishing its id, so stop() can always post to it.
            _PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_NOREMOVE)
            with self._lock:
                self._thread_id = int(_GetCurrentThreadId())
                if self._stop.is_set():
                    return
            hook = _SetWinEventHook(
                EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, None, self._callback, 0, 0, WINEVENT_OUTOFCONTEXT
            )
            if not hook:
                log.error("Couldn't watch focus changes (error %d)", ctypes.get_last_error())
                return
            self._watching.set()
            while True:
                result = _GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result in (0, -1) or msg.message == _WM_STOP:
                    break
                _TranslateMessage(ctypes.byref(msg))
                _DispatchMessageW(ctypes.byref(msg))
        except Exception:
            log.exception("The foreground watcher stopped unexpectedly")
        finally:
            self._watching.clear()
            if hook:
                _UnhookWinEvent(hook)
            with self._lock:
                self._thread_id = 0

    def _on_event(self, _hook, event, hwnd, id_object, id_child, _thread, _time) -> None:
        if event != EVENT_SYSTEM_FOREGROUND or self._stop.is_set():
            return
        try:
            self.changed.emit(get_foreground_target())
        except Exception:
            log.exception("Couldn't report a focus change")
