"""Shared helpers for tests that drive the real Windows desktop (marked ``desktop``).

The owner may be using this machine while tests run, so every desktop test:

* takes the focus only from the app that started the test run (a terminal or
  editor the person is watching) and only after nobody has touched the
  keyboard or mouse for a couple of seconds — never from an app they're
  working in. It waits up to ``WHISP_DESKTOP_WAIT_S`` seconds (default 120)
  for that, then skips;
* sends keys only after checking that its own test window has the focus,
  using short sequences, and releases anything it pressed even if a test fails;
* saves the clipboard first and puts it back afterwards;
* hands the focus back to the window that had it when the run ends. (The
  tests keep it in between: Windows lets an app take the focus only while it
  has it, so a window that handed it back after every test couldn't take it
  again.)
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from collections.abc import Callable, Sequence
from ctypes import wintypes
from typing import Any

import pytest
from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QLineEdit, QTextEdit, QVBoxLayout, QWidget
from pytestqt.exceptions import TimeoutError as WaitTimeout

from whisp.platform.win32 import keys
from whisp.platform.win32.foreground import foreground_window, process_image_path, window_process_id

TEST_MARKER = 0x7E570001
"""dwExtraInfo for keys the tests inject. Not Whisp's marker, so Whisp's hook treats them as the user's."""

IDLE_REQUIRED_S = 2.0
IDLE_WAIT_LIMIT_S = float(os.environ.get("WHISP_DESKTOP_WAIT_S", "120"))

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = (("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD))


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = (
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    )


_CreateToolhelp32Snapshot = _kernel32.CreateToolhelp32Snapshot
_CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_Process32FirstW = _kernel32.Process32FirstW
_Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
_Process32FirstW.restype = wintypes.BOOL
_Process32NextW = _kernel32.Process32NextW
_Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
_Process32NextW.restype = wintypes.BOOL
_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL
_TH32CS_SNAPPROCESS = 0x00000002
_INVALID_HANDLE = wintypes.HANDLE(-1).value


_SHELL_PROCESSES = frozenset({"explorer.exe", "userinit.exe", "winlogon.exe", "services.exe", "svchost.exe"})


def launcher_pids() -> set[int]:
    """This process and the processes above it, up to (not including) the Windows shell.

    That is the terminal or app that started the test run. The shell itself
    (explorer.exe, which also owns File Explorer windows) doesn't count.
    """
    snapshot = _CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == _INVALID_HANDLE:
        return {os.getpid()}
    processes: dict[int, tuple[int, str]] = {}
    try:
        entry = _PROCESSENTRY32W(dwSize=ctypes.sizeof(_PROCESSENTRY32W))
        more = _Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            processes[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), entry.szExeFile.lower())
            more = _Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        _CloseHandle(snapshot)
    chain: set[int] = set()
    pid = os.getpid()
    while pid and pid not in chain and pid in processes:
        parent, name = processes[pid]
        if name in _SHELL_PROCESSES:
            break
        chain.add(pid)
        pid = parent
    return chain or {os.getpid()}


_GetLastInputInfo = _user32.GetLastInputInfo
_GetLastInputInfo.argtypes = [ctypes.POINTER(_LASTINPUTINFO)]
_GetLastInputInfo.restype = wintypes.BOOL
_GetTickCount = _kernel32.GetTickCount
_GetTickCount.argtypes = []
_GetTickCount.restype = wintypes.DWORD
_SetForegroundWindow = _user32.SetForegroundWindow
_SetForegroundWindow.argtypes = [wintypes.HWND]
_SetForegroundWindow.restype = wintypes.BOOL
_IsWindow = _user32.IsWindow
_IsWindow.argtypes = [wintypes.HWND]
_IsWindow.restype = wintypes.BOOL
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
_SetFocus = _user32.SetFocus
_SetFocus.argtypes = [wintypes.HWND]
_SetFocus.restype = wintypes.HWND
_SendMessageW = _user32.SendMessageW
_SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_SendMessageW.restype = wintypes.LPARAM

_WS_CHILD, _WS_VISIBLE, _WS_BORDER, _ES_MULTILINE = 0x40000000, 0x10000000, 0x00800000, 0x0004
_WM_SETTEXT, _WM_GETTEXT, _WM_GETTEXTLENGTH = 0x000C, 0x000D, 0x000E
_RICH_EDIT_HEIGHT = 60
"""Logical pixels kept free at the bottom of the test window for the native rich text box."""
_msftedit: ctypes.WinDLL | None = None
"""Windows' rich text library; loading it registers the RICHEDIT50W window class."""

START_MENU_HOSTS = frozenset({"startmenuexperiencehost.exe", "searchhost.exe", "searchapp.exe"})


def user_idle_seconds() -> float:
    info = _LASTINPUTINFO(cbSize=ctypes.sizeof(_LASTINPUTINFO))
    if not _GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    return ((_GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


def wait_for_idle_user(deadline: float | None = None) -> None:
    if deadline is None:
        deadline = time.monotonic() + IDLE_WAIT_LIMIT_S
    while user_idle_seconds() < IDLE_REQUIRED_S:
        if time.monotonic() > deadline:
            pytest.skip("Someone is using the keyboard or mouse; not taking the focus for a desktop test.")
        time.sleep(0.2)


def set_foreground(hwnd: int) -> bool:
    return bool(_SetForegroundWindow(hwnd))


def is_window(hwnd: int) -> bool:
    return bool(hwnd) and bool(_IsWindow(hwnd))


def foreground_process_name() -> str:
    path = process_image_path(window_process_id(foreground_window()))
    return path.rsplit("\\", 1)[-1].lower() if path else ""


def press(vk: int) -> keys.INPUT:
    return keys.key_event(vk, marker=TEST_MARKER)


def release(vk: int) -> keys.INPUT:
    return keys.key_event(vk, up=True, marker=TEST_MARKER)


def wait_until_equal(
    qtbot: Any,
    read: Callable[[], Any],
    expected: Any,
    *,
    timeout_ms: int = 3000,
    describe: Callable[[], str] | None = None,
) -> None:
    """Wait for ``read()`` to return ``expected``. On timeout, fail showing what it returned instead."""
    try:
        qtbot.waitUntil(lambda: read() == expected, timeout=timeout_ms)
    except WaitTimeout:
        actual = read()
        detail = f" ({describe()})" if describe is not None else ""
        pytest.fail(f"After {timeout_ms} ms: expected {expected!r}, got {actual!r}{detail}")


def run_in_worker(qtbot: Any, fn: Callable[[], Any], timeout_ms: int = 10_000) -> Any:
    """Run a blocking call on another thread while the Qt event loop keeps running (like Whisp's workers)."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # re-raised on the test thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    qtbot.waitUntil(lambda: not thread.is_alive(), timeout=timeout_ms)
    if "error" in box:
        raise box["error"]
    return box.get("result")


class KeyLog(QObject):
    """Records key events that reach the test window: (pressed, native virtual key, Qt modifiers)."""

    def __init__(self, widgets: Sequence[QWidget]) -> None:
        super().__init__()
        self._widgets = set(widgets)
        self.entries: list[tuple[bool, int, int]] = []

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj in self._widgets and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            pressed = event.type() == QEvent.Type.KeyPress
            self.entries.append((pressed, int(event.nativeVirtualKey()), int(event.modifiers().value)))
        return False

    def pressed_vks(self) -> list[int]:
        return [vk for pressed, vk, _ in self.entries if pressed]


class DesktopTestWindow(QWidget):
    """Qt text boxes (``line``, ``edit``) and, below them, a native Windows rich text box (``rich``)."""

    def __init__(self, title: str = "Whisp platform test window") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.line = QLineEdit(self)
        self.edit = QTextEdit(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.line)
        layout.addWidget(self.edit)
        layout.setContentsMargins(9, 9, 9, 9 + _RICH_EDIT_HEIGHT)
        self.resize(520, 320 + _RICH_EDIT_HEIGHT)
        self.keys = KeyLog([self, self.line, self.edit, self.edit.viewport()])
        QApplication.instance().installEventFilter(self.keys)
        self.focus_from: int | None = None
        """The window that had the focus before the first desktop test took it."""
        self._rich = 0

    @property
    def hwnd(self) -> int:
        return int(self.winId())

    @property
    def rich(self) -> int:
        """A native Windows rich text box, the kind most Windows apps use (created on first use).

        A plain child window of this one, outside Qt: Qt's own text boxes handle typed emoji differently.
        """
        global _msftedit
        if not self._rich:
            if _msftedit is None:
                _msftedit = ctypes.WinDLL("msftedit")
            ratio = self.devicePixelRatioF()
            x, y = round(9 * ratio), round((self.height() - _RICH_EDIT_HEIGHT) * ratio)
            width, height = round((self.width() - 18) * ratio), round((_RICH_EDIT_HEIGHT - 9) * ratio)
            style = _WS_CHILD | _WS_VISIBLE | _WS_BORDER | _ES_MULTILINE
            hwnd = _CreateWindowExW(0, "RICHEDIT50W", None, style, x, y, width, height, self.hwnd, None, None, None)
            assert hwnd, f"Couldn't create a rich text box (error {ctypes.get_last_error()})"
            self._rich = int(hwnd)
        return self._rich

    def focus_rich(self) -> None:
        _SetFocus(self.rich)

    def rich_text(self) -> str:
        """The rich text box's text, with "\\n" line breaks."""
        length = _SendMessageW(self.rich, _WM_GETTEXTLENGTH, 0, 0)
        buffer = ctypes.create_unicode_buffer(length + 1)
        _SendMessageW(self.rich, _WM_GETTEXT, length + 1, ctypes.addressof(buffer))
        return buffer.value.replace("\r\n", "\n").replace("\r", "\n")

    def reset(self) -> None:
        """Empty every text box and give the keyboard focus to the Qt line box (the rich text box may have it)."""
        self.line.clear()
        self.edit.clear()
        if self._rich:
            _SendMessageW(self._rich, _WM_SETTEXT, 0, 0)
        _SetFocus(self.hwnd)
        self.line.setFocus()

    def dispose(self) -> None:
        QApplication.instance().removeEventFilter(self.keys)
        self.close()
        self.deleteLater()


class Desktop:
    """The focused test window plus a guarded way to send keys to it."""

    def __init__(self, window: DesktopTestWindow) -> None:
        self.window = window
        self._down: list[int] = []

    def ensure_focus(self) -> None:
        if foreground_window() != self.window.hwnd:
            pytest.skip("The test window lost the focus to another window; not sending keys.")

    def send(self, events: Sequence[keys.INPUT]) -> None:
        self.ensure_focus()
        sent = keys.send_input(events)
        for event in events[:sent]:
            vk = event.ki.wVk
            if event.ki.dwFlags & keys.KEYEVENTF_KEYUP:
                if vk in self._down:
                    self._down.remove(vk)
            elif vk not in self._down:
                self._down.append(vk)
        assert sent == len(events), "Windows rejected injected keys"

    def release_leftovers(self) -> None:
        """Release keys a failed test left down, so the user never gets a stuck Ctrl or Win."""
        if not self._down:
            return
        events = []
        if any(vk in (keys.VK_LWIN, keys.VK_RWIN, keys.VK_LMENU, keys.VK_RMENU) for vk in self._down):
            events += [press(keys.VK_MASK_KEY), release(keys.VK_MASK_KEY)]
        events += [release(vk) for vk in reversed(self._down)]
        keys.send_input(events)
        self._down.clear()


def close_start_menu_if_open() -> bool:
    """If a test opened the Start menu, close it again (only when the Start menu itself has the focus)."""
    if foreground_process_name() in START_MENU_HOSTS:
        keys.send_input([press(keys.VK_ESCAPE), release(keys.VK_ESCAPE)])
        time.sleep(0.3)
        return True
    return False


def take_focus(qtbot: Any, window: QWidget) -> int:
    """Focus ``window``. Returns the window that had the focus before.

    Only takes the focus from the app that launched the tests, and only while
    nobody is typing; otherwise waits, and skips the test if that doesn't
    happen in time. (Windows' foreground lock refuses other focus changes
    anyway, flashing the taskbar instead.)
    """
    deadline = time.monotonic() + IDLE_WAIT_LIMIT_S
    hwnd = int(window.winId())
    launchers = launcher_pids()
    while True:
        wait_for_idle_user(deadline)
        previous = foreground_window()
        if previous != hwnd and window_process_id(previous) not in launchers:
            if time.monotonic() > deadline:
                pytest.skip("Another app has the focus; desktop tests only take it from the app that started them.")
            time.sleep(1.0)
            continue
        window.show()
        window.raise_()
        window.activateWindow()
        set_foreground(hwnd)
        try:
            qtbot.waitUntil(lambda: foreground_window() == hwnd, timeout=1000)
            return previous
        except WaitTimeout:
            if time.monotonic() > deadline:
                window.hide()
                pytest.skip("Windows didn't give the test window the focus (foreground lock).")
            qtbot.wait(1500)


def give_focus_back(previous: int, window: QWidget) -> None:
    """Hand the focus back while we still own it (Windows only lets the focused app choose), then hide."""
    if previous and previous != int(window.winId()) and is_window(previous):
        set_foreground(previous)
    window.hide()
