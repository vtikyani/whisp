"""Focused-window queries that only read state (no focus changes, no input)."""

from __future__ import annotations

import os
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from whisp.core.types import AppCategory, TargetWindow
from whisp.platform.win32 import foreground
from whisp.platform.win32.foreground import (
    app_name_for,
    covers_monitor,
    current_process_elevated,
    get_foreground_target,
    is_process_elevated,
)


def test_covers_monitor() -> None:
    monitor = (0, 0, 1920, 1080)
    assert covers_monitor((0, 0, 1920, 1080), monitor)
    assert covers_monitor((-8, -8, 1928, 1088), monitor)
    assert not covers_monitor((-8, -8, 1928, 1040), monitor)  # maximised above the taskbar
    assert not covers_monitor((1920, 0, 3840, 1080), monitor)


def test_app_names_come_from_version_info() -> None:
    assert app_name_for(sys.executable) == "Python"
    windir = os.environ.get("WINDIR", r"C:\Windows")
    assert app_name_for(os.path.join(windir, "explorer.exe")) == "Windows Explorer"


def test_app_name_falls_back_to_the_file_name() -> None:
    assert app_name_for(r"C:\nowhere\slack.exe") == "Slack"
    assert app_name_for(r"C:\nowhere\WINWORD.EXE") == "Winword"
    assert app_name_for(r"C:\nowhere\WindowsTerminal.exe") == "WindowsTerminal"


def test_elevation_queries() -> None:
    assert isinstance(current_process_elevated(), bool)
    assert is_process_elevated(os.getpid()) == current_process_elevated()
    assert is_process_elevated(0) is None
    assert is_process_elevated(-1) is None
    # The System process can't be opened from a normal process; that's "don't know", not an error.
    assert is_process_elevated(4) in (None, True)


def test_foreground_target_is_fast_and_well_formed() -> None:
    target = get_foreground_target()  # warm the version-info cache
    started = time.perf_counter()
    target = get_foreground_target()
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 50  # typically well under 1 ms; generous for a loaded test machine
    if target is None:
        return  # e.g. the lock screen or a desktop switch: nothing has the focus
    assert isinstance(target, TargetWindow)
    assert target.hwnd > 0
    assert target.category is AppCategory.OTHER
    assert target.process_name == target.process_name.lower()
    if target.monitor_rect is not None:
        left, top, right, bottom = target.monitor_rect
        assert right > left and bottom > top


def test_target_for_a_window_that_does_not_exist() -> None:
    assert foreground.target_for_window(0) is None


def test_the_watcher_says_when_it_is_watching(qtbot) -> None:
    watcher = foreground.ForegroundWatcher()
    assert not watcher.is_watching
    watcher.start()
    try:
        qtbot.waitUntil(lambda: watcher.is_watching, timeout=2000)
    finally:
        watcher.stop()
    assert not watcher.is_watching


def test_own_hidden_windows_are_told_apart_from_visible_and_other_apps_windows(qtbot, monkeypatch) -> None:
    window = QWidget()
    window.setWindowFlags(
        Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowDoesNotAcceptFocus
    )
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.resize(1, 1)
    qtbot.addWidget(window)
    hwnd = int(window.winId())  # a real window that is still invisible, like the one that owns the tray menu
    assert foreground.is_own_hidden_window(hwnd)
    window.show()  # never takes the focus
    assert not foreground.is_own_hidden_window(hwnd)  # e.g. the main window's practice box: fine to dictate into
    assert not foreground.is_own_hidden_window(0)

    desktop = int(foreground._GetDesktopWindow())
    monkeypatch.setattr(foreground, "_IsWindowVisible", lambda hwnd: False)
    assert not foreground.is_own_hidden_window(desktop)  # hidden or not, another app's window is never Whisp's
