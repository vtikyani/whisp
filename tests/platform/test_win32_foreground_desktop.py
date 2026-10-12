"""Focus tracking with real windows. No keys are sent; the test only moves the focus between its own windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from desktop_support import Desktop, DesktopTestWindow, set_foreground, wait_until_equal
from PySide6.QtCore import Qt

from whisp.platform.win32.foreground import ForegroundWatcher, foreground_window, get_foreground_target

pytestmark = pytest.mark.desktop


def test_target_describes_our_window(desktop: Desktop) -> None:
    target = get_foreground_target()
    assert target is not None
    assert target.hwnd == desktop.window.hwnd
    assert target.pid == os.getpid()
    assert target.process_name == Path(sys.executable).name.lower()
    assert target.app_name == "Python"
    assert target.title == "Whisp platform test window"
    assert target.is_fullscreen is False
    assert target.monitor_rect is not None


def test_target_reports_a_fullscreen_window(qtbot, desktop: Desktop) -> None:
    desktop.window.showFullScreen()
    try:
        qtbot.waitUntil(lambda: get_foreground_target().is_fullscreen, timeout=2000)
    finally:
        desktop.window.showNormal()
    qtbot.waitUntil(lambda: not get_foreground_target().is_fullscreen, timeout=2000)


def test_watcher_reports_focus_changes(qtbot, desktop: Desktop) -> None:
    second = DesktopTestWindow("Whisp platform test window 2")
    second.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)  # the focus moves only when the test says
    watcher = ForegroundWatcher()
    reported: list[int | None] = []
    watcher.changed.connect(lambda target: reported.append(target.hwnd if target else None))
    watcher.start()
    try:
        qtbot.waitUntil(lambda: watcher.is_watching, timeout=2000)
        second.show()
        second.activateWindow()
        set_foreground(second.hwnd)
        wait_until_equal(qtbot, lambda: second.hwnd in reported, True, describe=lambda: _focus_report(reported))
        reported.clear()
        desktop.window.activateWindow()
        set_foreground(desktop.window.hwnd)
        wait_until_equal(qtbot, lambda: desktop.window.hwnd in reported, True, describe=lambda: _focus_report(reported))
        assert foreground_window() == desktop.window.hwnd
    finally:
        watcher.stop()
        watcher.stop()  # idempotent
        second.dispose()
    assert not watcher.is_watching


def _focus_report(reported: list[int | None]) -> str:
    return f"the watcher reported {reported}; the focus is on {foreground_window()}"
