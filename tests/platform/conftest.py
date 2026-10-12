"""Fixtures for the platform tests. Desktop safety rules are described in ``desktop_support``."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from desktop_support import Desktop, DesktopTestWindow, close_start_menu_if_open, give_focus_back, take_focus

from whisp.platform.win32 import clipboard


@pytest.fixture
def clipboard_guard() -> Iterator[None]:
    """Save the user's clipboard and put it back afterwards, whatever happens."""
    saved = clipboard.snapshot()
    try:
        yield
    finally:
        clipboard.restore(saved)


@pytest.fixture(scope="session")
def test_window(qapp) -> Iterator[DesktopTestWindow]:
    """One test window for the whole run, which hands the focus back when the run ends.

    Windows lets an app take the focus only while it has it (or once, just after it starts). A window that gave
    the focus back after each test could never take it again, so the first desktop test takes it and the rest
    keep it.
    """
    window = DesktopTestWindow()
    try:
        yield window
    finally:
        if window.focus_from is not None:
            give_focus_back(window.focus_from, window)
        window.dispose()


@pytest.fixture
def desktop(qtbot, test_window: DesktopTestWindow, clipboard_guard: None) -> Iterator[Desktop]:
    """Our test window, focused — only while the user is idle. It keeps the focus for the next test."""
    previous = take_focus(qtbot, test_window)
    if test_window.focus_from is None:
        test_window.focus_from = previous
    test_window.reset()
    test_window.keys.entries.clear()
    desk = Desktop(test_window)
    try:
        yield desk
    finally:
        desk.release_leftovers()
        close_start_menu_if_open()
