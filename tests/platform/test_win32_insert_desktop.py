"""TextInserter against a real Qt window that this test creates and focuses."""

from __future__ import annotations

import dataclasses

import pytest
from desktop_support import Desktop, run_in_worker, wait_until_equal
from PySide6.QtCore import Qt

from whisp.core.types import AppCategory, TargetWindow
from whisp.platform.win32 import clipboard, keys
from whisp.platform.win32.foreground import get_foreground_target
from whisp.platform.win32.insert import TextInserter

pytestmark = pytest.mark.desktop


def _target(desktop: Desktop) -> TargetWindow:
    desktop.ensure_focus()
    target = get_foreground_target()
    assert target is not None and target.hwnd == desktop.window.hwnd
    return target


def _insert(qtbot, desktop: Desktop, text: str, *, method: str, target: TargetWindow | None, restore: bool = True):
    desktop.ensure_focus()
    return run_in_worker(
        qtbot,
        lambda: TextInserter().insert(
            text, target=target, method=method, restore_clipboard=restore, on_focus_change="clipboard"
        ),
    )


def test_paste_inserts_text_and_restores_the_clipboard(qtbot, desktop: Desktop) -> None:
    clipboard.set_text("the user's clipboard")
    text = "Hello from Whisp — ünïcödé 😀"
    result = _insert(qtbot, desktop, text, method="paste", target=_target(desktop))
    assert result.ok, result.error
    assert result.method == "paste"
    assert not result.focus_changed
    qtbot.waitUntil(lambda: desktop.window.line.text() == text, timeout=2000)
    # Restored in the background once the app has had time to read the paste.
    qtbot.waitUntil(lambda: clipboard.get_text() == "the user's clipboard", timeout=3000)


def test_paste_with_shift_insert(qtbot, desktop: Desktop) -> None:
    clipboard.set_text("kept")
    result = _insert(qtbot, desktop, "via shift insert", method="paste_shift_insert", target=_target(desktop))
    assert result.ok and result.method == "paste"
    qtbot.waitUntil(lambda: desktop.window.line.text() == "via shift insert", timeout=2000)
    qtbot.waitUntil(lambda: clipboard.get_text() == "kept", timeout=3000)


def test_paste_without_restoring_leaves_the_text_on_the_clipboard(qtbot, desktop: Desktop) -> None:
    clipboard.set_text("old")
    result = _insert(qtbot, desktop, "left behind", method="paste", target=_target(desktop), restore=False)
    assert result.ok
    qtbot.waitUntil(lambda: desktop.window.line.text() == "left behind", timeout=2000)
    assert clipboard.get_text() == "left behind"


def test_type_handles_emoji_and_line_breaks(qtbot, desktop: Desktop) -> None:
    desktop.window.focus_rich()  # a native Windows text box, like most apps have
    clipboard.set_text("untouched")
    seq = clipboard.get_sequence_number()
    text = "line one 👍\nline two — done"
    result = _insert(qtbot, desktop, text, method="type", target=_target(desktop))
    assert result.ok and result.method == "type"
    wait_until_equal(qtbot, desktop.window.rich_text, text)
    assert clipboard.get_sequence_number() == seq  # typing never touches the clipboard


def test_emoji_for_a_qt_text_box_are_pasted_to_stay_in_place(qtbot, desktop: Desktop) -> None:
    # Typed into a Qt app (Telegram, Zoom, OBS…), "thanks 👍" would arrive as "👍thanks ".
    desktop.window.edit.setFocus()
    clipboard.set_text("the user's clipboard")
    text = "thanks 👍 see you"
    result = _insert(qtbot, desktop, text, method="type", target=_target(desktop))
    assert result.ok and result.method == "paste"
    wait_until_equal(qtbot, desktop.window.edit.toPlainText, text)
    wait_until_equal(qtbot, clipboard.get_text, "the user's clipboard")


def test_type_uses_shift_enter_in_messaging_apps(qtbot, desktop: Desktop) -> None:
    desktop.window.edit.setFocus()
    target = dataclasses.replace(_target(desktop), category=AppCategory.MESSAGING)
    result = _insert(qtbot, desktop, "hi\nthere", method="type", target=target)
    assert result.ok
    qtbot.waitUntil(lambda: "there" in desktop.window.edit.toPlainText(), timeout=3000)
    enter_presses = [mods for pressed, vk, mods in desktop.window.keys.entries if pressed and vk == keys.VK_RETURN]
    assert enter_presses, "Enter never reached the window"
    assert all(mods & Qt.KeyboardModifier.ShiftModifier.value for mods in enter_presses)


def test_focus_change_leaves_text_on_the_clipboard(qtbot, desktop: Desktop) -> None:
    elsewhere = dataclasses.replace(_target(desktop), hwnd=desktop.window.hwnd + 1_000_000)
    result = _insert(qtbot, desktop, "for another window", method="paste", target=elsewhere)
    assert result.ok
    assert result.method == "clipboard"
    assert result.focus_changed
    assert clipboard.get_text() == "for another window"
    assert desktop.window.line.text() == ""


def test_no_target_inserts_into_whatever_is_focused(qtbot, desktop: Desktop) -> None:
    result = _insert(qtbot, desktop, "no target given", method="paste", target=None)
    assert result.ok and result.method == "paste"
    qtbot.waitUntil(lambda: desktop.window.line.text() == "no target given", timeout=2000)


def test_capture_selection_reads_selected_text_and_restores_clipboard(qtbot, desktop: Desktop) -> None:
    desktop.window.line.setText("select me please")
    desktop.window.line.selectAll()
    clipboard.set_text("before capture")
    desktop.ensure_focus()
    selected = run_in_worker(qtbot, lambda: TextInserter().capture_selection())
    assert selected == "select me please"
    assert clipboard.get_text() == "before capture"


def test_capture_selection_without_a_selection_returns_none(qtbot, desktop: Desktop) -> None:
    desktop.window.line.setText("nothing selected")
    desktop.window.line.deselect()
    clipboard.set_text("unchanged")
    seq = clipboard.get_sequence_number()
    desktop.ensure_focus()
    assert run_in_worker(qtbot, lambda: TextInserter().capture_selection(timeout_ms=300)) is None
    assert clipboard.get_sequence_number() == seq
    assert clipboard.get_text() == "unchanged"
