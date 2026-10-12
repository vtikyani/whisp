"""Pure parts of text insertion and the clipboard: event building, method tables, format rules.

The paste tests replace every Windows call TextInserter makes with ``FakeWindows``, so nothing real is pasted,
typed or copied.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from PySide6.QtWidgets import QWidget

from whisp.core.types import InsertionResult, TargetWindow
from whisp.platform.win32 import clipboard, keys
from whisp.platform.win32 import insert as insert_module
from whisp.platform.win32.insert import (
    CLIPBOARD_RESTORE_DELAY_S,
    REMOTE_RESTORE_DELAY_S,
    SLOW_READER_RESTORE_DELAY_S,
    TextInserter,
    build_type_events,
    restore_delay_for,
)


def _describe(events: list[keys.INPUT]) -> list[tuple[str, int, bool]]:
    out = []
    for event in events:
        up = bool(event.ki.dwFlags & keys.KEYEVENTF_KEYUP)
        if event.ki.dwFlags & keys.KEYEVENTF_UNICODE:
            out.append(("char", event.ki.wScan, up))
        else:
            out.append(("vk", event.ki.wVk, up))
    return out


def test_type_events_are_unicode_down_up_pairs() -> None:
    assert _describe(build_type_events("Hi")) == [
        ("char", ord("H"), False),
        ("char", ord("H"), True),
        ("char", ord("i"), False),
        ("char", ord("i"), True),
    ]


def test_emoji_become_surrogate_pairs() -> None:
    events = _describe(build_type_events("😀"))
    assert events == [("char", 0xD83D, False), ("char", 0xD83D, True), ("char", 0xDE00, False), ("char", 0xDE00, True)]


def test_newlines_become_enter_or_shift_enter() -> None:
    assert _describe(build_type_events("a\nb")) == [
        ("char", ord("a"), False),
        ("char", ord("a"), True),
        ("vk", keys.VK_RETURN, False),
        ("vk", keys.VK_RETURN, True),
        ("char", ord("b"), False),
        ("char", ord("b"), True),
    ]
    assert _describe(build_type_events("\r\n", messaging=True)) == [
        ("vk", keys.VK_SHIFT, False),
        ("vk", keys.VK_RETURN, False),
        ("vk", keys.VK_RETURN, True),
        ("vk", keys.VK_SHIFT, True),
    ]


def test_every_injected_event_carries_the_marker() -> None:
    events = build_type_events("x\ny😀", messaging=True)
    assert events
    assert all(e.ki.dwExtraInfo == keys.INJECTED_MARKER for e in events)


def test_built_in_methods_for_terminals() -> None:
    assert TextInserter.default_method_for("mintty.exe") == "paste_shift_insert"
    assert TextInserter.default_method_for("PuTTY.EXE") == "paste_shift_insert"
    assert TextInserter.default_method_for("kitty.exe") == "paste_shift_insert"
    assert TextInserter.default_method_for("notepad.exe") is None
    assert TextInserter().default_method_for("") is None


def test_restore_delay_per_app() -> None:
    assert CLIPBOARD_RESTORE_DELAY_S >= 0.6
    assert 2.5 <= SLOW_READER_RESTORE_DELAY_S <= 3.5
    assert 4.5 <= REMOTE_RESTORE_DELAY_S <= 5.5
    for app in ("outlook.exe", "olk.exe", "winword.exe", "whatsapp.exe", "WhatsApp.Root.exe"):
        assert restore_delay_for(app) == SLOW_READER_RESTORE_DELAY_S, app
    for app in ("mstsc.exe", "msrdc.exe", "vmconnect.exe", "VirtualBoxVM.exe", "vmware.exe", " MSTSC.EXE "):
        assert restore_delay_for(app) == REMOTE_RESTORE_DELAY_S, app
    for app in ("notepad.exe", "chrome.exe", "slack.exe", ""):
        assert restore_delay_for(app) == CLIPBOARD_RESTORE_DELAY_S, app
    assert TextInserter.restore_delay_for("outlook.exe") == SLOW_READER_RESTORE_DELAY_S


def _insert(text: str, method: str) -> InsertionResult:
    return TextInserter().insert(text, target=None, method=method, restore_clipboard=True, on_focus_change="insert")


def test_unknown_method_fails_without_touching_anything() -> None:
    result = _insert("hello", "telepathy")
    assert isinstance(result, InsertionResult)
    assert not result.ok
    assert result.error and "telepathy" in result.error


def test_empty_text_is_a_successful_no_op() -> None:
    assert _insert("", "type") == InsertionResult(ok=True, method="type")
    assert _insert("", "paste_shift_insert") == InsertionResult(ok=True, method="paste")


NOTEPAD = TargetWindow(hwnd=101, pid=11, process_name="notepad.exe", app_name="Notepad", title="")
CTRL_V = [
    ("vk", keys.VK_CONTROL, False),
    ("vk", keys.VK_V, False),
    ("vk", keys.VK_V, True),
    ("vk", keys.VK_CONTROL, True),
]


class FakeWindows:
    """The focused window, held modifiers, keystrokes sent and clipboard contents, all simulated."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, focused: int, restore_delay: float = 0.3) -> None:
        self.focused = focused
        self.modifiers_down = False
        self.sent: list[list[tuple[str, int, bool]]] = []
        self.clipboard = ["the user's own clipboard"]  # every content it had, newest last
        self.sequence = 1
        self.while_saving: Callable[[], None] = lambda: None
        """Runs while the clipboard is being saved (which can take a while)."""
        self.own_hidden: set[int] = set()
        """Whisp's own invisible windows, such as the one that owns the tray menu."""
        self.qt_apps: set[int] = set()
        """Windows of apps built with Qt, where typed emoji land out of order."""
        monkeypatch.setattr(insert_module, "foreground_window", lambda: self.focused)
        monkeypatch.setattr(insert_module, "is_own_hidden_window", lambda hwnd: hwnd in self.own_hidden)
        monkeypatch.setattr(insert_module, "_types_emoji_out_of_order", lambda hwnd: hwnd in self.qt_apps)
        monkeypatch.setattr(insert_module, "window_process_id", lambda hwnd: 4242)
        monkeypatch.setattr(insert_module, "window_thread_id", lambda hwnd: 0)
        monkeypatch.setattr(insert_module, "current_process_elevated", lambda: False)
        monkeypatch.setattr(insert_module, "is_process_elevated", lambda pid: False)
        monkeypatch.setattr(insert_module, "restore_delay_for", lambda process_name: restore_delay)
        monkeypatch.setattr(keys, "any_modifier_down", lambda: self.modifiers_down)
        monkeypatch.setattr(keys, "send_input", self._send)
        monkeypatch.setattr(clipboard, "snapshot", self._snapshot)
        monkeypatch.setattr(clipboard, "set_text", self._set_text)
        monkeypatch.setattr(clipboard, "restore", self._restore)
        monkeypatch.setattr(clipboard, "get_sequence_number", lambda: self.sequence)

    def _send(self, events: list[keys.INPUT]) -> int:
        self.sent.append(_describe(events))
        return len(events)

    def _snapshot(self) -> clipboard.ClipboardSnapshot:
        self.while_saving()
        data = self.clipboard[-1].encode("utf-16-le")
        return clipboard.ClipboardSnapshot(((clipboard.CF_UNICODETEXT, data),), self.sequence)

    def _set_text(self, text: str, *, exclude_from_history: bool = True) -> None:
        self.clipboard.append(text)
        self.sequence += 1

    def _restore(self, snap: clipboard.ClipboardSnapshot) -> None:
        self.clipboard.append(snap.formats[0][1].decode("utf-16-le"))
        self.sequence += 1


def _paste(inserter: TextInserter, text: str = "dictated words") -> InsertionResult:
    return inserter.insert(text, target=NOTEPAD, method="paste", restore_clipboard=True, on_focus_change="clipboard")


def test_paste_returns_once_the_text_is_in_and_restores_the_clipboard_afterwards(monkeypatch) -> None:
    windows = FakeWindows(monkeypatch, focused=NOTEPAD.hwnd, restore_delay=1.0)
    inserter = TextInserter()
    started = time.monotonic()
    result = _paste(inserter)
    assert time.monotonic() - started < 0.5  # not held up while the app reads the paste
    assert (result.ok, result.method, result.focus_changed) == (True, "paste", False)
    assert windows.sent == [CTRL_V]
    assert windows.clipboard[-1] == "dictated words"
    # Other clipboard work waits for the restore, so it can never save the pasted text as the user's.
    assert inserter.copy_to_clipboard("copied later")
    assert windows.clipboard == [
        "the user's own clipboard",
        "dictated words",
        "the user's own clipboard",
        "copied later",
    ]


def test_no_paste_into_a_window_the_user_switched_to_while_the_clipboard_was_saved(monkeypatch) -> None:
    windows = FakeWindows(monkeypatch, focused=NOTEPAD.hwnd)
    windows.while_saving = lambda: setattr(windows, "focused", 999)
    result = _paste(TextInserter())
    assert (result.ok, result.method, result.focus_changed) == (True, "clipboard", True)
    assert windows.sent == []  # no Ctrl+V into the other window
    assert windows.clipboard[-1] == "dictated words"  # waiting there to be pasted by hand


def test_no_paste_while_a_key_pressed_during_the_save_is_held(monkeypatch) -> None:
    monkeypatch.setattr(insert_module, "MODIFIER_RELEASE_TIMEOUT_S", 0.05)
    windows = FakeWindows(monkeypatch, focused=NOTEPAD.hwnd)
    windows.while_saving = lambda: setattr(windows, "modifiers_down", True)  # e.g. Ctrl+Win for the next dictation
    result = _paste(TextInserter())
    assert (result.ok, result.method) == (True, "clipboard")
    assert windows.sent == []  # Ctrl+V under a held Win would be a different shortcut
    assert windows.clipboard[-1] == "dictated words"


TRAY_MENU_OWNER = TargetWindow(hwnd=202, pid=4242, process_name="whisp.exe", app_name="Whisp", title="")


def test_text_is_copied_not_inserted_when_whisps_own_hidden_window_has_the_focus(monkeypatch) -> None:
    # Starting and stopping from the tray menu can leave the menu's invisible window focused; text sent there is lost.
    windows = FakeWindows(monkeypatch, focused=TRAY_MENU_OWNER.hwnd)
    windows.own_hidden.add(TRAY_MENU_OWNER.hwnd)
    for method in ("paste", "type"):
        result = TextInserter().insert(
            "dictated words",
            target=TRAY_MENU_OWNER,
            method=method,
            restore_clipboard=True,
            on_focus_change="clipboard",
        )
        assert (result.ok, result.method, result.error) == (True, "clipboard", insert_module.MSG_NO_WINDOW)
    assert windows.sent == []
    assert windows.clipboard[-1] == "dictated words"  # waiting there to be pasted by hand


def _type(text: str) -> InsertionResult:
    inserter = TextInserter()
    return inserter.insert(text, target=NOTEPAD, method="type", restore_clipboard=True, on_focus_change="clipboard")


def test_emoji_for_a_qt_app_are_pasted_so_they_stay_in_place(monkeypatch) -> None:
    windows = FakeWindows(monkeypatch, focused=NOTEPAD.hwnd)
    windows.qt_apps.add(NOTEPAD.hwnd)
    result = _type("thanks 👍")  # typed, Qt would show "👍thanks "
    assert (result.ok, result.method) == (True, "paste")
    assert windows.sent == [CTRL_V]
    windows.sent.clear()
    assert _type("thanks — see you").method == "type"  # no emoji: typed as asked
    windows.qt_apps.clear()
    assert _type("thanks 👍").method == "type"  # other apps keep typed emoji in place


def test_only_qt_windows_put_typed_emoji_out_of_order(qtbot) -> None:
    qt_window = QWidget()
    qtbot.addWidget(qt_window)
    assert insert_module._types_emoji_out_of_order(int(qt_window.winId()))
    assert insert_module._QT_WINDOW_CLASS.match("Qt5152QWindowIcon")  # Qt 5 apps, such as older Telegram builds
    assert not insert_module._QT_WINDOW_CLASS.match("Chrome_WidgetWin_1")
    assert not insert_module._QT_WINDOW_CLASS.match("RICHEDIT50W")
    assert not insert_module._types_emoji_out_of_order(0)


def test_gdi_and_private_formats_are_not_copied_as_memory() -> None:
    for fmt in (clipboard.CF_BITMAP, clipboard.CF_ENHMETAFILE, clipboard.CF_METAFILEPICT, clipboard.CF_PALETTE):
        assert not clipboard.is_memory_format(fmt)
    assert not clipboard.is_memory_format(clipboard.CF_OWNERDISPLAY)
    assert not clipboard.is_memory_format(0x0201)  # private range
    assert not clipboard.is_memory_format(0x0305)  # GDI object range
    for fmt in (clipboard.CF_UNICODETEXT, clipboard.CF_DIB, clipboard.CF_DIBV5, clipboard.CF_HDROP, 0xC123):
        assert clipboard.is_memory_format(fmt)


def test_decoding_clipboard_text_stops_at_the_terminator() -> None:
    raw = "héllo 😀".encode("utf-16-le") + b"\x00\x00garbage"
    assert clipboard._decode_text(raw) == "héllo 😀"
    assert clipboard._decode_text("odd".encode("utf-16-le") + b"\x00") == "odd"


def test_snapshot_repr_never_shows_contents() -> None:
    snap = clipboard.ClipboardSnapshot(((clipboard.CF_UNICODETEXT, "secret".encode("utf-16-le")),), sequence=7)
    assert "secret" not in repr(snap)
    assert "s\\x00e" not in repr(snap)
    assert snap.size_bytes == 12
    assert snap.format_ids == (clipboard.CF_UNICODETEXT,)
    assert not snap.is_empty
    assert clipboard.ClipboardSnapshot((), 0).is_empty


def test_format_names() -> None:
    assert clipboard.format_name(clipboard.CF_UNICODETEXT) == "CF_UNICODETEXT"
    fmt = clipboard.registered_format("Whisp Test Format")
    assert fmt >= 0xC000
    assert clipboard.format_name(fmt) == "Whisp Test Format"
    assert clipboard.registered_format("Whisp Test Format") == fmt
