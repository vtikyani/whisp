"""HotkeyService's keyboard-hook logic, driven with simulated key events.

Nothing here installs a hook, registers a shortcut or sends input: the service's thread never starts, and the
key state Windows would report is simulated (updated after the hook, as Windows does, and only for keys the hook
let through).
"""

from __future__ import annotations

from typing import Any

import pytest

from whisp.core.types import KeyKind
from whisp.platform.win32 import hotkeys as hotkeys_module
from whisp.platform.win32 import keys
from whisp.platform.win32.hotkeys import KBDLLHOOKSTRUCT, LLKHF_UP, HotkeyService, validate_bindings

VK_L = 0x4C
_EITHER_SIDE = {
    keys.VK_CONTROL: (keys.VK_LCONTROL, keys.VK_RCONTROL),
    keys.VK_MENU: (keys.VK_LMENU, keys.VK_RMENU),
    keys.VK_SHIFT: (keys.VK_LSHIFT, keys.VK_RSHIFT),
}


class Keyboard:
    def __init__(self, service: HotkeyService, monkeypatch: pytest.MonkeyPatch) -> None:
        self.service = service
        self.down: set[int] = set()
        self.focused = 100
        """The window that has the focus."""
        monkeypatch.setattr(keys, "is_key_down", self.is_down)
        monkeypatch.setattr(keys, "send_input", lambda events: len(events))  # mask keys go nowhere
        monkeypatch.setattr(hotkeys_module, "foreground_window", lambda: self.focused)

    def is_down(self, vk: int) -> bool:
        return vk in self.down or any(side in self.down for side in _EITHER_SIDE.get(vk, ()))

    def press(self, vk: int, *, scan: int = 0) -> bool:
        return self._event(vk, up=False, scan=scan)

    def release(self, vk: int, *, scan: int = 0) -> bool:
        return self._event(vk, up=True, scan=scan)

    def chord(self, *vks: int) -> None:
        """Press the keys in order, then let go in reverse."""
        for vk in vks:
            self.press(vk)
        for vk in reversed(vks):
            self.release(vk)

    def typed_in(self, hwnd: int) -> KeyKind | None:
        """The kind of the last key smart spacing saw typed into ``hwnd``."""
        activity = self.service._activity.get(hwnd)
        return activity.kind if activity is not None else None

    def _event(self, vk: int, *, up: bool, scan: int = 0) -> bool:
        swallowed = self.service._on_key(KBDLLHOOKSTRUCT(vkCode=vk, scanCode=scan, flags=LLKHF_UP if up else 0))
        if not swallowed:
            (self.down.discard if up else self.down.add)(vk)
        return swallowed

    def events(self) -> list[tuple[str, tuple[Any, ...]]]:
        out = list(self.service._outbox)
        self.service._outbox.clear()
        return out


def make_service(**bindings: str) -> HotkeyService:
    service = HotkeyService()
    chords, errors = validate_bindings(bindings)
    assert not errors
    service._bindings = chords
    service._apply_tracker_bindings()
    return service


@pytest.mark.parametrize(
    ("away", "back"),
    [
        # Win+L locks the PC; Win and L come up on the lock screen. Then Win is pressed first.
        ((keys.VK_LWIN, VK_L), (keys.VK_LWIN, keys.VK_LCONTROL)),
        # Ctrl+Alt+Del opens the security screen; the keys come up there. Then Ctrl is pressed first.
        ((keys.VK_LCONTROL, keys.VK_LMENU, keys.VK_DELETE), (keys.VK_LCONTROL, keys.VK_LWIN)),
    ],
    ids=["win+l", "ctrl+alt+del"],
)
def test_keys_let_go_where_the_hook_cant_see_dont_block_the_next_press(
    qapp, monkeypatch: pytest.MonkeyPatch, away: tuple[int, ...], back: tuple[int, ...]
) -> None:
    service = make_service(dictate="ctrl+win")
    kb = Keyboard(service, monkeypatch)
    for vk in away:
        kb.press(vk)
    kb.down.clear()  # released on the secure desktop: the hook never hears about it
    kb.events()
    for vk in back:
        kb.press(vk)
    assert kb.events() == [("pressed", ("dictate",))]


def test_auto_repeat_is_still_not_a_new_press(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    service = make_service(dictate="ctrl+win")
    kb = Keyboard(service, monkeypatch)
    kb.press(keys.VK_LCONTROL)
    kb.press(keys.VK_LWIN)
    assert kb.events() == [("pressed", ("dictate",))]
    kb.press(keys.VK_LWIN)  # held down: Windows repeats the key
    kb.press(keys.VK_LCONTROL)
    assert kb.events() == []
    kb.release(keys.VK_LWIN)
    assert kb.events() == [("released", ("dictate",))]


def test_esc_cancels_while_a_registered_shortcut_is_held(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    # RegisterHotKey only catches Esc on its own; with Ctrl+Shift still held, Windows would see Ctrl+Shift+Esc
    # and open Task Manager instead.
    service = make_service(dictate="ctrl+shift+space")
    chord = service._bindings["dictate"]
    service._registered = {"dictate": chord}
    assert not service._tracking()  # no recording: the hook isn't needed for this shortcut
    service._cancel_registered = True  # recording: Esc cancels
    assert service._tracking()
    kb = Keyboard(service, monkeypatch)
    for vk in (keys.VK_LCONTROL, keys.VK_LSHIFT, keys.VK_SPACE):
        assert not kb.press(vk)
    service._held_registered = {"dictate": chord}  # what the WM_HOTKEY for it records
    kb.events()
    assert kb.press(keys.VK_ESCAPE)  # swallowed: the app (and Windows) never see Esc
    assert kb.events() == [("pressed", ("cancel",))]
    assert kb.release(keys.VK_ESCAPE)


ALTGR_CTRL_SCAN = 0x21D
"""The made-up Left Ctrl that Windows sends ahead of Right Alt on keyboard layouts with AltGr."""


def test_letters_typed_with_altgr_count_as_typing(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    # Polish "się" ends with AltGr+E. Smart spacing must see a letter, or the next dictation is glued on.
    service = make_service(dictate="ctrl+win")
    kb = Keyboard(service, monkeypatch)
    kb.press(keys.VK_LCONTROL, scan=ALTGR_CTRL_SCAN)
    kb.press(keys.VK_RMENU)
    kb.press(0x45)  # E
    assert kb.typed_in(kb.focused) is KeyKind.TEXT
    for vk, scan in ((0x45, 0), (keys.VK_RMENU, 0), (keys.VK_LCONTROL, ALTGR_CTRL_SCAN)):
        kb.release(vk, scan=scan)
    # Real shortcuts still count as editing.
    kb.press(keys.VK_LCONTROL)
    kb.press(0x56)  # Ctrl+V
    assert kb.typed_in(kb.focused) is KeyKind.EDIT
    kb.release(0x56)
    kb.release(keys.VK_LCONTROL)
    kb.press(keys.VK_LMENU)
    kb.press(keys.VK_LCONTROL)
    kb.press(0x45)  # Ctrl+Left Alt+E isn't AltGr
    assert kb.typed_in(kb.focused) is KeyKind.EDIT


def test_typing_is_kept_per_window_and_switching_apps_isnt_typing(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    # Dictate into an email, Alt+Tab to a chat, send a message, Alt+Tab back: the email's cursor hasn't moved.
    service = make_service(dictate="ctrl+win")
    kb = Keyboard(service, monkeypatch)
    email, chat = 100, 200
    kb.focused = email
    kb.chord(0x48)  # H
    kb.chord(keys.VK_LMENU, keys.VK_TAB)
    kb.focused = chat
    kb.chord(0x4F)  # O
    kb.chord(keys.VK_RETURN)
    kb.chord(keys.VK_LMENU, keys.VK_TAB)
    kb.focused = email
    kb.chord(keys.VK_LCONTROL, 0x53)  # Ctrl+S saves; the cursor stays put
    assert kb.typed_in(email) is KeyKind.TEXT
    assert kb.typed_in(chat) is KeyKind.NEWLINE


def test_what_was_typed_is_forgotten_when_the_hook_restarts(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keys pressed while the hook wasn't running went unseen, so what was known before no longer holds.
    service = make_service(dictate="ctrl+win")
    kb = Keyboard(service, monkeypatch)
    kb.chord(0x48)  # H
    typed = service._activity[kb.focused]
    service._forget_typing()  # what installing or reinstalling the hook does
    assert kb.typed_in(kb.focused) is None
    baseline = service._activity_baseline
    assert baseline.serial > typed.serial  # never matches a serial recorded before
    assert baseline.at == float("-inf")  # and never counts as recent typing
    kb.chord(0x48)
    assert service._activity[kb.focused].serial > baseline.serial


@pytest.mark.parametrize(
    ("change", "locked"),
    [
        (hotkeys_module.WTS_SESSION_LOCK, True),
        (hotkeys_module.WTS_REMOTE_DISCONNECT, True),
        (hotkeys_module.WTS_CONSOLE_DISCONNECT, True),
        (0x8, False),  # WTS_SESSION_UNLOCK
        (0x5, False),  # WTS_SESSION_LOGON
    ],
)
def test_locking_the_pc_is_reported(qapp, change: int, locked: bool) -> None:
    service = HotkeyService()
    signals: list[str] = []
    service.session_locked.connect(lambda: signals.append("locked"))
    assert service._on_window_message(0, hotkeys_module.WM_WTSSESSION_CHANGE, change, 1) == 0
    assert signals == (["locked"] if locked else [])


def test_esc_with_nothing_held_is_left_to_the_registered_hotkey(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    service = make_service(dictate="ctrl+shift+space")
    service._registered = {"dictate": service._bindings["dictate"]}
    service._cancel_registered = True
    kb = Keyboard(service, monkeypatch)
    assert not kb.press(keys.VK_ESCAPE)  # RegisterHotKey reports this one
    assert kb.events() == []
