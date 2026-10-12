"""keys.key_kind: what a key press does to the text, for smart spacing."""

from __future__ import annotations

import pytest

from whisp.core.types import KeyKind
from whisp.platform.win32 import keys

VK_B, VK_D, VK_F, VK_S, VK_Z = 0x42, 0x44, 0x46, 0x53, 0x5A


@pytest.mark.parametrize(
    ("vk", "kind"),
    [
        (keys.VK_SPACE, KeyKind.SPACE),
        (keys.VK_TAB, KeyKind.SPACE),
        (keys.VK_RETURN, KeyKind.NEWLINE),
        (keys.VK_BACK, KeyKind.EDIT),
        (keys.VK_DELETE, KeyKind.EDIT),
        (keys.VK_LEFT, KeyKind.MOVE),
        (keys.VK_HOME, KeyKind.MOVE),
        (keys.VK_NEXT, KeyKind.MOVE),
        (0x41, KeyKind.TEXT),  # A
        (0x37, KeyKind.TEXT),  # 7
        (0xBE, KeyKind.TEXT),  # period
        (0x65, KeyKind.TEXT),  # numpad 5
    ],
)
def test_plain_keys(vk: int, kind: KeyKind) -> None:
    assert keys.key_kind(vk) is kind


def test_shift_still_types() -> None:
    assert keys.key_kind(0x41, shift=True) is KeyKind.TEXT  # a capital A
    assert keys.key_kind(keys.VK_LEFT, shift=True) is KeyKind.MOVE  # selecting


@pytest.mark.parametrize(
    ("vk", "held"),
    [
        (keys.VK_V, {"ctrl": True}),  # paste
        (keys.VK_V, {"ctrl": True, "shift": True}),  # paste as plain text
        (VK_Z, {"ctrl": True}),  # undo
        (VK_F, {"ctrl": True}),  # find
        (VK_F, {"alt": True}),  # the File menu
        (VK_Z, {"ctrl": True, "alt": True}),
        (keys.VK_SPACE, {"ctrl": True}),
        (keys.VK_TAB, {"ctrl": True}),  # another browser tab
    ],
)
def test_shortcuts_change_text_unpredictably(vk: int, held: dict[str, bool]) -> None:
    # The text changed in a way Whisp can't see, or the cursor may be anywhere now.
    assert keys.key_kind(vk, **held) is KeyKind.EDIT


def test_shift_insert_pastes() -> None:
    assert keys.key_kind(keys.VK_INSERT, shift=True) is KeyKind.EDIT


def test_enter_and_navigation_keep_their_kind_with_modifiers() -> None:
    assert keys.key_kind(keys.VK_RETURN, ctrl=True) is KeyKind.NEWLINE
    assert keys.key_kind(keys.VK_LEFT, ctrl=True) is KeyKind.MOVE
    assert keys.key_kind(keys.VK_BACK, ctrl=True) is KeyKind.EDIT


@pytest.mark.parametrize(
    ("vk", "held"),
    [
        (keys.VK_TAB, {"alt": True}),  # Alt+Tab
        (keys.VK_TAB, {"alt": True, "shift": True}),
        (keys.VK_TAB, {"ctrl": True, "alt": True}),  # the Alt+Tab switcher that stays open
        (keys.VK_ESCAPE, {"alt": True}),  # Alt+Esc
        (keys.VK_TAB, {"win": True}),  # Task view
        (VK_D, {"win": True}),  # show the desktop
        (keys.VK_LEFT, {"win": True}),  # snap the window
        (0x31, {"win": True}),  # Win+1: the first app on the taskbar
        (keys.VK_C, {"ctrl": True}),  # copy
        (keys.VK_C, {"ctrl": True, "shift": True}),  # copy in a terminal
        (keys.VK_INSERT, {"ctrl": True}),  # copy
        (VK_S, {"ctrl": True}),  # save
        (VK_B, {"ctrl": True}),  # bold
    ],
)
def test_shortcuts_that_leave_the_text_and_cursor_alone(vk: int, held: dict[str, bool]) -> None:
    # Switching to another app and back, or saving, mustn't make the next dictation lose its space.
    assert keys.key_kind(vk, **held) is None


def test_copy_with_ctrl_and_alt_is_a_different_shortcut() -> None:
    assert keys.key_kind(keys.VK_C, ctrl=True, alt=True) is KeyKind.EDIT


# Esc, F1, Insert, Print Screen, volume mute
@pytest.mark.parametrize("vk", [keys.VK_ESCAPE, 0x70, keys.VK_INSERT, keys.VK_SNAPSHOT, 0xAD])
def test_keys_that_dont_touch_the_text(vk: int) -> None:
    assert keys.key_kind(vk) is None
