"""Chord parsing, validation, formatting and key-cap labels (no desktop interaction)."""

from __future__ import annotations

import pytest

from whisp.platform.win32 import keys
from whisp.platform.win32.keys import Chord, chord_from_keys, display_chord, format_chord, parse_chord


@pytest.mark.parametrize(
    ("text", "canonical"),
    [
        ("ctrl+win", "ctrl+win"),
        ("Ctrl + Shift + Space", "ctrl+shift+space"),
        ("f8", "f8"),
        ("F24", "f24"),
        ("rctrl", "rctrl"),
        ("Right Ctrl", "rctrl"),
        ("alt+shift+v", "alt+shift+v"),
        ("ctrl+win+alt", "ctrl+alt+win"),
        ("win+shift+alt+ctrl+x", "ctrl+alt+shift+win+x"),
        ("CONTROL+Windows+Alt", "ctrl+alt+win"),
        ("  lalt  ", "lalt"),
        ("rctrl+rshift", "rctrl+rshift"),
        ("ctrl+page up", "ctrl+pageup"),
        ("ctrl+,", "ctrl+comma"),
        ("ctrl+Num 7", "ctrl+num7"),
        ("pause", "pause"),
        ("Scroll Lock", "scrolllock"),
        ("insert", "insert"),
        ("shift+f9", "shift+f9"),
        ("ctrl+alt+escape", "ctrl+alt+esc"),
        ("win+shift+f23", "shift+win+f23"),
    ],
)
def test_parse_and_format_canonical(text: str, canonical: str) -> None:
    chord = parse_chord(text)
    assert format_chord(chord) == canonical
    assert str(chord) == canonical
    # The canonical form parses back to the same chord.
    assert parse_chord(canonical) == chord


@pytest.mark.parametrize(
    ("text", "message_part"),
    [
        ("", "No shortcut"),
        ("ctrl", "Ctrl on its own"),
        ("shift", "Shift on its own"),
        ("win", "Win on its own"),
        ("a", "A on its own would get in the way of normal typing"),
        ("7", "7 on its own"),
        ("space", "Space on its own"),
        ("enter", "Enter on its own"),
        ("tab", "Tab on its own"),
        ("home", "Home on its own"),
        ("shift+a", "Shift + A is used for typing"),
        ("shift+space", "Shift + Space is used for typing"),
        ("shift+home", "Shift + Home is used for typing"),
        ("esc", "reserved for cancelling"),
        ("ctrl+c", "standard Windows shortcut for copy"),
        ("ctrl+v", "standard Windows shortcut for paste"),
        ("shift+insert", "standard Windows shortcut for paste"),
        ("alt+f4", "closing windows"),
        ("alt+tab", "switching windows"),
        ("win+l", "locking the PC"),
        ("rctrl+f8", "Left/right-specific keys"),
        ("lalt+space", "Left/right-specific keys"),
        ("ctrl+ctrl", "Ctrl appears more than once"),
        ("lctrl+rctrl", "Ctrl appears more than once"),
        ("ctrl+foo", '"foo" isn\'t a key'),
        ("ctrl++", "empty part"),
        ("a+b", "only one key besides"),
    ],
)
def test_rejected_chords_have_friendly_messages(text: str, message_part: str) -> None:
    with pytest.raises(ValueError) as info:
        parse_chord(text)
    assert message_part in str(info.value)


@pytest.mark.parametrize(
    "text", ["f1", "f12", "f24", "pause", "scrolllock", "insert", "rctrl", "lctrl", "ralt", "rwin", "ctrl+win"]
)
def test_allowed_without_regular_modifiers(text: str) -> None:
    parse_chord(text)


@pytest.mark.parametrize(
    ("text", "labels"),
    [
        ("ctrl+shift+space", ["Ctrl", "Shift", "Space"]),
        ("ctrl+win", ["Ctrl", "Win"]),
        ("rctrl", ["Right Ctrl"]),
        ("lalt", ["Left Alt"]),
        ("rwin", ["Right Win"]),
        ("f8", ["F8"]),
        ("alt+shift+v", ["Alt", "Shift", "V"]),
        ("ctrl+pagedown", ["Ctrl", "Page Down"]),
        ("ctrl+num5", ["Ctrl", "Num 5"]),
        ("scrolllock", ["Scroll Lock"]),
        ("ctrl+win+alt", ["Ctrl", "Alt", "Win"]),
    ],
)
def test_display_labels(text: str, labels: list[str]) -> None:
    assert display_chord(text) == labels


def test_display_accepts_chords_that_parse_would_reject_and_empty_text() -> None:
    # A stored setting must always be showable, even if it isn't a valid shortcut any more.
    assert display_chord("a") == ["A"]
    assert display_chord("ctrl") == ["Ctrl"]
    assert display_chord("") == []
    assert display_chord("   ") == []
    with pytest.raises(ValueError):
        display_chord("ctrl+nonsense")


def test_display_labels_parse_back() -> None:
    for text in ("ctrl+shift+space", "rctrl", "lalt+lwin", "ctrl+pageup", "ctrl+scrolllock", "f13"):
        labels = display_chord(text)
        assert format_chord(parse_chord("+".join(labels))) == text


def test_punctuation_label_is_a_single_visible_character() -> None:
    label = display_chord("ctrl+quote")[1]
    assert len(label) == 1 and not label.isspace()


def test_describe_chord() -> None:
    assert keys.describe_chord("ctrl+shift+space") == "Ctrl + Shift + Space"
    assert keys.describe_chord(parse_chord("rctrl")) == "Right Ctrl"


def test_chord_properties() -> None:
    chord = parse_chord("ctrl+alt+shift+win+f23")
    assert chord.vk == 0x86
    assert chord.mod_flags == keys.MOD_CONTROL | keys.MOD_ALT | keys.MOD_SHIFT | keys.MOD_WIN
    assert not chord.is_modifier_only
    mod_only = parse_chord("rctrl+shift")
    assert mod_only.is_modifier_only
    assert mod_only.vk is None
    assert mod_only.families == {"ctrl", "shift"}
    assert mod_only.side("ctrl") == keys.RIGHT
    assert mod_only.side("shift") is None
    assert mod_only.has_side_specific


def test_chord_structure_is_checked_on_construction() -> None:
    with pytest.raises(ValueError):
        Chord(frozenset({"ctrl", "rctrl"}), "a")
    with pytest.raises(ValueError):
        Chord(frozenset({"hyper"}), None)
    with pytest.raises(ValueError):
        Chord(frozenset(), "notakey")
    with pytest.raises(ValueError):
        Chord(frozenset(), None)


def test_chord_from_pressed_keys() -> None:
    assert format_chord(chord_from_keys([("ctrl", keys.RIGHT)], None)) == "rctrl"
    assert format_chord(chord_from_keys([("alt", keys.LEFT)], None)) == "lalt"
    assert format_chord(chord_from_keys([("ctrl", keys.LEFT), ("win", keys.LEFT)], None)) == "ctrl+win"
    assert format_chord(chord_from_keys([("ctrl", keys.LEFT), ("shift", keys.RIGHT)], "f8")) == "ctrl+shift+f8"
    assert format_chord(chord_from_keys([], "f9")) == "f9"


def test_modifier_from_vk() -> None:
    assert keys.modifier_from_vk(keys.VK_RCONTROL) == ("ctrl", keys.RIGHT)
    assert keys.modifier_from_vk(keys.VK_LWIN) == ("win", keys.LEFT)
    assert keys.modifier_from_vk(keys.VK_CONTROL, extended=True) == ("ctrl", keys.RIGHT)
    assert keys.modifier_from_vk(keys.VK_MENU) == ("alt", keys.LEFT)
    assert keys.modifier_from_vk(keys.VK_SHIFT, scan_code=0x36) == ("shift", keys.RIGHT)
    assert keys.modifier_from_vk(keys.VK_SHIFT, scan_code=0x2A) == ("shift", keys.LEFT)
    assert keys.modifier_from_vk(0x41) is None


def test_key_name_lookup() -> None:
    assert keys.key_name_for_vk(0x41) == "a"
    assert keys.key_name_for_vk(keys.VK_SPACE) == "space"
    assert keys.key_name_for_vk(0x87) == "f24"
    assert keys.key_name_for_vk(keys.VK_MASK_KEY) is None
    assert keys.key_vk("f8") == 0x77


def test_injected_events_carry_the_marker_and_extended_flag() -> None:
    insert = keys.key_event(keys.VK_INSERT)
    assert insert.type == keys.INPUT_KEYBOARD
    assert insert.ki.dwExtraInfo == keys.INJECTED_MARKER
    assert insert.ki.dwFlags & keys.KEYEVENTF_EXTENDEDKEY
    assert insert.ki.wScan == 0x52
    up = keys.key_event(keys.VK_CONTROL, up=True)
    assert up.ki.dwFlags & keys.KEYEVENTF_KEYUP
    assert not up.ki.dwFlags & keys.KEYEVENTF_EXTENDEDKEY
    assert up.ki.wScan == 0x1D
    char = keys.unicode_event(0xD83D, up=True)
    assert char.ki.wVk == 0
    assert char.ki.wScan == 0xD83D
    assert char.ki.dwFlags == keys.KEYEVENTF_UNICODE | keys.KEYEVENTF_KEYUP
    mask = keys.mask_key_events()
    assert [e.ki.wVk for e in mask] == [keys.VK_MASK_KEY, keys.VK_MASK_KEY]


def test_combo_releases_modifiers_in_reverse_order() -> None:
    events = keys.combo([keys.VK_CONTROL, keys.VK_SHIFT], keys.VK_V)
    sequence = [(e.ki.wVk, bool(e.ki.dwFlags & keys.KEYEVENTF_KEYUP)) for e in events]
    assert sequence == [
        (keys.VK_CONTROL, False),
        (keys.VK_SHIFT, False),
        (keys.VK_V, False),
        (keys.VK_V, True),
        (keys.VK_SHIFT, True),
        (keys.VK_CONTROL, True),
    ]


def test_send_input_with_nothing_sends_nothing() -> None:
    assert keys.send_input([]) == 0
