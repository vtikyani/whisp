"""The pure state machines behind HotkeyService: modifier-only chords, recording and binding validation."""

from __future__ import annotations

from whisp.platform.win32 import keys
from whisp.platform.win32.hotkeys import ChordRecorder, ModifierChordTracker, validate_bindings
from whisp.platform.win32.keys import LEFT, RIGHT, parse_chord

CTRL_L = ("ctrl", LEFT)
CTRL_R = ("ctrl", RIGHT)
WIN_L = ("win", LEFT)
ALT_L = ("alt", LEFT)
SHIFT_L = ("shift", LEFT)


def tracker_with(**bindings: str) -> ModifierChordTracker:
    tracker = ModifierChordTracker()
    assert tracker.set_bindings({action: parse_chord(text) for action, text in bindings.items()}) == []
    return tracker


def test_press_and_release_in_either_order() -> None:
    t = tracker_with(dictate="ctrl+win")
    assert t.key_down(*CTRL_L) == []
    assert t.key_down(*WIN_L) == [("pressed", "dictate")]
    assert t.key_up(*WIN_L) == [("released", "dictate")]
    assert t.key_up(*CTRL_L) == []

    assert t.key_down(*WIN_L) == []
    assert t.key_down(*CTRL_L) == [("pressed", "dictate")]
    assert t.key_up(*CTRL_L) == [("released", "dictate")]
    assert t.key_up(*WIN_L) == []


def test_auto_repeat_is_ignored() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == [("pressed", "dictate")]
    assert t.key_down(*WIN_L) == []
    assert t.key_down(*CTRL_L) == []
    assert t.active == ("dictate",)


def test_regular_key_interrupts_and_release_still_follows() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*CTRL_L)
    t.key_down(*WIN_L)
    assert t.other_key_down() == [("interrupted", "dictate")]
    assert t.other_key_down() == []  # reported once per press
    assert t.key_up(*WIN_L) == [("released", "dictate")]
    assert t.key_up(*CTRL_L) == []


def test_superset_upgrade_interrupts_then_presses_the_bigger_chord() -> None:
    t = tracker_with(dictate="ctrl+win", command="ctrl+win+alt")
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == [("pressed", "dictate")]
    assert t.key_down(*ALT_L) == [("interrupted", "dictate"), ("pressed", "command")]
    assert t.active == ("dictate", "command")
    # Releasing a key both chords need releases both, newest first.
    assert t.key_up(*CTRL_L) == [("released", "command"), ("released", "dictate")]
    assert t.key_up(*WIN_L) == []
    assert t.key_up(*ALT_L) == []


def test_releasing_only_the_extra_key_after_an_upgrade() -> None:
    t = tracker_with(dictate="ctrl+win", command="ctrl+win+alt")
    t.key_down(*CTRL_L)
    t.key_down(*WIN_L)
    t.key_down(*ALT_L)
    assert t.key_up(*ALT_L) == [("released", "command")]
    # Back to exactly ctrl+win, but that's a release, not a new press.
    assert t.active == ("dictate",)
    assert t.key_up(*WIN_L) == [("released", "dictate")]


def test_bigger_chord_pressed_directly() -> None:
    t = tracker_with(dictate="ctrl+win", command="ctrl+win+alt")
    t.key_down(*ALT_L)
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == [("pressed", "command")]
    assert t.key_up(*ALT_L) == [("released", "command")]
    assert t.key_up(*CTRL_L) == []
    assert t.key_up(*WIN_L) == []


def test_extra_modifier_without_a_matching_chord_interrupts() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*CTRL_L)
    t.key_down(*WIN_L)
    assert t.key_down(*SHIFT_L) == [("interrupted", "dictate")]
    assert t.key_up(*SHIFT_L) == []
    assert t.key_up(*CTRL_L) == [("released", "dictate")]


def test_no_retrigger_until_everything_is_released() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*CTRL_L)
    t.key_down(*WIN_L)
    t.key_up(*WIN_L)
    assert t.key_down(*WIN_L) == []  # Ctrl was never released
    t.key_up(*WIN_L)
    t.key_up(*CTRL_L)
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == [("pressed", "dictate")]


def test_shortcut_typed_while_holding_ctrl_blocks_activation() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*CTRL_L)
    assert t.other_key_down() == []  # Ctrl+C: nothing active yet
    assert t.key_down(*WIN_L) == []
    t.key_up(*WIN_L)
    t.key_up(*CTRL_L)
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == [("pressed", "dictate")]


def test_partial_chord_with_unbound_extra_key_does_not_fire_on_release() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*SHIFT_L)
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == []  # ctrl+shift+win isn't bound
    assert t.key_up(*SHIFT_L) == []  # now exactly ctrl+win, but reached by letting go: no press
    assert t.active == ()


def test_side_specific_chords() -> None:
    t = tracker_with(dictate="rctrl")
    assert t.key_down(*CTRL_L) == []
    t.key_up(*CTRL_L)
    assert t.key_down(*CTRL_R) == [("pressed", "dictate")]
    assert t.key_up(*CTRL_R) == [("released", "dictate")]
    # Both Ctrl keys held is not exactly "right Ctrl".
    t.key_down(*CTRL_L)
    assert t.key_down(*CTRL_R) == []


def test_either_side_modifier_ends_when_neither_side_is_held() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*CTRL_L)
    t.key_down(*WIN_L)
    assert t.key_down(*CTRL_R) == []  # still exactly ctrl+win: not an interruption
    assert t.key_up(*CTRL_L) == []
    assert t.key_up(*CTRL_R) == [("released", "dictate")]


def test_release_all_and_rebinding() -> None:
    t = tracker_with(dictate="ctrl+win", command="ctrl+win+alt")
    t.key_down(*CTRL_L)
    t.key_down(*WIN_L)
    t.key_down(*ALT_L)
    assert t.set_bindings({"dictate": parse_chord("ctrl+win")}) == [
        ("released", "command"),
        ("released", "dictate"),
    ]
    # Keys are still held: the new bindings don't fire until they're released.
    t.key_up(*ALT_L)
    assert t.active == ()
    t.key_up(*CTRL_L)
    t.key_up(*WIN_L)
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == [("pressed", "dictate")]


def test_regular_key_chords_are_ignored_by_the_tracker() -> None:
    t = tracker_with(paste_last="alt+shift+v")
    t.key_down(*ALT_L)
    assert t.key_down(*SHIFT_L) == []


def test_resync_releases_chords_whose_keys_went_up_unseen() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*CTRL_L)
    t.key_down(*WIN_L)
    assert t.resync({"ctrl": {LEFT}}) == [("released", "dictate")]
    assert t.resync({}) == []
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == [("pressed", "dictate")]


def test_disarm_blocks_until_all_released() -> None:
    t = tracker_with(dictate="ctrl+win")
    t.key_down(*CTRL_L)
    t.disarm()
    assert t.key_down(*WIN_L) == []
    t.key_up(*WIN_L)
    t.key_up(*CTRL_L)
    t.key_down(*CTRL_L)
    assert t.key_down(*WIN_L) == [("pressed", "dictate")]


# -- recording -----------------------------------------------------------------------


def test_record_chord_with_a_regular_key() -> None:
    r = ChordRecorder()
    assert r.key_down(keys.VK_LCONTROL, CTRL_L) is None
    assert r.key_down(keys.VK_RSHIFT, ("shift", RIGHT)) is None
    assert r.key_down(keys.VK_SPACE, None) == "ctrl+shift+space"


def test_record_single_modifier_keeps_its_side() -> None:
    r = ChordRecorder()
    r.key_down(keys.VK_RCONTROL, CTRL_R)
    assert r.key_up(CTRL_R) == "rctrl"


def test_record_several_modifiers_on_release() -> None:
    r = ChordRecorder()
    r.key_down(keys.VK_LCONTROL, CTRL_L)
    r.key_down(keys.VK_LWIN, WIN_L)
    r.key_down(keys.VK_LMENU, ALT_L)
    assert r.key_up(WIN_L) == "ctrl+alt+win"


def test_record_escape_cancels_but_not_with_modifiers() -> None:
    assert ChordRecorder().key_down(keys.VK_ESCAPE, None) == ""
    r = ChordRecorder()
    r.key_down(keys.VK_LMENU, ALT_L)
    assert r.key_down(keys.VK_ESCAPE, None) == "alt+esc"


def test_record_reports_invalid_chords_so_the_ui_can_explain() -> None:
    assert ChordRecorder().key_down(0x41, None) == "a"


def test_record_ignores_unsupported_keys_and_releases_of_earlier_keys() -> None:
    r = ChordRecorder()
    assert r.key_down(0xAD, None) is None  # volume mute
    assert r.key_up(CTRL_L) is None  # held before recording started
    assert r.key_up(None) is None
    assert r.key_down(0x77, None) == "f8"


# -- binding validation ----------------------------------------------------------------


def test_validate_bindings() -> None:
    chords, errors = validate_bindings(
        {"dictate": "ctrl+win", "command": "ctrl+win+alt", "paste_last": "alt+shift+v", "hands_free": ""}
    )
    assert {a: str(c) for a, c in chords.items()} == {
        "dictate": "ctrl+win",
        "command": "ctrl+alt+win",
        "paste_last": "alt+shift+v",
    }
    assert errors == {}


def test_validate_bindings_reports_problems_per_action() -> None:
    chords, errors = validate_bindings({"dictate": "ctrl+win", "command": "WIN + CTRL", "paste_last": "a"})
    assert set(chords) == {"dictate"}
    assert errors["command"] == "Ctrl + Win is already used for Dictate."
    assert "A on its own" in errors["paste_last"]


def test_validate_bindings_with_the_new_defaults() -> None:
    chords, errors = validate_bindings(
        {
            "dictate": "ctrl+win",
            "command": "ctrl+win+alt",
            "paste_last": "alt+shift+z",
            "copy_last": "alt+shift+x",
            "hands_free": "ctrl+win+space",
        }
    )
    assert errors == {}
    assert {a: str(c) for a, c in chords.items()} == {
        "dictate": "ctrl+win",
        "command": "ctrl+alt+win",
        "paste_last": "alt+shift+z",
        "copy_last": "alt+shift+x",
        "hands_free": "ctrl+win+space",
    }


def test_validate_bindings_accepts_any_well_formed_action_name() -> None:
    chords, errors = validate_bindings({"some_future_action": "ctrl+alt+f9", "copy_last": "ctrl+alt+f9"})
    # Known actions take precedence for a shared chord; the newcomer gets the error.
    assert set(chords) == {"copy_last"}
    assert errors == {"some_future_action": "Ctrl + Alt + F9 is already used for Copy last."}

    chords, errors = validate_bindings({"Bad Name": "f9", "cancel": "f10", "": "f11", "ok_name2": "f12"})
    assert set(chords) == {"ok_name2"}
    assert set(errors) == {"Bad Name", "cancel", ""}
    assert all("can't be used as a shortcut name" in message for message in errors.values())
