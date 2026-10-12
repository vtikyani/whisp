"""HotkeyService against real keyboard input, injected into a test window this test owns.

Keys are only sent while the test window (or the separate-process test window
for the hook-health test) has the focus. Chords use keys nothing else reacts to
(F21/F22, an unassigned virtual key) so a mistake can't trigger other shortcuts.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from desktop_support import Desktop, foreground_process_name, press, release

from whisp.platform.win32 import hotkeys as hotkeys_module
from whisp.platform.win32 import keys
from whisp.platform.win32.foreground import foreground_window
from whisp.platform.win32.hotkeys import HotkeyService

pytestmark = pytest.mark.desktop

C, A, S, W = keys.VK_LCONTROL, keys.VK_LMENU, keys.VK_LSHIFT, keys.VK_LWIN
RC = keys.VK_RCONTROL
ESC = keys.VK_ESCAPE
F21, F22 = 0x84, 0x85
UNASSIGNED = 0x97


class Signals:
    def __init__(self, service: HotkeyService) -> None:
        self.events: list[tuple[str, str]] = []
        service.pressed.connect(lambda a: self.events.append(("pressed", a)))
        service.released.connect(lambda a: self.events.append(("released", a)))
        service.interrupted.connect(lambda a: self.events.append(("interrupted", a)))
        service.recorded.connect(lambda c: self.events.append(("recorded", c)))
        service.binding_error.connect(lambda a, m: self.events.append(("binding_error", a)))

    def wait_for(self, qtbot, *expected: tuple[str, str], timeout: int = 2000) -> None:
        qtbot.waitUntil(lambda: all(e in self.events for e in expected), timeout=timeout)


@pytest.fixture
def service(qapp) -> Iterator[HotkeyService]:
    svc = HotkeyService()
    svc.start()
    try:
        yield svc
    finally:
        svc.stop()


def on_thread(service: HotkeyService, fn):
    """Run ``fn`` on the hotkey thread after every earlier command (a barrier for async calls)."""
    return service._call(fn)


def bind(service: HotkeyService, **bindings: str) -> None:
    errors = service.set_bindings(bindings)
    if errors:
        pytest.skip(f"Shortcut unavailable on this machine: {errors}")


def test_registered_chord_press_release_and_swallow(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    bind(service, paste_last="ctrl+alt+f22")
    signals = Signals(service)
    desktop.send([press(C), press(A), press(F22)])
    signals.wait_for(qtbot, ("pressed", "paste_last"))
    assert ("released", "paste_last") not in signals.events
    desktop.send([release(F22)])
    signals.wait_for(qtbot, ("released", "paste_last"))
    desktop.send([release(A), release(C)])
    qtbot.wait(100)
    assert F22 not in desktop.window.keys.pressed_vks()  # Windows swallowed the chord's key
    assert signals.events == [("pressed", "paste_last"), ("released", "paste_last")]


@pytest.mark.parametrize(
    ("down", "up"),
    [([C, W], [W, C]), ([W, C], [C, W])],
    ids=["ctrl-first-win-released-first", "win-first-win-released-last"],
)
def test_modifier_only_chord_without_opening_start(
    qtbot, desktop: Desktop, service: HotkeyService, down: list[int], up: list[int]
) -> None:
    bind(service, dictate="ctrl+win")
    assert on_thread(service, lambda: service._hook is not None)
    # Raw input must be off while our own window has the focus, or Windows stops calling the hook.
    assert on_thread(service, lambda: service._raw_registered) is False
    signals = Signals(service)
    desktop.send([press(down[0])])
    qtbot.wait(30)
    desktop.send([press(down[1])])
    signals.wait_for(qtbot, ("pressed", "dictate"))
    qtbot.wait(150)  # a human-length hold
    desktop.send([release(up[0])])
    desktop.send([release(up[1])])
    signals.wait_for(qtbot, ("released", "dictate"))
    qtbot.wait(700)
    opened = foreground_window() != desktop.window.hwnd
    assert not opened, f"The Start menu (or {foreground_process_name()}) took the focus"
    assert signals.events == [("pressed", "dictate"), ("released", "dictate")]
    entries = desktop.window.keys.entries
    mask_press = next(i for i, (pressed, vk, _) in enumerate(entries) if pressed and vk == keys.VK_MASK_KEY)
    win_release = next(i for i, (pressed, vk, _) in enumerate(entries) if not pressed and vk == keys.VK_LWIN)
    assert mask_press < win_release, "the mask key must reach the app before Win is released"


def test_superset_upgrade(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    bind(service, dictate="ctrl+win", command="ctrl+win+alt")
    signals = Signals(service)
    desktop.send([press(C), press(W)])
    signals.wait_for(qtbot, ("pressed", "dictate"))
    desktop.send([press(A)])
    signals.wait_for(qtbot, ("pressed", "command"))
    desktop.send([release(A), release(W), release(C)])
    signals.wait_for(qtbot, ("released", "dictate"))
    assert signals.events == [
        ("pressed", "dictate"),
        ("interrupted", "dictate"),
        ("pressed", "command"),
        ("released", "command"),
        ("released", "dictate"),
    ]
    qtbot.wait(500)
    assert foreground_window() == desktop.window.hwnd


def test_hands_free_chord_overlapping_the_dictate_chord(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    """Ctrl+Win (dictate) then Space (hands_free = ctrl+win+space): interrupted before the new press."""
    bind(service, dictate="ctrl+win", hands_free="ctrl+win+space")
    handled_by = on_thread(service, lambda: "hook" if "hands_free" in service._hook_triggers else "RegisterHotKey")
    signals = Signals(service)
    desktop.send([press(C), press(W)])
    signals.wait_for(qtbot, ("pressed", "dictate"))
    qtbot.wait(100)
    desktop.send([press(keys.VK_SPACE)])
    signals.wait_for(qtbot, ("pressed", "hands_free"))
    assert signals.events == [("pressed", "dictate"), ("interrupted", "dictate"), ("pressed", "hands_free")]
    desktop.send([release(keys.VK_SPACE), release(W), release(C)])
    signals.wait_for(qtbot, ("released", "hands_free"), ("released", "dictate"))
    qtbot.wait(700)
    print(f"\nctrl+win+space handled by {handled_by}; observed order: {signals.events}")
    assert foreground_window() == desktop.window.hwnd, f"{foreground_process_name()} took the focus"
    assert keys.VK_SPACE not in desktop.window.keys.pressed_vks()  # Space was swallowed


def test_windows_reserved_chord_is_taken_over(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    bind(service, hands_free="ctrl+win+space")
    signals = Signals(service)
    desktop.send([press(C), press(W), press(keys.VK_SPACE)])
    signals.wait_for(qtbot, ("pressed", "hands_free"))
    desktop.send([release(keys.VK_SPACE)])
    signals.wait_for(qtbot, ("released", "hands_free"))
    desktop.send([release(W), release(C)])
    qtbot.wait(700)
    assert signals.events == [("pressed", "hands_free"), ("released", "hands_free")]
    assert foreground_window() == desktop.window.hwnd, f"{foreground_process_name()} took the focus"
    assert keys.VK_SPACE not in desktop.window.keys.pressed_vks()


def test_other_key_interrupts_a_modifier_only_chord(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    bind(service, dictate="ctrl+win")
    signals = Signals(service)
    desktop.send([press(C), press(W)])
    signals.wait_for(qtbot, ("pressed", "dictate"))
    desktop.send([press(UNASSIGNED), release(UNASSIGNED)])
    signals.wait_for(qtbot, ("interrupted", "dictate"))
    desktop.send([release(W), release(C)])
    signals.wait_for(qtbot, ("released", "dictate"))
    assert signals.events == [("pressed", "dictate"), ("interrupted", "dictate"), ("released", "dictate")]
    assert UNASSIGNED in desktop.window.keys.pressed_vks()  # modifier-only chords never swallow keys


def test_escape_cancels_only_while_active(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    signals = Signals(service)
    service.set_cancel_active(True)
    on_thread(service, lambda: None)
    desktop.send([press(ESC), release(ESC)])
    signals.wait_for(qtbot, ("pressed", "cancel"))
    assert ESC not in desktop.window.keys.pressed_vks()
    service.set_cancel_active(False)
    on_thread(service, lambda: None)
    desktop.send([press(ESC), release(ESC)])
    qtbot.waitUntil(lambda: ESC in desktop.window.keys.pressed_vks(), timeout=2000)
    assert signals.events == [("pressed", "cancel")]


def test_escape_while_holding_the_chord_cancels(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    bind(service, dictate="ctrl+win")
    service.set_cancel_active(True)
    signals = Signals(service)
    desktop.send([press(C), press(W)])
    signals.wait_for(qtbot, ("pressed", "dictate"))
    desktop.send([press(ESC), release(ESC)])
    signals.wait_for(qtbot, ("pressed", "cancel"))
    desktop.send([release(W), release(C)])
    signals.wait_for(qtbot, ("released", "dictate"))
    assert signals.events == [("pressed", "dictate"), ("pressed", "cancel"), ("released", "dictate")]
    assert ESC not in desktop.window.keys.pressed_vks()


def test_recording_captures_and_swallows_the_chord(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    bind(service, dictate="ctrl+win")
    signals = Signals(service)
    service.start_recording()
    on_thread(service, lambda: None)
    desktop.send([press(C), press(S), press(F21)])
    signals.wait_for(qtbot, ("recorded", "ctrl+shift+f21"))
    desktop.send([release(F21), release(S), release(C)])
    qtbot.wait(100)
    assert not set(desktop.window.keys.pressed_vks()) & {keys.VK_CONTROL, keys.VK_SHIFT, F21}

    service.start_recording()
    on_thread(service, lambda: None)
    desktop.send([press(RC), release(RC)])
    signals.wait_for(qtbot, ("recorded", "rctrl"))

    service.start_recording()
    on_thread(service, lambda: None)
    desktop.send([press(ESC), release(ESC)])
    signals.wait_for(qtbot, ("recorded", ""))
    # Recording never dispatched the bound shortcut.
    assert [e for e in signals.events if e[0] != "recorded"] == []


def test_paused_service_ignores_shortcuts(qtbot, desktop: Desktop, service: HotkeyService) -> None:
    bind(service, dictate="rctrl", paste_last="ctrl+alt+f22")
    signals = Signals(service)
    service.set_paused(True)
    assert on_thread(service, lambda: service._hook is None and not service._registered)
    desktop.send([press(RC), release(RC)])
    desktop.send([press(C), press(A), press(F22), release(F22), release(A), release(C)])
    qtbot.waitUntil(lambda: F22 in desktop.window.keys.pressed_vks(), timeout=2000)  # not swallowed while paused
    assert signals.events == []
    service.set_paused(False)
    on_thread(service, lambda: None)
    desktop.send([press(RC)])
    signals.wait_for(qtbot, ("pressed", "dictate"))
    desktop.send([release(RC)])
    signals.wait_for(qtbot, ("released", "dictate"))


def test_hook_is_reinstalled_after_windows_drops_it(
    qtbot, desktop: Desktop, service: HotkeyService, caplog: pytest.LogCaptureFixture
) -> None:
    bind(service, dictate="ctrl+win")
    script = Path(__file__).with_name("desktop_target.py")
    # Started while our window has the focus, so Windows lets the helper take it; it hands the
    # focus back to our window when it quits.
    helper = subprocess.Popen(
        [sys.executable, str(script), str(desktop.window.hwnd)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        line = helper.stdout.readline().strip()
        if not line.startswith("ready"):
            pytest.skip(f"The helper window didn't get the focus ({line or 'no reply'}).")

        def keys_in_helper(*items: str) -> None:
            helper.stdin.write("keys " + " ".join(items) + "\n")
            reply = helper.stdout.readline().strip()
            if reply == "lost-focus":
                pytest.skip("The helper window lost the focus; not sending keys.")
            assert reply == "ok", reply

        signals = Signals(service)
        qtbot.waitUntil(lambda: on_thread(service, lambda: service._raw_registered), timeout=2000)
        # Simulate Windows silently removing the hook (as it does after a slow callback).
        on_thread(service, lambda: hooks_unhook(service._hook))
        caplog.set_level(logging.WARNING, logger=hotkeys_module.__name__)
        keys_in_helper(f"{UNASSIGNED}:d", f"{UNASSIGNED}:u")
        qtbot.waitUntil(
            lambda: "reinstalling" in caplog.text,
            timeout=int((hotkeys_module.HOOK_SILENCE_LIMIT_S + 3) * 1000),
        )
        keys_in_helper(f"{C}:d", f"{W}:d")
        signals.wait_for(qtbot, ("pressed", "dictate"))
        keys_in_helper(f"{W}:u", f"{C}:u")
        signals.wait_for(qtbot, ("released", "dictate"))
    finally:
        if helper.poll() is None:
            helper.stdin.write("quit\n")
            helper.stdin.flush()
            try:
                helper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                helper.kill()


def hooks_unhook(hook: int | None) -> None:
    assert hook is not None
    hotkeys_module._UnhookWindowsHookEx(hook)
