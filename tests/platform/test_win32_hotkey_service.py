"""HotkeyService lifecycle and RegisterHotKey bookkeeping. Sends no input and installs no keyboard hook."""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes

from whisp.platform.win32 import hotkeys as hotkeys_module
from whisp.platform.win32 import keys
from whisp.platform.win32.hotkeys import HotkeyService
from whisp.platform.win32.insert import TextInserter
from whisp.platform.win32.instance import SingleInstance

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_RegisterHotKey = _user32.RegisterHotKey
_RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
_RegisterHotKey.restype = wintypes.BOOL
_UnregisterHotKey = _user32.UnregisterHotKey
_UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
_UnregisterHotKey.restype = wintypes.BOOL

RARE_CHORD = "ctrl+alt+shift+f20"
F20 = 0x83


def test_services_construct_without_arguments(qapp) -> None:
    HotkeyService()
    TextInserter()
    SingleInstance()


def test_bindings_before_start_report_validation_errors_only(qapp) -> None:
    service = HotkeyService()
    reported: list[tuple[str, str]] = []
    service.binding_error.connect(lambda a, m: reported.append((a, m)))
    errors = service.set_bindings({"dictate": "ctrl", "paste_last": RARE_CHORD})
    assert set(errors) == {"dictate"}
    assert reported == [("dictate", errors["dictate"])]
    assert service.binding_errors() == errors


def test_start_and_stop_are_idempotent(qapp) -> None:
    service = HotkeyService()
    service.stop()  # never started
    service.start()
    service.start()
    names = [t.name for t in threading.enumerate()]
    assert names.count("whisp-hotkeys") >= 1
    assert service.set_bindings({"paste_last": RARE_CHORD}) == {}
    assert service._call(lambda: dict(service._registered)) == {"paste_last": keys.parse_chord(RARE_CHORD)}
    service.stop()
    service.stop()
    assert service._thread is None
    # Restartable, with the same bindings re-applied.
    service.start()
    try:
        assert service.set_bindings({"paste_last": RARE_CHORD}) == {}
    finally:
        service.stop()


def test_unbinding_releases_the_registration(qapp) -> None:
    service = HotkeyService()
    service.start()
    try:
        assert service.set_bindings({"paste_last": RARE_CHORD}) == {}
        assert service.set_bindings({}) == {}
        assert service._call(lambda: dict(service._registered)) == {}
        # Now another program can take it.
        assert _RegisterHotKey(None, 0x2FFF, keys.MOD_CONTROL | keys.MOD_ALT | keys.MOD_SHIFT, F20)
        _UnregisterHotKey(None, 0x2FFF)
    finally:
        service.stop()


def _keep_busy(service: HotkeyService) -> threading.Event:
    """Keep the hotkey thread busy until the returned event is set, like a PC that's slow just after sign-in."""
    started, done = threading.Event(), threading.Event()

    def busy() -> None:
        started.set()
        done.wait(10)

    service._post_command(busy)
    assert started.wait(5)
    return done


def test_a_busy_service_applies_shortcuts_late_without_reporting_a_failure(qtbot, monkeypatch) -> None:
    monkeypatch.setattr(hotkeys_module, "COMMAND_TIMEOUT_S", 0.2)
    service = HotkeyService()
    reported: list[tuple[str, str]] = []
    service.binding_error.connect(lambda a, m: reported.append((a, m)))
    service.start()
    try:
        assert service.set_bindings({}) == {}
        done = _keep_busy(service)
        assert service.set_bindings({"paste_last": RARE_CHORD}) == {}  # no made-up failure
        done.set()
        qtbot.waitUntil(lambda: service._call(lambda: "paste_last" in service._registered) is True, timeout=3000)
        qtbot.wait(100)
        assert reported == []
        assert service.binding_errors() == {}
    finally:
        service.stop()


def test_a_stuck_service_is_reported(qtbot, monkeypatch) -> None:
    monkeypatch.setattr(hotkeys_module, "COMMAND_TIMEOUT_S", 0.2)
    monkeypatch.setattr(hotkeys_module, "APPLY_DEADLINE_S", 0.5)
    service = HotkeyService()
    service.start()
    try:
        assert service.set_bindings({}) == {}
        done = _keep_busy(service)
        try:
            with qtbot.waitSignal(service.binding_error, timeout=3000) as blocker:
                assert service.set_bindings({"paste_last": RARE_CHORD}) == {}
            assert blocker.args == ["paste_last", hotkeys_module.MSG_NOT_RESPONDING]
            assert service.binding_errors() == {"paste_last": hotkeys_module.MSG_NOT_RESPONDING}
        finally:
            done.set()
    finally:
        service.stop()


def test_conflict_with_another_program_is_reported(qtbot) -> None:
    taken = _RegisterHotKey(None, 0x2FFE, keys.MOD_CONTROL | keys.MOD_ALT | keys.MOD_SHIFT, F20)
    assert taken, "the test chord is already registered elsewhere"
    service = HotkeyService()
    service.start()
    try:
        with qtbot.waitSignal(service.binding_error, timeout=3000) as blocker:
            errors = service.set_bindings({"paste_last": RARE_CHORD})
        message = "Ctrl + Alt + Shift + F20 is already used by another app"
        assert errors.get("paste_last") == message
        assert blocker.args == ["paste_last", message]
        assert service.binding_errors().get("paste_last") == message
        # The conflict clears once the other program lets go.
        _UnregisterHotKey(None, 0x2FFE)
        taken = False
        assert "paste_last" not in service.set_bindings({"paste_last": RARE_CHORD})
    finally:
        if taken:
            _UnregisterHotKey(None, 0x2FFE)
        service.stop()
