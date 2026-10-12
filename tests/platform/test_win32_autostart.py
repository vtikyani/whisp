"""Launch-at-sign-in through HKCU\\...\\Run, using a throwaway value name that is always removed."""

from __future__ import annotations

import sys
import uuid
import winreg
from collections.abc import Iterator
from pathlib import Path

import pytest

from whisp.platform.win32 import autostart
from whisp.platform.win32.autostart import APPROVED_KEY, RUN_KEY, autostart_command, autostart_status, set_autostart


def _delete(key_path: str, name: str) -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, name)
    except FileNotFoundError:
        pass


def _read(key_path: str, name: str) -> object | None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            return winreg.QueryValueEx(key, name)[0]
    except FileNotFoundError:
        return None


@pytest.fixture
def value_name() -> Iterator[str]:
    name = f"WhispTest-{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        _delete(RUN_KEY, name)
        _delete(APPROVED_KEY, name)
        assert _read(RUN_KEY, name) is None
        assert _read(APPROVED_KEY, name) is None


def test_enable_and_disable(value_name: str) -> None:
    assert autostart_status(value_name) == "disabled"
    set_autostart(True, value_name)
    assert autostart_status(value_name) == "enabled"
    assert _read(RUN_KEY, value_name) == autostart_command()
    set_autostart(True, value_name)  # idempotent
    assert autostart_status(value_name) == "enabled"
    set_autostart(False, value_name)
    assert autostart_status(value_name) == "disabled"
    assert _read(RUN_KEY, value_name) is None
    set_autostart(False, value_name)  # idempotent
    assert autostart_status(value_name) == "disabled"


def test_switched_off_in_windows_is_blocked_and_enabling_clears_it(value_name: str) -> None:
    set_autostart(True, value_name)
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, APPROVED_KEY, 0, winreg.KEY_SET_VALUE) as key:
        # What Task Manager writes when the user turns a startup app off: odd first byte + a FILETIME.
        winreg.SetValueEx(key, value_name, 0, winreg.REG_BINARY, bytes([0x03]) + bytes(11))
    assert autostart_status(value_name) == "blocked"

    set_autostart(True, value_name)
    assert autostart_status(value_name) == "enabled"
    approved = _read(APPROVED_KEY, value_name)
    assert isinstance(approved, bytes) and approved[0] % 2 == 0

    set_autostart(False, value_name)
    assert autostart_status(value_name) == "disabled"
    assert _read(APPROVED_KEY, value_name) is None


def test_switched_on_entry_in_startup_approved_counts_as_enabled(value_name: str) -> None:
    set_autostart(True, value_name)
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, APPROVED_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, value_name, 0, winreg.REG_BINARY, bytes([0x02]) + bytes(11))
    assert autostart_status(value_name) == "enabled"


def test_command_for_a_source_checkout() -> None:
    command = autostart_command()
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    expected = pythonw if pythonw.exists() else Path(sys.executable)
    assert command == f'"{expected}" -m whisp --background'


def test_command_for_a_frozen_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autostart.sys, "frozen", True, raising=False)
    monkeypatch.setattr(autostart.sys, "executable", r"C:\Program Files\Whisp\Whisp.exe")
    assert autostart_command() == '"C:\\Program Files\\Whisp\\Whisp.exe" --background'


def test_an_installed_copy_takes_over_a_stale_entry_but_keeps_windows_off_switch(
    value_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, '"D:\\Old place\\Whisp.exe" --background')
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, APPROVED_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, value_name, 0, winreg.REG_BINARY, bytes([0x03]) + bytes(11))  # off in Windows

    assert autostart.refresh_autostart_command(value_name) is False  # running from source never takes over

    monkeypatch.setattr(autostart.sys, "frozen", True, raising=False)
    monkeypatch.setattr(autostart.sys, "executable", r"C:\Users\Sam\AppData\Local\Programs\Whisp\Whisp.exe")
    assert autostart.refresh_autostart_command(value_name) is True
    assert _read(RUN_KEY, value_name) == autostart_command()
    assert autostart_status(value_name) == "blocked"  # the user's off switch is untouched
    assert autostart.refresh_autostart_command(value_name) is False  # already current


def test_refresh_leaves_a_missing_entry_alone(value_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autostart.sys, "frozen", True, raising=False)
    assert autostart.refresh_autostart_command(value_name) is False
    assert _read(RUN_KEY, value_name) is None
