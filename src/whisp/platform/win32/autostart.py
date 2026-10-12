"""Launch at sign-in, through the per-user Run key (no admin rights or shortcuts needed).

Windows keeps a second, separate switch for every Run entry: Task Manager's
Startup tab (and Settings → Apps → Startup) record the user's on/off choice
under ``Explorer\\StartupApproved\\Run``. An entry whose first byte there is
odd is turned off — Whisp reports that as ``"blocked"``.
"""

from __future__ import annotations

import logging
import sys
import winreg
from pathlib import Path

log = logging.getLogger(__name__)

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"
VALUE_NAME = "Whisp"

_APPROVED_ENABLED = bytes([0x02]) + bytes(11)
"""What Windows itself writes when a startup entry is switched on."""


class AutostartError(OSError):
    """The startup setting couldn't be changed. ``str(error)`` is a plain-language message."""


def autostart_command() -> str:
    """The command Windows runs at sign-in.

    A packaged build starts its own executable; a source checkout starts the
    windowless ``pythonw.exe`` next to the running interpreter with
    ``-m whisp`` (the package must be installed in that environment).
    """
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --background'
    interpreter = Path(sys.executable)
    pythonw = interpreter.with_name("pythonw.exe")
    if pythonw.exists():
        interpreter = pythonw
    return f'"{interpreter}" -m whisp --background'


def _read_value(key_path: str, value_name: str) -> object | None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            value, _kind = winreg.QueryValueEx(key, value_name)
            return value
    except FileNotFoundError:
        return None


def _delete_value(key_path: str, value_name: str) -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, value_name)
    except FileNotFoundError:
        pass


def _is_blocked(approved: object | None) -> bool:
    return isinstance(approved, bytes) and len(approved) > 0 and bool(approved[0] & 1)


def autostart_status(value_name: str = VALUE_NAME) -> str:
    """``"enabled"``, ``"disabled"``, or ``"blocked"`` (registered, but switched off in Windows' startup apps)."""
    command = _read_value(RUN_KEY, value_name)
    if not isinstance(command, str) or not command.strip():
        return "disabled"
    if _is_blocked(_read_value(APPROVED_KEY, value_name)):
        return "blocked"
    return "enabled"


def refresh_autostart_command(value_name: str = VALUE_NAME) -> bool:
    """Point an existing startup entry at this copy of Whisp, if it names another one. Returns True if changed.

    Only installed (packaged) builds do this, so running Whisp from source never takes over the installed
    copy's entry. The user's on/off choice in Windows' startup apps is left as it is.
    """
    if not getattr(sys, "frozen", False):
        return False
    stored = _read_value(RUN_KEY, value_name)
    command = autostart_command()
    if not isinstance(stored, str) or not stored.strip() or stored == command:
        return False
    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, command)
    except OSError as exc:
        log.exception("Couldn't update the startup entry")
        raise AutostartError("Windows didn't let Whisp change the startup setting.") from exc
    log.info("Startup entry now points at this copy of Whisp")
    return True


def set_autostart(enabled: bool, value_name: str = VALUE_NAME) -> None:
    """Turn launch-at-sign-in on or off, then read the registry back to confirm.

    Turning it on also clears Windows' own "off" switch for Whisp, because the
    user just asked for it. Raises ``AutostartError`` if Windows doesn't keep
    the change.
    """
    command = autostart_command()
    try:
        if enabled:
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, command)
            if _is_blocked(_read_value(APPROVED_KEY, value_name)):
                with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, APPROVED_KEY, 0, winreg.KEY_SET_VALUE) as key:
                    winreg.SetValueEx(key, value_name, 0, winreg.REG_BINARY, _APPROVED_ENABLED)
        else:
            _delete_value(RUN_KEY, value_name)
            _delete_value(APPROVED_KEY, value_name)
    except OSError as exc:
        log.exception("Couldn't change the startup setting")
        raise AutostartError("Windows didn't let Whisp change the startup setting.") from exc

    status = autostart_status(value_name)
    if enabled and (status != "enabled" or _read_value(RUN_KEY, value_name) != command):
        log.error("Launch at sign-in didn't stick (status %s)", status)
        raise AutostartError("Windows didn't keep the startup setting. Try again, or check Settings → Apps → Startup.")
    if not enabled and status != "disabled":
        log.error("Launch at sign-in couldn't be removed (status %s)", status)
        raise AutostartError("Windows didn't remove Whisp from the startup apps. Try again.")
    log.info("Launch at sign-in %s", "enabled" if enabled else "disabled")
