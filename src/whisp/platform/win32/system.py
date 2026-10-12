"""Small queries about the Windows environment."""

from __future__ import annotations

import ctypes
import logging
import winreg
from ctypes import wintypes

log = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_SystemParametersInfoW = _user32.SystemParametersInfoW
_SystemParametersInfoW.argtypes = [wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
_SystemParametersInfoW.restype = wintypes.BOOL

SPI_GETCLIENTAREAANIMATION = 0x1042
_PERSONALIZE = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def animations_enabled() -> bool:
    """False when the user turned off "Animation effects" (Settings → Accessibility → Visual effects)."""
    value = wintypes.BOOL(True)
    if not _SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(value), 0):
        return True
    return bool(value.value)


def _personalize_flag(name: str, default: bool) -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PERSONALIZE) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return bool(value)
    except OSError:
        return default


def apps_use_light_theme() -> bool:
    """The "Choose your app mode" setting."""
    return _personalize_flag("AppsUseLightTheme", default=False)


def taskbar_uses_light_theme() -> bool:
    """The "Choose your default Windows mode" setting, which colours the taskbar and tray."""
    return _personalize_flag("SystemUsesLightTheme", default=False)
