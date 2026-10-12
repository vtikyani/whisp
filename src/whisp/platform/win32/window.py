"""Win32 styling for Qt top-level widgets: overlay behaviour and themed title bars."""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_dwmapi = ctypes.WinDLL("dwmapi")

_GetWindowLongPtrW = _user32.GetWindowLongPtrW
_GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
_GetWindowLongPtrW.restype = ctypes.c_ssize_t

_SetWindowLongPtrW = _user32.SetWindowLongPtrW
_SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
_SetWindowLongPtrW.restype = ctypes.c_ssize_t

_SetWindowPos = _user32.SetWindowPos
_SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
_SetWindowPos.restype = wintypes.BOOL

_DwmSetWindowAttribute = _dwmapi.DwmSetWindowAttribute
_DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
_DwmSetWindowAttribute.restype = ctypes.c_long  # HRESULT

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000

HWND_TOPMOST = wintypes.HWND(-1)
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020

DWMWA_TRANSITIONS_FORCEDISABLED = 3
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36
DWMWCP_DONOTROUND = 1


def _hwnd(widget: QWidget) -> wintypes.HWND:
    # winId() creates the native window if it doesn't exist yet.
    return wintypes.HWND(int(widget.winId()))


def _set_dword(hwnd: wintypes.HWND, attribute: int, value: int) -> bool:
    data = wintypes.DWORD(value)
    result = _DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(data), ctypes.sizeof(data))
    if result != 0:
        # Older Windows builds don't know some attributes; that only costs cosmetics.
        log.debug("DwmSetWindowAttribute(%d) returned 0x%08X", attribute, result & 0xFFFFFFFF)
        return False
    return True


def _colorref(color: QColor) -> int:
    return color.red() | (color.green() << 8) | (color.blue() << 16)


def _update_ex_style(widget: QWidget, add: int = 0, remove: int = 0) -> None:
    hwnd = _hwnd(widget)
    style = _GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    new_style = (style | add) & ~remove
    if new_style != style:
        ctypes.set_last_error(0)
        if _SetWindowLongPtrW(hwnd, GWL_EXSTYLE, new_style) == 0 and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
    _SetWindowPos(hwnd, None, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED)


def style_overlay_window(widget: QWidget) -> None:
    """Make a frameless, translucent widget behave like a system overlay.

    It never takes keyboard focus (even when clicked), stays above other
    windows, and is hidden from Alt+Tab and the taskbar. DWM's own rounded
    corners and show/hide transitions are turned off because the overlay draws
    and animates itself.
    """
    _update_ex_style(widget, add=WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST, remove=WS_EX_APPWINDOW)
    hwnd = _hwnd(widget)
    _SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
    _set_dword(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_DONOTROUND)
    _set_dword(hwnd, DWMWA_TRANSITIONS_FORCEDISABLED, 1)


def set_click_through(widget: QWidget, enabled: bool) -> None:
    """Let mouse clicks pass through a translucent (layered) window, or stop them."""
    if enabled:
        _update_ex_style(widget, add=WS_EX_TRANSPARENT)
    else:
        _update_ex_style(widget, remove=WS_EX_TRANSPARENT)


def style_app_window(widget: QWidget, *, dark: bool) -> None:
    """Colour the native title bar and border to match Whisp's background.

    Uses the Windows 11 caption/border/text colour attributes; on Windows 10 only
    the dark-mode flag applies.
    """
    hwnd = _hwnd(widget)
    _set_dword(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
    if dark:
        caption, border, text = QColor("#0B0B0E"), QColor("#202028"), QColor("#F4F4F6")
    else:
        caption, border, text = QColor("#F7F7F9"), QColor("#E2E2E8"), QColor("#0E0E12")
    _set_dword(hwnd, DWMWA_CAPTION_COLOR, _colorref(caption))
    _set_dword(hwnd, DWMWA_BORDER_COLOR, _colorref(border))
    _set_dword(hwnd, DWMWA_TEXT_COLOR, _colorref(text))
