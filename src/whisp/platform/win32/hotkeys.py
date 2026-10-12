"""HotkeyService: Whisp's global keyboard shortcuts.

Everything runs on one thread, ``whisp-hotkeys``, with its own Win32 message
loop. Public methods may be called from any thread; they record what the
caller wants and ask the hotkey thread to apply it. Signals are emitted from
the hotkey thread (Qt queues them to receivers on the GUI thread).

Two kinds of shortcut:

* **Chords with a regular key** (``ctrl+shift+space``, ``f8``) use
  ``RegisterHotKey``. Windows swallows the key press and reports conflicts
  with other apps. Release is detected by polling the key state every
  ~10 ms while the chord is held. Windows keeps many Win combinations for
  itself (``ctrl+win+space`` switches input languages), so a Win chord that
  Windows refuses to register is handled by the hook instead: it swallows
  that key while exactly the chord's modifiers are held.
* **Modifier-only chords** (``ctrl+win``, ``rctrl``) use a low-level keyboard
  hook. Nothing is swallowed. A chord is *pressed* when exactly its modifiers
  are held, *interrupted* when another key joins (the user is pressing some
  other shortcut, like Ctrl+Win+D), and *released* when one of its modifiers
  goes up. If a held chord grows into another bound chord (Alt joins
  ``ctrl+win`` to make ``ctrl+alt+win``), the first is interrupted and the
  second pressed.

Releasing Win (or Alt) on its own normally opens the Start menu (or a menu
bar). After a Win/Alt chord fires, the hook injects an unassigned key just
before that release reaches Windows, so the release isn't a "tap".

While Esc cancels a recording (``set_cancel_active``), Esc on its own is a
RegisterHotKey shortcut, and the hook catches Esc pressed while a shortcut is
still held (RegisterHotKey would miss it, and Windows would take Ctrl+Shift+Esc
as "open Task Manager"). So the hook also runs during a recording when only
regular-key shortcuts are bound.

Windows silently removes low-level hooks that respond too slowly. A raw-input
listener watches for keyboard activity the hook never saw and reinstalls the
hook. Raw input is turned off while one of Whisp's own windows has the focus:
Windows stops calling a process's low-level hook for its own windows while
that process is registered for raw keyboard input.
"""

from __future__ import annotations

import collections
import ctypes
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from ctypes import wintypes
from typing import Any, TypeVar

from PySide6.QtCore import QObject, Signal

from whisp.core.types import KeyActivity, KeyKind
from whisp.platform.win32 import keys
from whisp.platform.win32.foreground import (
    current_process_elevated,
    current_process_id,
    foreground_window,
    is_process_elevated,
    window_process_id,
)
from whisp.platform.win32.keys import Chord

log = logging.getLogger(__name__)

T = TypeVar("T")

ACTIONS: tuple[str, ...] = ("dictate", "command", "paste_last", "copy_last", "hands_free")
"""The actions Whisp binds today, in priority order for duplicate checks. Other well-formed names are accepted
too, so new actions need no change here."""
CANCEL = "cancel"
"""Reserved: Esc emits ``pressed("cancel")`` while ``set_cancel_active(True)``."""
ACTION_LABELS = {
    "dictate": "Dictate",
    "command": "Command mode",
    "paste_last": "Paste last",
    "copy_last": "Copy last",
    "hands_free": "Hands-free",
}
_ACTION_NAME = re.compile(r"[a-z][a-z0-9_]{0,47}")

RELEASE_POLL_MS = 10
"""How often a held RegisterHotKey chord is checked for release."""

HEALTH_CHECK_MS = 1000
HOOK_SILENCE_LIMIT_S = 2.0
"""Raw input saw keys this long ago that the hook never did: Windows removed the hook."""
REINSTALL_WARNING_INTERVAL_S = 60.0

_RAW_GRACE_S = 0.25
COMMAND_TIMEOUT_S = 2.0
STOP_TIMEOUT_S = 2.0
APPLY_DEADLINE_S = 20.0
"""How long a hotkey thread too busy to answer within ``COMMAND_TIMEOUT_S`` (Windows can be very slow just after
sign-in, when Whisp starts) still gets to apply new shortcuts before it counts as stuck."""
MSG_NOT_RESPONDING = "Whisp's shortcut service isn't responding. Restart Whisp to try again."

Event = tuple[str, str]
"""("pressed" | "released" | "interrupted", action)."""


# -- pure state machines -------------------------------------------------------------


class ModifierChordTracker:
    """Turns modifier key events into pressed/interrupted/released for modifier-only chords.

    Pure logic (no Win32), driven by the hook. Rules:

    * ``pressed`` when the held modifiers exactly match a bound chord, built up
      from nothing held. After any release or interruption, nothing new fires
      until every modifier is up again (no surprise re-triggers).
    * ``interrupted`` once per press when a regular key or an extra modifier
      joins the chord.
    * ``released`` when a modifier the chord needs goes up — always, even after
      an interruption, so every ``pressed`` gets exactly one ``released``.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, Chord] = {}
        self._held: dict[str, set[str]] = {f: set() for f in keys.MODIFIER_FAMILIES}
        self._active: list[str] = []
        self._interrupted: set[str] = set()
        self._armed = True

    @property
    def active(self) -> tuple[str, ...]:
        return tuple(self._active)

    @property
    def any_held(self) -> bool:
        return any(self._held.values())

    def held(self) -> dict[str, frozenset[str]]:
        return {f: frozenset(s) for f, s in self._held.items()}

    def is_held(self, family: str) -> bool:
        return bool(self._held[family])

    def set_bindings(self, bindings: Mapping[str, Chord]) -> list[Event]:
        events = self.release_all()
        self._bindings = {action: chord for action, chord in bindings.items() if chord.is_modifier_only}
        return events

    def key_down(self, family: str, side: str) -> list[Event]:
        if side in self._held[family]:
            return []  # auto-repeat
        self._held[family].add(side)
        if not self._armed:
            return []
        if any(self._matches_exactly(self._bindings[action]) for action in self._active):
            return []  # e.g. the other Ctrl key joined an either-side "ctrl+win": still the same chord
        match = self._match()
        if match is None:
            if not self._active:
                return []  # still building up a chord
            self._armed = False
            return self._interrupt_all()
        events = self._interrupt_all()
        self._active.append(match)
        events.append(("pressed", match))
        return events

    def key_up(self, family: str, side: str) -> list[Event]:
        if side not in self._held[family]:
            return []
        self._held[family].discard(side)
        events: list[Event] = []
        for action in reversed(list(self._active)):
            if self._releases(self._bindings[action], family, side):
                self._active.remove(action)
                self._interrupted.discard(action)
                events.append(("released", action))
        if self.any_held:
            self._armed = False
        else:
            self._armed = True
            self._interrupted.clear()
        return events

    def other_key_down(self) -> list[Event]:
        """A regular (non-modifier) key went down."""
        if not self.any_held:
            return []
        self._armed = False
        return self._interrupt_all()

    def disarm(self) -> None:
        """Ignore the current key combination until every modifier is released."""
        if self.any_held:
            self._armed = False

    def release_all(self) -> list[Event]:
        events: list[Event] = [("released", action) for action in reversed(self._active)]
        self._active.clear()
        self._interrupted.clear()
        self._armed = not self.any_held
        return events

    def resync(self, held: Mapping[str, Any]) -> list[Event]:
        """Adopt a fresh picture of the held modifiers (after events were missed); release chords no longer held."""
        self._held = {f: set(held.get(f, ())) for f in keys.MODIFIER_FAMILIES}
        events: list[Event] = []
        for action in reversed(list(self._active)):
            if not self._still_held(self._bindings[action]):
                self._active.remove(action)
                self._interrupted.discard(action)
                events.append(("released", action))
        self._armed = not self.any_held
        if self._armed:
            self._interrupted.clear()
        return events

    def _interrupt_all(self) -> list[Event]:
        events: list[Event] = [("interrupted", a) for a in self._active if a not in self._interrupted]
        self._interrupted.update(self._active)
        return events

    def _match(self) -> str | None:
        for action, chord in self._bindings.items():
            if action not in self._active and self._matches_exactly(chord):
                return action
        return None

    def _matches_exactly(self, chord: Chord) -> bool:
        families = chord.families
        for family in keys.MODIFIER_FAMILIES:
            sides = self._held[family]
            if family not in families:
                if sides:
                    return False
                continue
            wanted = chord.side(family)
            if (wanted is None and not sides) or (wanted is not None and sides != {wanted}):
                return False
        return True

    def _still_held(self, chord: Chord) -> bool:
        for family in chord.families:
            wanted = chord.side(family)
            sides = self._held[family]
            if (wanted is None and not sides) or (wanted is not None and wanted not in sides):
                return False
        return True

    def _releases(self, chord: Chord, family: str, side: str) -> bool:
        """Does releasing (family, side) — already removed from the held set — end ``chord``?"""
        if family not in chord.families:
            return False
        wanted = chord.side(family)
        if wanted is None:
            return not self._held[family]  # an either-side modifier ends when neither side is held
        return wanted == side


class ChordRecorder:
    """Captures the next shortcut the user presses. Pure logic.

    A regular key completes the chord on key-down; modifiers alone complete it
    when the first of them is released. Esc on its own cancels (result ``""``).
    """

    def __init__(self) -> None:
        self._held: set[tuple[str, str]] = set()

    def key_down(self, vk: int, modifier: tuple[str, str] | None) -> str | None:
        """Returns the canonical chord when complete, ``""`` if cancelled, or None to keep listening."""
        if modifier is not None:
            self._held.add(modifier)
            return None
        name = keys.key_name_for_vk(vk)
        if name is None:
            return None  # a key Whisp can't bind (media keys, IME keys, ...): keep listening
        if name == "esc" and not self._held:
            return ""
        return keys.format_chord(keys.chord_from_keys(self._held, name))

    def key_up(self, modifier: tuple[str, str] | None) -> str | None:
        if modifier is None or modifier not in self._held:
            return None
        return keys.format_chord(keys.chord_from_keys(self._held, None))


def action_label(action: str) -> str:
    """Readable name of an action for messages, e.g. "Paste last"."""
    return ACTION_LABELS.get(action) or action.replace("_", " ").capitalize()


def validate_bindings(bindings: Mapping[str, str]) -> tuple[dict[str, Chord], dict[str, str]]:
    """Parse a {action: chord text} mapping. Returns (usable chords, {action: error message}).

    Any lower-case action name (``[a-z][a-z0-9_]*``) is accepted except
    "cancel", which is Esc's. Empty text means unbound. A chord already used by
    an earlier action (``ACTIONS`` first, then the others in the given order)
    is an error for the later one.
    """
    chords: dict[str, Chord] = {}
    errors: dict[str, str] = {}
    names: list[str] = []
    for action in bindings:
        if not isinstance(action, str) or action == CANCEL or not _ACTION_NAME.fullmatch(action):
            errors[str(action)] = f"“{action}” can't be used as a shortcut name."
        else:
            names.append(action)
    ordered = [a for a in ACTIONS if a in names] + [a for a in names if a not in ACTIONS]
    used: dict[str, str] = {}
    for action in ordered:
        text = bindings[action]
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            chord = keys.parse_chord(text)
        except ValueError as exc:
            errors[action] = str(exc)
            continue
        canonical = keys.format_chord(chord)
        if canonical in used:
            errors[action] = f"{keys.describe_chord(chord)} is already used for {action_label(used[canonical])}."
            continue
        used[canonical] = action
        chords[action] = chord
    return chords, errors


# -- Win32 ---------------------------------------------------------------------------

LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
WINEVENTPROC = ctypes.WINFUNCTYPE(
    None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND, wintypes.LONG, wintypes.LONG, wintypes.DWORD, wintypes.DWORD
)


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = (
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    )


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = (
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    )


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_SetWindowsHookExW = _user32.SetWindowsHookExW
_SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
_SetWindowsHookExW.restype = wintypes.HHOOK

_UnhookWindowsHookEx = _user32.UnhookWindowsHookEx
_UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
_UnhookWindowsHookEx.restype = wintypes.BOOL

_CallNextHookEx = _user32.CallNextHookEx
_CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
_CallNextHookEx.restype = LRESULT

_GetMessageW = _user32.GetMessageW
_GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
_GetMessageW.restype = wintypes.BOOL

_PeekMessageW = _user32.PeekMessageW
_PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
_PeekMessageW.restype = wintypes.BOOL

_DispatchMessageW = _user32.DispatchMessageW
_DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
_DispatchMessageW.restype = LRESULT

_PostThreadMessageW = _user32.PostThreadMessageW
_PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_PostThreadMessageW.restype = wintypes.BOOL

_RegisterHotKey = _user32.RegisterHotKey
_RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
_RegisterHotKey.restype = wintypes.BOOL

_UnregisterHotKey = _user32.UnregisterHotKey
_UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
_UnregisterHotKey.restype = wintypes.BOOL

_SetTimer = _user32.SetTimer
_SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, wintypes.LPVOID]
_SetTimer.restype = ctypes.c_size_t

_KillTimer = _user32.KillTimer
_KillTimer.argtypes = [wintypes.HWND, ctypes.c_size_t]
_KillTimer.restype = wintypes.BOOL

_RegisterClassExW = _user32.RegisterClassExW
_RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
_RegisterClassExW.restype = wintypes.ATOM

_UnregisterClassW = _user32.UnregisterClassW
_UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
_UnregisterClassW.restype = wintypes.BOOL

_CreateWindowExW = _user32.CreateWindowExW
_CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]
_CreateWindowExW.restype = wintypes.HWND

_DestroyWindow = _user32.DestroyWindow
_DestroyWindow.argtypes = [wintypes.HWND]
_DestroyWindow.restype = wintypes.BOOL

_DefWindowProcW = _user32.DefWindowProcW
_DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_DefWindowProcW.restype = LRESULT

_RegisterRawInputDevices = _user32.RegisterRawInputDevices
_RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
_RegisterRawInputDevices.restype = wintypes.BOOL

_SetWinEventHook = _user32.SetWinEventHook
_SetWinEventHook.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HMODULE,
    WINEVENTPROC,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
]
_SetWinEventHook.restype = wintypes.HANDLE

_UnhookWinEvent = _user32.UnhookWinEvent
_UnhookWinEvent.argtypes = [wintypes.HANDLE]
_UnhookWinEvent.restype = wintypes.BOOL

_GetModuleHandleW = _kernel32.GetModuleHandleW
_GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
_GetModuleHandleW.restype = wintypes.HMODULE

_GetCurrentThreadId = _kernel32.GetCurrentThreadId
_GetCurrentThreadId.argtypes = []
_GetCurrentThreadId.restype = wintypes.DWORD

_wtsapi32 = ctypes.WinDLL("wtsapi32", use_last_error=True)

_WTSRegisterSessionNotification = _wtsapi32.WTSRegisterSessionNotification
_WTSRegisterSessionNotification.argtypes = [wintypes.HWND, wintypes.DWORD]
_WTSRegisterSessionNotification.restype = wintypes.BOOL

_WTSUnRegisterSessionNotification = _wtsapi32.WTSUnRegisterSessionNotification
_WTSUnRegisterSessionNotification.argtypes = [wintypes.HWND]
_WTSUnRegisterSessionNotification.restype = wintypes.BOOL

WH_KEYBOARD_LL = 13
HC_ACTION = 0
LLKHF_EXTENDED = 0x01
LLKHF_INJECTED = 0x10
LLKHF_UP = 0x80
WM_TIMER = 0x0113
WM_HOTKEY = 0x0312
WM_INPUT = 0x00FF
WM_APP = 0x8000
_WM_WAKE = WM_APP + 1
_WM_FLUSH = WM_APP + 2
PM_NOREMOVE = 0x0000
HWND_MESSAGE = wintypes.HWND(-3)
RIDEV_REMOVE = 0x00000001
RIDEV_INPUTSINK = 0x00000100
HID_USAGE_PAGE_GENERIC = 0x01
HID_USAGE_GENERIC_KEYBOARD = 0x06
EVENT_SYSTEM_FOREGROUND = 0x0003
WINEVENT_OUTOFCONTEXT = 0x0000
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
WM_WTSSESSION_CHANGE = 0x02B1
NOTIFY_FOR_THIS_SESSION = 0
WTS_CONSOLE_DISCONNECT = 0x2
WTS_REMOTE_DISCONNECT = 0x4
WTS_SESSION_LOCK = 0x7
_AWAY_EVENTS = frozenset({WTS_CONSOLE_DISCONNECT, WTS_REMOTE_DISCONNECT, WTS_SESSION_LOCK})
"""Session changes after which nobody is at this session's keyboard (see ``session_locked``)."""

_TIMER_HEALTH = 1
_TIMER_RELEASE = 2

_CANCEL_ID = 0xBFFF
"""RegisterHotKey id for Esc; actions get ids from 1 upwards (applications may use 0x0000-0xBFFF)."""

_ALTGR_FAKE_CTRL_SCAN = 0x200
"""Layouts with AltGr send a fake Left Ctrl before Right Alt, flagged with this bit in its scan code."""

_MASK_FAMILIES = ("win", "alt")

ACTIVITY_WINDOWS = 256
"""How many windows' last keystroke kinds are kept for smart spacing; the least recently typed-in go first."""


def _registration_error(chord: Chord, error: int) -> str:
    label = keys.describe_chord(chord)
    if error == ERROR_HOTKEY_ALREADY_REGISTERED:
        return f"{label} is already used by another app"
    return f"Windows wouldn't register {label} (error {error})"


# -- the service ---------------------------------------------------------------------


class HotkeyService(QObject):
    """Global shortcuts. See the module docstring for how each kind works."""

    pressed = Signal(str)
    """"dictate" | "command" | "paste_last" | "hands_free" | "cancel"."""
    released = Signal(str)
    interrupted = Signal(str)
    """A held modifier-only chord was combined with another key."""
    recorded = Signal(str)
    """The chord captured by ``start_recording`` (canonical text), or "" if the user pressed Esc."""
    binding_error = Signal(str, str)
    """(action, message) — a shortcut couldn't be set up."""
    session_locked = Signal()
    """The PC was locked (Win+L, or by itself) or this remote session was disconnected: nobody is at the keyboard."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._commands: queue.SimpleQueue[tuple[Callable[[], Any], Future[Any]]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._thread_ident: int | None = None

        # What callers asked for (guarded by _lock); the hotkey thread makes it so.
        self._want_bindings: dict[str, Chord] = {}
        self._validation_errors: dict[str, str] = {}
        self._want_paused = False
        self._want_cancel = False
        self._errors: dict[str, str] = {}

        # ctypes callbacks must outlive every call Windows may make into them.
        self._hook_proc = HOOKPROC(self._on_hook)
        self._wnd_proc = WNDPROC(self._on_window_message)
        self._win_event_proc = WINEVENTPROC(self._on_foreground_event)
        self._class_name = f"WhispHotkeys{id(self):x}"

        # Smart spacing: the kind of the last key typed into each window (never the key itself). Written by
        # the hook, read from any thread. Serials only ever grow, so they stay comparable across hook restarts.
        self._activity_lock = threading.Lock()
        self._activity: collections.OrderedDict[int, KeyActivity] = collections.OrderedDict()
        self._activity_serial = 0
        self._activity_baseline = KeyActivity(serial=0, kind=KeyKind.MOVE, hwnd=0, at=float("-inf"))
        """Reported for a window nothing was typed into since the hook started watching."""
        self._reset_thread_state()

    def _reset_thread_state(self) -> None:
        """State owned by the hotkey thread (only touched there while it runs)."""
        self._hinstance: int | None = None
        self._hwnd: int | None = None
        self._hook: int | None = None
        self._foreground_hook: int | None = None
        self._session_notifications = False
        self._quit = False
        self._bindings: dict[str, Chord] = {}
        self._registered: dict[str, Chord] = {}
        self._hotkey_ids: dict[str, int] = {}
        self._id_actions: dict[int, str] = {}
        self._hook_triggers: dict[str, Chord] = {}
        """Regular-key chords Windows keeps for itself (e.g. ctrl+win+space), handled by the hook instead."""
        self._held_hook_triggers: dict[str, Chord] = {}
        self._cancel_registered = False
        self._paused = False
        self._held_registered: dict[str, Chord] = {}
        self._release_timer = False
        self._tracker = ModifierChordTracker()
        self._recorder: ChordRecorder | None = None
        self._swallowed: set[int] = set()
        self._mask_pending: dict[str, bool] = dict.fromkeys(_MASK_FAMILIES, False)
        self._outbox: collections.deque[tuple[str, tuple[Any, ...]]] = collections.deque()
        self._flush_posted = False
        self._hook_dirty = False
        self._raw_registered = False
        self._hook_seen = 0.0
        self._raw_unseen_since: float | None = None
        self._last_reinstall_warning = float("-inf")

    # -- public API (any thread) -------------------------------------------------------

    def start(self) -> None:
        """Start the hotkey thread and apply the current bindings. Problems arrive as ``binding_error``."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread_id = 0
            self._thread = threading.Thread(target=self._run, name="whisp-hotkeys", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Unhook, unregister everything and end the thread. Safe to call more than once."""
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        self._post_command(self._cmd_quit)
        if thread.ident == threading.get_ident():
            return  # called from a signal handler on the hotkey thread; the loop ends after this message
        thread.join(timeout=STOP_TIMEOUT_S)
        if thread.is_alive():
            # Keep the reference so start() can't run a second message loop alongside this one.
            log.error("The hotkey thread didn't stop within %.0f s", STOP_TIMEOUT_S)
            return
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def set_bindings(self, bindings: Mapping[str, str]) -> dict[str, str]:
        """Bind actions to chords ({"dictate": "ctrl+win", ...}; missing or "" = unbound).

        Returns {action: message} for every chord that is invalid or couldn't be
        registered (e.g. "Ctrl + Shift + Space is already used by another app").
        The same problems are also emitted as ``binding_error``. While the
        service runs, this waits (briefly) for the hotkey thread to apply the
        change; before ``start()`` only validation errors can be known.
        """
        chords, errors = validate_bindings(bindings)
        with self._lock:
            self._want_bindings = chords
            self._validation_errors = errors
            running = self._thread is not None and self._thread.is_alive()
        if not running:
            self._store_errors(errors)
            for action, message in errors.items():
                self.binding_error.emit(action, message)
            return dict(errors)
        if self._thread_ident == threading.get_ident():
            return self._call(self._cmd_apply_bindings) or {}
        future = self._post_command(self._cmd_apply_bindings)
        try:
            return future.result(timeout=COMMAND_TIMEOUT_S)
        except FutureTimeoutError:
            # Busy, not broken: the thread applies them when it gets to them and reports any problem then, as
            # binding_error. Claiming they failed now would leave a warning up after they start working.
            log.warning("The hotkey thread is busy; the shortcuts will be applied as soon as it's free")
            self._report_if_stuck(future, list(chords))
            return dict(errors)
        except Exception:
            log.exception("Couldn't apply the shortcuts")
            return {**dict.fromkeys(chords, MSG_NOT_RESPONDING), **errors}

    def _report_if_stuck(self, future: Future[dict[str, str]], actions: list[str]) -> None:
        """Report ``actions`` as unavailable if the busy thread still hasn't applied them after ``APPLY_DEADLINE_S``."""

        def check() -> None:
            if future.done():
                return  # applied (any problem was reported then), or the service was stopped
            log.error("The hotkey thread hasn't applied the shortcuts after %.0f s", APPLY_DEADLINE_S)
            with self._lock:
                self._errors = {**self._errors, **dict.fromkeys(actions, MSG_NOT_RESPONDING)}
            for action in actions:
                self.binding_error.emit(action, MSG_NOT_RESPONDING)

        timer = threading.Timer(APPLY_DEADLINE_S, check)
        timer.daemon = True
        timer.start()

    def binding_errors(self) -> dict[str, str]:
        """The current {action: message} problems, for UI that opens after they were reported."""
        with self._lock:
            return dict(self._errors)

    def typing_activity(self, hwnd: int) -> KeyActivity | None:
        """The kind of key the user last typed into window ``hwnd``, or None if unknown (the hook isn't installed).

        Only the kind (space, Enter, a character…) is kept, never which key it was. Any thread.
        """
        if self._hook is None:
            return None
        with self._activity_lock:
            return self._activity.get(hwnd, self._activity_baseline)

    def set_cancel_active(self, active: bool) -> None:
        """While active, Esc on its own emits ``pressed("cancel")`` and doesn't reach the focused app."""
        with self._lock:
            self._want_cancel = bool(active)
        self._post_command(self._cmd_apply_cancel)

    def start_recording(self) -> None:
        """Capture the next chord the user presses and emit it as ``recorded`` instead of acting on it."""
        self._post_command(self._cmd_start_recording)

    def stop_recording(self) -> None:
        """Abandon a recording started with ``start_recording`` (nothing is emitted)."""
        self._post_command(self._cmd_stop_recording)

    def set_paused(self, paused: bool) -> None:
        """While paused every shortcut is off (recording still works)."""
        with self._lock:
            self._want_paused = bool(paused)
        self._post_command(self._cmd_apply_paused)

    # -- command plumbing --------------------------------------------------------------

    def _post_command(self, fn: Callable[[], T]) -> Future[T]:
        future: Future[T] = Future()
        self._commands.put((fn, future))
        thread_id = self._thread_id
        if thread_id:
            _PostThreadMessageW(thread_id, _WM_WAKE, 0, 0)
        # If the thread hasn't published its id yet it drains the queue as soon as it does.
        return future

    def _call(self, fn: Callable[[], T]) -> T | None:
        """Run ``fn`` on the hotkey thread and return its result (None if the thread isn't there)."""
        if self._thread_ident == threading.get_ident():
            result = fn()
            self._flush()
            return result
        with self._lock:
            thread = self._thread
        if thread is None or not thread.is_alive():
            log.error("The hotkey thread isn't running")
            return None
        future = self._post_command(fn)
        try:
            return future.result(timeout=COMMAND_TIMEOUT_S)
        except FutureTimeoutError:
            log.error("The hotkey thread didn't respond within %.0f s", COMMAND_TIMEOUT_S)
            return None
        except Exception:
            log.exception("A hotkey command failed")
            return None

    def _drain_commands(self) -> None:
        while True:
            try:
                fn, future = self._commands.get_nowait()
            except queue.Empty:
                return
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except Exception as exc:
                log.exception("A hotkey command failed")
                future.set_exception(exc)
            self._flush()

    def _store_errors(self, errors: Mapping[str, str]) -> None:
        with self._lock:
            self._errors = dict(errors)

    # -- the hotkey thread -------------------------------------------------------------

    def _run(self) -> None:
        self._reset_thread_state()
        self._thread_ident = threading.get_ident()
        try:
            msg = wintypes.MSG()
            # Create the thread's message queue before publishing its id, so posts can't be lost.
            _PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_NOREMOVE)
            if not self._create_window():
                self._fail_startup()
                return
            self._thread_id = int(_GetCurrentThreadId())
            self._foreground_hook = _SetWinEventHook(
                EVENT_SYSTEM_FOREGROUND,
                EVENT_SYSTEM_FOREGROUND,
                None,
                self._win_event_proc,
                0,
                0,
                WINEVENT_OUTOFCONTEXT,
            )
            if not self._foreground_hook:
                log.warning("Couldn't watch focus changes (error %d)", ctypes.get_last_error())
            self._session_notifications = bool(_WTSRegisterSessionNotification(self._hwnd, NOTIFY_FOR_THIS_SESSION))
            if not self._session_notifications:
                log.warning("Couldn't watch for the PC being locked (error %d)", ctypes.get_last_error())
            with self._lock:
                self._paused = self._want_paused
            self._cmd_apply_bindings()
            self._cmd_apply_cancel()
            self._flush()
            self._drain_commands()
            while not self._quit:
                result = _GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result in (0, -1):
                    break
                if not msg.hWnd and msg.message == _WM_WAKE:
                    self._drain_commands()
                elif not msg.hWnd and msg.message == _WM_FLUSH:
                    self._flush()
                else:
                    _DispatchMessageW(ctypes.byref(msg))
        except Exception:
            log.exception("The hotkey thread stopped unexpectedly")
        finally:
            self._teardown()

    def _create_window(self) -> bool:
        self._hinstance = _GetModuleHandleW(None)
        wc = WNDCLASSEXW(
            cbSize=ctypes.sizeof(WNDCLASSEXW),
            lpfnWndProc=self._wnd_proc,
            hInstance=self._hinstance,
            lpszClassName=self._class_name,
        )
        if not _RegisterClassExW(ctypes.byref(wc)):
            log.error("Couldn't register the hotkey window class (error %d)", ctypes.get_last_error())
            return False
        hwnd = _CreateWindowExW(0, self._class_name, None, 0, 0, 0, 0, 0, HWND_MESSAGE, None, self._hinstance, None)
        if not hwnd:
            log.error("Couldn't create the hotkey window (error %d)", ctypes.get_last_error())
            _UnregisterClassW(self._class_name, self._hinstance)
            return False
        self._hwnd = int(hwnd)
        return True

    def _fail_startup(self) -> None:
        message = "Windows didn't let Whisp set up its keyboard shortcuts. Restart Whisp to try again."
        with self._lock:
            actions = list(self._want_bindings)
        errors = dict.fromkeys(actions, message)
        self._store_errors(errors)
        for action in actions:
            self.binding_error.emit(action, message)

    def _teardown(self) -> None:
        try:
            if self._hwnd:
                _KillTimer(self._hwnd, _TIMER_RELEASE)
                _KillTimer(self._hwnd, _TIMER_HEALTH)
                for action in list(self._registered):
                    _UnregisterHotKey(self._hwnd, self._hotkey_ids[action])
                if self._cancel_registered:
                    _UnregisterHotKey(self._hwnd, _CANCEL_ID)
                if self._session_notifications:
                    _WTSUnRegisterSessionNotification(self._hwnd)
                    self._session_notifications = False
            self._registered.clear()
            self._hook_triggers.clear()
            self._held_hook_triggers.clear()
            self._cancel_registered = False
            if self._hook:
                _UnhookWindowsHookEx(self._hook)
                self._hook = None
            self._set_raw_input(False)
            if self._foreground_hook:
                _UnhookWinEvent(self._foreground_hook)
                self._foreground_hook = None
            if self._hwnd:
                _DestroyWindow(self._hwnd)
                self._hwnd = None
                _UnregisterClassW(self._class_name, self._hinstance)
        except Exception:
            log.exception("Couldn't clean up the hotkey thread")
        finally:
            self._thread_id = 0
            self._thread_ident = None
            # Anything still queued will never run; don't leave callers waiting.
            while True:
                try:
                    _fn, future = self._commands.get_nowait()
                except queue.Empty:
                    break
                if future.set_running_or_notify_cancel():
                    future.set_exception(RuntimeError("the hotkey service has stopped"))
            log.info("Hotkey thread stopped")

    # -- commands (run on the hotkey thread) -------------------------------------------

    def _cmd_quit(self) -> None:
        self._quit = True

    def _cmd_apply_bindings(self) -> dict[str, str]:
        with self._lock:
            chords = dict(self._want_bindings)
            validation = dict(self._validation_errors)
        if chords != self._bindings:
            self._bindings = chords
            self._apply_tracker_bindings()
        errors = {**validation, **self._apply_registrations()}
        errors.update(self._update_hook())
        self._store_errors(errors)
        for action, message in errors.items():
            self._queue("binding_error", action, message)
        return errors

    def _cmd_apply_paused(self) -> None:
        with self._lock:
            paused = self._want_paused
            validation = dict(self._validation_errors)
        if paused == self._paused:
            return
        self._paused = paused
        log.info("Shortcuts %s", "paused" if paused else "resumed")
        self._apply_tracker_bindings()
        errors = self._apply_registrations()
        self._apply_cancel()
        errors.update(self._update_hook())
        if not paused:
            # Another app may have taken a shortcut while Whisp was paused.
            all_errors = {**validation, **errors}
            self._store_errors(all_errors)
            for action, message in errors.items():
                self._queue("binding_error", action, message)

    def _cmd_apply_cancel(self) -> None:
        self._apply_cancel()
        self._update_hook()  # Esc held with a RegisterHotKey shortcut needs the hook (see _tracking)

    def _cmd_start_recording(self) -> None:
        self._release_everything()
        self._recorder = ChordRecorder()
        self._update_hook()
        if self._hook is None:
            log.error("Can't record a shortcut without the keyboard hook")
            self._recorder = None
            self._queue("recorded", "")

    def _cmd_stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder = None
            self._update_hook()

    # -- RegisterHotKey chords ---------------------------------------------------------

    def _hotkey_id(self, action: str) -> int:
        hotkey_id = self._hotkey_ids.get(action)
        if hotkey_id is None:
            hotkey_id = len(self._hotkey_ids) + 1
            self._hotkey_ids[action] = hotkey_id
            self._id_actions[hotkey_id] = action
        return hotkey_id

    def _apply_registrations(self) -> dict[str, str]:
        """Set up exactly the wanted regular-key chords. Returns {action: error} for failures.

        Each chord is registered with RegisterHotKey. A chord with Win that
        Windows already holds (Windows keeps many Win combinations, e.g.
        ctrl+win+space for switching input languages) is handled by the
        keyboard hook instead, which swallows its key the same way. Any other
        conflict belongs to another app and is reported.
        """
        wanted = {} if self._paused else {a: c for a, c in self._bindings.items() if not c.is_modifier_only}
        for action, chord in list(self._registered.items()):
            if wanted.get(action) != chord:
                _UnregisterHotKey(self._hwnd, self._hotkey_ids[action])
                del self._registered[action]
                if self._held_registered.pop(action, None) is not None:
                    self._queue("released", action)
        for action, chord in list(self._hook_triggers.items()):
            if wanted.get(action) != chord:
                del self._hook_triggers[action]
                if self._held_hook_triggers.pop(action, None) is not None:
                    self._queue("released", action)
        errors: dict[str, str] = {}
        for action, chord in wanted.items():
            if action in self._registered or action in self._hook_triggers:
                continue
            flags = chord.mod_flags | keys.MOD_NOREPEAT
            if _RegisterHotKey(self._hwnd, self._hotkey_id(action), flags, chord.vk):
                self._registered[action] = chord
                log.info("Shortcut for %s: %s", action, keys.format_chord(chord))
                continue
            error = ctypes.get_last_error()
            if error == ERROR_HOTKEY_ALREADY_REGISTERED and "win" in chord.families:
                self._hook_triggers[action] = chord
                log.info("Shortcut for %s: %s (kept by Windows; handled by the keyboard hook)", action, chord)
                continue
            errors[action] = _registration_error(chord, error)
            log.warning("Shortcut for %s isn't available: %s", action, errors[action])
        return errors

    def _apply_cancel(self) -> None:
        with self._lock:
            want = self._want_cancel and not self._paused
        if want == self._cancel_registered:
            return
        if not want:
            _UnregisterHotKey(self._hwnd, _CANCEL_ID)
            self._cancel_registered = False
            return
        if _RegisterHotKey(self._hwnd, _CANCEL_ID, keys.MOD_NOREPEAT, keys.VK_ESCAPE):
            self._cancel_registered = True
        else:
            error = ctypes.get_last_error()
            log.warning("Couldn't register Esc to cancel (error %d)", error)
            message = "Esc is already used by another app, so it can't cancel a dictation."
            if error != ERROR_HOTKEY_ALREADY_REGISTERED:
                message = f"Windows wouldn't register Esc to cancel a dictation (error {error})."
            self._queue("binding_error", CANCEL, message)

    def _on_hotkey(self, hotkey_id: int) -> None:
        if hotkey_id == _CANCEL_ID:
            if self._recorder is None and self._cancel_registered:
                self._queue("pressed", CANCEL)
            return
        action = self._id_actions.get(hotkey_id)
        chord = self._registered.get(action) if action else None
        if action is None or chord is None or self._recorder is not None or self._paused:
            return
        # The hook saw this key press first; whatever it reported (e.g. that a held modifier-only chord was
        # interrupted) must reach listeners before this press does.
        self._flush()
        if action in self._held_registered:
            # Released and pressed again between two polls: keep presses and releases paired.
            self._queue("released", action)
        self._held_registered[action] = chord
        if chord.families & {"win", "alt"}:
            # The chord's key never reaches the app, so the app sees Win/Alt pressed and released on
            # their own; a masking key in between keeps the Start menu or menu bar from opening.
            keys.send_input(keys.mask_key_events())
        self._queue("pressed", action)
        if not self._release_timer:
            self._release_timer = bool(_SetTimer(self._hwnd, _TIMER_RELEASE, RELEASE_POLL_MS, None))

    def _poll_releases(self) -> None:
        for action, chord in list(self._held_registered.items()):
            if not _registered_chord_held(chord):
                del self._held_registered[action]
                self._queue("released", action)
        if not self._held_registered and self._release_timer:
            _KillTimer(self._hwnd, _TIMER_RELEASE)
            self._release_timer = False

    def _release_everything(self) -> None:
        """End every held shortcut (used when pausing or starting to record)."""
        for action in list(self._held_registered):
            self._queue("released", action)
        self._held_registered.clear()
        for action in list(self._held_hook_triggers):
            self._queue("released", action)
        self._held_hook_triggers.clear()
        for kind, action in self._tracker.release_all():
            self._queue(kind, action)
        self._tracker.disarm()

    # -- the low-level hook ------------------------------------------------------------

    def _tracking(self) -> bool:
        """Does the hook have shortcuts to watch? (Recording is handled separately.)"""
        if self._paused:
            return False
        if self._hook_triggers or any(c.is_modifier_only for c in self._bindings.values()):
            return True
        # While Esc cancels, it must also work with a RegisterHotKey shortcut still held: RegisterHotKey only
        # catches Esc on its own, and Windows would take Ctrl+Shift+Esc as "open Task Manager".
        return self._cancel_registered and bool(self._registered)

    def _hook_actions(self) -> list[str]:
        return [a for a, c in self._bindings.items() if c.is_modifier_only] + list(self._hook_triggers)

    def _apply_tracker_bindings(self) -> None:
        bindings = self._bindings if not self._paused else {}
        for kind, action in self._tracker.set_bindings(bindings):
            self._queue(kind, action)
        self._tracker.disarm()

    def _update_hook(self) -> dict[str, str]:
        """Install the hook when something needs it, remove it when nothing does. Returns hook errors."""
        needed = self._recorder is not None or bool(self._swallowed) or self._tracking()
        if needed and self._hook is None:
            if not self._install_hook():
                message = "Windows didn't let Whisp watch the keyboard, so this shortcut can't work."
                return dict.fromkeys(self._hook_actions(), message)
        elif not needed and self._hook is not None:
            self._remove_hook()
        return {}

    def _install_hook(self) -> bool:
        hook = _SetWindowsHookExW(WH_KEYBOARD_LL, self._hook_proc, self._hinstance, 0)
        if not hook:
            log.error("Couldn't install the keyboard hook (error %d)", ctypes.get_last_error())
            return False
        self._hook = int(hook)
        self._hook_seen = time.monotonic()
        self._raw_unseen_since = None
        self._forget_typing()
        self._resync_from_key_state()
        self._update_raw_input()
        _SetTimer(self._hwnd, _TIMER_HEALTH, HEALTH_CHECK_MS, None)
        log.info("Keyboard hook installed")
        return True

    def _remove_hook(self) -> None:
        if self._hook is not None:
            _UnhookWindowsHookEx(self._hook)
            self._hook = None
        _KillTimer(self._hwnd, _TIMER_HEALTH)
        self._set_raw_input(False)
        for action in list(self._held_hook_triggers):
            self._queue("released", action)
        self._held_hook_triggers.clear()
        for kind, action in self._tracker.release_all():
            self._queue(kind, action)
        log.info("Keyboard hook removed")

    def _reinstall_hook(self) -> None:
        old = self._hook
        hook = _SetWindowsHookExW(WH_KEYBOARD_LL, self._hook_proc, self._hinstance, 0)
        if not hook:
            log.error("Couldn't reinstall the keyboard hook (error %d)", ctypes.get_last_error())
            return
        self._hook = int(hook)
        if old is not None:
            # Windows has usually removed it already; unhooking is then a harmless no-op.
            _UnhookWindowsHookEx(old)
        self._hook_seen = time.monotonic()
        self._raw_unseen_since = None
        self._forget_typing()
        self._resync_from_key_state()

    def _resync_from_key_state(self) -> None:
        held: dict[str, set[str]] = {f: set() for f in keys.MODIFIER_FAMILIES}
        for (family, side), vk in keys.MODIFIER_VKS.items():
            if keys.is_key_down(vk):
                held[family].add(side)
        for kind, action in self._tracker.resync(held):
            self._queue(kind, action)

    def _on_hook(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code == HC_ACTION:
            try:
                if self._on_key(KBDLLHOOKSTRUCT.from_address(l_param)):
                    return 1
            except Exception:
                log.exception("The keyboard hook failed to handle a key")
        return _CallNextHookEx(None, n_code, w_param, l_param)

    def _on_key(self, info: KBDLLHOOKSTRUCT) -> bool:
        """Handle one key event inside the hook. Returns True to swallow it. Must stay fast."""
        self._hook_seen = time.monotonic()
        flags = info.flags
        if flags & LLKHF_INJECTED and info.dwExtraInfo == keys.INJECTED_MARKER:
            return False  # our own paste, typing or mask key
        vk = info.vkCode
        if vk == keys.VK_LCONTROL and info.scanCode & _ALTGR_FAKE_CTRL_SCAN:
            return False
        up = bool(flags & LLKHF_UP)
        modifier = keys.modifier_from_vk(vk, scan_code=info.scanCode, extended=bool(flags & LLKHF_EXTENDED))
        if self._recorder is not None:
            return self._record_key(self._recorder, vk, modifier, up)
        # A key whose press Whisp swallowed (recording, Esc-to-cancel, a hook-handled shortcut) has its
        # repeats and release swallowed too, so apps never see half a keystroke.
        swallowed = vk in self._swallowed
        if swallowed:
            if not up:
                return True
            self._swallowed.discard(vk)
            if not self._swallowed:
                self._hook_dirty = True
                self._request_flush()
        swallow = (self._track_key(vk, modifier, up) or swallowed) if self._tracking() else swallowed
        if not up and modifier is None and not swallow:
            self._note_typing(vk)
        return swallow

    def _note_typing(self, vk: int) -> None:
        """Remember what kind of key just reached the focused window (smart spacing). Never the key itself."""
        ctrl, alt, win, shift = _held_modifiers()
        kind = keys.key_kind(vk, ctrl=ctrl, alt=alt, win=win, shift=shift)
        if kind is None:
            return
        hwnd = foreground_window()
        at = time.monotonic()
        with self._activity_lock:
            self._activity_serial += 1
            self._activity[hwnd] = KeyActivity(serial=self._activity_serial, kind=kind, hwnd=hwnd, at=at)
            self._activity.move_to_end(hwnd)
            if len(self._activity) > ACTIVITY_WINDOWS:
                self._activity.popitem(last=False)

    def _forget_typing(self) -> None:
        """Drop what's known about typing: keys may have gone unseen while the hook wasn't running.

        The new baseline gets a new serial, so no window looks untouched since an earlier dictation.
        """
        with self._activity_lock:
            self._activity_serial += 1
            self._activity.clear()
            self._activity_baseline = KeyActivity(
                serial=self._activity_serial, kind=KeyKind.MOVE, hwnd=0, at=float("-inf")
            )

    def _record_key(self, recorder: ChordRecorder, vk: int, modifier: tuple[str, str] | None, up: bool) -> bool:
        if up:
            if vk not in self._swallowed:
                # Held since before recording began: let the release through, masked if it's Win/Alt
                # (Windows saw it pressed on its own).
                if modifier is not None and modifier[0] in _MASK_FAMILIES:
                    keys.send_input(keys.mask_key_events())
                return False
            self._swallowed.discard(vk)
            self._finish_recording(recorder.key_up(modifier))
            return True
        if vk in self._swallowed:
            return True  # auto-repeat
        self._swallowed.add(vk)
        self._finish_recording(recorder.key_down(vk, modifier))
        return True

    def _finish_recording(self, result: str | None) -> None:
        if result is None:
            return
        self._recorder = None
        self._hook_dirty = True
        log.info("Recorded a shortcut" if result else "Shortcut recording cancelled")
        self._queue("recorded", result)
        self._request_flush()

    def _track_key(self, vk: int, modifier: tuple[str, str] | None, up: bool) -> bool:
        """Feed one key event to the shortcut logic. Returns True to swallow it."""
        tracker = self._tracker
        swallow = False
        if modifier is None:
            if up:
                events = self._release_hook_triggers(vk=vk)
            elif vk == keys.VK_ESCAPE and self._cancel_registered and self._any_shortcut_held():
                # Esc while holding a dictation shortcut cancels, like Esc on its own.
                tracker.disarm()
                self._swallowed.add(vk)
                self._queue("pressed", CANCEL)
                self._request_flush()
                return True
            else:
                # Reported in this order on purpose: whatever was held is interrupted before a
                # shortcut that includes this key is pressed.
                events = tracker.other_key_down()
                if self._hook_triggers:
                    events += self._release_stale_modifiers()
                    action = self._match_hook_trigger(vk)
                    if action is not None:
                        swallow = True
                        self._swallowed.add(vk)
                        self._held_hook_triggers[action] = self._hook_triggers[action]
                        events.append(("pressed", action))
        else:
            family, side = modifier
            if up:
                if self._mask_pending.get(family, False):
                    # Must reach Windows before this release does, so it's sent from inside the hook
                    # (SendInput here is processed ahead of the event being hooked).
                    keys.send_input(keys.mask_key_events())
                events = tracker.key_up(family, side)
                events += self._release_hook_triggers(family=family)
                if family in self._mask_pending and not tracker.is_held(family):
                    self._mask_pending[family] = False
            else:
                events = self._release_stale_modifiers()
                events += tracker.key_down(family, side)
        for kind, action in events:
            if kind == "pressed":
                for family in self._chord_families(action) & set(_MASK_FAMILIES):
                    self._mask_pending[family] = True
            self._queue(kind, action)
        if events:
            self._request_flush()
        return swallow

    def _any_shortcut_held(self) -> bool:
        return bool(self._tracker.active or self._held_hook_triggers or self._held_registered)

    def _match_hook_trigger(self, vk: int) -> str | None:
        """The hook-handled shortcut whose key is ``vk`` and whose modifiers are exactly the held ones."""
        held = frozenset(family for family, sides in self._tracker.held().items() if sides)
        for action, chord in self._hook_triggers.items():
            if chord.vk == vk and chord.families == held and action not in self._held_hook_triggers:
                return action
        return None

    def _release_hook_triggers(self, *, vk: int | None = None, family: str | None = None) -> list[Event]:
        """Release hook-handled shortcuts whose key (``vk``) or one of whose modifiers (``family``) went up."""
        events: list[Event] = []
        for action, chord in list(self._held_hook_triggers.items()):
            key_released = vk is not None and chord.vk == vk
            modifier_released = family is not None and family in chord.families and not self._tracker.is_held(family)
            if key_released or modifier_released:
                del self._held_hook_triggers[action]
                events.append(("released", action))
        return events

    def _chord_families(self, action: str) -> frozenset[str]:
        chord = self._bindings.get(action)
        return chord.families if chord is not None else frozenset()

    def _release_stale_modifiers(self) -> list[Event]:
        """Forget modifiers whose release we never saw (e.g. let go on the lock screen).

        Windows updates its key state only after the hook has run, so a modifier the tracker holds but Windows
        reports up was released unseen — including the one being pressed right now, which is then a new press
        rather than auto-repeat.
        """
        events: list[Event] = []
        for family, sides in self._tracker.held().items():
            for side in sides:
                if not keys.is_key_down(keys.MODIFIER_VKS[(family, side)]):
                    events += self._tracker.key_up(family, side)
                    events += self._release_hook_triggers(family=family)
                    if family in self._mask_pending and not self._tracker.is_held(family):
                        self._mask_pending[family] = False
        return events

    # -- raw input health check --------------------------------------------------------

    def _update_raw_input(self) -> None:
        own_window_focused = window_process_id(foreground_window()) == current_process_id()
        self._set_raw_input(self._hook is not None and not own_window_focused)

    def _set_raw_input(self, enabled: bool) -> None:
        if enabled == self._raw_registered:
            return
        flags = RIDEV_INPUTSINK if enabled else RIDEV_REMOVE
        target = self._hwnd if enabled else None
        device = RAWINPUTDEVICE(HID_USAGE_PAGE_GENERIC, HID_USAGE_GENERIC_KEYBOARD, flags, target)
        if _RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
            self._raw_registered = enabled
            self._raw_unseen_since = None
        else:
            action = "register" if enabled else "remove"
            log.warning("Couldn't %s raw keyboard input (error %d)", action, ctypes.get_last_error())

    def _on_raw_input(self) -> None:
        now = time.monotonic()
        if now - self._hook_seen > _RAW_GRACE_S:
            if self._raw_unseen_since is None:
                self._raw_unseen_since = now
        else:
            self._raw_unseen_since = None

    def _check_hook_health(self) -> None:
        since = self._raw_unseen_since
        if since is None or self._hook is None:
            return
        if self._hook_seen >= since:
            self._raw_unseen_since = None
            return
        if time.monotonic() - since < HOOK_SILENCE_LIMIT_S:
            return
        self._raw_unseen_since = None
        pid = window_process_id(foreground_window())
        if pid and not current_process_elevated() and is_process_elevated(pid):
            return  # Windows keeps keys typed into admin windows away from our hook; that's expected
        now = time.monotonic()
        if now - self._last_reinstall_warning >= REINSTALL_WARNING_INTERVAL_S:
            self._last_reinstall_warning = now
            log.warning("Windows stopped calling the keyboard hook; reinstalling it")
        else:
            log.debug("Keyboard hook silent again; reinstalling it")
        self._reinstall_hook()

    # -- window procedure and callbacks --------------------------------------------------

    def _on_window_message(self, hwnd: int, message: int, w_param: int, l_param: int) -> int:
        try:
            if message == WM_HOTKEY:
                self._on_hotkey(int(w_param))
                self._flush()
                return 0
            if message == WM_WTSSESSION_CHANGE:
                if w_param in _AWAY_EVENTS:
                    log.info("The PC was locked or the session disconnected")
                    self._queue("session_locked")
                    self._flush()
                return 0
            if message == WM_TIMER:
                if w_param == _TIMER_RELEASE:
                    self._poll_releases()
                elif w_param == _TIMER_HEALTH:
                    self._check_hook_health()
                self._flush()
                return 0
            if message == WM_INPUT:
                self._on_raw_input()
        except Exception:
            log.exception("The hotkey window failed to handle message 0x%04X", message)
        return _DefWindowProcW(hwnd, message, w_param, l_param)

    def _on_foreground_event(self, _hook, event, _hwnd, _id_object, _id_child, _thread, _time) -> None:
        if event != EVENT_SYSTEM_FOREGROUND:
            return
        try:
            if self._hook is not None:
                self._update_raw_input()
        except Exception:
            log.exception("Couldn't follow a focus change")

    # -- emitting ------------------------------------------------------------------------

    def _queue(self, signal_name: str, *args: Any) -> None:
        self._outbox.append((signal_name, args))

    def _request_flush(self) -> None:
        """From inside the hook: emit queued signals once the hook has returned."""
        if not self._flush_posted and self._thread_id:
            self._flush_posted = bool(_PostThreadMessageW(self._thread_id, _WM_FLUSH, 0, 0))

    def _flush(self) -> None:
        self._flush_posted = False
        if self._hook_dirty:
            self._hook_dirty = False
            self._update_hook()
        while self._outbox:
            signal_name, args = self._outbox.popleft()
            if signal_name == "binding_error":
                log.debug("Reporting a shortcut problem for %s", args[0])
            getattr(self, signal_name).emit(*args)


def _held_modifiers() -> tuple[bool, bool, bool, bool]:
    """Which of Ctrl, Alt, Win and Shift are down, as ``(ctrl, alt, win, shift)``, for the key being pressed.

    AltGr counts as neither Ctrl nor Alt: it's Right Alt with a Ctrl that Windows adds itself, and on many
    layouts it types letters and symbols (Polish ę, Czech ř, €, @, {).
    """
    win = keys.is_key_down(keys.VK_LWIN) or keys.is_key_down(keys.VK_RWIN)
    ctrl, alt = keys.is_key_down(keys.VK_CONTROL), keys.is_key_down(keys.VK_MENU)
    if ctrl and alt and keys.is_key_down(keys.VK_RMENU) and not keys.is_key_down(keys.VK_LMENU):
        ctrl = alt = False
    return ctrl, alt, win, keys.is_key_down(keys.VK_SHIFT)


def _registered_chord_held(chord: Chord) -> bool:
    if chord.vk is None or not keys.is_key_down(chord.vk):
        return False
    return all(any(keys.is_key_down(vk) for vk in keys.family_vks(f)) for f in chord.families)
