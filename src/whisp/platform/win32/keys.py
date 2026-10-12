"""Keyboard primitives: virtual-key tables, shortcut chords and synthetic key input.

A *chord* is a keyboard shortcut written as text. The canonical form is
lower-case, joined with ``+``, with modifiers in a fixed order (ctrl, alt,
shift, win) followed by at most one regular key::

    "ctrl+shift+space"   "f8"   "ctrl+win"   "rctrl"   "alt+shift+v"

Chords made only of modifier keys may name a side ("rctrl" is the right Ctrl
key); chords with a regular key always use either-side modifiers, because
that is all ``RegisterHotKey`` can express.

Every event Whisp injects with ``SendInput`` carries ``INJECTED_MARKER`` in
``dwExtraInfo`` so Whisp's own keyboard hook can tell them apart from the
user's keystrokes.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Sequence
from ctypes import wintypes
from dataclasses import dataclass

from whisp.core.types import KeyKind

# -- virtual-key codes ---------------------------------------------------------------

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_PAUSE = 0x13
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_PRIOR = 0x21
VK_NEXT = 0x22
VK_END = 0x23
VK_HOME = 0x24
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_SNAPSHOT = 0x2C
VK_INSERT = 0x2D
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_APPS = 0x5D
VK_NUMPAD0 = 0x60
VK_MULTIPLY = 0x6A
VK_ADD = 0x6B
VK_SUBTRACT = 0x6D
VK_DECIMAL = 0x6E
VK_DIVIDE = 0x6F
VK_F1 = 0x70
VK_SCROLL = 0x91
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5
VK_C = 0x43
VK_V = 0x56

VK_MASK_KEY = 0xE8
"""An unassigned virtual key. Pressing it while Win or Alt is held "uses up" that key, so its release
doesn't open the Start menu or a menu bar (the same trick AutoHotkey uses for its menu mask key)."""

# -- modifiers -----------------------------------------------------------------------

MODIFIER_FAMILIES: tuple[str, ...] = ("ctrl", "alt", "shift", "win")
"""Canonical modifier order."""

LEFT = "left"
RIGHT = "right"

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

_MODIFIER_TOKENS: dict[str, tuple[str, str | None]] = {
    "ctrl": ("ctrl", None),
    "lctrl": ("ctrl", LEFT),
    "rctrl": ("ctrl", RIGHT),
    "alt": ("alt", None),
    "lalt": ("alt", LEFT),
    "ralt": ("alt", RIGHT),
    "shift": ("shift", None),
    "lshift": ("shift", LEFT),
    "rshift": ("shift", RIGHT),
    "win": ("win", None),
    "lwin": ("win", LEFT),
    "rwin": ("win", RIGHT),
}
_MODIFIER_ALIASES: dict[str, str] = {
    "control": "ctrl",
    "lcontrol": "lctrl",
    "rcontrol": "rctrl",
    "leftctrl": "lctrl",
    "rightctrl": "rctrl",
    "leftcontrol": "lctrl",
    "rightcontrol": "rctrl",
    "leftalt": "lalt",
    "rightalt": "ralt",
    "altgr": "ralt",
    "leftshift": "lshift",
    "rightshift": "rshift",
    "windows": "win",
    "super": "win",
    "meta": "win",
    "leftwin": "lwin",
    "rightwin": "rwin",
    "leftwindows": "lwin",
    "rightwindows": "rwin",
}
_FAMILY_LABELS = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}
_FAMILY_MOD_FLAGS = {"ctrl": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}
_FAMILY_ANY_SIDE_VKS: dict[str, tuple[int, ...]] = {
    "ctrl": (VK_CONTROL,),
    "alt": (VK_MENU,),
    "shift": (VK_SHIFT,),
    "win": (VK_LWIN, VK_RWIN),
}

MODIFIER_VKS: dict[tuple[str, str], int] = {
    ("ctrl", LEFT): VK_LCONTROL,
    ("ctrl", RIGHT): VK_RCONTROL,
    ("alt", LEFT): VK_LMENU,
    ("alt", RIGHT): VK_RMENU,
    ("shift", LEFT): VK_LSHIFT,
    ("shift", RIGHT): VK_RSHIFT,
    ("win", LEFT): VK_LWIN,
    ("win", RIGHT): VK_RWIN,
}
"""Side-specific virtual key for each (family, side)."""

_VK_TO_MODIFIER: dict[int, tuple[str, str]] = {vk: fam_side for fam_side, vk in MODIFIER_VKS.items()}

_SCAN_RSHIFT = 0x36


def modifier_from_vk(vk: int, *, scan_code: int = 0, extended: bool = False) -> tuple[str, str] | None:
    """(family, side) for a modifier virtual key, or None for any other key.

    Low-level hooks normally report side-specific codes (``VK_LCONTROL``); the
    generic codes some programs inject are resolved from the scan code or the
    extended-key flag, the way Windows does.
    """
    found = _VK_TO_MODIFIER.get(vk)
    if found is not None:
        return found
    if vk == VK_CONTROL:
        return ("ctrl", RIGHT if extended else LEFT)
    if vk == VK_MENU:
        return ("alt", RIGHT if extended else LEFT)
    if vk == VK_SHIFT:
        return ("shift", RIGHT if (scan_code & 0xFF) == _SCAN_RSHIFT else LEFT)
    return None


def family_vks(family: str) -> tuple[int, ...]:
    """Virtual keys whose async state says whether ``family`` is held on either side."""
    return _FAMILY_ANY_SIDE_VKS[family]


_MOVE_VKS = frozenset({VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN, VK_HOME, VK_END, VK_PRIOR, VK_NEXT})
_TEXT_VK_RANGES = (
    (0x30, 0x39),  # digits
    (0x41, 0x5A),  # letters
    (0x60, 0x6F),  # numeric keypad digits and operators
    (0xBA, 0xC0),  # OEM punctuation ; = , - . / `
    (0xDB, 0xDF),  # OEM punctuation [ \ ] ' and layout-specific
    (0xE2, 0xE2),  # the extra key on ISO keyboards
)


_CTRL_KEEPS_TEXT_VKS = frozenset({VK_C, VK_INSERT, 0x53, 0x42, 0x49, 0x55})
"""Ctrl+C and Ctrl+Insert (copy), Ctrl+S (save), Ctrl+B/I/U (bold, italic, underline): text and cursor stay put."""


def key_kind(
    vk: int, *, ctrl: bool = False, alt: bool = False, win: bool = False, shift: bool = False
) -> KeyKind | None:
    """What a key press does to the text, for smart spacing; None for keys that leave the text and cursor alone.

    ``ctrl``, ``alt``, ``win``, ``shift``: which of those keys are down. With Ctrl, Alt or Win the key is a
    shortcut (paste, undo, select all…) rather than typing. Pass AltGr as neither Ctrl nor Alt: on many layouts
    it types letters and symbols. Only the kind is kept — never the key or character.
    """
    if win or (alt and vk in (VK_TAB, VK_ESCAPE)):
        return None  # Windows' own shortcuts: switching apps, showing the desktop, snapping windows…
    if ctrl and not alt and vk in _CTRL_KEEPS_TEXT_VKS:
        return None
    if vk == VK_INSERT:
        return KeyKind.EDIT if shift and not (ctrl or alt) else None  # Shift+Insert pastes
    command = ctrl or alt
    if vk in (VK_SPACE, VK_TAB):
        return KeyKind.EDIT if command else KeyKind.SPACE
    if vk == VK_RETURN:
        return KeyKind.NEWLINE
    if vk in (VK_BACK, VK_DELETE):
        return KeyKind.EDIT
    if vk in _MOVE_VKS:
        return KeyKind.MOVE
    if any(lo <= vk <= hi for lo, hi in _TEXT_VK_RANGES):
        return KeyKind.EDIT if command else KeyKind.TEXT
    return None


# -- regular keys --------------------------------------------------------------------


@dataclass(frozen=True)
class _KeyInfo:
    name: str
    vk: int
    label: str
    layout_char: bool = False
    """Punctuation: show the character the user's keyboard layout prints on the key."""


def _build_key_table() -> tuple[dict[str, _KeyInfo], dict[str, str]]:
    keys: list[_KeyInfo] = []
    aliases: dict[str, str] = {}
    for i in range(26):
        ch = chr(ord("a") + i)
        keys.append(_KeyInfo(ch, 0x41 + i, ch.upper()))
    for i in range(10):
        keys.append(_KeyInfo(str(i), 0x30 + i, str(i)))
        keys.append(_KeyInfo(f"num{i}", VK_NUMPAD0 + i, f"Num {i}"))
        aliases[f"numpad{i}"] = f"num{i}"
    for i in range(1, 25):
        keys.append(_KeyInfo(f"f{i}", VK_F1 + i - 1, f"F{i}"))
    named: list[tuple[str, int, str, tuple[str, ...]]] = [
        ("space", VK_SPACE, "Space", ("spacebar", "spc")),
        ("enter", VK_RETURN, "Enter", ("return",)),
        ("tab", VK_TAB, "Tab", ()),
        ("backspace", VK_BACK, "Backspace", ("back", "bksp")),
        ("esc", VK_ESCAPE, "Esc", ("escape",)),
        ("delete", VK_DELETE, "Delete", ("del",)),
        ("insert", VK_INSERT, "Insert", ("ins",)),
        ("home", VK_HOME, "Home", ()),
        ("end", VK_END, "End", ()),
        ("pageup", VK_PRIOR, "Page Up", ("pgup", "prior")),
        ("pagedown", VK_NEXT, "Page Down", ("pgdn", "pgdown", "next")),
        ("up", VK_UP, "Up", ("uparrow", "arrowup")),
        ("down", VK_DOWN, "Down", ("downarrow", "arrowdown")),
        ("left", VK_LEFT, "Left", ("leftarrow", "arrowleft")),
        ("right", VK_RIGHT, "Right", ("rightarrow", "arrowright")),
        ("pause", VK_PAUSE, "Pause", ("break",)),
        ("scrolllock", VK_SCROLL, "Scroll Lock", ("scroll", "scrlk")),
        ("printscreen", VK_SNAPSHOT, "Print Screen", ("prtsc", "prtscn", "print", "snapshot")),
        ("menu", VK_APPS, "Menu", ("apps", "contextmenu", "application")),
        ("nummultiply", VK_MULTIPLY, "Num *", ("num*", "numpad*", "multiply")),
        ("numadd", VK_ADD, "Num +", ("numplus", "add")),
        ("numsubtract", VK_SUBTRACT, "Num -", ("num-", "numpad-", "numminus", "subtract")),
        ("numdecimal", VK_DECIMAL, "Num .", ("num.", "numpad.", "numperiod", "decimal")),
        ("numdivide", VK_DIVIDE, "Num /", ("num/", "numpad/", "numslash", "divide")),
    ]
    for name, vk, label, extra in named:
        keys.append(_KeyInfo(name, vk, label))
        aliases.update(dict.fromkeys(extra, name))
    # Punctuation keys. Names and fallback labels follow the US layout; on screen
    # the label is whatever the user's own layout prints on that key.
    punctuation: list[tuple[str, int, str, tuple[str, ...]]] = [
        ("semicolon", 0xBA, ";", (";",)),
        ("equal", 0xBB, "=", ("=", "equals", "plus")),
        ("comma", 0xBC, ",", (",",)),
        ("minus", 0xBD, "-", ("-", "dash", "hyphen")),
        ("period", 0xBE, ".", (".", "dot")),
        ("slash", 0xBF, "/", ("/", "forwardslash")),
        ("backquote", 0xC0, "`", ("`", "grave", "backtick")),
        ("bracketleft", 0xDB, "[", ("[", "lbracket", "leftbracket", "openbracket")),
        ("backslash", 0xDC, "\\", ("\\",)),
        ("bracketright", 0xDD, "]", ("]", "rbracket", "rightbracket", "closebracket")),
        ("quote", 0xDE, "'", ("'", "apostrophe", "singlequote")),
    ]
    for name, vk, label, extra in punctuation:
        keys.append(_KeyInfo(name, vk, label, layout_char=True))
        aliases.update(dict.fromkeys(extra, name))
    return {k.name: k for k in keys}, aliases


_KEYS, _KEY_ALIASES = _build_key_table()
_VK_TO_KEY: dict[int, str] = {info.vk: name for name, info in _KEYS.items()}

_BARE_OK = frozenset({*(f"f{i}" for i in range(1, 25)), "pause", "scrolllock", "insert"})
"""Keys that may be a shortcut on their own (or with only Shift): they never type anything."""

_RESERVED: dict[str, str] = {
    "ctrl+a": "select all",
    "ctrl+c": "copy",
    "ctrl+v": "paste",
    "ctrl+x": "cut",
    "ctrl+z": "undo",
    "ctrl+y": "redo",
    "ctrl+insert": "copy",
    "shift+insert": "paste",
    "shift+delete": "cut",
    "alt+tab": "switching windows",
    "alt+shift+tab": "switching windows",
    "alt+esc": "switching windows",
    "alt+f4": "closing windows",
    "ctrl+esc": "opening Start",
    "ctrl+shift+esc": "opening Task Manager",
    "ctrl+alt+delete": "the security screen",
    "win+l": "locking the PC",
}


def key_name_for_vk(vk: int) -> str | None:
    """Canonical name of a regular (non-modifier) key, or None if Whisp doesn't support it."""
    return _VK_TO_KEY.get(vk)


def key_vk(name: str) -> int:
    """Virtual key of a canonical key name. Raises KeyError for unknown names."""
    return _KEYS[name].vk


# -- chords --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chord:
    """A keyboard shortcut: zero or more modifiers plus an optional regular key.

    ``modifiers`` holds canonical modifier tokens ("ctrl", "rctrl", "win", ...),
    at most one per family. ``key`` is a canonical key name ("space", "f8") or
    None for a modifier-only chord. Construction checks structure only; use
    ``parse_chord`` (or ``validate_chord``) for the rules about which chords
    make sensible shortcuts.
    """

    modifiers: frozenset[str] = frozenset()
    key: str | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for token in self.modifiers:
            if token not in _MODIFIER_TOKENS:
                raise ValueError(f'"{token}" is not a modifier key.')
            family = _MODIFIER_TOKENS[token][0]
            if family in seen:
                raise ValueError(f"{_FAMILY_LABELS[family]} appears more than once.")
            seen.add(family)
        if self.key is not None and self.key not in _KEYS:
            raise ValueError(f'"{self.key}" isn\'t a key Whisp recognises.')
        if self.key is None and not self.modifiers:
            raise ValueError("A shortcut needs at least one key.")

    @property
    def is_modifier_only(self) -> bool:
        return self.key is None

    @property
    def families(self) -> frozenset[str]:
        return frozenset(_MODIFIER_TOKENS[t][0] for t in self.modifiers)

    def side(self, family: str) -> str | None:
        """The required side for ``family`` ("left"/"right"), or None for either side."""
        for token in self.modifiers:
            fam, side = _MODIFIER_TOKENS[token]
            if fam == family:
                return side
        raise KeyError(family)

    @property
    def has_side_specific(self) -> bool:
        return any(_MODIFIER_TOKENS[t][1] is not None for t in self.modifiers)

    @property
    def vk(self) -> int | None:
        return _KEYS[self.key].vk if self.key is not None else None

    @property
    def mod_flags(self) -> int:
        """``RegisterHotKey`` modifier flags (without MOD_NOREPEAT)."""
        flags = 0
        for family in self.families:
            flags |= _FAMILY_MOD_FLAGS[family]
        return flags

    def __str__(self) -> str:
        return format_chord(self)


def _ordered_modifiers(chord: Chord) -> list[str]:
    by_family = {_MODIFIER_TOKENS[t][0]: t for t in chord.modifiers}
    return [by_family[f] for f in MODIFIER_FAMILIES if f in by_family]


def format_chord(chord: Chord) -> str:
    """Canonical text, e.g. ``"ctrl+alt+shift+win+space"`` (modifiers in a fixed order, then the key)."""
    parts = _ordered_modifiers(chord)
    if chord.key is not None:
        parts.append(chord.key)
    return "+".join(parts)


def _normalize_token(raw: str) -> str:
    return "".join(raw.split()).replace("_", "").lower()


def _parse_structure(text: str) -> Chord:
    """Parse ``text`` into a Chord, checking only that it names real keys."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No shortcut given.")
    raw_parts = text.split("+")
    modifiers: list[str] = []
    key: str | None = None
    key_raw = ""
    for raw in raw_parts:
        token = _normalize_token(raw)
        if not token:
            raise ValueError("This shortcut has an empty part. Join keys with a single + (like ctrl+shift+space).")
        token = _MODIFIER_ALIASES.get(token, token)
        if token in _MODIFIER_TOKENS:
            modifiers.append(token)
            continue
        name = _KEY_ALIASES.get(token, token)
        if name not in _KEYS:
            raise ValueError(f'"{raw.strip()}" isn\'t a key Whisp recognises.')
        if key is not None:
            raise ValueError(
                "A shortcut can have only one key besides Ctrl, Alt, Shift and Win "
                f"({key_raw.strip()} and {raw.strip()} were given)."
            )
        key, key_raw = name, raw
    families = [_MODIFIER_TOKENS[t][0] for t in modifiers]
    for family in MODIFIER_FAMILIES:
        if families.count(family) > 1:
            raise ValueError(f"{_FAMILY_LABELS[family]} appears more than once.")
    return Chord(frozenset(modifiers), key)


def validate_chord(chord: Chord) -> None:
    """Raise ValueError with a friendly message if ``chord`` would make a poor global shortcut.

    Rejected: chords that would get in the way of normal typing (a letter,
    digit, Space, Enter, … on its own or with only Shift), standard Windows
    editing and system shortcuts, a lone either-side modifier ("ctrl"), and
    left/right-specific modifiers combined with a regular key (Windows can't
    register those).
    """
    families = chord.families
    if chord.key is None:
        if len(chord.modifiers) == 1 and not chord.has_side_specific:
            label = _FAMILY_LABELS[next(iter(families))]
            raise ValueError(
                f"{label} on its own would fire every time you use a {label} shortcut. "
                f"Use Right {label} or Left {label}, or combine it with another key."
            )
        return
    if chord.has_side_specific:
        raise ValueError(
            "Left/right-specific keys (like Right Ctrl) only work in shortcuts made of modifier keys alone, "
            "such as rctrl or ctrl+win. With a regular key, use Ctrl, Alt, Shift or Win."
        )
    label = _KEYS[chord.key].label
    if not families or families == {"shift"}:
        if chord.key == "esc" and not families:
            raise ValueError("Esc on its own is reserved for cancelling a dictation.")
        if chord.key not in _BARE_OK:
            if families:
                raise ValueError(f"Shift + {label} is used for typing and selecting text. Add Ctrl, Alt or Win.")
            raise ValueError(f"{label} on its own would get in the way of normal typing. Add Ctrl, Alt or Win.")
    canonical = format_chord(chord)
    purpose = _RESERVED.get(canonical)
    if purpose is not None:
        raise ValueError(
            f"{describe_chord(chord)} is a standard Windows shortcut for {purpose}. Pick a different combination."
        )


def parse_chord(text: str) -> Chord:
    """Parse and validate a shortcut such as ``"Ctrl + Shift + Space"`` (case and spaces don't matter).

    Raises ValueError with a message that can be shown to the user as is.
    """
    chord = _parse_structure(text)
    validate_chord(chord)
    return chord


def _modifier_label(token: str) -> str:
    family, side = _MODIFIER_TOKENS[token]
    base = _FAMILY_LABELS[family]
    if side is None:
        return base
    return f"{'Left' if side == LEFT else 'Right'} {base}"


def _key_label(name: str) -> str:
    info = _KEYS[name]
    if info.layout_char:
        char = _layout_char(info.vk)
        if char:
            return char
    return info.label


def chord_labels(chord: Chord) -> list[str]:
    """Key-cap labels for a parsed chord, e.g. ``["Ctrl", "Shift", "Space"]``."""
    labels = [_modifier_label(t) for t in _ordered_modifiers(chord)]
    if chord.key is not None:
        labels.append(_key_label(chord.key))
    return labels


def display_chord(text: str) -> list[str]:
    """Key-cap labels for a chord string: ``"rctrl"`` → ``["Right Ctrl"]``; ``""`` → ``[]``.

    Accepts any chord that names real keys, even one ``parse_chord`` would
    reject, so a stored setting can always be shown. Raises ValueError for text
    that doesn't name keys at all.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return chord_labels(_parse_structure(text))


def describe_chord(chord: Chord | str) -> str:
    """Human-readable chord for messages, e.g. ``"Ctrl + Shift + Space"``."""
    labels = chord_labels(chord) if isinstance(chord, Chord) else display_chord(chord)
    return " + ".join(labels)


def chord_from_keys(families_sides: Iterable[tuple[str, str]], key: str | None) -> Chord:
    """Build a chord from keys the user actually pressed (used when recording a shortcut).

    With a regular key, modifiers are recorded for either side ("ctrl+shift+f8").
    A single modifier pressed on its own is recorded with its side ("rctrl");
    several modifiers on their own are recorded for either side ("ctrl+win").
    """
    pressed = set(families_sides)
    if key is None and len(pressed) == 1:
        family, side = next(iter(pressed))
        token = ("l" if side == LEFT else "r") + family
        return Chord(frozenset({token}), None)
    return Chord(frozenset(family for family, _ in pressed), key)


# -- Win32: layout labels, key state and synthetic input ------------------------------

_user32 = ctypes.WinDLL("user32", use_last_error=True)

_MapVirtualKeyW = _user32.MapVirtualKeyW
_MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
_MapVirtualKeyW.restype = wintypes.UINT

_MapVirtualKeyExW = _user32.MapVirtualKeyExW
_MapVirtualKeyExW.argtypes = [wintypes.UINT, wintypes.UINT, wintypes.HKL]
_MapVirtualKeyExW.restype = wintypes.UINT

_GetAsyncKeyState = _user32.GetAsyncKeyState
_GetAsyncKeyState.argtypes = [ctypes.c_int]
_GetAsyncKeyState.restype = ctypes.c_short

_GetKeyboardLayout = _user32.GetKeyboardLayout
_GetKeyboardLayout.argtypes = [wintypes.DWORD]
_GetKeyboardLayout.restype = wintypes.HKL

MAPVK_VK_TO_CHAR = 2
MAPVK_VK_TO_VSC_EX = 4


def _map_virtual_key(code: int, map_type: int, layout: int | None) -> int:
    if layout is None:
        return int(_MapVirtualKeyW(code, map_type))
    return int(_MapVirtualKeyExW(code, map_type, layout))


def _layout_char(vk: int) -> str | None:
    """The character the calling thread's keyboard layout prints on a key, if it's a visible one."""
    value = _map_virtual_key(vk, MAPVK_VK_TO_CHAR, None) & 0xFFFF
    if not value:
        return None
    char = chr(value)
    return char.upper() if char.isprintable() and not char.isspace() else None


def is_key_down(vk: int) -> bool:
    """True while ``vk`` is held, as Windows currently sees it (physical keys plus injected ones)."""
    return bool(_GetAsyncKeyState(vk) & 0x8000)


MODIFIER_STATE_VKS: tuple[int, ...] = (VK_CONTROL, VK_SHIFT, VK_MENU, VK_LWIN, VK_RWIN)


def any_modifier_down() -> bool:
    """True while Ctrl, Shift, Alt or a Windows key is held."""
    return any(is_key_down(vk) for vk in MODIFIER_STATE_VKS)


def keyboard_layout(thread_id: int) -> int | None:
    """The keyboard layout (HKL) active for a thread, or None if Windows doesn't say."""
    layout = _GetKeyboardLayout(thread_id)
    return int(layout) if layout else None


INJECTED_MARKER = 0x57685370
"""``dwExtraInfo`` on every event Whisp injects ("WhSp"), so Whisp's hook can ignore its own input."""

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD))


class _INPUTUNION(ctypes.Union):
    # MOUSEINPUT is the largest member; it must be present so INPUT has the size SendInput expects.
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


_SendInput = _user32.SendInput
_SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
_SendInput.restype = wintypes.UINT


_EXTENDED_VKS = frozenset(
    {
        VK_INSERT,
        VK_DELETE,
        VK_HOME,
        VK_END,
        VK_PRIOR,
        VK_NEXT,
        VK_LEFT,
        VK_UP,
        VK_RIGHT,
        VK_DOWN,
        VK_RCONTROL,
        VK_RMENU,
        VK_LWIN,
        VK_RWIN,
        VK_APPS,
        VK_DIVIDE,
        VK_SNAPSHOT,
    }
)
"""Keys on the navigation cluster and other "E0" keys. MapVirtualKey maps several of them to their numeric
keypad twins, so the extended flag is set explicitly; terminals use it to tell Insert from keypad 0."""


def key_event(vk: int, *, up: bool = False, layout: int | None = None, marker: int = INJECTED_MARKER) -> INPUT:
    """A virtual-key event with a matching scan code (some apps read one, some the other)."""
    scan = _map_virtual_key(vk, MAPVK_VK_TO_VSC_EX, layout) if vk != VK_MASK_KEY else 0
    flags = KEYEVENTF_KEYUP if up else 0
    if vk in _EXTENDED_VKS or (scan & 0xFF00) in (0xE000, 0xE100):
        flags |= KEYEVENTF_EXTENDEDKEY
    event = INPUT(type=INPUT_KEYBOARD)
    event.ki = KEYBDINPUT(wVk=vk, wScan=scan & 0xFF, dwFlags=flags, time=0, dwExtraInfo=marker)
    return event


def unicode_event(code_unit: int, *, up: bool = False, marker: int = INJECTED_MARKER) -> INPUT:
    """A KEYEVENTF_UNICODE event for one UTF-16 code unit; held modifier keys don't change what it types."""
    event = INPUT(type=INPUT_KEYBOARD)
    flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
    event.ki = KEYBDINPUT(wVk=0, wScan=code_unit, dwFlags=flags, time=0, dwExtraInfo=marker)
    return event


def key_tap(vk: int, *, layout: int | None = None) -> list[INPUT]:
    return [key_event(vk, layout=layout), key_event(vk, up=True, layout=layout)]


def combo(modifier_vks: Sequence[int], vk: int, *, layout: int | None = None) -> list[INPUT]:
    """Press modifiers, tap ``vk``, release the modifiers in reverse order."""
    events = [key_event(m, layout=layout) for m in modifier_vks]
    events += key_tap(vk, layout=layout)
    events += [key_event(m, up=True, layout=layout) for m in reversed(modifier_vks)]
    return events


def mask_key_events() -> list[INPUT]:
    """Down and up of ``VK_MASK_KEY``: harmless to every app, but it counts as "another key" for Win/Alt."""
    return key_tap(VK_MASK_KEY)


def send_input(events: Sequence[INPUT]) -> int:
    """Inject ``events`` as one uninterruptible batch. Returns how many Windows accepted.

    Fewer than ``len(events)`` usually means the focused window belongs to a
    program running as administrator (Windows blocks input into it), or the
    secure desktop is showing.
    """
    if not events:
        return 0
    array = (INPUT * len(events))(*events)
    return int(_SendInput(len(events), array, ctypes.sizeof(INPUT)))
