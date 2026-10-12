"""Smart spacing between consecutive dictations.

When you dictate "Hello there." and then, a moment later, "How are you?" into
the same window, the second piece needs a leading space or the words run
together. ``SpacingMemory`` remembers the last dictation into each window and
how it ended, and decides whether the next one there needs that space.

Modes (``settings.text.smart_spacing``):

- ``"auto"``: decide from what happened in the window since its last dictation.
  Nothing typed there in between → a continuation, so add a space (unless the
  previous text ended in whitespace or an opening bracket/quote). Working in other
  apps meanwhile doesn't count: their keys don't move this window's cursor. Keys
  typed there in between → look only at the *kind* of the last one: a character
  means the user typed a word and the dictation follows it (space); Space, Enter,
  Backspace, arrows or a shortcut mean the cursor is somewhere new or already
  spaced (no space). Without keyboard information, fall back to: a dictation into
  the same window within the last 10 minutes.
- ``"always"``: add a space unless the new text starts with punctuation or a
  line break.
- ``"never"``: no space.

New text that starts with punctuation or a line break never gets a space, and
neither does text next to Chinese or Japanese, which are written without spaces.
Whisp learns only the kind of the last key in each window (see ``KeyActivity``),
never what was typed. Thread-safe.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from whisp.core.text.scripts import is_spaceless
from whisp.core.types import KeyActivity, KeyKind, TargetWindow

log = logging.getLogger(__name__)

AUTO_WINDOW_S = 10 * 60
"""How long after a dictation the next one into the same window still counts as a continuation."""
WINDOWS_REMEMBERED = 32
"""How many windows' last dictations are kept; the one dictated into least recently is dropped first."""

_NO_SPACE_BEFORE = frozenset(",.!?;:)]}…%'’”»")
"""A new dictation starting with one of these attaches to the previous text without a space."""
_NO_SPACE_AFTER = frozenset("([{“‘«/")
"""Previous text ending with one of these needs no space after it."""


@dataclass(frozen=True, slots=True)
class _Last:
    """The last dictation into one window."""

    pid: int
    at: float
    ends_open: bool
    """True when the text ended with whitespace, a line break, an opening bracket/quote, or Chinese or Japanese."""
    key_serial: int | None
    """The window's ``KeyActivity.serial`` right after the insertion, or None if keystrokes aren't known."""


def _starts_attached(text: str) -> bool:
    first = text[0]
    return first.isspace() or first in _NO_SPACE_BEFORE or is_spaceless(first)


class SpacingMemory:
    """Remembers the last dictation and decides on a leading space for the next one."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        window_s: float = AUTO_WINDOW_S,
        activity: Callable[[int], KeyActivity | None] | None = None,
    ) -> None:
        """``activity(hwnd)`` reports the kind of the user's last keystroke in a window
        (``HotkeyService.typing_activity``); its clock must match ``clock``."""
        self._clock = clock
        self._window_s = window_s
        self._activity = activity
        self._lock = threading.Lock()
        self._last: OrderedDict[int, _Last] = OrderedDict()
        """The last dictation into each window, by window handle; least recently dictated into first."""

    def _activity_in(self, hwnd: int) -> KeyActivity | None:
        """The user's last keystroke kind in window ``hwnd``, or None when Whisp has no keyboard information."""
        if self._activity is None:
            return None
        try:
            # None while the keyboard hook isn't installed (e.g. only regular-key shortcuts are bound).
            return self._activity(hwnd)
        except Exception:
            log.exception("Couldn't read keyboard activity; spacing falls back to timing")
            return None

    def remember(self, target: TargetWindow | None, text: str) -> None:
        """Record text that was just inserted into ``target`` (call after a successful insertion)."""
        if not text:
            return
        if target is None:
            self.forget()  # it went into a window Whisp can't name, which may be any of the remembered ones
            return
        activity = self._activity_in(target.hwnd)
        last_char = text[-1]
        last = _Last(
            pid=target.pid,
            at=self._clock(),
            ends_open=last_char.isspace() or last_char in _NO_SPACE_AFTER or is_spaceless(last_char),
            key_serial=activity.serial if activity is not None else None,
        )
        with self._lock:
            self._last[target.hwnd] = last
            self._last.move_to_end(target.hwnd)
            if len(self._last) > WINDOWS_REMEMBERED:
                self._last.popitem(last=False)

    def forget(self) -> None:
        """Drop the memory of every window."""
        with self._lock:
            self._last.clear()

    def prefix_for(self, target: TargetWindow | None, text: str, mode: str) -> str:
        """``" "`` when ``text`` should be inserted with a leading space, else ``""``."""
        if not text or mode == "never":
            return ""
        if _starts_attached(text):
            return ""
        if mode == "always":
            return " "
        if mode != "auto":
            log.warning("Unknown smart spacing mode %r; adding no space", mode)
            return ""
        if target is None:
            return ""
        activity = self._activity_in(target.hwnd)
        with self._lock:
            last = self._last.get(target.hwnd)
            now = self._clock()
        if last is not None and (last.pid != target.pid or now - last.at > self._window_s):
            last = None  # the handle now belongs to another app's window, or the dictation was long ago
        if activity is None or (last is not None and last.key_serial is None):
            # No keyboard information: a recent dictation into the same window is a continuation.
            return " " if last is not None and not last.ends_open else ""
        if last is not None and last.key_serial == activity.serial:
            return "" if last.ends_open else " "  # nothing typed in this window since its last dictation
        typed_recently = activity.serial > 0 and now - activity.at <= self._window_s
        # A character means the user typed a word and the dictation follows it. Space, Enter,
        # Backspace, arrows or a shortcut mean the cursor is already spaced or somewhere new.
        return " " if typed_recently and activity.kind is KeyKind.TEXT else ""
