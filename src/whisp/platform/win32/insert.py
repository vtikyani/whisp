"""TextInserter: put text into the focused app by pasting it or typing it.

All methods block and are meant for worker threads. ``insert`` never raises;
every failure becomes an ``InsertionResult`` with a plain-language ``error``.

Safety rules, in order:

1. Never type while the user holds Ctrl, Shift, Alt or Win — the text would
   turn into shortcuts. Wait for the keys to be released first.
2. Never put text into a window the user didn't dictate into, unless they
   chose that (``on_focus_change="insert"``).
3. Leave the user's clipboard as it was, but only after the app has had time
   to read the pasted text, and only if nobody changed the clipboard since.

``insert`` returns as soon as the text is in. The clipboard is put back by a
short-lived ``whisp-clipboard-restore`` thread after the app's reading time;
every other clipboard round-trip waits for it.
"""

from __future__ import annotations

import logging
import re
import threading
import time

from whisp.core.types import AppCategory, InsertionResult, TargetWindow
from whisp.platform.win32 import clipboard, keys
from whisp.platform.win32.foreground import (
    current_process_elevated,
    focused_control,
    foreground_window,
    is_own_hidden_window,
    is_process_elevated,
    target_for_window,
    window_class,
    window_process_id,
    window_thread_id,
)

log = logging.getLogger(__name__)

METHODS = ("paste", "paste_shift_insert", "type")

MODIFIER_RELEASE_TIMEOUT_S = 2.0
"""How long ``insert`` and ``capture_selection`` wait for the user to let go of Ctrl/Shift/Alt/Win.
Keys sent while one of them is held reach the app as a different shortcut (Ctrl+Alt+Win+C isn't a copy)."""

CLIPBOARD_RESTORE_DELAY_S = 0.6
"""Wait this long after pasting before restoring the clipboard. Apps such as Chromium and Electron read the
clipboard asynchronously; restoring sooner makes them paste the user's old clipboard instead of the text."""

SLOW_READER_RESTORE_DELAY_S = 3.0
"""For apps known to read a paste late (Office, WhatsApp)."""

REMOTE_RESTORE_DELAY_S = 5.0
"""For remote-desktop and virtual-machine windows, which first copy the clipboard to the other computer."""

_RESTORE_DELAYS: dict[str, float] = {
    **dict.fromkeys(
        ("outlook.exe", "olk.exe", "winword.exe", "whatsapp.exe", "whatsapp.root.exe"),
        SLOW_READER_RESTORE_DELAY_S,
    ),
    **dict.fromkeys(
        (
            "mstsc.exe",  # Remote Desktop Connection
            "msrdc.exe",  # Remote Desktop client (Azure Virtual Desktop, Windows App, RemoteApp windows)
            "msrdcw.exe",
            "rdcman.exe",  # Remote Desktop Connection Manager
            "vmconnect.exe",  # Hyper-V
            "vmware.exe",  # VMware Workstation
            "vmplayer.exe",
            "vmware-view.exe",  # VMware Horizon client
            "virtualboxvm.exe",
            "wfica32.exe",  # Citrix Workspace
            "cdviewer.exe",
            "anydesk.exe",
            "teamviewer.exe",
            "rustdesk.exe",
            "parsecd.exe",
            "vncviewer.exe",
            "tvnviewer.exe",
            "remote-viewer.exe",  # virt-viewer (SPICE/VNC)
        ),
        REMOTE_RESTORE_DELAY_S,
    ),
}


def restore_delay_for(process_name: str) -> float:
    """Seconds to wait after pasting into ``process_name`` (lower-case exe name) before restoring the clipboard."""
    return _RESTORE_DELAYS.get(process_name.strip().lower(), CLIPBOARD_RESTORE_DELAY_S)


_KEY_POLL_S = 0.01

_BUILTIN_METHODS: dict[str, str] = {
    # Terminals where Ctrl+V sends a control character instead of pasting; Shift+Insert pastes.
    "mintty.exe": "paste_shift_insert",  # Git Bash, Cygwin, MSYS2
    "putty.exe": "paste_shift_insert",
    "puttytel.exe": "paste_shift_insert",
    "kitty.exe": "paste_shift_insert",  # KiTTY, a PuTTY fork
    "kitty_portable.exe": "paste_shift_insert",
}

MSG_HELD_KEYS = "A key was still held down, so the text was copied instead. Press Ctrl+V to paste it."
MSG_ELEVATED = (
    "That app is running as administrator, so Windows won't let Whisp type into it. "
    "The text is on the clipboard — press Ctrl+V to paste it."
)
MSG_NO_WINDOW = "No window had the focus, so the text was copied. Press Ctrl+V to paste it."
MSG_BLOCKED = "Windows blocked the keystrokes, so the text was copied instead. Press Ctrl+V to paste it."
MSG_PARTIAL = "Only part of the text could be typed."
MSG_CLIPBOARD_FAILED = "The clipboard is busy, so the text couldn't be inserted."
MSG_UNEXPECTED = "Something went wrong while inserting the text."


def _wait_for_modifiers_released(timeout_s: float) -> bool:
    """Wait until Ctrl, Shift, Alt and Win are all up. Returns False if they're still held at the deadline."""
    deadline = time.monotonic() + timeout_s
    while keys.any_modifier_down():
        if time.monotonic() >= deadline:
            return False
        time.sleep(_KEY_POLL_S)
    return True


def _changed_since_checked(hwnd: int) -> tuple[bool, bool]:
    """(a modifier is still held after waiting, the focus has left ``hwnd``) since ``insert`` checked both.

    For keystrokes sent after a step that can take a while, such as saving the clipboard.
    """
    held = not _wait_for_modifiers_released(MODIFIER_RELEASE_TIMEOUT_S)
    return held, foreground_window() != hwnd


def _utf16_units(text: str) -> list[int]:
    data = text.encode("utf-16-le", errors="surrogatepass")
    return [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]


_QT_WINDOW_CLASS = re.compile(r"Qt\d*QWindow")
"""Window classes of apps built with Qt (Telegram, Zoom, OBS…), such as "Qt6120QWindowIcon"."""


def _has_supplementary(text: str) -> bool:
    """True if ``text`` has characters beyond U+FFFF (most emoji): each takes two UTF-16 units to type."""
    return any(ord(ch) > 0xFFFF for ch in text)


def _types_emoji_out_of_order(hwnd: int) -> bool:
    """True if the text box with the focus in ``hwnd`` belongs to a Qt app.

    Qt hands a typed emoji to the text box at once, while the keys typed before it are still queued, so typing
    "ab👍" there gives "👍ab". (Windows' own text boxes, rich text boxes and browsers keep the order.)
    """
    return _QT_WINDOW_CLASS.match(window_class(focused_control(hwnd) or hwnd)) is not None


def build_type_events(text: str, *, messaging: bool = False, layout: int | None = None) -> list[keys.INPUT]:
    """Key events that type ``text``: one Unicode down/up per UTF-16 code unit, so emoji work.

    Line breaks become Enter — or Shift+Enter in messaging apps, where Enter
    would send the message.
    """
    events: list[keys.INPUT] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for index, line in enumerate(normalized.split("\n")):
        if index:
            if messaging:
                events += keys.combo([keys.VK_SHIFT], keys.VK_RETURN, layout=layout)
            else:
                events += keys.key_tap(keys.VK_RETURN, layout=layout)
        for unit in _utf16_units(line):
            events.append(keys.unicode_event(unit))
            events.append(keys.unicode_event(unit, up=True))
    return events


def _paste_events(method: str, layout: int | None) -> list[keys.INPUT]:
    if method == "paste_shift_insert":
        return keys.combo([keys.VK_SHIFT], keys.VK_INSERT, layout=layout)
    return keys.combo([keys.VK_CONTROL], keys.VK_V, layout=layout)


def _layout_for(hwnd: int) -> int | None:
    thread_id = window_thread_id(hwnd)
    return keys.keyboard_layout(thread_id) if thread_id else None


def _process_name(hwnd: int, target: TargetWindow | None) -> str:
    """Lower-case exe name of the app that owns ``hwnd`` (Store apps resolved past their frame host)."""
    if target is not None and target.hwnd == hwnd:
        return target.process_name
    window = target_for_window(hwnd)
    return window.process_name if window is not None else ""


class TextInserter:
    """Inserts text into the focused app. Thread-safe: clipboard round-trips never overlap."""

    def __init__(self) -> None:
        self._clipboard_lock = threading.Lock()

    @staticmethod
    def default_method_for(process_name: str) -> str | None:
        """Whisp's built-in insertion method for apps that don't paste with Ctrl+V, or None."""
        return _BUILTIN_METHODS.get(process_name.strip().lower())

    @staticmethod
    def restore_delay_for(process_name: str) -> float:
        """Seconds to wait after pasting into an app before restoring the clipboard (module function of the
        same name)."""
        return restore_delay_for(process_name)

    # -- insert --------------------------------------------------------------------------

    def insert(
        self,
        text: str,
        *,
        target: TargetWindow | None,
        method: str,
        restore_clipboard: bool,
        on_focus_change: str,
    ) -> InsertionResult:
        """Insert ``text`` into the focused window. Blocking; call from a worker thread. Never raises."""
        try:
            return self._insert(
                text,
                target=target,
                method=method,
                restore_clipboard=restore_clipboard,
                on_focus_change=on_focus_change,
            )
        except Exception:
            log.exception("Text insertion failed (method %s, %d chars)", method, len(text))
            return InsertionResult(ok=False, method=method, error=MSG_UNEXPECTED)

    def _insert(
        self,
        text: str,
        *,
        target: TargetWindow | None,
        method: str,
        restore_clipboard: bool,
        on_focus_change: str,
    ) -> InsertionResult:
        if method not in METHODS:
            return InsertionResult(ok=False, method=method, error=f"Unknown insertion method “{method}”.")
        result_method = "type" if method == "type" else "paste"
        if not text:
            return InsertionResult(ok=True, method=result_method)

        released = _wait_for_modifiers_released(MODIFIER_RELEASE_TIMEOUT_S)
        hwnd = foreground_window()
        # Whisp's overlay never takes the focus, so any different foreground window is a real focus change.
        focus_changed = target is not None and hwnd != target.hwnd
        if focus_changed:
            log.info("The focused window changed since dictation started (on_focus_change=%s)", on_focus_change)
            if on_focus_change == "clipboard":
                return self._leave_on_clipboard(text, focus_changed=True, reason=None)
        if not released:
            log.warning("Modifier keys still held after %.1f s; copying instead", MODIFIER_RELEASE_TIMEOUT_S)
            return self._leave_on_clipboard(text, focus_changed=focus_changed, reason=MSG_HELD_KEYS)
        if not hwnd or is_own_hidden_window(hwnd):
            # E.g. after starting and stopping from the tray menu, Whisp's hidden menu window can hold the focus.
            return self._leave_on_clipboard(text, focus_changed=focus_changed, reason=MSG_NO_WINDOW)
        if not current_process_elevated() and is_process_elevated(window_process_id(hwnd)):
            log.info("The focused app runs as administrator; leaving the text on the clipboard")
            return self._leave_on_clipboard(text, focus_changed=focus_changed, reason=MSG_ELEVATED)

        layout = _layout_for(hwnd)
        if method == "type" and _has_supplementary(text) and _types_emoji_out_of_order(hwnd):
            log.info("This app would put typed emoji in the wrong place; pasting instead")
            method = "paste"
        if method == "type":
            # The category describes the window the user dictated into; it says nothing about a different one.
            category = target.category if target is not None and not focus_changed else AppCategory.OTHER
            messaging = category is AppCategory.MESSAGING
            return self._type(text, layout=layout, messaging=messaging, focus_changed=focus_changed)
        restore_delay = restore_delay_for(_process_name(hwnd, target)) if restore_clipboard else 0.0
        return self._paste(
            text,
            method=method,
            layout=layout,
            restore=restore_clipboard,
            restore_delay=restore_delay,
            focus_changed=focus_changed,
            hwnd=hwnd,
        )

    def _type(self, text: str, *, layout: int | None, messaging: bool, focus_changed: bool) -> InsertionResult:
        events = build_type_events(text, messaging=messaging, layout=layout)
        sent = keys.send_input(events)
        if sent == len(events):
            log.info("Typed %d chars", len(text))
            return InsertionResult(ok=True, method="type", focus_changed=focus_changed)
        if sent == 0:
            log.warning("Windows rejected the typed input (%d events)", len(events))
            return self._leave_on_clipboard(text, focus_changed=focus_changed, reason=MSG_BLOCKED)
        log.warning("Windows accepted only %d of %d typed events", sent, len(events))
        return InsertionResult(ok=False, method="type", focus_changed=focus_changed, error=MSG_PARTIAL)

    def _paste(
        self,
        text: str,
        *,
        method: str,
        layout: int | None,
        restore: bool,
        restore_delay: float,
        focus_changed: bool,
        hwnd: int,
    ) -> InsertionResult:
        # The lock is held until the user's clipboard is back, after the app has had time to read the paste: a
        # selection capture or another paste in between would save the pasted text and leave it behind, losing
        # the user's own clipboard. That wait happens on a helper thread, which then releases the lock, so this
        # returns as soon as the text is in.
        self._clipboard_lock.acquire()
        handed_over = False
        try:
            try:
                saved = clipboard.snapshot() if restore else None
                clipboard.set_text(text, exclude_from_history=True)
            except clipboard.ClipboardError:
                log.warning("Couldn't use the clipboard; typing the text instead", exc_info=True)
            else:
                written = clipboard.get_sequence_number()
                result = self._send_paste(text, method=method, layout=layout, focus_changed=focus_changed, hwnd=hwnd)
                if saved is not None and result.method == "paste":
                    handed_over = self._restore_later(saved, written, restore_delay)
                return result
        finally:
            if not handed_over:
                self._clipboard_lock.release()
        # Outside the lock: typing may fall back to the clipboard itself.
        held, moved = _changed_since_checked(hwnd)
        if held or moved:
            return self._leave_on_clipboard(
                text, focus_changed=focus_changed or moved, reason=MSG_HELD_KEYS if held else None
            )
        return self._type(text, layout=layout, messaging=False, focus_changed=focus_changed)

    def _send_paste(
        self, text: str, *, method: str, layout: int | None, focus_changed: bool, hwnd: int
    ) -> InsertionResult:
        """Send the paste keystroke for text already on the clipboard."""
        # Saving and filling the clipboard can take a while (apps that render clipboard data on demand). A key
        # pressed or a window switched to meanwhile would turn Ctrl+V into another shortcut, or paste into the
        # wrong window: the text stays on the clipboard instead.
        held, moved = _changed_since_checked(hwnd)
        if held or moved:
            log.info(
                "%s while the clipboard was prepared; leaving the text on it", "Keys held" if held else "Focus moved"
            )
            return InsertionResult(
                ok=True, method="clipboard", focus_changed=focus_changed or moved, error=MSG_HELD_KEYS if held else None
            )
        events = _paste_events(method, layout)
        sent = keys.send_input(events)
        if sent != len(events):
            # The text is already on the clipboard; leave it there for the user.
            log.warning("Windows rejected the paste keystrokes (%d of %d sent)", sent, len(events))
            return InsertionResult(ok=True, method="clipboard", focus_changed=focus_changed, error=MSG_BLOCKED)
        log.info("Pasted %d chars (%s)", len(text), method)
        return InsertionResult(ok=True, method="paste", focus_changed=focus_changed)

    def _restore_later(self, saved: clipboard.ClipboardSnapshot, written: int, delay: float) -> bool:
        """Restore the clipboard after ``delay`` on a helper thread, which then releases the clipboard lock.

        Returns False if the thread couldn't start: the clipboard was restored right here, and the caller still
        holds the lock.
        """

        def restore_then_unlock() -> None:
            try:
                self._restore_after_paste(saved, written, delay)
            finally:
                self._clipboard_lock.release()

        # Not a daemon: quitting right after a paste still gives the user their clipboard back.
        thread = threading.Thread(target=restore_then_unlock, name="whisp-clipboard-restore")
        try:
            thread.start()
        except RuntimeError:
            log.warning("Couldn't restore the clipboard in the background; restoring it now")
            self._restore_after_paste(saved, written, delay)
            return False
        return True

    @staticmethod
    def _restore_after_paste(saved: clipboard.ClipboardSnapshot, written: int, delay: float) -> None:
        # There is no notification for "the app has read the clipboard", so give it a head start
        # that suits the app (see restore_delay_for).
        time.sleep(delay)
        if clipboard.get_sequence_number() != written:
            log.info("The clipboard changed after pasting; leaving it as it is")
            return
        try:
            clipboard.restore(saved)
        except clipboard.ClipboardError:
            log.warning("Couldn't restore the clipboard after pasting", exc_info=True)

    def _leave_on_clipboard(self, text: str, *, focus_changed: bool, reason: str | None) -> InsertionResult:
        if self.copy_to_clipboard(text):
            return InsertionResult(ok=True, method="clipboard", focus_changed=focus_changed, error=reason)
        return InsertionResult(ok=False, method="clipboard", focus_changed=focus_changed, error=MSG_CLIPBOARD_FAILED)

    # -- clipboard helpers ---------------------------------------------------------------

    def copy_to_clipboard(self, text: str) -> bool:
        """Put ``text`` on the clipboard, kept out of clipboard history. Returns False if that failed."""
        try:
            with self._clipboard_lock:
                clipboard.set_text(text, exclude_from_history=True)
            return True
        except clipboard.ClipboardError:
            log.warning("Couldn't copy %d chars to the clipboard", len(text), exc_info=True)
            return False
        except Exception:
            log.exception("Copying to the clipboard failed")
            return False

    def capture_selection(self, timeout_ms: int = 500) -> str | None:
        """The text selected in the focused app, copied with Ctrl+C; the clipboard is restored afterwards.

        Blocking. First waits (up to ``MODIFIER_RELEASE_TIMEOUT_S``) for Ctrl,
        Shift, Alt and Win to be released — Ctrl+C with Alt or Win still held
        would be a different shortcut — and returns None without sending any
        keys if they stay down. Then waits up to ``timeout_ms`` for the app to
        copy. Returns None if nothing was selected or the selection holds no
        text. Never raises.
        """
        try:
            return self._capture_selection(timeout_ms / 1000.0)
        except Exception:
            log.exception("Couldn't read the selected text")
            return None

    def _capture_selection(self, timeout_s: float) -> str | None:
        if not _wait_for_modifiers_released(MODIFIER_RELEASE_TIMEOUT_S):
            log.warning("Modifier keys still held after %.1f s; not reading the selection", MODIFIER_RELEASE_TIMEOUT_S)
            return None
        hwnd = foreground_window()
        if not hwnd:
            return None
        if not current_process_elevated() and is_process_elevated(window_process_id(hwnd)):
            log.info("The focused app runs as administrator; Whisp can't copy its selection")
            return None
        layout = _layout_for(hwnd)
        with self._clipboard_lock:
            try:
                saved = clipboard.snapshot()
            except clipboard.ClipboardError:
                log.warning("Couldn't save the clipboard; not reading the selection", exc_info=True)
                return None
            before = clipboard.get_sequence_number()
            events = keys.combo([keys.VK_CONTROL], keys.VK_C, layout=layout)
            if keys.send_input(events) != len(events):
                log.warning("Windows rejected the copy keystrokes")
                return None
            deadline = time.monotonic() + timeout_s
            if not clipboard.wait_for_change(before, timeout_s):
                log.info("Nothing was copied; treating the selection as empty")
                return None
            try:
                text = self._read_copied_text(deadline)
            finally:
                try:
                    clipboard.restore(saved)
                except clipboard.ClipboardError:
                    log.warning("Couldn't restore the clipboard after reading the selection", exc_info=True)
        if text:
            log.info("Read a selection of %d chars", len(text))
        return text or None

    @staticmethod
    def _read_copied_text(deadline: float) -> str | None:
        # Some apps empty the clipboard and add their formats a moment later; give them until the deadline.
        while True:
            try:
                text = clipboard.get_text()
            except clipboard.ClipboardError:
                text = None
            if text or time.monotonic() >= deadline:
                return text
            time.sleep(_KEY_POLL_S)
