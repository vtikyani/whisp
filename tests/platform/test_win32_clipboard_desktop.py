"""Clipboard round-trips on the real clipboard. The user's clipboard is saved first and restored afterwards."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

import pytest

from whisp.platform.win32 import clipboard

pytestmark = [pytest.mark.desktop, pytest.mark.usefixtures("clipboard_guard")]

_IsClipboardFormatAvailable = ctypes.WinDLL("user32").IsClipboardFormatAvailable
_IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
_IsClipboardFormatAvailable.restype = wintypes.BOOL


def _available(fmt: int) -> bool:
    return bool(_IsClipboardFormatAvailable(fmt))


def test_text_round_trip_with_emoji_and_line_breaks() -> None:
    text = "Whisp ✓ — naïve café 😀👍\nsecond line\r\nthird"
    before = clipboard.get_sequence_number()
    clipboard.set_text(text)
    assert clipboard.get_sequence_number() != before
    assert clipboard.get_text() == text
    assert clipboard.has_text()


def test_dictations_are_kept_out_of_clipboard_history() -> None:
    clipboard.set_text("private dictation")
    for name in (clipboard.EXCLUDE_FROM_MONITORS, clipboard.CAN_INCLUDE_IN_HISTORY, clipboard.CAN_UPLOAD_TO_CLOUD):
        assert _available(clipboard.registered_format(name)), name
    clipboard.set_text("ordinary copy", exclude_from_history=False)
    assert not _available(clipboard.registered_format(clipboard.EXCLUDE_FROM_MONITORS))


def test_snapshot_and_restore_every_memory_format() -> None:
    payload_format = clipboard.registered_format("Whisp Test Payload")
    payload = bytes(range(256)) * 64
    with clipboard._session(write=True):
        clipboard._EmptyClipboard()
        clipboard._set_data(clipboard.CF_UNICODETEXT, "original text\x00".encode("utf-16-le"))
        clipboard._set_data(payload_format, payload)
    snap = clipboard.snapshot()
    assert clipboard.CF_UNICODETEXT in snap.format_ids
    assert payload_format in snap.format_ids

    clipboard.set_text("something else")
    clipboard.restore(snap)

    assert clipboard.get_text() == "original text"
    restored = dict(clipboard.snapshot().formats)
    assert restored[payload_format][: len(payload)] == payload
    # A restore is marked private too, so clipboard managers don't log the user's content twice.
    assert _available(clipboard.registered_format(clipboard.EXCLUDE_FROM_MONITORS))


def test_restoring_an_empty_snapshot_empties_the_clipboard() -> None:
    clipboard.set_text("temporary")
    clipboard.restore(clipboard.ClipboardSnapshot((), 0))
    assert clipboard.get_text() is None
    assert clipboard.snapshot().is_empty


def test_change_detection() -> None:
    clipboard.set_text("one")
    seq = clipboard.get_sequence_number()
    assert not clipboard.wait_for_change(seq, 0.05)
    clipboard.set_text("two")
    assert clipboard.wait_for_change(seq, 0.05)
