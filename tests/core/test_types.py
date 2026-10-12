"""Shared data types: what they let a stray log line print."""

from __future__ import annotations

import numpy as np

from whisp.core.text.polish import PolishResult
from whisp.core.types import (
    AsrResult,
    DeliveryResult,
    Outcome,
    RecordingMode,
    Segment,
    SessionInfo,
    SessionKind,
    TargetWindow,
    Transcript,
)

DICTATED = "the secret plan for monday"
TITLE = "Re: salary review - Inbox"


def test_dictated_text_and_window_titles_never_show_in_repr() -> None:
    target = TargetWindow(1, 2, "outlook.exe", "Outlook", TITLE)
    result = AsrResult("s1", 0, DICTATED, 1200, 40, "en")
    values = [
        target,
        result,
        Transcript("s1", DICTATED, 1200, 900, 40, segments=(result,), language="en"),
        Segment("s1", 0, np.zeros(16, dtype=np.float32), 900, True),
        SessionInfo("s1", SessionKind.COMMAND, RecordingMode.HOLD, target, 0.0, 1.0, 1.0, selection=DICTATED),
        DeliveryResult("s1", SessionKind.DICTATION, Outcome.INSERTED, DICTATED, 5, 300, target=target),
        PolishResult(DICTATED, True, None, 250),
    ]
    for value in values:
        text = repr(value)
        assert DICTATED not in text and TITLE not in text, text
    assert "outlook.exe" in repr(target)  # the rest stays useful for debugging
