"""End to end: real speech through the real pipeline, model, text rules, delivery and history.

Only the microphone (it plays synthesised speech at real-time pace) and the final keystrokes (an
inserter that records the text instead of typing it) are stand-ins, so this runs on a locked or
headless desktop. Needs Parakeet TDT 0.6B v2 in the shared model cache (``-m models``).
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from helpers.speech import concat, silence, synthesize
from whisp.app.delivery import DeliveryService
from whisp.core.asr.manager import ModelManager
from whisp.core.asr.service import AsrService
from whisp.core.session import SessionPipeline
from whisp.core.types import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    DeliveryResult,
    EngineState,
    InsertionResult,
    Outcome,
    RecordingMode,
    SessionInfo,
    SessionKind,
    Transcript,
)
from whisp.storage.history import HistoryStore
from whisp.storage.paths import app_paths
from whisp.storage.settings import SettingsStore

pytestmark = pytest.mark.models

MODEL_ID = "parakeet-tdt-0.6b-v2"
LOAD_TIMEOUT_MS = 120_000
RESULT_TIMEOUT_MS = 60_000
PLAYBACK_SPEED = 3.0
"""Frames arrive this many times faster than real time — still paced, like a real device."""


class PlayedMicrophone:
    """A microphone that plays prepared audio, paced like real capture, then sends silence until stopped."""

    def __init__(self, audio: np.ndarray, on_level: Callable[[float], None] | None = None) -> None:
        pad = (-len(audio)) % FRAME_SAMPLES
        audio = np.concatenate([audio.astype(np.float32), np.zeros(pad, np.float32)])
        self._frames = list(audio.reshape(-1, FRAME_SAMPLES))
        self._on_level = on_level
        self._index = 0
        self._started_at = 0.0
        self._stopped = threading.Event()
        self.played_all = threading.Event()

    @property
    def device_name(self) -> str:
        return "Played speech"

    @property
    def error(self) -> str | None:
        return None

    def start(self) -> None:
        self._started_at = time.monotonic()

    def read(self, timeout: float) -> np.ndarray | None:
        if self._stopped.is_set():
            return None
        due = self._started_at + self._index * (FRAME_SAMPLES / SAMPLE_RATE) / PLAYBACK_SPEED
        wait = due - time.monotonic()
        if wait > timeout:
            time.sleep(timeout)
            return None
        if wait > 0:
            time.sleep(wait)
        if self._index < len(self._frames):
            frame = self._frames[self._index]
        else:
            frame = np.zeros(FRAME_SAMPLES, np.float32)  # a real microphone keeps sending room silence
            self.played_all.set()
        self._index += 1
        if self._on_level is not None:
            self._on_level(0.3)
        return frame

    def stop(self) -> None:
        self._stopped.set()


class RecordingInserter:
    """Stands in for TextInserter: records what would be typed."""

    def __init__(self) -> None:
        self.inserted: list[str] = []

    def default_method_for(self, process_name: str) -> str | None:
        return None

    def insert(self, text: str, **_options: object) -> InsertionResult:
        self.inserted.append(text)
        return InsertionResult(ok=True, method="paste")

    def copy_to_clipboard(self, text: str) -> bool:
        return True


class NoPolish:
    def is_ready(self) -> bool:
        return False


class Rig:
    def __init__(self, settings: SettingsStore, models_dir: Path) -> None:
        self.settings = settings
        self.models = ModelManager(models_dir)
        self.asr = AsrService(settings, self.models)
        self.microphones: list[PlayedMicrophone] = []
        self.next_audio: np.ndarray | None = None
        self.pipeline = SessionPipeline(settings, self.asr, stream_factory=self._open_microphone)
        self.history = HistoryStore(app_paths().history_db, settings)
        self.inserter = RecordingInserter()
        self.delivery = DeliveryService(settings, self.history, NoPolish(), self.inserter, get_target=lambda: None)
        self.transcripts: list[Transcript] = []
        self.results: list[DeliveryResult] = []
        self.problems: dict[str, str] = {}
        self.pipeline.finished.connect(self.transcripts.append)
        self.pipeline.no_speech.connect(lambda sid: self.problems.setdefault(sid, "no speech detected"))
        self.pipeline.failed.connect(lambda sid, message: self.problems.setdefault(sid, message))
        self.delivery.delivered.connect(self.results.append)

    def _open_microphone(self, device: str | None, on_level: Callable[[float], None] | None = None) -> PlayedMicrophone:
        assert self.next_audio is not None, "the test must queue audio before starting a session"
        microphone = PlayedMicrophone(self.next_audio, on_level)
        self.next_audio = None
        self.microphones.append(microphone)
        return microphone

    def dictate(self, qtbot: QtBot, audio: np.ndarray, *, stop_after_s: float | None = None) -> DeliveryResult:
        """Record ``audio`` as one session, stop (after it all played, or ``stop_after_s`` of playback), deliver."""
        self.next_audio = audio
        session_id = uuid.uuid4().hex
        started = time.time()
        opened = len(self.microphones)
        self.pipeline.start(session_id)
        qtbot.waitUntil(lambda: len(self.microphones) > opened and self.microphones[-1]._started_at > 0, timeout=10_000)
        microphone = self.microphones[-1]
        if stop_after_s is None:
            qtbot.waitUntil(microphone.played_all.is_set, timeout=RESULT_TIMEOUT_MS)
        else:
            qtbot.wait(int(stop_after_s * 1000 / PLAYBACK_SPEED))
        self.pipeline.stop(session_id)
        stopped_monotonic = time.monotonic()

        def reported() -> bool:
            return session_id in self.problems or any(t.session_id == session_id for t in self.transcripts)

        qtbot.waitUntil(reported, timeout=RESULT_TIMEOUT_MS)
        assert session_id not in self.problems, f"the pipeline reported: {self.problems.get(session_id)}"
        transcript = next(t for t in self.transcripts if t.session_id == session_id)
        info = SessionInfo(
            session_id=session_id,
            kind=SessionKind.DICTATION,
            mode=RecordingMode.HOLD,
            target=None,
            started_at=started,
            stopped_at=time.time(),
            stopped_monotonic=stopped_monotonic,
        )
        count = len(self.results)
        self.delivery.submit(info, transcript)
        qtbot.waitUntil(lambda: len(self.results) > count, timeout=RESULT_TIMEOUT_MS)
        return self.results[-1]

    def close(self) -> None:
        self.pipeline.shutdown()
        self.delivery.shutdown()
        self.asr.shutdown()
        self.history.close()


@pytest.fixture
def rig(qtbot: QtBot, settings_store: SettingsStore, shared_models_dir: Path) -> Iterator[Rig]:
    manager = ModelManager(shared_models_dir)
    if not (manager.is_installed(MODEL_ID, "gpu") or manager.is_installed(MODEL_ID, "cpu")):
        pytest.skip("Parakeet TDT 0.6B v2 isn't in the shared model cache (run the -m models ASR tests first)")
    settings_store.update({"transcription": {"model_id": MODEL_ID, "live_preview": True}})
    rig = Rig(settings_store, shared_models_dir)
    rig.asr.start()
    qtbot.waitUntil(lambda: rig.asr.status.state is EngineState.READY, timeout=LOAD_TIMEOUT_MS)
    yield rig
    rig.close()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / max(1, len(ref))


def test_a_dictation_goes_from_speech_to_inserted_text_and_history(qtbot: QtBot, rig: Rig) -> None:
    sentence = "Please send the quarterly report to the whole team before Friday afternoon."
    audio = concat(silence(0.4), synthesize(sentence), silence(0.6))
    result = rig.dictate(qtbot, audio)

    assert result.outcome is Outcome.INSERTED
    [typed] = rig.inserter.inserted
    assert word_error_rate(sentence, typed) <= 0.1
    assert typed[0].isupper() and typed.rstrip()[-1] in ".!?"  # the model punctuates
    assert result.latency_ms < 5_000
    [entry] = rig.history.list()
    assert entry.final_text == typed and entry.word_count == len(words(typed))


def test_filler_words_are_removed(qtbot: QtBot, rig: Rig) -> None:
    speech = synthesize("Um, so I think we should, uh, move the meeting to Thursday.")
    rig.dictate(qtbot, concat(silence(0.3), speech, silence(0.6)))
    [typed] = rig.inserter.inserted
    assert not {"um", "uh"} & set(words(typed))
    assert word_error_rate("So I think we should move the meeting to Thursday.", typed) <= 0.15


def test_stopping_mid_sentence_still_keeps_the_last_words(qtbot: QtBot, rig: Rig) -> None:
    sentence = "The new design looks great and I would like to ship it next week."
    speech = synthesize(sentence)
    spoken_s = len(speech) / SAMPLE_RATE
    # Stop while the very last words are still being "spoken": nothing captured may be lost.
    result = rig.dictate(qtbot, concat(silence(0.3), speech), stop_after_s=0.3 + spoken_s - 0.15)
    assert result.outcome is Outcome.INSERTED
    [typed] = rig.inserter.inserted
    assert "week" in words(typed) or "next" in words(typed)
    assert word_error_rate(sentence, typed) <= 0.2
