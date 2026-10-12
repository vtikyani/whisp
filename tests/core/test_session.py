"""SessionPipeline end to end: real speech audio and the real Silero VAD.

The microphone is ``FakeStream`` (plays prepared audio frame by frame) and the
recogniser is ``FakeAsr`` (same methods as ``AsrService``, answers on its own
``whisp-asr`` thread). By default a segment is "recognised" as ``w<index>``, so
the transcript text shows exactly which segments were joined, in what order.
"""

from __future__ import annotations

import itertools
import math
import queue
import threading
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from helpers.speech import concat, noise, silence, synthesize
from whisp.core.audio import AudioError
from whisp.core.session import MSG_MIC_SILENT, SessionPipeline
from whisp.core.types import FRAME_SAMPLES, SAMPLE_RATE, AsrResult, Segment, Transcript
from whisp.storage.settings import SettingsStore

FAKE_INFERENCE_MS = 7
WAIT_MS = 10_000

Job = tuple[Segment | None, Callable[..., None]]


class RecognitionFailed(Exception):
    """Raised by a test's transcribe function to make FakeAsr report an error result."""


class FakeAsr:
    """Stands in for AsrService. With ``hold=True`` segment jobs wait for ``release()``."""

    def __init__(
        self,
        transcribe: Callable[[Segment], str] | None = None,
        *,
        hold: bool = False,
        preview_text: str = "",
    ) -> None:
        self.transcribe = transcribe or (lambda segment: f"w{segment.index}")
        self.preview_text = preview_text
        self.segments: list[Segment] = []
        self.previews: list[tuple[str, int]] = []
        self.discarded: list[str] = []
        self._hold = hold
        self._held: list[Job] = []
        self._lock = threading.Lock()
        self._jobs: queue.SimpleQueue[Job | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._work, name="whisp-asr", daemon=True)
        self._thread.start()

    def submit(self, segment: Segment, callback: Callable[[AsrResult], None]) -> None:
        with self._lock:
            self.segments.append(segment)
            if self._hold:
                self._held.append((segment, callback))
                return
        self._jobs.put((segment, callback))

    def submit_preview(self, session_id: str, samples: np.ndarray, callback: Callable[[str], None]) -> None:
        with self._lock:
            self.previews.append((session_id, int(samples.size)))
        self._jobs.put((None, callback))

    def discard(self, session_id: str) -> None:
        with self._lock:
            self.discarded.append(session_id)
            self._held = [job for job in self._held if job[0] is not None and job[0].session_id != session_id]

    def release(self, order: Callable[[list[Job]], list[Job]] | None = None) -> None:
        with self._lock:
            held, self._held, self._hold = self._held, [], False
        for job in order(held) if order else held:
            self._jobs.put(job)

    def segments_for(self, session_id: str) -> list[Segment]:
        with self._lock:
            return [s for s in self.segments if s.session_id == session_id]

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(2.0)

    def _work(self) -> None:
        while (job := self._jobs.get()) is not None:
            segment, callback = job
            if segment is None:
                callback(self.preview_text)
                continue
            audio_ms = segment.samples.size * 1000 // SAMPLE_RATE
            try:
                text, error = self.transcribe(segment), None
            except RecognitionFailed as exc:
                text, error = "", str(exc)
            language = "en" if error is None else None
            callback(AsrResult(segment.session_id, segment.index, text, audio_ms, FAKE_INFERENCE_MS, language, error))


class FakeStream:
    """A microphone that plays prepared audio, one frame per ``read()``.

    ``end_at`` limits how many frames are ever captured. ``hooks`` run on the
    pipeline thread right after that many frames have been read (tests press
    stop or cancel there). ``fail_at`` makes the device vanish after that many
    frames. Once all captured frames are read it behaves like a quiet device:
    ``read()`` waits for the timeout and returns None.
    """

    LOST = "The microphone was disconnected."

    def __init__(
        self,
        audio: np.ndarray,
        *,
        end_at: int | None = None,
        fail_at: int | None = None,
        hooks: dict[int, Callable[[], None]] | None = None,
        start_error: Exception | None = None,
    ) -> None:
        padded = np.concatenate([audio, np.zeros((-audio.size) % FRAME_SAMPLES, dtype=np.float32)])
        self.frames = list(padded.astype(np.float32).reshape(-1, FRAME_SAMPLES))
        self._limit = len(self.frames) if end_at is None else end_at
        if fail_at is not None:
            self._limit = min(self._limit, fail_at)
        self._fails = fail_at is not None
        self._hooks = dict(hooks or {})
        self._start_error = start_error
        self._position = 0
        self._error: str | None = None
        self.on_level: Callable[[float], None] | None = None
        self.stopped = threading.Event()

    @property
    def device_name(self) -> str:
        return "Test microphone"

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def captured(self) -> np.ndarray:
        return np.concatenate(self.frames[: self._limit])

    @property
    def exhausted(self) -> bool:
        return self._position >= self._limit

    def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error

    def read(self, timeout: float) -> np.ndarray | None:
        if self._position < self._limit:
            frame = self.frames[self._position]
            self._position += 1
            if self.on_level is not None:
                self.on_level(0.5)
            hook = self._hooks.pop(self._position, None)
            if hook is not None:
                hook()
            return frame
        if self._fails:
            self._error = self.LOST
            return None
        if timeout > 0 and not self.stopped.is_set():
            self.stopped.wait(timeout)
        return None

    def stop(self) -> None:
        self.stopped.set()


class StreamSource:
    """The pipeline's ``stream_factory``: hands out prepared streams in order."""

    def __init__(self) -> None:
        self._streams: list[FakeStream] = []
        self.devices: list[str | None] = []

    def add(self, stream: FakeStream) -> FakeStream:
        self._streams.append(stream)
        return stream

    def __call__(self, device: str | None, on_level: Callable[[float], None] | None) -> FakeStream:
        self.devices.append(device)
        stream = self._streams.pop(0)
        stream.on_level = on_level
        return stream


def frame_count(audio: np.ndarray) -> int:
    return math.ceil(audio.size / FRAME_SAMPLES)


def with_room_tone(audio: np.ndarray, seed: int = 0) -> np.ndarray:
    """Add a -60 dBFS noise floor, like any real microphone (it also makes every frame unique)."""
    return (audio + noise(audio.size / SAMPLE_RATE, -60.0, seed)).astype(np.float32)


def loud_frames(audio: np.ndarray, threshold_db: float = -35.0) -> list[int]:
    frames = audio[: audio.size // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return np.flatnonzero(db > threshold_db).tolist()


def frame_ranges(segments: list[Segment], captured: np.ndarray) -> list[range]:
    """Where each segment's audio sits in the captured audio, in frames."""
    frames = captured.reshape(-1, FRAME_SAMPLES)
    ranges = []
    for segment in segments:
        assert segment.samples.size % FRAME_SAMPLES == 0
        count = segment.samples.size // FRAME_SAMPLES
        starts = [
            i
            for i in range(len(frames) - count + 1)
            if np.array_equal(frames[i], segment.samples[:FRAME_SAMPLES])
            and np.array_equal(frames[i : i + count].reshape(-1), segment.samples)
        ]
        assert len(starts) == 1, "segment audio must be an exact, unique slice of what was captured"
        ranges.append(range(starts[0], starts[0] + count))
    return ranges


@pytest.fixture(scope="module")
def phrases() -> list[np.ndarray]:
    return [
        synthesize("Seven green turtles walked slowly across the long wooden bridge at noon."),
        synthesize("Please send the updated budget to the finance team before Friday."),
        synthesize("The library closes early on public holidays."),
    ]


@pytest.fixture(scope="module")
def long_speech() -> np.ndarray:
    return synthesize(
        "When you hold the hotkey and speak, the app listens, waits for a natural pause, "
        "and quietly sends each finished thought to the recogniser while you keep talking."
    )


@pytest.fixture
def make_pipeline(
    qtbot: QtBot, settings_store: SettingsStore
) -> Iterator[Callable[[FakeAsr], tuple[SessionPipeline, StreamSource]]]:
    made: list[tuple[SessionPipeline, FakeAsr]] = []

    def make(asr: FakeAsr) -> tuple[SessionPipeline, StreamSource]:
        source = StreamSource()
        pipeline = SessionPipeline(settings_store, asr, stream_factory=source)
        made.append((pipeline, asr))
        return pipeline, source

    yield make
    for pipeline, asr in made:
        pipeline.shutdown()
        asr.close()


def test_stop_mid_sentence_still_transcribes_the_tail(qtbot: QtBot, make_pipeline, phrases) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    audio = with_room_tone(concat(silence(0.6), phrases[0]))
    loud = loud_frames(audio)
    stop_at = loud[int(len(loud) * 0.7)]
    # Stop is pressed while the last five captured frames are still waiting to be read.
    stream = source.add(FakeStream(audio, end_at=stop_at + 1, hooks={stop_at - 4: lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")
    transcript: Transcript = blocker.args[0]
    segments = asr.segments_for("s1")
    captured = stream.captured
    assert segments[-1].is_final
    assert not any(s.is_final for s in segments[:-1])
    np.testing.assert_array_equal(segments[-1].samples[-FRAME_SAMPLES:], captured[-FRAME_SAMPLES:])
    covered = [i for r in frame_ranges(segments, captured) for i in r]
    assert covered == sorted(set(covered))
    assert set(range(loud[0], stop_at + 1)) <= set(covered)
    assert transcript.session_id == "s1"
    assert transcript.text == " ".join(f"w{i}" for i in range(len(segments)))
    assert transcript.audio_ms == captured.size * 1000 // SAMPLE_RATE
    assert 0 < transcript.speech_ms <= transcript.audio_ms
    assert stream.stopped.is_set()


def test_cancel_discards_the_session(qtbot: QtBot, make_pipeline, phrases) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    reported: list[str] = []
    pipeline.finished.connect(lambda transcript: reported.append(transcript.session_id))
    pipeline.no_speech.connect(lambda sid: reported.append(sid))
    pipeline.failed.connect(lambda sid, message: reported.append(sid))
    first = with_room_tone(concat(silence(0.3), phrases[0], silence(1.5), phrases[1]), seed=1)
    stream = source.add(FakeStream(first, hooks={frame_count(first) - 20: lambda: pipeline.cancel("s1")}))
    pipeline.start("s1")
    qtbot.waitUntil(lambda: "s1" in asr.discarded, timeout=WAIT_MS)
    assert stream.stopped.is_set()
    assert asr.segments_for("s1")  # the first phrase had already gone to recognition
    second = with_room_tone(concat(silence(0.3), phrases[2]), seed=2)
    source.add(FakeStream(second, hooks={frame_count(second): lambda: pipeline.stop("s2")}))
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        pipeline.start("s2")
    assert blocker.args[0].session_id == "s2"
    assert reported == ["s2"]


def test_room_tone_alone_is_no_speech(qtbot: QtBot, make_pipeline) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    audio = noise(2.0, -50.0, seed=4)
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.no_speech, timeout=WAIT_MS, check_params_cb=lambda sid: sid == "s1"):
        pipeline.start("s1")
    assert asr.segments == []


def one_step_of_16_bit_audio(seconds: float) -> np.ndarray:
    """What a switched-off microphone sends through MME: zeros with ±1 steps of 16-bit audio."""
    rng = np.random.default_rng(3)
    return (rng.integers(-1, 2, round(seconds * SAMPLE_RATE)) / 32768).astype(np.float32)


@pytest.mark.parametrize("audio", [silence(1.5), one_step_of_16_bit_audio(1.5)], ids=["wasapi-zeros", "mme-steps"])
def test_digital_silence_points_at_a_muted_or_switched_off_microphone(
    qtbot: QtBot, make_pipeline, audio: np.ndarray
) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.failed, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")
    assert blocker.args == ["s1", MSG_MIC_SILENT]


def test_a_very_quiet_working_microphone_is_just_no_speech(qtbot: QtBot, make_pipeline) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    audio = noise(1.5, -80.0, seed=6)
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.no_speech, timeout=WAIT_MS, check_params_cb=lambda sid: sid == "s1"):
        pipeline.start("s1")


def test_segments_keep_their_order_when_results_arrive_out_of_order(qtbot: QtBot, make_pipeline, phrases) -> None:
    asr = FakeAsr(hold=True)
    pipeline, source = make_pipeline(asr)
    gap = silence(1.4)
    audio = with_room_tone(concat(silence(0.3), phrases[0], gap, phrases[1], gap, phrases[2]))
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    pipeline.start("s1")
    qtbot.waitUntil(lambda: len(asr.segments_for("s1")) == 3, timeout=WAIT_MS)
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        asr.release(order=lambda jobs: list(reversed(jobs)))
    transcript: Transcript = blocker.args[0]
    assert transcript.text == "w0 w1 w2"
    assert [r.index for r in transcript.segments] == [0, 1, 2]
    assert transcript.inference_ms == 3 * FAKE_INFERENCE_MS
    assert transcript.language == "en"


def test_chinese_segments_join_without_spaces(qtbot: QtBot, make_pipeline, phrases) -> None:
    sentences = ["你好。", "我是小明。", "今天天气很好。"]
    pipeline, source = make_pipeline(FakeAsr(transcribe=lambda segment: sentences[segment.index]))
    gap = silence(1.4)
    audio = with_room_tone(concat(silence(0.3), phrases[0], gap, phrases[1], gap, phrases[2]))
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")
    assert blocker.args[0].text == "你好。我是小明。今天天气很好。"


def test_overlapping_sessions_both_finish_in_start_order(qtbot: QtBot, make_pipeline, phrases) -> None:
    asr = FakeAsr(transcribe=lambda segment: f"{segment.session_id}:{segment.index}", hold=True)
    pipeline, source = make_pipeline(asr)
    finished: list[Transcript] = []
    pipeline.finished.connect(finished.append)
    first = with_room_tone(concat(silence(0.3), phrases[0]), seed=1)
    second = with_room_tone(concat(silence(0.3), phrases[1], silence(1.4), phrases[2]), seed=2)
    source.add(FakeStream(first, hooks={frame_count(first): lambda: pipeline.stop("A")}))
    source.add(FakeStream(second, hooks={frame_count(second): lambda: pipeline.stop("B")}))
    pipeline.start("A")
    qtbot.waitUntil(lambda: len(asr.segments_for("A")) == 1, timeout=WAIT_MS)
    pipeline.start("B")  # A is still waiting for recognition while B records
    qtbot.waitUntil(lambda: len(asr.segments_for("B")) == 2, timeout=WAIT_MS)
    # B's results come back before A's; A must still be reported first.
    asr.release(order=lambda jobs: sorted(jobs, key=lambda job: job[0] is not None and job[0].session_id != "B"))
    qtbot.waitUntil(lambda: len(finished) == 2, timeout=WAIT_MS)
    assert [(t.session_id, t.text) for t in finished] == [("A", "A:0"), ("B", "B:0 B:1")]


def test_starting_while_recording_finishes_the_earlier_session(qtbot: QtBot, make_pipeline, phrases) -> None:
    asr = FakeAsr(transcribe=lambda segment: segment.session_id)
    pipeline, source = make_pipeline(asr)
    finished: list[Transcript] = []
    pipeline.finished.connect(finished.append)
    first = source.add(FakeStream(with_room_tone(concat(silence(0.3), phrases[0]), seed=1)))
    second_audio = with_room_tone(concat(silence(0.3), phrases[1]), seed=2)
    source.add(FakeStream(second_audio, hooks={frame_count(second_audio): lambda: pipeline.stop("B")}))
    pipeline.start("A")
    qtbot.waitUntil(lambda: first.exhausted, timeout=WAIT_MS)
    pipeline.start("B")
    qtbot.waitUntil(lambda: len(finished) == 2, timeout=WAIT_MS)
    assert [(t.session_id, t.text) for t in finished] == [("A", "A"), ("B", "B")]
    assert first.stopped.is_set()


def test_speech_longer_than_a_segment_is_cut_without_losing_audio(qtbot: QtBot, make_pipeline, long_speech) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    loud = loud_frames(long_speech, -50.0)
    sentence = long_speech[loud[0] * FRAME_SAMPLES : (loud[-1] + 1) * FRAME_SAMPLES]
    pause = silence(0.25)  # far shorter than the 900 ms that would end a segment
    audio = with_room_tone(concat(silence(0.3), sentence, pause, sentence, pause, sentence))
    assert audio.size / SAMPLE_RATE > 28.0
    stream = source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")
    segments = asr.segments_for("s1")
    assert len(segments) >= 2
    assert all(s.samples.size <= 28.0 * SAMPLE_RATE for s in segments)
    ranges = frame_ranges(segments, stream.captured)
    for before, after in itertools.pairwise(ranges):
        assert after.start == before.stop  # cut cleanly: nothing dropped, nothing repeated
    speech = loud_frames(audio)
    assert ranges[0].start <= speech[0] and ranges[-1].stop > speech[-1]
    assert blocker.args[0].text == " ".join(f"w{i}" for i in range(len(segments)))


def test_live_preview_arrives_during_long_speech(qtbot: QtBot, make_pipeline, long_speech) -> None:
    asr = FakeAsr(preview_text="so far so good")
    pipeline, source = make_pipeline(asr)
    source.add(FakeStream(with_room_tone(concat(silence(0.3), long_speech))))
    with qtbot.waitSignal(
        pipeline.preview,
        timeout=WAIT_MS,
        check_params_cb=lambda sid, text: sid == "s1" and text.endswith("so far so good"),
    ):
        pipeline.start("s1")
    assert asr.previews
    assert all(sid == "s1" and size > 0 for sid, size in asr.previews)
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        pipeline.stop("s1")
    assert "so far" not in blocker.args[0].text  # preview text never reaches the transcript


def test_no_previews_when_live_preview_is_off(
    qtbot: QtBot, make_pipeline, settings_store: SettingsStore, long_speech
) -> None:
    settings_store.update({"transcription": {"live_preview": False}})
    asr = FakeAsr(preview_text="should not appear")
    pipeline, source = make_pipeline(asr)
    previews: list[str] = []
    pipeline.preview.connect(lambda sid, text: previews.append(text))
    audio = with_room_tone(concat(silence(0.3), long_speech))
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS):
        pipeline.start("s1")
    assert asr.previews == []
    assert previews == []


def test_microphone_lost_mid_sentence_keeps_what_was_captured(qtbot: QtBot, make_pipeline, phrases) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    audio = with_room_tone(concat(silence(0.4), phrases[0]))
    loud = loud_frames(audio)
    stream = source.add(FakeStream(audio, fail_at=loud[len(loud) // 2]))
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")  # nobody presses stop: the pipeline ends the session itself
    segments = asr.segments_for("s1")
    np.testing.assert_array_equal(segments[-1].samples[-FRAME_SAMPLES:], stream.captured[-FRAME_SAMPLES:])
    assert segments[-1].is_final
    assert blocker.args[0].text == " ".join(f"w{i}" for i in range(len(segments)))
    assert stream.stopped.is_set()


def test_microphone_lost_before_any_speech_reports_failed(qtbot: QtBot, make_pipeline) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    source.add(FakeStream(noise(1.0, -50.0, seed=8), fail_at=12))
    with qtbot.waitSignal(pipeline.failed, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")
    assert blocker.args == ["s1", FakeStream.LOST]
    pipeline.stop("s1")  # the controller's late stop is harmless


def test_microphone_that_will_not_start_reports_failed(qtbot: QtBot, make_pipeline, phrases) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    source.add(FakeStream(noise(0.5, -50.0), start_error=AudioError("No microphone found.")))
    with qtbot.waitSignal(pipeline.failed, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")
    assert blocker.args == ["s1", "No microphone found."]
    audio = with_room_tone(concat(silence(0.3), phrases[2]))
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s2")}))
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        pipeline.start("s2")
    assert blocker.args[0].session_id == "s2"
    assert source.devices == [None, None]  # the configured device: the Windows default


def test_recognition_error_without_any_text_reports_failed(qtbot: QtBot, make_pipeline, phrases) -> None:
    def transcribe(segment: Segment) -> str:
        raise RecognitionFailed("The speech model isn't installed.")

    asr = FakeAsr(transcribe)
    pipeline, source = make_pipeline(asr)
    audio = with_room_tone(concat(silence(0.3), phrases[1]))
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.failed, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")
    assert blocker.args == ["s1", "The speech model isn't installed."]


def test_failed_segments_do_not_lose_the_recognised_ones(qtbot: QtBot, make_pipeline, phrases) -> None:
    def transcribe(segment: Segment) -> str:
        if segment.index == 0:
            raise RecognitionFailed("Recognition failed.")
        return f"w{segment.index}"

    asr = FakeAsr(transcribe)
    pipeline, source = make_pipeline(asr)
    audio = with_room_tone(concat(silence(0.3), phrases[0], silence(1.4), phrases[1]))
    source.add(FakeStream(audio, hooks={frame_count(audio): lambda: pipeline.stop("s1")}))
    with qtbot.waitSignal(pipeline.finished, timeout=WAIT_MS) as blocker:
        pipeline.start("s1")
    transcript: Transcript = blocker.args[0]
    assert transcript.text == "w1"
    assert [r.error for r in transcript.segments] == ["Recognition failed.", None]


def test_level_is_reported_while_recording(qtbot: QtBot, make_pipeline, phrases) -> None:
    asr = FakeAsr()
    pipeline, source = make_pipeline(asr)
    source.add(FakeStream(with_room_tone(concat(silence(0.3), phrases[2]))))
    with qtbot.waitSignal(pipeline.level, timeout=WAIT_MS, check_params_cb=lambda level: level == 0.5):
        pipeline.start("s1")
    pipeline.cancel("s1")


def test_shutdown_releases_the_microphone(qtbot: QtBot, settings_store: SettingsStore, phrases) -> None:
    asr = FakeAsr()
    source = StreamSource()
    stream = source.add(FakeStream(with_room_tone(concat(silence(0.3), phrases[2]))))
    before = set(threading.enumerate())
    pipeline = SessionPipeline(settings_store, asr, stream_factory=source)
    try:
        pipeline.start("s1")
        qtbot.waitUntil(lambda: stream.exhausted, timeout=WAIT_MS)
        pipeline.shutdown()
        assert stream.stopped.is_set()
        assert "s1" in asr.discarded
        pipeline.start("s2")  # ignored after shutdown
    finally:
        pipeline.shutdown()
        asr.close()
    assert not [t for t in threading.enumerate() if t.name == "whisp-pipeline" and t not in before]


@pytest.mark.desktop
def test_real_microphone_session_runs_end_to_end(qtbot: QtBot, settings_store: SettingsStore) -> None:
    asr = FakeAsr()
    pipeline = SessionPipeline(settings_store, asr)  # the real microphone
    outcomes: list[str] = []
    levels: list[float] = []
    pipeline.finished.connect(lambda transcript: outcomes.append("finished"))
    pipeline.no_speech.connect(lambda sid: outcomes.append("no_speech"))
    pipeline.failed.connect(lambda sid, message: outcomes.append(f"failed: {message}"))
    pipeline.level.connect(levels.append)
    try:
        pipeline.start("desktop")
        qtbot.wait(1500)  # record for a second and a half
        pipeline.stop("desktop")
        qtbot.waitUntil(lambda: bool(outcomes), timeout=WAIT_MS)
        qtbot.wait(200)  # a second outcome for the same session would be a bug
    finally:
        pipeline.shutdown()
        asr.close()
    # What was said (if anything) decides between these; a switched-off or muted microphone gives the last one.
    assert len(outcomes) == 1
    assert outcomes[0] in ("finished", "no_speech", f"failed: {MSG_MIC_SILENT}")
    assert levels
    assert all(0.0 <= v <= 1.0 for v in levels)
