"""Segmenter: deterministic tests driven by synthetic speech probabilities.

Every frame is filled with ``index + 1``, so a segment's samples reveal exactly
which frames it holds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pytest

from whisp.core.segmenter import CUT_LOOKBACK_MS, SegmentAudio, Segmenter, SegmenterConfig
from whisp.core.types import FRAME_MS, FRAME_SAMPLES

SPEECH = 0.9
QUIET = 0.05

CONFIG = SegmenterConfig()
PRE_ROLL = math.ceil(CONFIG.pre_roll_ms / FRAME_MS)
POST_ROLL = math.ceil(CONFIG.post_roll_ms / FRAME_MS)
COMMIT = math.ceil(CONFIG.commit_silence_ms / FRAME_MS)
MIN_SPEECH = math.ceil(CONFIG.min_speech_ms / FRAME_MS)
LOOKBACK = math.ceil(CUT_LOOKBACK_MS / FRAME_MS)


def frame(index: int) -> np.ndarray:
    return np.full(FRAME_SAMPLES, index + 1, dtype=np.float32)


def ids(samples: np.ndarray | SegmentAudio | None) -> list[int]:
    assert samples is not None
    if isinstance(samples, SegmentAudio):
        samples = samples.samples
    assert samples.size % FRAME_SAMPLES == 0
    return [int(v) - 1 for v in samples.reshape(-1, FRAME_SAMPLES)[:, 0]]


class Feeder:
    """Pushes numbered frames with the given probabilities and collects what comes out."""

    def __init__(self, segmenter: Segmenter | None = None) -> None:
        self.segmenter = segmenter or Segmenter()
        self.count = 0
        self.segments: list[SegmentAudio] = []

    def feed(self, probs: Iterable[float]) -> list[SegmentAudio]:
        out: list[SegmentAudio] = []
        for prob in probs:
            out += self.segmenter.push(frame(self.count), prob)
            self.count += 1
        self.segments += out
        return out

    def flush(self) -> list[SegmentAudio]:
        out = self.segmenter.flush()
        self.segments += out
        return out


def test_frame_conversions_match_the_documented_defaults() -> None:
    # 300 ms pre/post-roll, 900 ms commit, 200 ms minimum speech, 2 s look-back, in 32 ms frames.
    assert (PRE_ROLL, POST_ROLL, COMMIT, MIN_SPEECH, LOOKBACK) == (10, 10, 29, 7, 63)


def test_pre_roll_and_post_roll_surround_the_speech() -> None:
    f = Feeder()
    f.feed([QUIET] * 20 + [SPEECH] * 20 + [QUIET] * 40)
    assert len(f.segments) == 1
    segment = f.segments[0]
    assert ids(segment) == list(range(20 - PRE_ROLL, 40 + POST_ROLL))
    assert segment.speech_ms == 20 * FRAME_MS
    assert segment.samples.dtype == np.float32


def test_segment_is_committed_only_after_the_full_commit_silence() -> None:
    f = Feeder()
    f.feed([QUIET] * 5 + [SPEECH] * 20)
    assert f.feed([QUIET] * (COMMIT - 1)) == []
    assert len(f.feed([QUIET])) == 1


def test_hysteresis_keeps_speech_going_above_the_end_threshold() -> None:
    f = Feeder()
    between = (CONFIG.start_threshold + CONFIG.end_threshold) / 2
    f.feed([QUIET] * 5 + [SPEECH] * 10 + [between] * 40)
    assert f.segments == []  # 40 frames under start_threshold, but never below end_threshold
    f.feed([QUIET] * COMMIT)
    assert len(f.segments) == 1
    assert f.segments[0].speech_ms == 50 * FRAME_MS


def test_after_a_dip_only_the_start_threshold_resumes_speech() -> None:
    f = Feeder()
    between = (CONFIG.start_threshold + CONFIG.end_threshold) / 2
    f.feed([QUIET] * 5 + [SPEECH] * 10 + [0.2] * 3 + [between] * 30)
    assert len(f.segments) == 1
    assert f.segments[0].speech_ms == 10 * FRAME_MS


def test_short_blips_are_dropped() -> None:
    f = Feeder()
    f.feed([QUIET] * 10 + [SPEECH] * (MIN_SPEECH - 1) + [QUIET] * 40)
    assert f.segments == []
    assert f.flush() == []
    assert f.segmenter.total_speech_ms == 0


def test_speech_of_exactly_min_speech_is_kept() -> None:
    f = Feeder()
    f.feed([QUIET] * 10 + [SPEECH] * MIN_SPEECH + [QUIET] * 40)
    assert len(f.segments) == 1
    assert f.segments[0].speech_ms == MIN_SPEECH * FRAME_MS


def test_short_pauses_stay_inside_one_segment() -> None:
    f = Feeder()
    f.feed([QUIET] * 10 + [SPEECH] * 20 + [QUIET] * 15 + [SPEECH] * 20 + [QUIET] * 40)
    assert len(f.segments) == 1
    assert ids(f.segments[0]) == list(range(0, 65 + POST_ROLL))  # the 15-frame pause is kept
    assert f.segments[0].speech_ms == 40 * FRAME_MS


def test_long_pauses_split_segments_and_are_trimmed() -> None:
    f = Feeder()
    f.feed([QUIET] * 10 + [SPEECH] * 20 + [QUIET] * 100 + [SPEECH] * 20)
    f.flush()
    first, second = f.segments
    assert ids(first) == list(range(0, 30 + POST_ROLL))
    assert ids(second) == list(range(130 - PRE_ROLL, 150))  # 80 frames of silence never reach recognition


def test_max_length_cuts_at_the_quietest_recent_frame() -> None:
    max_frames = int(CONFIG.max_segment_s * 1000) // FRAME_MS
    probs = [SPEECH] * max_frames
    quietest = max_frames - 25  # inside the look-back window
    older_and_quieter = max_frames - LOOKBACK - 20  # outside it: must not be chosen
    probs[quietest] = 0.2
    probs[older_and_quieter] = 0.1
    f = Feeder()
    out = f.feed(probs)
    assert len(out) == 1
    assert ids(out[0]) == list(range(0, quietest + 1))
    assert out[0].speech_ms == (quietest + 1 - 2) * FRAME_MS
    tail = f.flush()
    assert ids(tail[0]) == list(range(quietest + 1, max_frames))  # nothing lost or duplicated at the cut


def test_cut_prefers_the_quieter_frame_when_probabilities_tie() -> None:
    config = SegmenterConfig(max_segment_s=3.2)  # 100 frames
    segmenter = Segmenter(config)
    loud, soft = 80, 90
    out: list[SegmentAudio] = []
    for i in range(100):
        prob = 0.6 if i in (loud, soft) else SPEECH
        amplitude = {loud: 0.5, soft: 0.001}.get(i, 0.1)
        out += segmenter.push(np.full(FRAME_SAMPLES, amplitude, dtype=np.float32), prob)
    assert len(out) == 1
    assert out[0].samples.size == (soft + 1) * FRAME_SAMPLES


def test_speech_after_a_cut_is_never_dropped_as_a_blip() -> None:
    f = Feeder(Segmenter(SegmenterConfig(max_segment_s=3.2)))  # 100 frames
    probs = [SPEECH] * 100
    probs[97] = 0.1
    f.feed(probs)  # cut after frame 97: frames 98-99 (64 ms of speech) continue in the next segment
    f.feed([QUIET] * 40)
    head, tail = f.segments
    assert ids(head) == list(range(0, 98))
    assert ids(tail) == list(range(98, 100 + POST_ROLL))
    assert tail.speech_ms == 2 * FRAME_MS


def test_pending_reports_speech_in_progress() -> None:
    f = Feeder()
    f.feed([QUIET] * 12)
    assert f.segmenter.pending() is None
    f.feed([SPEECH] * (MIN_SPEECH - 1))
    assert f.segmenter.pending() is None  # could still be a blip
    f.feed([SPEECH] * 6)
    assert ids(f.segmenter.pending()) == list(range(12 - PRE_ROLL, 12 + MIN_SPEECH + 5))
    f.feed([QUIET] * 5)
    assert ids(f.segmenter.pending()) == list(range(12 - PRE_ROLL, 12 + MIN_SPEECH + 10))
    f.feed([QUIET] * COMMIT)
    assert len(f.segments) == 1
    assert f.segmenter.pending() is None


def test_flush_returns_speech_still_in_progress() -> None:
    f = Feeder()
    f.feed([QUIET] * 12 + [SPEECH] * 30)
    tail = f.flush()
    assert ids(tail[0]) == list(range(12 - PRE_ROLL, 42))  # right up to the last frame
    assert f.flush() == []


def test_flush_trims_trailing_silence_to_the_post_roll() -> None:
    f = Feeder()
    f.feed([QUIET] * 12 + [SPEECH] * 20 + [QUIET] * (COMMIT - 4))
    tail = f.flush()
    assert ids(tail[0]) == list(range(12 - PRE_ROLL, 32 + POST_ROLL))


def test_flush_drops_a_trailing_blip() -> None:
    f = Feeder()
    f.feed([QUIET] * 12 + [SPEECH] * 3)
    assert f.flush() == []


def test_total_speech_counts_delivered_segments_only() -> None:
    f = Feeder()
    f.feed([QUIET] * 10 + [SPEECH] * 20 + [QUIET] * 40)  # 20 frames of speech
    f.feed([SPEECH] * 3 + [QUIET] * 40)  # blip
    f.feed([SPEECH] * 15)
    f.flush()
    assert len(f.segments) == 2
    assert f.segmenter.total_speech_ms == 35 * FRAME_MS == sum(s.speech_ms for s in f.segments)


def test_segments_never_duplicate_or_reorder_audio() -> None:
    rng = np.random.default_rng(7)
    probs: list[float] = []
    for _ in range(80):
        probs += [float(rng.uniform(0.55, 1.0)) for _ in range(int(rng.integers(1, 70)))]
        probs += [float(rng.uniform(0.0, 0.45)) for _ in range(int(rng.integers(1, 45)))]
    f = Feeder(Segmenter(SegmenterConfig(max_segment_s=4.0)))
    f.feed(probs)
    f.flush()
    covered = [i for segment in f.segments for i in ids(segment)]
    assert covered == sorted(set(covered))
    assert f.segmenter.total_speech_ms == sum(s.speech_ms for s in f.segments)
    assert all(s.samples.size <= 4000 * 16 for s in f.segments)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_threshold": 0.3, "end_threshold": 0.4},
        {"end_threshold": 0.0},
        {"start_threshold": 1.5},
        {"min_speech_ms": -1},
        {"pre_roll_ms": -1},
        {"commit_silence_ms": 10},
        {"max_segment_s": 0.01},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SegmenterConfig(**kwargs)


def test_frames_of_the_wrong_size_are_rejected() -> None:
    with pytest.raises(ValueError):
        Segmenter().push(np.zeros(FRAME_SAMPLES - 1, dtype=np.float32), SPEECH)


def test_nan_probability_counts_as_silence() -> None:
    f = Feeder()
    f.feed([QUIET] * 5 + [SPEECH] * 20 + [math.nan] * COMMIT)
    assert len(f.segments) == 1
    assert f.segments[0].speech_ms == 20 * FRAME_MS
