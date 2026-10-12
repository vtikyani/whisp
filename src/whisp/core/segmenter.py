"""Turns VAD frames into natural speech segments.

Pure logic: no threads, no Qt, no I/O. Push one 512-sample frame at a time
with the VAD's speech probability for it; get back finished segments.

How a segment is formed:

- Speech starts on a frame at or above ``start_threshold`` and continues while
  frames stay at or above ``end_threshold`` (hysteresis). Once a frame drops
  below ``end_threshold``, only a frame at or above ``start_threshold`` resumes
  speech, exactly like Silero's own iterator.
- About ``pre_roll_ms`` of audio before the onset is kept, so first syllables
  are never clipped.
- Short pauses stay inside the segment. A pause of ``commit_silence_ms`` ends
  it: the segment is trimmed to ``post_roll_ms`` after the last speech frame and
  returned. The rest of the pause is dropped, so long silences never reach
  speech recognition.
- A segment whose speech adds up to less than ``min_speech_ms`` (a click, a
  cough) is dropped.
- Speech that runs past ``max_segment_s`` without a long pause is cut at the
  quietest frame of the last ``CUT_LOOKBACK_MS`` (lowest speech probability,
  then lowest loudness), never at a fixed time. Nothing is lost or duplicated
  at the cut: the remainder simply continues as the next segment.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from whisp.core.types import FRAME_MS, FRAME_SAMPLES

CUT_LOOKBACK_MS = 2000
"""How far back an over-long segment looks for the quietest frame to cut at."""

_CUT_LOUDNESS_WEIGHT = 0.25
"""Loudness (0-1) counts this much against speech probability (0-1) when choosing a cut.

Probability leads because it dips between words; loudness breaks the tie when
probabilities are saturated during fast, continuous speech.
"""

_LOUDNESS_FLOOR_DB = -60.0


@dataclass(frozen=True)
class SegmenterConfig:
    start_threshold: float = 0.5
    """Speech starts at or above this probability..."""
    end_threshold: float = 0.35
    """...and continues until it falls below this."""
    min_speech_ms: int = 200
    """Shorter blips (clicks, coughs) are not speech."""
    commit_silence_ms: int = 900
    """A pause this long ends a segment while recording."""
    pre_roll_ms: int = 300
    """Audio kept before speech onset."""
    post_roll_ms: int = 300
    """Audio kept after speech ends."""
    max_segment_s: float = 28.0
    """Longer speech is cut at the quietest recent frame."""

    def __post_init__(self) -> None:
        if not 0.0 < self.end_threshold <= self.start_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 < end_threshold <= start_threshold <= 1")
        for name in ("min_speech_ms", "pre_roll_ms", "post_roll_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.commit_silence_ms < FRAME_MS:
            raise ValueError(f"commit_silence_ms must be at least one frame ({FRAME_MS} ms)")
        if self.max_segment_s * 1000 < 2 * FRAME_MS:
            raise ValueError(f"max_segment_s must cover at least two frames ({2 * FRAME_MS} ms)")


@dataclass(frozen=True)
class SegmentAudio:
    samples: np.ndarray
    """Mono float32 at 16 kHz: pre-roll, speech (with its short pauses), post-roll."""
    speech_ms: int
    """How much of it the VAD judged to be speech."""


@dataclass(slots=True)
class _Frame:
    samples: np.ndarray
    prob: float
    loudness: float
    voiced: bool = False


def _frames_for(ms: float) -> int:
    return math.ceil(ms / FRAME_MS)


def _loudness(samples: np.ndarray) -> float:
    """0-1 loudness: -60 dBFS or quieter is 0, full scale is 1."""
    power = float(np.dot(samples, samples)) / samples.size
    if power <= 1e-12:
        return 0.0
    db = 10.0 * math.log10(power)
    return min(1.0, max(0.0, (db - _LOUDNESS_FLOOR_DB) / -_LOUDNESS_FLOOR_DB))


class Segmenter:
    """Groups VAD frames into speech segments. One instance per session; not thread-safe."""

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self._config = config or SegmenterConfig()
        cfg = self._config
        self._pre_roll = _frames_for(cfg.pre_roll_ms)
        self._post_roll = _frames_for(cfg.post_roll_ms)
        self._commit_frames = _frames_for(cfg.commit_silence_ms)
        self._max_frames = max(2, int(cfg.max_segment_s * 1000) // FRAME_MS)
        self._lookback = min(_frames_for(CUT_LOOKBACK_MS), self._max_frames - 1)

        self._ring: deque[_Frame] = deque(maxlen=self._pre_roll)
        """Recent frames while no segment is open: the next segment's pre-roll."""
        self._frames: list[_Frame] = []
        """The open segment, pre-roll included."""
        self._open = False
        self._in_speech = False
        self._speech_frames = 0
        self._silence_run = 0
        """Frames since the last speech frame. Counts the physical pause, even across a cut."""
        self._last_voiced = -1
        """Index into ``_frames`` of the last speech frame, or -1."""
        self._continuation = False
        """The open segment continues speech that was cut at ``max_segment_s``."""
        self._total_speech_frames = 0

    @property
    def config(self) -> SegmenterConfig:
        return self._config

    @property
    def total_speech_ms(self) -> int:
        """Speech in every segment returned so far by ``push`` and ``flush`` (dropped blips excluded)."""
        return self._total_speech_frames * FRAME_MS

    def push(self, frame: np.ndarray, speech_prob: float) -> list[SegmentAudio]:
        """Add one frame and its speech probability. Returns any segments that are now complete."""
        samples = np.array(frame, dtype=np.float32).reshape(-1)
        if samples.size != FRAME_SAMPLES:
            raise ValueError(f"expected a {FRAME_SAMPLES}-sample frame, got {samples.size} samples")
        prob = float(speech_prob)
        if math.isnan(prob):
            prob = 0.0
        item = _Frame(samples, prob, _loudness(samples))

        if not self._open:
            if prob >= self._config.start_threshold:
                self._begin(item)
            else:
                self._ring.append(item)
            return []

        self._append(item)
        if self._silence_run >= self._commit_frames:
            return self._close()
        if len(self._frames) >= self._max_frames:
            return self._cut()
        return []

    def flush(self) -> list[SegmentAudio]:
        """On stop: return everything still held (trimmed to the post-roll) and reset."""
        out = self._close() if self._open else []
        self._ring.clear()
        self._silence_run = 0
        return out

    def pending(self) -> np.ndarray | None:
        """Audio of the speech in progress, for live preview; None when there is none.

        A possible blip (less than ``min_speech_ms`` of speech so far) is not
        reported yet.
        """
        if not self._open or not self._is_speech():
            return None
        return np.concatenate([f.samples for f in self._frames[: self._trimmed_end()]])

    # -- internals -----------------------------------------------------------

    def _begin(self, item: _Frame) -> None:
        self._frames = list(self._ring)
        self._ring.clear()
        self._open = True
        self._in_speech = False
        self._speech_frames = 0
        self._silence_run = 0
        self._last_voiced = -1
        self._continuation = False
        self._append(item)

    def _append(self, item: _Frame) -> None:
        threshold = self._config.end_threshold if self._in_speech else self._config.start_threshold
        item.voiced = item.prob >= threshold
        self._frames.append(item)
        if item.voiced:
            self._in_speech = True
            self._speech_frames += 1
            self._silence_run = 0
            self._last_voiced = len(self._frames) - 1
        else:
            self._in_speech = False
            self._silence_run += 1

    def _is_speech(self) -> bool:
        """Whether the open segment is real speech rather than a blip."""
        if self._speech_frames == 0:
            return False
        return self._continuation or self._speech_frames * FRAME_MS >= self._config.min_speech_ms

    def _trimmed_end(self) -> int:
        if self._last_voiced < 0:
            return 0
        return min(len(self._frames), self._last_voiced + 1 + self._post_roll)

    def _close(self) -> list[SegmentAudio]:
        end = self._trimmed_end()
        out = [self._emit(self._frames[:end])] if self._is_speech() else []
        # The rest of the pause becomes the next segment's pre-roll (only its newest frames are kept).
        self._ring.clear()
        self._ring.extend(self._frames[end:])
        self._frames = []
        self._open = False
        self._in_speech = False
        self._speech_frames = 0
        self._last_voiced = -1
        self._continuation = False
        return out

    def _cut(self) -> list[SegmentAudio]:
        frames = self._frames
        lo = max(0, len(frames) - self._lookback)
        scores = [f.prob + _CUT_LOUDNESS_WEIGHT * f.loudness for f in frames[lo:]]
        best = min(scores)
        first = scores.index(best)
        last = first
        while last + 1 < len(scores) and scores[last + 1] == best:
            last += 1
        # Cut after the middle of the quietest stretch; the quiet frame stays with the head.
        cut = lo + (first + last) // 2 + 1
        head, tail = frames[:cut], frames[cut:]

        out: list[SegmentAudio] = []
        is_speech = self._is_speech()
        if is_speech and any(f.voiced for f in head):
            out.append(self._emit(head))

        self._frames = tail
        self._speech_frames = sum(1 for f in tail if f.voiced)
        self._last_voiced = max((i for i, f in enumerate(tail) if f.voiced), default=-1)
        self._continuation = is_speech
        return out

    def _emit(self, frames: list[_Frame]) -> SegmentAudio:
        speech_frames = sum(1 for f in frames if f.voiced)
        self._total_speech_frames += speech_frames
        return SegmentAudio(
            samples=np.concatenate([f.samples for f in frames]),
            speech_ms=speech_frames * FRAME_MS,
        )
