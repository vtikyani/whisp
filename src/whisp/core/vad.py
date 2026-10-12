"""Silero VAD: the probability that a 32 ms frame contains speech.

Runs the bundled ``silero_vad.onnx`` (Silero VAD v6.2.1, MIT licence) with
ONNX Runtime on the CPU. It mirrors ``OnnxWrapper`` from the silero-vad
package: each 512-sample frame is fed with the last 64 samples of the previous
frame in front of it, and the recurrent state is carried from frame to frame.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from whisp.core.onnx_runtime import ort
from whisp.core.types import FRAME_SAMPLES, SAMPLE_RATE
from whisp.storage.paths import resource_path

_CONTEXT_SAMPLES = 64
"""Samples from the end of the previous frame prepended to each frame (Silero, 16 kHz)."""

_STATE_SHAPE = (2, 1, 128)


def default_model_path() -> Path:
    """The Silero model shipped inside the package."""
    return resource_path("assets", "models", "silero_vad.onnx")


class SileroVad:
    """Speech probability per frame. Not thread-safe; one instance per thread."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or default_model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found at {path}")
        options = ort.SessionOptions()
        # The model is tiny: extra threads only add scheduling overhead.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self._sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def reset(self) -> None:
        """Forget all previous audio. Call at the start of every recording."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability (0-1) for one ``FRAME_SAMPLES``-sample float32 frame at 16 kHz."""
        samples = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if samples.shape[1] != FRAME_SAMPLES:
            raise ValueError(f"expected a {FRAME_SAMPLES}-sample frame, got {samples.shape[1]} samples")
        model_input = np.concatenate((self._context, samples), axis=1)
        prob, state = self._session.run(None, {"input": model_input, "state": self._state, "sr": self._sample_rate})
        self._state = state
        self._context = model_input[:, -_CONTEXT_SAMPLES:].copy()
        value = float(prob[0, 0])
        if not math.isfinite(value):
            # A non-finite result means the recurrent state is corrupt; start afresh instead of carrying it on.
            self.reset()
            return 0.0
        return value
