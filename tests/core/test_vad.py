"""Silero VAD on real audio: silence and noise score low, speech scores high."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers.speech import concat, noise, silence, synthesize
from whisp.core.onnx_runtime import ort
from whisp.core.types import FRAME_SAMPLES, SAMPLE_RATE
from whisp.core.vad import SileroVad, default_model_path

SENTENCE = "Please schedule the quarterly review for next Thursday afternoon."


@pytest.fixture(scope="module")
def vad() -> SileroVad:
    return SileroVad()


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    return synthesize(SENTENCE)


def probabilities(vad: SileroVad, audio: np.ndarray) -> np.ndarray:
    vad.reset()
    count = audio.size // FRAME_SAMPLES
    return np.array([vad(audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]) for i in range(count)])


def test_digital_silence_is_not_speech(vad: SileroVad) -> None:
    assert probabilities(vad, silence(2.0)).max() < 0.2


@pytest.mark.parametrize("level_db", [-65.0, -45.0, -25.0])
def test_steady_noise_is_not_speech(vad: SileroVad, level_db: float) -> None:
    probs = probabilities(vad, noise(3.0, level_db, seed=11))
    assert probs.max() < 0.35


def test_speech_is_detected(vad: SileroVad, speech: np.ndarray) -> None:
    lead = round(0.5 * SAMPLE_RATE) // FRAME_SAMPLES
    probs = probabilities(vad, concat(silence(0.5), speech, silence(0.5)))
    assert probs[:lead].max() < 0.2
    assert probs.max() > 0.95
    assert np.mean(probs[lead:-lead] > 0.5) > 0.6


def test_speech_is_detected_over_room_noise(vad: SileroVad, speech: np.ndarray) -> None:
    probs = probabilities(vad, speech + noise(speech.size / SAMPLE_RATE, -45.0, seed=5))
    assert np.mean(probs > 0.5) > 0.6


def test_reset_makes_results_repeatable(vad: SileroVad, speech: np.ndarray) -> None:
    first = probabilities(vad, speech)
    second = probabilities(vad, speech)
    np.testing.assert_array_equal(first, second)


def test_matches_the_reference_onnx_wrapper(vad: SileroVad, speech: np.ndarray) -> None:
    # The call sequence of silero_vad.utils_vad.OnnxWrapper at 16 kHz, batch size 1.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(default_model_path()), sess_options=options, providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    expected = []
    for i in range(speech.size // FRAME_SAMPLES):
        x = np.concatenate([context, speech[None, i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]], axis=1)
        out, state = session.run(None, {"input": x, "state": state, "sr": np.array(16000, dtype="int64")})
        context = x[..., -64:]
        expected.append(float(out[0, 0]))
    np.testing.assert_allclose(probabilities(vad, speech), expected, rtol=0, atol=1e-6)


def test_frames_of_the_wrong_size_are_rejected(vad: SileroVad) -> None:
    with pytest.raises(ValueError):
        vad(np.zeros(FRAME_SAMPLES // 2, dtype=np.float32))


def test_non_finite_input_does_not_poison_later_frames(vad: SileroVad, speech: np.ndarray) -> None:
    vad.reset()
    assert 0.0 <= vad(np.full(FRAME_SAMPLES, np.nan, dtype=np.float32)) <= 1.0
    after = [vad(speech[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]) for i in range(speech.size // FRAME_SAMPLES)]
    assert all(np.isfinite(after))
    assert max(after) > 0.95


def test_missing_model_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SileroVad(tmp_path / "missing.onnx")
