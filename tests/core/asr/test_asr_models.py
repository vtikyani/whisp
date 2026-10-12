"""Real speech models: Parakeet v2 on the GPU and CPU, onnx-asr bug #138, a 1-60 s length sweep, long passes.

Marked ``models`` (and ``gpu`` where CUDA is needed). Parakeet TDT 0.6B v2 is
downloaded into the shared test cache on first use (about 3.2 GB for both
variants). Audio is real speech from Windows text-to-speech.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from onnx_asr.models.nemo import NemoConformerTdt
from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy
from onnx_asr.resolver import Resolver
from pytestqt.qtbot import QtBot

from helpers.speech import concat, noise, silence, synthesize
from whisp.core.asr.catalog import PARAKEET_V2
from whisp.core.asr.engine import EngineSpec, ParakeetTdt, load_engine
from whisp.core.asr.manager import ModelManager
from whisp.core.asr.service import AsrService
from whisp.core.types import SAMPLE_RATE, AsrResult, EngineState, Segment

pytestmark = pytest.mark.models

MODEL_ID = "parakeet-tdt-0.6b-v2"
SENTENCES = (
    "Please send the quarterly report to Maria before the meeting on Thursday afternoon.",
    "The weather was cold and windy, so we stayed inside and read our books.",
    "Can you remind me to call the dentist tomorrow morning?",
    "Our team finished the project early and the client was very happy with the results.",
    "I think we should take the train instead of driving through the city.",
    "Remember to buy milk, bread, and a few apples on the way home.",
    "The new design looks great, but the buttons are a little too small.",
    "She opened the window and listened to the birds singing in the garden.",
)


@pytest.fixture(scope="module")
def models(shared_models_dir: Path, qapp) -> ModelManager:
    manager = ModelManager(shared_models_dir)
    failures: list[str] = []
    manager.download_failed.connect(lambda _id, message: failures.append(message))
    for variant in ("gpu", "cpu"):
        if manager.is_installed(MODEL_ID, variant):
            continue
        manager.download(MODEL_ID, variant)
        deadline = time.monotonic() + 1800
        while manager.is_downloading(MODEL_ID) and time.monotonic() < deadline:
            time.sleep(0.5)
        assert manager.is_installed(MODEL_ID, variant), failures
    return manager


@pytest.fixture(scope="module")
def sentences() -> list[np.ndarray]:
    return [synthesize(text) for text in SENTENCES]


def folders(models: ModelManager, *variants: str) -> dict[str, Path]:
    return {variant: models.model_path(MODEL_ID, variant) for variant in variants}


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / max(1, len(ref))


def check_understanding(engine, sentences: list[np.ndarray]) -> float:
    """WER over the given synthesized sentences (the first ``len(sentences)`` of ``SENTENCES``)."""
    references, hypotheses = [], []
    for text, audio in zip(SENTENCES[: len(sentences)], sentences, strict=True):
        hypothesis, language = engine.transcribe(concat(silence(0.3), audio, silence(0.3)), "auto")
        assert language == "en"
        references.append(text)
        hypotheses.append(hypothesis)
    return wer(" ".join(references), " ".join(hypotheses))


def speech_track(sentences: list[np.ndarray], seconds: float) -> np.ndarray:
    gap = silence(0.4)
    parts: list[np.ndarray] = []
    total = 0
    index = 0
    while total < seconds * SAMPLE_RATE:
        parts.extend((sentences[index % len(sentences)], gap))
        total += sentences[index % len(sentences)].size + gap.size
        index += 1
    return concat(*parts)


# -- accuracy on each device ------------------------------------------------------------------------------


@pytest.mark.gpu
def test_parakeet_on_the_gpu(models: ModelManager, sentences) -> None:
    engine = load_engine(EngineSpec(PARAKEET_V2, folders(models, "gpu", "cpu"), use_cuda=True))
    try:
        assert (engine.device, engine.variant, engine.notice) == ("cuda", "gpu", None)
        assert check_understanding(engine, sentences) < 0.06
        assert engine.transcribe(silence(2.0), "auto")[0] == ""
        assert engine.transcribe(noise(2.0, -45.0), "auto")[0] == ""
    finally:
        engine.close()


def test_parakeet_int8_on_the_cpu(models: ModelManager, sentences) -> None:
    engine = load_engine(EngineSpec(PARAKEET_V2, folders(models, "gpu", "cpu"), use_cuda=False))
    try:
        assert (engine.device, engine.variant, engine.device_name) == ("cpu", "cpu", "CPU")
        assert check_understanding(engine, sentences) < 0.06
        assert engine.transcribe(silence(2.0), "auto")[0] == ""
    finally:
        engine.close()


def test_the_gpu_files_also_run_on_the_cpu(models: ModelManager, sentences) -> None:
    engine = load_engine(EngineSpec(PARAKEET_V2, folders(models, "gpu"), use_cuda=False))
    try:
        assert (engine.device, engine.variant) == ("cpu", "gpu")
        assert check_understanding(engine, sentences[:3]) < 0.06
    finally:
        engine.close()


# -- onnx-asr bug #138 and long audio -----------------------------------------------------------------------


def test_bug_138_crashes_onnx_asr_and_not_whisp(models: ModelManager, sentences) -> None:
    """When the lengths fed in overstate the audio (as in the issue), stock onnx-asr asserts; ParakeetTdt doesn't."""
    folder = models.model_path(MODEL_ID, "cpu")
    files = Resolver(NemoConformerTdt, None, folder, offline=True).resolve_model(quantization="int8")
    options = {"providers": ["CPUExecutionProvider"]}
    stock = NemoConformerTdt(files, NemoPreprocessorNumpy, options)
    fixed = ParakeetTdt(files, NemoPreprocessorNumpy, options)
    audio = speech_track(sentences, 45.0)[: 45 * SAMPLE_RATE][None, :]
    overstated = np.array([audio.shape[1] + 5 * 160], dtype=np.int64)  # five feature frames too many

    with pytest.raises(AssertionError):
        list(stock.recognize_batch(audio, overstated))
    text = next(iter(fixed.recognize_batch(audio, overstated))).text
    assert wer(" ".join(SENTENCES), text) < 0.5  # 45 s covers most of the sentences once


@pytest.mark.gpu
def test_length_sweep_from_1_to_60_seconds_never_raises(models: ModelManager, sentences) -> None:
    """Every clip length from 1 s to 60 s in 130 ms steps (all residues of the 80 ms encoder frame), on the GPU.

    Each length also runs with the lengths overstated by five frames, the condition behind bug #138.
    """
    engine = load_engine(EngineSpec(PARAKEET_V2, folders(models, "gpu"), use_cuda=True))
    model = engine._model
    track = speech_track(sentences, 61.0)
    try:
        lengths = np.arange(1.0, 60.0 + 1e-9, 0.13)
        empty = []
        for seconds in lengths:
            clip = track[: int(seconds * SAMPLE_RATE)]
            text, _ = engine.transcribe(clip, "auto")
            if seconds >= 3.0 and not text:
                empty.append(round(float(seconds), 2))
            overstated = np.array([clip.size + 5 * 160], dtype=np.int64)
            next(iter(model.recognize_batch(clip[None, :], overstated)))
        assert len(lengths) > 450
        assert empty == []
    finally:
        engine.close()


@pytest.mark.gpu
def test_long_single_passes_do_not_drift(models: ModelManager, sentences) -> None:
    """30 s and 58 s in one pass: accuracy holds to the end (the last sentence is recognised too)."""
    engine = load_engine(EngineSpec(PARAKEET_V2, folders(models, "gpu"), use_cuda=True))
    try:
        for seconds in (30.0, 58.0):
            parts, references, total = [], [], 0
            index = 0
            while True:
                audio = sentences[index % len(sentences)]
                if total + audio.size + 0.4 * SAMPLE_RATE > seconds * SAMPLE_RATE:
                    break
                parts.extend((audio, silence(0.4)))
                references.append(SENTENCES[index % len(SENTENCES)])
                total += audio.size + int(0.4 * SAMPLE_RATE)
                index += 1
            text, _ = engine.transcribe(concat(*parts), "auto")
            assert wer(" ".join(references), text) < 0.06, seconds
            last = references[-1]
            assert wer(last, " ".join(words(text)[-len(words(last)) :])) < 0.2, seconds
    finally:
        engine.close()


def test_audio_longer_than_one_pass_is_split_and_fully_recognised(models: ModelManager, sentences) -> None:
    engine = load_engine(EngineSpec(PARAKEET_V2, folders(models, "cpu"), use_cuda=False))
    try:
        parts, references = [], []
        for index in range(len(SENTENCES) * 3):  # ~100 s: two cuts at quiet moments
            parts.extend((sentences[index % len(sentences)], silence(0.6)))
            references.append(SENTENCES[index % len(SENTENCES)])
        audio = concat(*parts)
        assert audio.size > 90 * SAMPLE_RATE
        text, _ = engine.transcribe(audio, "auto")
        assert wer(" ".join(references), text) < 0.06
    finally:
        engine.close()


# -- Whisper (only if it is installed in the shared cache) -----------------------------------------------------


@pytest.mark.gpu
def test_whisper_on_the_gpu_if_installed(shared_models_dir: Path, sentences, qapp) -> None:
    manager = ModelManager(shared_models_dir)
    info = manager.model_info("whisper-large-v3-turbo")
    if not manager.is_installed(info.id, "gpu"):
        pytest.skip("Whisper Large v3 Turbo isn't in the shared model cache")
    engine = load_engine(EngineSpec(info, {"gpu": manager.model_path(info.id, "gpu")}, use_cuda=True))
    try:
        assert engine.device == "cuda"
        assert check_understanding(engine, sentences) < 0.08
        assert engine.transcribe(silence(1.5), "auto")[0] == ""
        assert engine.transcribe(noise(2.0, -45.0), "auto")[0] == ""
    finally:
        engine.close()


# -- the service end to end ----------------------------------------------------------------------------------


@pytest.mark.gpu
def test_the_service_recognises_segments_on_the_gpu(
    qtbot: QtBot, models: ModelManager, settings_store, sentences
) -> None:
    service = AsrService(settings_store, models)
    results: list[AsrResult] = []
    lock = threading.Lock()

    def receive(result: AsrResult) -> None:
        with lock:
            results.append(result)

    try:
        service.start()
        for index, audio in enumerate(sentences[:3]):
            service.submit(Segment("s", index, concat(silence(0.3), audio, silence(0.3)), 2000, index == 2), receive)
        qtbot.waitUntil(lambda: len(results) == 3, timeout=60_000)
        status = service.status
        assert status.state is EngineState.READY and status.device == "cuda"
        assert [r.index for r in results] == [0, 1, 2]
        for result, reference in zip(results, SENTENCES[:3], strict=True):
            assert result.error is None and wer(reference, result.text) < 0.1
            assert result.language == "en" and result.inference_ms < 1000
        assert models.in_use(MODEL_ID)
    finally:
        service.shutdown()
    assert not models.in_use(MODEL_ID)
