"""OnnxAsrEngine without real models: device fallback rules, the onnx-asr #138 fix, audio handling, Whisper guards."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.speech import concat, noise, silence, synthesize
from whisp.core.asr import engine as engine_module
from whisp.core.asr import gpu
from whisp.core.asr.catalog import PARAKEET_V2, PARAKEET_V3, WHISPER_TURBO, ModelInfo
from whisp.core.asr.engine import (
    MSG_FILES_MISSING,
    MSG_GPU_INACTIVE,
    MSG_GPU_LOAD_FAILED,
    MSG_GPU_NO_VARIANT,
    MSG_NO_MEMORY,
    MSG_NOT_INSTALLED,
    EngineError,
    EngineSpec,
    OnnxAsrEngine,
    ParakeetTdt,
    WhisperTurbo,
    _clean_audio,
    _is_language_token,
    _pad,
    _quietest,
    _repeated_tail,
    _split_long,
    load_engine,
)
from whisp.core.types import SAMPLE_RATE

READY = gpu.GpuInfo(True, "NVIDIA Test GPU", 12000, True, "NVIDIA Test GPU · ready")
NOT_READY = gpu.GpuInfo(True, "NVIDIA Test GPU", 12000, False, "Your NVIDIA driver is too old. Using the CPU.")


class FakeSession:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers


class FakeRecognizer:
    """Stands in for an onnx-asr model: records calls, answers with a fixed text."""

    def __init__(self, providers: list[str], text: str = "hello there", language: str | None = None) -> None:
        self.providers = providers
        self.text = text
        self.language = language
        self.calls: list[tuple[int, str | None]] = []

    def sessions(self) -> tuple[FakeSession, ...]:
        return (FakeSession(self.providers), FakeSession(self.providers))

    def transcribe_one(self, samples: np.ndarray, language: str | None) -> tuple[str, str | None]:
        self.calls.append((samples.size, language))
        return self.text, self.language


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """Replace model building: records (variant, cuda); CUDA builds get the CUDA provider."""
    calls: list[tuple[str, bool]] = []

    def build(self: OnnxAsrEngine, variant: str, cuda: bool) -> FakeRecognizer:
        calls.append((variant, cuda))
        return FakeRecognizer(["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"])

    monkeypatch.setattr(OnnxAsrEngine, "_build", build)
    monkeypatch.setattr(gpu, "detect_gpu", lambda: READY)
    monkeypatch.setattr(engine_module.hardware, "nvidia_free_vram_mb", lambda: 11000)
    return calls


def spec(info: ModelInfo = PARAKEET_V2, variants: tuple[str, ...] = ("gpu", "cpu"), cuda: bool = True) -> EngineSpec:
    return EngineSpec(info, {v: Path(f"/models/{info.id}/{v}") for v in variants}, cuda, cpu_threads=4)


# -- device and variant choice ------------------------------------------------------------------------


def test_gpu_files_run_on_cuda_when_it_is_ready(builds) -> None:
    engine = OnnxAsrEngine(spec())
    assert builds == [("gpu", True)]
    assert (engine.device, engine.device_name, engine.variant, engine.notice) == (
        "cuda",
        "NVIDIA Test GPU",
        "gpu",
        None,
    )


def test_cpu_prefers_the_int8_variant(builds) -> None:
    engine = OnnxAsrEngine(spec(cuda=False))
    assert builds == [("cpu", False)]
    assert (engine.device, engine.device_name, engine.variant) == ("cpu", "CPU", "cpu")


def test_cpu_uses_the_gpu_files_when_only_they_are_installed(builds) -> None:
    engine = OnnxAsrEngine(spec(variants=("gpu",), cuda=False))
    assert builds == [("gpu", False)]
    assert engine.variant == "gpu" and engine.device == "cpu"


def test_gpu_not_ready_falls_back_to_the_cpu_with_the_reason(builds, monkeypatch) -> None:
    monkeypatch.setattr(gpu, "detect_gpu", lambda: NOT_READY)
    engine = OnnxAsrEngine(spec())
    assert builds == [("cpu", False)]
    assert engine.device == "cpu" and engine.notice == NOT_READY.detail


def test_asking_for_the_gpu_without_its_files_explains(builds) -> None:
    engine = OnnxAsrEngine(spec(variants=("cpu",)))
    assert builds == [("cpu", False)]
    assert engine.notice == MSG_GPU_NO_VARIANT


def test_a_failed_gpu_load_falls_back_to_the_cpu(builds, monkeypatch) -> None:
    def build(self: OnnxAsrEngine, variant: str, cuda: bool) -> FakeRecognizer:
        builds.append((variant, cuda))
        if cuda:
            raise RuntimeError("CUDA failure 2: out of memory")
        return FakeRecognizer(["CPUExecutionProvider"])

    monkeypatch.setattr(OnnxAsrEngine, "_build", build)
    engine = OnnxAsrEngine(spec())
    assert builds == [("gpu", True), ("cpu", False)]
    assert engine.device == "cpu" and engine.notice == MSG_GPU_LOAD_FAILED


class FailingRecognizer(FakeRecognizer):
    """Loads fine, then fails every run with ``error``."""

    def __init__(self, providers: list[str], error: Exception) -> None:
        super().__init__(providers)
        self.error = error

    def transcribe_one(self, samples: np.ndarray, language: str | None) -> tuple[str, str | None]:
        raise self.error


def test_a_gpu_that_fails_its_first_runs_falls_back_to_the_cpu(builds, monkeypatch) -> None:
    def build(self: OnnxAsrEngine, variant: str, cuda: bool) -> FakeRecognizer:
        builds.append((variant, cuda))
        if cuda:  # the weights fit, but the first real run runs out of graphics memory
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return FailingRecognizer(providers, RuntimeError("CUDA failure 2: out of memory"))
        return FakeRecognizer(["CPUExecutionProvider"])

    monkeypatch.setattr(OnnxAsrEngine, "_build", build)
    engine = load_engine(spec())
    assert builds == [("gpu", True), ("cpu", False)]
    assert (engine.device, engine.variant, engine.notice) == ("cpu", "cpu", MSG_GPU_LOAD_FAILED)
    assert engine.transcribe(np.ones(16000, dtype=np.float32) * 0.01, "auto")[0] == "hello there"


def test_a_cpu_that_fails_its_first_runs_reports_why(builds, monkeypatch) -> None:
    def build(self: OnnxAsrEngine, variant: str, cuda: bool) -> FakeRecognizer:
        return FailingRecognizer(["CPUExecutionProvider"], MemoryError("bad allocation"))

    monkeypatch.setattr(OnnxAsrEngine, "_build", build)
    with pytest.raises(EngineError) as caught:
        load_engine(spec(cuda=False))
    assert str(caught.value) == MSG_NO_MEMORY


def test_onnx_runtime_quietly_running_on_the_cpu_is_noticed(builds, monkeypatch) -> None:
    def build(self: OnnxAsrEngine, variant: str, cuda: bool) -> FakeRecognizer:
        builds.append((variant, cuda))
        return FakeRecognizer(["CPUExecutionProvider"])  # asked for CUDA, got the CPU

    monkeypatch.setattr(OnnxAsrEngine, "_build", build)
    engine = OnnxAsrEngine(spec())
    assert builds == [("gpu", True), ("cpu", False)]
    assert engine.device == "cpu" and engine.notice == MSG_GPU_INACTIVE


def test_too_little_free_vram_skips_the_gpu(builds, monkeypatch) -> None:
    monkeypatch.setattr(engine_module.hardware, "nvidia_free_vram_mb", lambda: 1500)
    engine = OnnxAsrEngine(spec())
    assert builds == [("cpu", False)]
    assert engine.device == "cpu"
    assert engine.notice is not None and "1.5 GB free" in engine.notice and "3.7 GB needed" in engine.notice


def test_nothing_installed_is_an_error(builds) -> None:
    with pytest.raises(EngineError) as caught:
        OnnxAsrEngine(spec(variants=()))
    assert str(caught.value) == MSG_NOT_INSTALLED


def test_missing_files_on_the_cpu_are_a_plain_error(builds, monkeypatch) -> None:
    def build(self: OnnxAsrEngine, variant: str, cuda: bool) -> FakeRecognizer:
        raise FileNotFoundError("encoder-model.int8.onnx")

    monkeypatch.setattr(OnnxAsrEngine, "_build", build)
    with pytest.raises(EngineError) as caught:
        OnnxAsrEngine(spec(cuda=False))
    assert str(caught.value) == MSG_FILES_MISSING


def test_close_frees_the_model_and_further_use_fails(builds) -> None:
    engine = OnnxAsrEngine(spec())
    engine.close()
    with pytest.raises(EngineError):
        engine.transcribe(np.zeros(16000, dtype=np.float32), "auto")
    engine.close()  # twice is fine


# -- transcription ---------------------------------------------------------------------------------------


def test_short_audio_is_padded_and_empty_audio_skips_the_model(builds) -> None:
    engine = OnnxAsrEngine(spec())
    model = engine._model
    assert engine.transcribe(np.zeros(0, dtype=np.float32), "auto") == ("", "en")
    assert model.calls == []
    engine.transcribe(np.zeros(1000, dtype=np.float32), "auto")
    assert model.calls == [(SAMPLE_RATE // 2, None)]


def test_long_audio_is_split_for_parakeet(builds) -> None:
    engine = OnnxAsrEngine(spec())
    audio = np.asarray(noise(130.0, -30.0), dtype=np.float32)
    text, _ = engine.transcribe(audio, "auto")
    sizes = [size for size, _ in engine._model.calls]
    assert len(sizes) == 3 and sum(sizes) == audio.size and max(sizes) <= 60 * SAMPLE_RATE
    assert text == "hello there hello there hello there"


def test_language_results(builds) -> None:
    english = OnnxAsrEngine(spec(PARAKEET_V2))
    assert english.transcribe(np.ones(16000, dtype=np.float32) * 0.01, "de")[1] == "en"
    multilingual = OnnxAsrEngine(spec(PARAKEET_V3))
    assert multilingual.transcribe(np.ones(16000, dtype=np.float32) * 0.01, "auto")[1] is None
    assert multilingual.transcribe(np.ones(16000, dtype=np.float32) * 0.01, "de")[1] == "de"
    assert multilingual.transcribe(np.ones(16000, dtype=np.float32) * 0.01, "ja")[1] is None


def test_whisper_is_told_the_language_in_its_own_spelling(builds) -> None:
    engine = OnnxAsrEngine(spec(WHISPER_TURBO))
    engine._model.language = "jw"
    audio = concat(silence(0.2), synthesize("Good morning, everyone, and welcome."))
    _, language = engine.transcribe(audio, "jv")
    assert engine._model.calls[-1][1] == "jw" and language == "jv"
    engine.transcribe(audio, "auto")
    assert engine._model.calls[-1][1] is None
    engine.transcribe(audio, "tlh")  # not a Whisper language: detect instead
    assert engine._model.calls[-1][1] is None


def test_punctuation_alone_is_dropped(builds) -> None:
    engine = OnnxAsrEngine(spec())
    engine._model.text = " . "
    assert engine.transcribe(np.ones(16000, dtype=np.float32) * 0.01, "auto")[0] == ""


def test_short_whisper_results_without_speech_are_dropped(builds) -> None:
    engine = OnnxAsrEngine(spec(WHISPER_TURBO))
    engine._model.text = "you"
    assert engine.transcribe(silence(1.5), "auto")[0] == ""
    assert engine.transcribe(noise(2.0, -40.0), "auto")[0] == ""
    speech = concat(silence(0.3), synthesize("You."), silence(0.3))
    assert engine.transcribe(speech, "auto")[0] == "you"


def test_parakeet_results_are_not_second_guessed(builds) -> None:
    engine = OnnxAsrEngine(spec())
    engine._model.text = "you"
    assert engine.transcribe(silence(1.5), "auto")[0] == "you"


def test_nan_samples_are_cleaned(builds) -> None:
    audio = np.array([0.1, np.nan, np.inf, -np.inf], dtype=np.float32)
    assert _clean_audio(audio).tolist() == pytest.approx([0.1, 0.0, 1.0, -1.0])


# -- Whisper's decoding loop ---------------------------------------------------------------------------------


class FakeWhisper(WhisperTurbo):
    """WhisperTurbo's own decoding loop over a fake model that writes ``hear(audio)``, then end-of-text."""

    EOT = 9000

    def __init__(self) -> None:  # no model files: only what transcribe_one uses
        self.passes: list[float] = []
        """Seconds of audio in each decoding pass."""
        self._eos_token_id = self.EOT
        self._tokens = {"<|ml|>": 9001}
        self._vocab = {9001: "<|ml|>"}
        self._transcribe_input = np.array([[9002, self.EOT, 9003, 9004]], dtype=np.int64)
        self._first_suppressed = np.array([220, self.EOT], dtype=np.int64)

    def hear(self, audio: np.ndarray) -> list[int]:
        raise NotImplementedError

    def _encode(self, waveforms: np.ndarray, waveforms_len: np.ndarray) -> list[int]:
        self.passes.append(waveforms.shape[1] / SAMPLE_RATE)
        return self.hear(waveforms[0])

    def _create_state(self) -> None:
        return None

    def _decode(self, tokens: np.ndarray, state: None, encoded: list[int]) -> tuple[np.ndarray, None]:
        step = tokens.shape[1] - self._transcribe_input.shape[1]
        logits = np.zeros((1, tokens.shape[1], self.EOT + 10), dtype=np.float32)
        logits[0, -1, encoded[step] if step < len(encoded) else self.EOT] = 1.0
        return logits, state

    def _decode_tokens(self, tokens: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(text=" ".join(f"t{t}" for t in tokens))


class ScriptedWhisper(FakeWhisper):
    """Writes ``script`` whatever the audio."""

    def __init__(self, script: list[int]) -> None:
        super().__init__()
        self._script = script

    def hear(self, audio: np.ndarray) -> list[int]:
        return self._script


class WordsWhisper(FakeWhisper):
    """Hears "words" in the audio: runs of one constant, non-zero sample value that names the word, with silence
    between them. Each word takes ``tokens_per_word`` tokens, as words of some languages take many."""

    def __init__(self, tokens_per_word: int) -> None:
        super().__init__()
        self.tokens_per_word = tokens_per_word

    @staticmethod
    def speak(words: int, word_s: float, gap_s: float) -> np.ndarray:
        gap = np.zeros(round(gap_s * SAMPLE_RATE), dtype=np.float32)
        parts = [np.full(round(word_s * SAMPLE_RATE), (w + 1) / 1000, dtype=np.float32) for w in range(words)]
        return np.concatenate([part for word in parts for part in (word, gap)])

    def hear(self, audio: np.ndarray) -> list[int]:
        starts = np.flatnonzero(np.diff(audio, prepend=0.0) != 0)
        words = [round(float(audio[i]) * 1000) - 1 for i in starts if audio[i] != 0]
        return [300 + w * self.tokens_per_word + j for w in words for j in range(self.tokens_per_word)]

    def tokens(self, text: str) -> list[int]:
        return [int(t[1:]) for t in text.split()]


def test_whisper_writes_out_scripts_that_take_many_tokens() -> None:
    # Malayalam, Telugu, Georgian and Burmese take 35-60 of Whisper's tokens per second of speech.
    script = list(range(300, 500))  # 200 tokens in 5 seconds
    text, language = ScriptedWhisper(script).transcribe_one(np.zeros(5 * SAMPLE_RATE, dtype=np.float32), "ml")
    assert text.split() == [f"t{t}" for t in script]
    assert language == "ml"


def test_whisper_output_is_still_bounded_by_the_clip_length() -> None:
    endless = list(range(300, 999))  # no end-of-text and never repeating itself
    text, _ = ScriptedWhisper(endless).transcribe_one(np.zeros(SAMPLE_RATE, dtype=np.float32), "ml")
    assert 0 < len(text.split()) <= 16 + WhisperTurbo.TOKENS_PER_SECOND < len(endless)


def test_a_pass_that_fills_whispers_window_is_split_and_nothing_is_lost() -> None:
    # 28 s of speech at about 53 tokens a second (Burmese-like): 1,488 tokens, more than three windows' worth.
    model = WordsWhisper(tokens_per_word=8)
    audio = model.speak(186, word_s=0.1, gap_s=0.05)
    expected = model.hear(audio)
    text, language = model.transcribe_one(audio, "ml")
    assert model.tokens(text) == expected  # every word once, in order: nothing lost or doubled at the seams
    assert language == "ml"
    assert len(model.passes) > 1 and model.passes[0] == pytest.approx(audio.size / SAMPLE_RATE)


def test_splitting_has_a_hard_bound() -> None:
    model = ScriptedWhisper(list(range(300, 999)))  # fills every window, however short the audio
    model.transcribe_one(np.zeros(60 * SAMPLE_RATE, dtype=np.float32), "ml")
    assert len(model.passes) <= 2 ** (WhisperTurbo.MAX_SPLITS + 1) - 1  # 1 + 2 + 4 + ... passes, no more


def test_english_speech_is_never_split() -> None:
    model = WordsWhisper(tokens_per_word=1)  # about 4 tokens a second, like English
    audio = model.speak(112, word_s=0.2, gap_s=0.05)
    text, _ = model.transcribe_one(audio, "ml")
    assert model.tokens(text) == model.hear(audio)
    assert len(model.passes) == 1


# -- onnx-asr bug #138 ---------------------------------------------------------------------------------------


class _Encoder:
    """An encoder session that reports one frame more than it returns, as in onnx-asr issue #138."""

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        frames = 563
        return [np.zeros((1, 1024, frames), dtype=np.float32), np.array([frames + 1], dtype=np.int64)]


def test_parakeet_clamps_encoder_lengths_to_the_frames_returned() -> None:
    model = ParakeetTdt.__new__(ParakeetTdt)  # the encoder alone, without loading a model
    model._encoder = _Encoder()
    features = np.zeros((1, 128, 4501), dtype=np.float32)
    encoder_out, lengths = model._encode(features, np.array([4505], dtype=np.int64))
    assert encoder_out.shape == (1, 563, 1024)
    assert lengths.tolist() == [563]


def test_lengths_that_fit_are_left_alone() -> None:
    class Encoder:
        def run(self, names, feeds):
            return [np.zeros((2, 1024, 100), dtype=np.float32), np.array([100, 60], dtype=np.int64)]

    model = ParakeetTdt.__new__(ParakeetTdt)
    model._encoder = Encoder()
    _, lengths = model._encode(np.zeros((2, 128, 800), dtype=np.float32), np.array([800, 480], dtype=np.int64))
    assert lengths.tolist() == [100, 60]


# -- helpers -----------------------------------------------------------------------------------------------------


def test_split_long_cuts_at_the_quietest_moment() -> None:
    rng = np.random.default_rng(0)
    loud = (rng.standard_normal(70 * SAMPLE_RATE) * 0.3).astype(np.float32)
    gap_at = 55 * SAMPLE_RATE
    loud[gap_at : gap_at + SAMPLE_RATE // 5] = 0.0  # a 200 ms pause at 55 s
    pieces = _split_long(loud, 60 * SAMPLE_RATE, 8 * SAMPLE_RATE)
    assert len(pieces) == 2
    assert gap_at <= pieces[0].size <= gap_at + SAMPLE_RATE // 5
    assert np.array_equal(np.concatenate(pieces), loud)


def test_split_long_leaves_short_audio_whole() -> None:
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    assert [p.size for p in _split_long(audio, 2 * SAMPLE_RATE, SAMPLE_RATE)] == [SAMPLE_RATE]


def test_split_long_always_makes_progress() -> None:
    audio = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)  # silent everywhere: any cut is "quietest"
    pieces = _split_long(audio, SAMPLE_RATE, 3 * SAMPLE_RATE)
    assert all(0 < p.size <= SAMPLE_RATE for p in pieces)
    assert sum(p.size for p in pieces) == audio.size


def test_quietest_and_pad() -> None:
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    audio[8000:8320] = 0.0
    assert 8000 <= _quietest(audio) < 8320
    assert _pad(np.ones(10, dtype=np.float32), 16).tolist() == [1.0] * 10 + [0.0] * 6
    assert _pad(np.ones(20, dtype=np.float32), 16).size == 20


def test_repeated_tail_finds_where_a_copy_begins() -> None:
    original = list(range(30))
    tokens = [*original, 100, 101, *original[5:20]]
    assert _repeated_tail(tokens, 10) == 32  # the copy of tokens 5..19 starts after 100, 101
    assert _repeated_tail(list(range(40)), 10) is None
    assert _repeated_tail([7] * 9, 10) is None


def test_language_tokens() -> None:
    assert _is_language_token("<|en|>") and _is_language_token("<|haw|>") and _is_language_token("<|yue|>")
    for token in ("<|transcribe|>", "<|nospeech|>", "<|0.00|>", "hello", "<|notimestamps|>", "<|endoftext|>"):
        assert not _is_language_token(token)


def test_multilingual_flag() -> None:
    assert not PARAKEET_V2.multilingual and PARAKEET_V3.multilingual
    assert replace(PARAKEET_V2, languages=("en", "de")).multilingual
