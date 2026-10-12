"""OnnxAsrEngine: load a speech model with onnx-asr and turn audio into text.

Loading
-------
The model is built from onnx-asr's own pieces: its resolver finds the files of
the installed variant, and its model class runs them. Two of those classes are
extended here (see ``ParakeetTdt`` and ``WhisperTurbo``), which is why the engine
doesn't go through ``onnx_asr.load_model``: that function can't be told to use a
subclass, and it would also build seven resampler sessions Whisp never uses
(audio is always 16 kHz already). Execution providers are always passed
explicitly: CUDA then CPU on the GPU, CPU alone otherwise (onnx-asr's default
puts TensorRT first).

After loading, the providers ONNX Runtime really chose are checked, because it
silently falls back to the CPU when CUDA can't start. If the GPU was asked for
but isn't running, the engine reloads on the CPU (the int8 ``cpu`` variant if
it's installed, else the same fp32 files) and says why in ``notice``.

Speed choices, all measured on an RTX 4070 and an i9-12900K (see
scripts/benchmark_asr.py):

* CUDA uses cuDNN's ``HEURISTIC`` convolution algorithms (``gpu.cuda_provider_options``).
* Features are computed with onnx-asr's NumPy preprocessor on every device. Its ONNX
  version on CUDA saved ~20 ms on 30-second clips but needed 3 GB more VRAM.
* The Parakeet decoder stays on the GPU with the encoder: it was 2-3x faster
  there than on the CPU.
* CPU sessions use one thread per performance core with thread spinning off:
  efficiency cores and spinning both made inference markedly slower on a hybrid CPU.

onnx-asr bug #138
-----------------
``_AsrWithTransducerDecoding._decoding`` (onnx_asr/asr.py) asserts
``encoder_out_lens <= encoder_out.shape[1]``; the lengths come straight from the
encoder graph (``NemoConformerRnnt._encode``) and only TensorRT fp16 gets them
clamped. When the lengths fed in overstate the audio by about 50 ms or more, the
encoder reports one frame more than it returns at some clip lengths (45.0 s,
47.0 s, 49.0 s, 58.6 s, ...) and recognition crashes with ``AssertionError``.
Whisp passes exact lengths, and with them the lengths never overshot in a sweep
of every clip length from 1 to 60 s; ``ParakeetTdt`` still clamps the encoder
lengths for every provider, which is exactly onnx-asr's own fix, applied
unconditionally and to nothing else.

Whisper and silence
-------------------
Whisper Large v3 Turbo writes "you" or "." for silence, breaths, clicks and hum,
and its no-speech token never rises above zero, so the usual no-speech check
can't work. Whisp's segments are already cut by voice activity detection, but a
short Whisper result is double-checked with the bundled Silero VAD and dropped
if the audio holds no speech. Parakeet doesn't have the problem.
"""

from __future__ import annotations

import gc
import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from onnx_asr.loader import create_asr_resolver
from onnx_asr.models.nemo import NemoConformerTdt
from onnx_asr.models.whisper import WhisperHf
from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy, WhisperPreprocessorNumpy
from onnx_asr.utils import ModelLoadingError

from whisp.core import hardware
from whisp.core.asr import gpu
from whisp.core.asr.catalog import ModelInfo
from whisp.core.onnx_runtime import ort
from whisp.core.text.scripts import join_spoken
from whisp.core.types import FRAME_SAMPLES, SAMPLE_RATE
from whisp.core.vad import SileroVad

log = logging.getLogger(__name__)

CUDA = "cuda"
CPU = "cpu"
CPU_NAME = "CPU"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

MIN_SAMPLES = SAMPLE_RATE // 2
"""Shorter clips are padded with silence; a few frames of audio give the models too little to work with."""
PARAKEET_MAX_PASS_S = 60.0
"""Longer audio is cut at quiet moments and recognised in parts (Whisp's segments are under 30 s anyway)."""
WHISPER_MAX_PASS_S = 29.5
"""Whisper hears 30 seconds at a time; anything beyond is dropped by its preprocessor, so audio is cut before."""
CUT_SEARCH_S = 8.0
"""Where a long clip is cut: the quietest moment in the last few seconds before the limit."""
PARAKEET_VRAM_FACTOR = 1.35
"""Measured: Parakeet's 2.4 GB of fp32 weights take 3.2-3.5 GB of VRAM once running."""
WHISPER_VRAM_FACTOR = 2.05
"""Measured: Whisper's 1.6 GB of fp16 weights take 3.3 GB (its 30-second window and decoder cache are large)."""
VRAM_MARGIN_MB = 512
WHISPER_CHECK_MAX_WORDS = 3
"""Whisper results this short are checked for speech: on silence, breaths, clicks and hum Whisper Large v3
Turbo confidently writes "you" or "." (its no-speech token stays at probability 0, so OpenAI's usual check
can't catch it)."""
VAD_SPEECH_PROBABILITY = 0.5
MIN_VOICED_FRAMES = 6
"""About 190 ms of speech, in line with the session segmenter's 200 ms minimum."""

MSG_NOT_INSTALLED = "The speech model isn't downloaded. Download it in Settings."
MSG_FILES_MISSING = "Some of the speech model's files are missing. Download it again in Settings."
MSG_LOAD_FAILED = "The speech model couldn't be loaded. Delete it and download it again in Settings."
MSG_NO_MEMORY = "There isn't enough free memory to load the speech model. Close some apps and try again."
MSG_CLOSED = "The speech model was unloaded."
MSG_GPU_NO_VARIANT = "The GPU version of this model isn't downloaded, so it runs on the CPU."
MSG_GPU_LOW_MEMORY = (
    "The graphics card is short of free memory ({free:.1f} GB free, about {need:.1f} GB needed), "
    "so the model runs on the CPU."
)
MSG_GPU_LOAD_FAILED = "The graphics card couldn't load the model, so it runs on the CPU."
MSG_GPU_INACTIVE = "The GPU libraries didn't start, so the model runs on the CPU."


class EngineError(Exception):
    """The model couldn't be loaded. ``str(error)`` is a short message for the user."""


@dataclass(frozen=True)
class EngineSpec:
    """What to load."""

    model: ModelInfo
    folders: Mapping[str, Path]
    """Installed variants: variant key -> folder holding its files."""
    use_cuda: bool
    """Try the GPU first (falls back to the CPU on its own)."""
    cpu_threads: int = 0
    """Threads for CPU inference; 0 = one per performance core."""


class AsrEngine(Protocol):
    """A loaded speech model. Used by one thread at a time."""

    @property
    def device(self) -> str:
        """``"cuda"`` or ``"cpu"``: where the model really runs."""
        ...

    @property
    def device_name(self) -> str: ...

    @property
    def variant(self) -> str:
        """Key of the variant whose files are loaded."""
        ...

    @property
    def notice(self) -> str | None:
        """Why the model runs somewhere other than asked, in plain words; None if it doesn't."""
        ...

    def transcribe(self, samples: np.ndarray, language: str) -> tuple[str, str | None]:
        """Mono float32 16 kHz audio -> (text, language code or None). ``language``: ``"auto"`` or a code."""
        ...

    def close(self) -> None:
        """Free the model's memory (RAM and VRAM). The engine can't be used afterwards."""
        ...


EngineFactory = Callable[[EngineSpec], AsrEngine]
"""Loads an engine (blocking; called on the speech recognition thread). Raises ``EngineError``."""


def load_engine(spec: EngineSpec) -> OnnxAsrEngine:
    """Load the model described by ``spec`` and warm it up. Blocking. Raises ``EngineError``.

    A model that loads on the GPU but fails its first runs there (typically short of graphics
    memory once it really works) is loaded on the CPU instead, like a GPU load failure.
    """
    engine = OnnxAsrEngine(spec)
    try:
        engine.warm_up()
    except Exception as exc:
        on_gpu = engine.device == CUDA
        engine.close()
        if not on_gpu:
            raise _load_error(exc) from exc
        log.exception("Running %s on the GPU failed; loading it on the CPU", spec.model.id)
        engine = OnnxAsrEngine(replace(spec, use_cuda=False))
        engine._notice = MSG_GPU_LOAD_FAILED  # this module's own engine: say why it isn't on the GPU
        try:
            engine.warm_up()
        except Exception as cpu_exc:
            engine.close()
            raise _load_error(cpu_exc) from cpu_exc
    return engine


class OnnxAsrEngine:
    """A speech model loaded in ONNX Runtime. See the module docstring."""

    def __init__(self, spec: EngineSpec) -> None:
        self._spec = spec
        self._info = spec.model
        self._threads = spec.cpu_threads or max(1, hardware.detect().performance_cores)
        self._model: _Recognizer | None = None
        self._device = CPU
        self._device_name = CPU_NAME
        self._variant = ""
        self._notice: str | None = None
        self._max_pass = int((WHISPER_MAX_PASS_S if self._is_whisper else PARAKEET_MAX_PASS_S) * SAMPLE_RATE)
        self._vad: SileroVad | None = None
        self._load()

    # -- AsrEngine ---------------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def notice(self) -> str | None:
        return self._notice

    def transcribe(self, samples: np.ndarray, language: str) -> tuple[str, str | None]:
        model = self._model
        if model is None:
            raise EngineError(MSG_CLOSED)
        audio = _clean_audio(samples)
        forced = self._forced_language(language)
        if audio.size == 0:
            return "", self._result_language(language, [])
        texts: list[str] = []
        detected: list[str] = []
        for piece in _split_long(audio, self._max_pass, int(CUT_SEARCH_S * SAMPLE_RATE)):
            text, found = model.transcribe_one(_pad(piece, MIN_SAMPLES), forced)
            if text:
                texts.append(text)
                if found:
                    detected.append(found)
        text = join_spoken(texts)
        if not any(ch.isalnum() for ch in text):
            text = ""  # punctuation alone isn't something anyone said
        elif self._is_whisper and len(text.split()) <= WHISPER_CHECK_MAX_WORDS and not self._has_speech(audio):
            log.info("Dropped a %d-word Whisper result: the audio has no speech in it", len(text.split()))
            text = ""
        return text, self._result_language(language, detected)

    def close(self) -> None:
        model, self._model = self._model, None
        self._vad = None
        if model is not None:
            del model
            gc.collect()  # ONNX Runtime frees a session's memory (including VRAM) when it's destroyed

    def _has_speech(self, audio: npt.NDArray[np.float32]) -> bool:
        """Silero VAD's verdict: at least ``MIN_VOICED_FRAMES`` frames of speech."""
        if self._vad is None:
            self._vad = SileroVad()
        vad = self._vad
        vad.reset()
        voiced = 0
        for start in range(0, audio.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            if vad(audio[start : start + FRAME_SAMPLES]) >= VAD_SPEECH_PROBABILITY:
                voiced += 1
                if voiced >= MIN_VOICED_FRAMES:
                    return True
        return False

    # -- loading -------------------------------------------------------------------------------

    def warm_up(self) -> None:
        """Run the model once on silence and on speech-like sound of a few lengths.

        The first inference on a fresh session pays for cuDNN/cuBLAS set-up, kernel
        loading and memory arena growth; doing it here keeps it out of the first dictation.
        """
        model = self._model
        if model is None:
            return
        started = time.perf_counter()
        lengths = (1.0, 4.0) if self._is_whisper else (2.0, 5.0, 12.0)
        model.transcribe_one(np.zeros(MIN_SAMPLES, dtype=np.float32), None)
        for index, seconds in enumerate(lengths):
            model.transcribe_one(_speech_like(seconds, seed=index), None)
        log.info("Warmed up %s in %.0f ms", self._info.id, (time.perf_counter() - started) * 1000)

    @property
    def _is_whisper(self) -> bool:
        return self._info.onnx_asr_name.startswith("whisper")

    def _load(self) -> None:
        folders = self._spec.folders
        attempts: list[tuple[str, bool]] = []
        if self._spec.use_cuda:
            if "gpu" in folders:
                attempts.append(("gpu", True))
            else:
                self._notice = MSG_GPU_NO_VARIANT
        cpu_variant = "cpu" if "cpu" in folders else "gpu" if "gpu" in folders else None
        if cpu_variant is None:
            raise EngineError(MSG_NOT_INSTALLED)
        attempts.append((cpu_variant, False))
        for variant, cuda in attempts:
            if cuda:
                obstacle = self._cuda_obstacle(variant)
                if obstacle is not None:
                    self._notice = obstacle
                    continue
            started = time.perf_counter()
            try:
                model = self._build(variant, cuda)
            except Exception as exc:
                if not cuda:
                    raise _load_error(exc) from exc
                log.exception("Loading %s (%s) on the GPU failed", self._info.id, variant)
                self._notice = MSG_GPU_LOAD_FAILED
                gc.collect()
                continue
            if cuda and not all(CUDA_PROVIDER in session.get_providers() for session in model.sessions()):
                log.warning("ONNX Runtime didn't run %s on CUDA; trying the CPU", self._info.id)
                self._notice = MSG_GPU_INACTIVE
                del model
                gc.collect()
                continue
            self._model = model
            self._variant = variant
            self._device = CUDA if cuda else CPU
            self._device_name = (gpu.detect_gpu().name or "NVIDIA GPU") if cuda else CPU_NAME
            log.info(
                "Loaded %s (%s files) on %s in %.1f s%s",
                self._info.id,
                variant,
                self._device_name if cuda else f"the CPU ({self._threads} threads)",
                time.perf_counter() - started,
                f"; {self._notice}" if self._notice else "",
            )
            return
        raise EngineError(MSG_LOAD_FAILED)

    def _cuda_obstacle(self, variant: str) -> str | None:
        """Why the GPU can't be used for this variant right now, or None."""
        info = gpu.detect_gpu()
        if not info.cuda_ready:
            return info.detail
        weights = sum(f.size for f in self._info.variant(variant).files if f.path.endswith((".onnx", ".data")))
        factor = WHISPER_VRAM_FACTOR if self._is_whisper else PARAKEET_VRAM_FACTOR
        need_mb = weights * factor / 2**20 + VRAM_MARGIN_MB
        free_mb = hardware.nvidia_free_vram_mb()
        if free_mb is not None and free_mb < need_mb:
            log.warning("Only %d MB of VRAM free; %s needs about %.0f MB", free_mb, self._info.id, need_mb)
            return MSG_GPU_LOW_MEMORY.format(free=free_mb / 1024, need=need_mb / 1024)
        return None

    def _build(self, variant: str, cuda: bool) -> _Recognizer:
        resolver = create_asr_resolver(self._info.onnx_asr_name, self._spec.folders[variant], offline=True)
        model_class = _MODEL_CLASSES.get(resolver.model_type)
        if model_class is None:
            raise EngineError(MSG_LOAD_FAILED)
        files = resolver.resolve_model(quantization=self._info.variant(variant).quantization)
        providers: list[Any] = [(CUDA_PROVIDER, gpu.cuda_provider_options()), CPU_PROVIDER] if cuda else [CPU_PROVIDER]
        if cuda and not gpu.prepare_cuda():
            raise RuntimeError("the NVIDIA runtime libraries aren't available")
        options = {"sess_options": self._session_options(cuda), "providers": providers}
        return model_class(files, _numpy_preprocessor, options)

    def _session_options(self, cuda: bool) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.log_severity_level = 3  # ONNX Runtime's own warnings (memcpy nodes etc.) are noise here
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        # On the GPU only shape arithmetic runs on the CPU; one thread is plenty.
        options.intra_op_num_threads = 1 if cuda else self._threads
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return options

    # -- languages -------------------------------------------------------------------------------

    def _forced_language(self, language: str) -> str | None:
        """The language to make the model use, as the model spells it; None = let it detect."""
        if not self._is_whisper or language == "auto" or language not in self._info.languages:
            return None
        return _WHISPER_SPELLINGS.get(language, language)

    def _result_language(self, language: str, detected: list[str]) -> str | None:
        if not self._info.multilingual:
            return self._info.languages[0]
        if detected:
            code = Counter(detected).most_common(1)[0][0]
            return _ISO_SPELLINGS.get(code, code)
        if language != "auto" and language in self._info.languages:
            return language
        return None


# -- model classes ---------------------------------------------------------------------------------


class _Recognizer(Protocol):
    def sessions(self) -> tuple[ort.InferenceSession, ...]: ...

    def transcribe_one(self, samples: npt.NDArray[np.float32], language: str | None) -> tuple[str, str | None]: ...


class ParakeetTdt(NemoConformerTdt):
    """onnx-asr's NeMo Parakeet TDT model with bug #138 fixed (see the module docstring)."""

    def _encode(
        self, features: npt.NDArray[np.float32], features_lens: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        encoder_out, encoder_out_lens = super()._encode(features, features_lens)
        return encoder_out, np.minimum(encoder_out_lens, encoder_out.shape[1])

    def sessions(self) -> tuple[ort.InferenceSession, ...]:
        return (self._encoder, self._decoder_joint)

    def transcribe_one(self, samples: npt.NDArray[np.float32], language: str | None) -> tuple[str, str | None]:
        lengths = np.array([samples.shape[0]], dtype=np.int64)
        result = next(iter(self.recognize_batch(samples[None, :], lengths)))
        return result.text.strip(), None


class WhisperTurbo(WhisperHf):
    """onnx-asr's Whisper (onnx-community export) with real language detection and a loop guard.

    onnx-asr picks the language with an argmax over the whole vocabulary and
    never reports it; here only language tokens compete, and the winner is
    returned. As in OpenAI's reference decoder, special tokens can't be produced
    as text and a blank or end-of-text can't be the first token.

    Greedy decoding sometimes falls into a loop and copies an earlier passage
    again (seen on 1 of 11 thirty-second LibriSpeech clips, adding 23 wrong
    words). Once the last ``REPEAT_TOKENS`` tokens repeat an earlier run exactly,
    the copy is cut back to where it began and decoding stops. The token count
    is also bounded by the clip length, generously enough for every language Whisper writes.

    One pass writes at most ``MAX_TOKENS`` tokens, which a few seconds of Burmese,
    Malayalam, Georgian or Telugu already fill. A pass that fills them without
    finishing would lose the rest, so its audio is cut in two at its quietest
    moment near the middle and each part is recognised on its own, at most
    ``MAX_SPLITS`` times over. English never comes close to filling a pass.
    """

    MAX_TOKENS = 444
    """Whisper's 448-token window minus the 4-token prompt."""
    REPEAT_TOKENS = 10
    """About seven words: a genuinely repeated phrase in one segment is shorter than this."""
    TOKENS_PER_SECOND = 64
    """Most tokens one second of speech can need. Measured with Whisper's own vocabulary: about 4 for English,
    9 for Arabic, 20 for Hindi, 35-45 for Telugu, Malayalam and Georgian and about 55 for Burmese, whose letters
    it spells out byte by byte. A tighter bound cut those languages off mid-sentence."""
    MAX_SPLITS = 3
    """How many times over a full pass is halved: at most 15 passes for one clip. Three halvings bring Whisp's
    longest segments (28 s) down to passes even Burmese fits in."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._language_ids = np.array(
            sorted(i for token, i in self._tokens.items() if _is_language_token(token)), dtype=np.int64
        )
        self._first_suppressed = np.array([self._tokens.get("Ġ", 220), self._eos_token_id], dtype=np.int64)

    def sessions(self) -> tuple[ort.InferenceSession, ...]:
        return (self._encoder, self._decoder)

    def transcribe_one(self, samples: npt.NDArray[np.float32], language: str | None) -> tuple[str, str | None]:
        return self._transcribe(samples, language, self.MAX_SPLITS)

    def _transcribe(
        self, samples: npt.NDArray[np.float32], language: str | None, splits_left: int
    ) -> tuple[str, str | None]:
        lengths = np.array([samples.shape[0]], dtype=np.int64)
        encoded = self._encode(samples[None, :], lengths)
        if language is None:
            logits, _ = self._decode(np.array([[self._bos_token_id]], dtype=np.int64), self._create_state(), encoded)
            language_id = int(self._language_ids[int(np.argmax(logits[0, -1, self._language_ids]))])
        else:
            language_id = self._tokens[f"<|{language}|>"]
        language = self._vocab[language_id][2:-2]
        tokens = self._transcribe_input.copy()
        tokens[0, 1] = language_id
        state = self._create_state()
        generated: list[int] = []
        limit = min(self.MAX_TOKENS, 16 + int(self.TOKENS_PER_SECOND * samples.shape[0] / SAMPLE_RATE))
        for step in range(limit):
            logits, state = self._decode(tokens, state, encoded)
            scores = logits[0, -1]
            scores[self._eos_token_id + 1 :] = -np.inf  # language, task, timestamp and other special tokens
            if step == 0:
                scores[self._first_suppressed] = -np.inf
            token = int(np.argmax(scores))
            if token == self._eos_token_id:
                break
            generated.append(token)
            copy_start = _repeated_tail(generated, self.REPEAT_TOKENS)
            if copy_start is not None:
                log.info("Whisper repeated itself; dropped %d looping tokens", len(generated) - copy_start)
                del generated[copy_start:]
                break
            tokens = np.concatenate((tokens, np.array([[token]], dtype=np.int64)), axis=1)
        else:
            if limit == self.MAX_TOKENS and splits_left > 0:
                # The pass is full and the speech goes on: recognise the two halves on their own instead.
                cut = _middle_cut(samples)
                seconds = samples.size / SAMPLE_RATE
                log.info("Whisper filled a pass with %.1f s of audio; recognising it in two parts", seconds)
                head, _ = self._transcribe(samples[:cut], language, splits_left - 1)
                tail, _ = self._transcribe(samples[cut:], language, splits_left - 1)
                return join_spoken((head, tail)), language
        text = self._decode_tokens(np.array(generated, dtype=np.int64)).text.strip()
        return text, language


_MODEL_CLASSES: dict[type, type] = {NemoConformerTdt: ParakeetTdt, WhisperHf: WhisperTurbo}
"""onnx-asr model type -> the class Whisp loads it with."""

_WHISPER_SPELLINGS = {"jv": "jw"}
"""ISO code -> Whisper's token, where they differ."""
_ISO_SPELLINGS = {value: key for key, value in _WHISPER_SPELLINGS.items()}
_NOT_LANGUAGES = frozenset(
    {
        "endoftext",
        "startoftranscript",
        "translate",
        "transcribe",
        "startoflm",
        "startofprev",
        "nospeech",
        "nocaptions",
        "notimestamps",
    }
)


def _repeated_tail(tokens: list[int], length: int) -> int | None:
    """Where a copy of earlier text begins, if the last ``length`` tokens repeat an earlier run; else None.

    The match is extended backwards as far as the copy agrees with the original,
    so everything that was copied is removed, not just the last ``length`` tokens.
    """
    end = len(tokens)
    if end < 2 * length:
        return None
    tail = tokens[end - length :]
    last = tail[-1]
    for start in range(end - 2 * length, -1, -1):
        if tokens[start + length - 1] == last and tokens[start : start + length] == tail:
            copy, original = end - length, start
            while original > 0 and copy > start + length and tokens[copy - 1] == tokens[original - 1]:
                copy -= 1
                original -= 1
            return copy
    return None


def _is_language_token(token: str) -> bool:
    if not (token.startswith("<|") and token.endswith("|>")):
        return False
    body = token[2:-2]
    return body not in _NOT_LANGUAGES and body.isalpha() and body.islower() and len(body) <= 3


def _numpy_preprocessor(name: str) -> Any:
    """onnx-asr preprocessor factory: always the NumPy implementation (see the module docstring)."""
    if name.startswith("nemo"):
        return NemoPreprocessorNumpy(name)
    if name.startswith("whisper"):
        return WhisperPreprocessorNumpy(name)
    raise EngineError(MSG_LOAD_FAILED)


def _load_error(exc: Exception) -> EngineError:
    if isinstance(exc, EngineError):
        return exc
    log.error("Loading the speech model failed: %s: %s", type(exc).__name__, exc)
    text = str(exc).lower()
    if isinstance(exc, MemoryError) or "bad allocation" in text or "out of memory" in text:
        return EngineError(MSG_NO_MEMORY)
    if isinstance(exc, (FileNotFoundError, ModelLoadingError)):
        return EngineError(MSG_FILES_MISSING)
    return EngineError(MSG_LOAD_FAILED)


# -- audio helpers -----------------------------------------------------------------------------------


def _clean_audio(samples: np.ndarray) -> npt.NDArray[np.float32]:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(audio)


def _pad(audio: npt.NDArray[np.float32], minimum: int) -> npt.NDArray[np.float32]:
    if audio.size >= minimum:
        return audio
    return np.concatenate((audio, np.zeros(minimum - audio.size, dtype=np.float32)))


def _split_long(audio: npt.NDArray[np.float32], max_samples: int, search: int) -> list[npt.NDArray[np.float32]]:
    """Cut audio longer than ``max_samples`` at its quietest moments, so no word is split if it can be helped."""
    pieces: list[npt.NDArray[np.float32]] = []
    start = 0
    while audio.size - start > max_samples:
        window_end = start + max_samples
        window_start = max(start + 1, window_end - search)
        cut = window_start + _quietest(audio[window_start:window_end])
        pieces.append(audio[start:cut])
        start = cut
    pieces.append(audio[start:])
    return pieces


def _quietest(audio: npt.NDArray[np.float32], frame: int = SAMPLE_RATE // 50) -> int:
    """Offset of the middle of the quietest 20 ms frame."""
    count = audio.size // frame
    if count == 0:
        return audio.size // 2
    energy = np.square(audio[: count * frame].reshape(count, frame)).mean(axis=1)
    return int(np.argmin(energy)) * frame + frame // 2


def _middle_cut(audio: npt.NDArray[np.float32], frame: int = SAMPLE_RATE // 50) -> int:
    """Where to cut audio in two: the middle of its quietest 20 ms in the middle third (nearest the centre on a tie),
    so no word is split and each part is at least a third of the whole."""
    start = audio.size // 3
    count = (audio.size - 2 * start) // frame
    if count == 0:
        return audio.size // 2
    energy = np.square(audio[start : start + count * frame].reshape(count, frame)).mean(axis=1)
    quietest = np.flatnonzero(energy == energy.min())
    centre = (audio.size / 2 - start) / frame - 0.5
    return start + int(quietest[np.argmin(np.abs(quietest - centre))]) * frame + frame // 2


def _speech_like(seconds: float, seed: int) -> npt.NDArray[np.float32]:
    """Deterministic voice-like sound for warming up: a buzzing, gliding pitch in syllable-sized bursts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    pitch = 110.0 + 25.0 * np.sin(2 * np.pi * 0.6 * t + seed)
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    voice = sum(np.sin(k * phase) * (0.8**k) for k in range(1, 16))
    syllables = np.clip(np.sin(2 * np.pi * 3.5 * t), 0.0, None) ** 0.5
    audio = 0.08 * voice * syllables + 0.003 * rng.standard_normal(n)
    return audio.astype(np.float32)
