"""The speech recognition models Whisp supports, pinned to exact files.

Every model points at one commit of its Hugging Face repo, and every file is
listed with its exact size and SHA-256 (the LFS object id for large files, a
hash computed from the downloaded file for small ones). Downloads are checked
against these values, so what runs on a user's machine is exactly what was
tested.

Each model comes in two variants:

``gpu``
    Full-precision weights for NVIDIA GPUs: fp32 for Parakeet, fp16 for Whisper
    (half the download of its fp32 export, with the same transcripts in testing).
    They also run on the CPU if the GPU can't be used, just more slowly than ``cpu``.
``cpu``
    Weights quantized to int8: about a quarter of the download and much faster
    on a CPU, at a tiny cost in accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFile:
    path: str
    """Path relative to the repo root (and to the model's install folder)."""
    size: int
    """Bytes."""
    sha256: str


@dataclass(frozen=True)
class ModelVariant:
    key: str
    """``"gpu"`` or ``"cpu"``."""
    quantization: str | None
    """Passed to onnx-asr to pick the weight files: ``None`` (fp32) or e.g. ``"int8"``."""
    files: tuple[ModelFile, ...]

    @property
    def size(self) -> int:
        """Total download size in bytes."""
        return sum(f.size for f in self.files)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    """Stable id used in settings."""
    onnx_asr_name: str
    """onnx-asr model name or type, which selects the model implementation."""
    name: str
    """Display name."""
    description: str
    """One plain-English line for the model picker."""
    languages: tuple[str, ...]
    """ISO language codes; ``("en",)`` for English-only models."""
    accuracy: int
    """1-5, for the UI."""
    speed: int
    """1-5, for the UI."""
    license: str
    license_url: str
    repo_id: str
    revision: str
    """Pinned commit of ``repo_id``."""
    variants: tuple[ModelVariant, ...]
    recommended: bool
    attribution: str = ""
    """Credit line for the About page, as the model's licence asks."""

    def variant(self, key: str) -> ModelVariant:
        """The variant called ``key``. Raises ``KeyError`` if this model doesn't have it."""
        for variant in self.variants:
            if variant.key == key:
                return variant
        raise KeyError(f"{self.id} has no {key!r} variant")

    @property
    def multilingual(self) -> bool:
        return len(self.languages) > 1


CC_BY_4 = "CC-BY-4.0"
CC_BY_4_URL = "https://creativecommons.org/licenses/by/4.0/"


def _codes(text: str) -> tuple[str, ...]:
    """``"en de fr"`` -> ``("en", "de", "fr")``: keeps long language lists compact."""
    return tuple(text.split())


# Files shared by every Parakeet export from istupakov: the onnx-asr config and the vocabulary.
_PARAKEET_CONFIG = ModelFile("config.json", 97, "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466")

PARAKEET_V2 = ModelInfo(
    id="parakeet-tdt-0.6b-v2",
    onnx_asr_name="nemo-parakeet-tdt-0.6b-v2",
    name="Parakeet TDT 0.6B v2",
    description="NVIDIA's best English model: very accurate, punctuated text, and fast even without a GPU.",
    languages=("en",),
    accuracy=5,
    speed=5,
    license=CC_BY_4,
    license_url=CC_BY_4_URL,
    repo_id="istupakov/parakeet-tdt-0.6b-v2-onnx",
    revision="0bbb45a3365852604aef28b538a8f066f4ccaa85",
    variants=(
        ModelVariant(
            key="gpu",
            quantization=None,
            files=(
                _PARAKEET_CONFIG,
                ModelFile("vocab.txt", 9384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
                ModelFile(
                    "encoder-model.onnx", 41770866, "3987bcd28175d829d12888a996a84e8f62a0e374d9ffd640662c1515adc679d3"
                ),
                ModelFile(
                    "encoder-model.onnx.data",
                    2435420160,
                    "4dab7362d4874d85965045b1e41b2d61dd2cc0fb25671a7f6b3dc47bf120cc41",
                ),
                ModelFile(
                    "decoder_joint-model.onnx",
                    35792059,
                    "cbb52a07bd70ab5b67f8439d4b3cd8704b18467b4430bcacb5adabe154b8d191",
                ),
            ),
        ),
        ModelVariant(
            key="cpu",
            quantization="int8",
            files=(
                _PARAKEET_CONFIG,
                ModelFile("vocab.txt", 9384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
                ModelFile(
                    "encoder-model.int8.onnx",
                    652184014,
                    "3e0581fda6ab843888b51e56d7ee78b6d5bc3237ec113af1f732d1d5286aa155",
                ),
                ModelFile(
                    "decoder_joint-model.int8.onnx",
                    8998286,
                    "a449f49acd68979d418651dd2dcb737cc0f1bf0225e009e29ee326354edbf7d3",
                ),
            ),
        ),
    ),
    recommended=True,
    attribution=(
        "Parakeet TDT 0.6B v2 by NVIDIA (huggingface.co/nvidia/parakeet-tdt-0.6b-v2), licensed under CC BY 4.0. "
        "Converted to ONNX by Ilya Stupakov (huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx). "
        "Whisp uses the converted model without changes."
    ),
)

PARAKEET_V3 = ModelInfo(
    id="parakeet-tdt-0.6b-v3",
    onnx_asr_name="nemo-parakeet-tdt-0.6b-v3",
    name="Parakeet TDT 0.6B v3",
    description="The same fast Parakeet for 25 European languages. It works out which one you're speaking.",
    languages=_codes("en bg cs da de el es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk"),
    accuracy=4,
    speed=5,
    license=CC_BY_4,
    license_url=CC_BY_4_URL,
    repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
    revision="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    variants=(
        ModelVariant(
            key="gpu",
            quantization=None,
            files=(
                _PARAKEET_CONFIG,
                ModelFile("vocab.txt", 93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
                ModelFile(
                    "encoder-model.onnx", 41770866, "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1"
                ),
                ModelFile(
                    "encoder-model.onnx.data",
                    2435420160,
                    "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36",
                ),
                ModelFile(
                    "decoder_joint-model.onnx",
                    72520893,
                    "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1",
                ),
            ),
        ),
        ModelVariant(
            key="cpu",
            quantization="int8",
            files=(
                _PARAKEET_CONFIG,
                ModelFile("vocab.txt", 93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
                ModelFile(
                    "encoder-model.int8.onnx",
                    652183999,
                    "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
                ),
                ModelFile(
                    "decoder_joint-model.int8.onnx",
                    18202004,
                    "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
                ),
            ),
        ),
    ),
    recommended=False,
    attribution=(
        "Parakeet TDT 0.6B v3 by NVIDIA (huggingface.co/nvidia/parakeet-tdt-0.6b-v3), licensed under CC BY 4.0. "
        "Converted to ONNX by Ilya Stupakov (huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx). "
        "Whisp uses the converted model without changes."
    ),
)

_WHISPER_TOKENIZER = (
    ModelFile("config.json", 1332, "35cd83669f75bc2867f3b3a4461850392d5e308cd6ea951c3700539883c28df1"),
    ModelFile("vocab.json", 1036558, "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387"),
    ModelFile("added_tokens.json", 34648, "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0"),
)

WHISPER_TURBO = ModelInfo(
    id="whisper-large-v3-turbo",
    onnx_asr_name="whisper",
    name="Whisper Large v3 Turbo",
    description="OpenAI's Whisper for 99 languages, including Asian and Middle Eastern ones. Best with an NVIDIA GPU.",
    # Whisper's own language tokens, in its order, as ISO codes (Whisper writes Javanese as "jw").
    languages=_codes(
        "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la "
        "mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be "
        "tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jv su yue"
    ),
    accuracy=4,
    speed=2,
    license="MIT",
    license_url="https://github.com/openai/whisper/blob/main/LICENSE",
    repo_id="onnx-community/whisper-large-v3-turbo",
    revision="360ebcde2559d60bb474678be3c1de9ef347d01a",
    variants=(
        ModelVariant(
            key="gpu",
            quantization="fp16",
            files=(
                *_WHISPER_TOKENIZER,
                ModelFile(
                    "onnx/encoder_model_fp16.onnx",
                    1274342603,
                    "fdadc70836e6b028fd5e580417c312208dad073d2d01e509e2d127c1373399d8",
                ),
                ModelFile(
                    "onnx/decoder_model_merged_fp16.onnx",
                    344227339,
                    "fdf10afca73a0c7bf87286cfb96cf7028a9edbc9bb02512509a526f95b126c9d",
                ),
            ),
        ),
        ModelVariant(
            key="cpu",
            quantization="int8",
            files=(
                *_WHISPER_TOKENIZER,
                ModelFile(
                    "onnx/encoder_model_int8.onnx",
                    644822094,
                    "e44c0d5cfcc6ad283011602a738fa28dfa1ad7f7540c9503205479072a9cc1ef",
                ),
                ModelFile(
                    "onnx/decoder_model_merged_int8.onnx",
                    439936716,
                    "61481bd3be3a445d5a4b9070e8f8b2c6cc4fbbbbdc9f0e7ed048a132b8b84e0d",
                ),
            ),
        ),
    ),
    recommended=False,
    attribution=(
        "Whisper Large v3 Turbo by OpenAI (github.com/openai/whisper), licensed under the MIT License. "
        "ONNX conversion by the ONNX Community (huggingface.co/onnx-community/whisper-large-v3-turbo)."
    ),
)

CATALOG: tuple[ModelInfo, ...] = (PARAKEET_V2, PARAKEET_V3, WHISPER_TURBO)


def get_model(model_id: str) -> ModelInfo:
    """The catalog entry for ``model_id``. Raises ``KeyError`` for unknown ids."""
    for info in CATALOG:
        if info.id == model_id:
            return info
    raise KeyError(f"unknown speech model: {model_id!r}")
