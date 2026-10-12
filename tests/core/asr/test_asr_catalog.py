"""The speech model catalog: pinned, complete and consistent with onnx-asr and the settings."""

from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath

import pytest
from onnx_asr.loader import create_asr_resolver

from whisp.core.asr.catalog import CATALOG, ModelInfo, get_model
from whisp.storage.settings import TranscriptionSettings

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(params=CATALOG, ids=lambda info: info.id)
def model(request: pytest.FixtureRequest) -> ModelInfo:
    return request.param


def test_ids_are_unique_and_lookup_works() -> None:
    ids = [info.id for info in CATALOG]
    assert len(ids) == len(set(ids))
    for info in CATALOG:
        assert get_model(info.id) is info
    with pytest.raises(KeyError):
        get_model("no-such-model")


def test_the_default_setting_is_the_one_recommended_model() -> None:
    default = TranscriptionSettings().model_id
    assert [info.id for info in CATALOG if info.recommended] == [default]
    assert get_model(default).languages == ("en",)


def test_display_fields(model: ModelInfo) -> None:
    assert model.name and model.description and model.description.endswith(".")
    assert 1 <= model.accuracy <= 5 and 1 <= model.speed <= 5
    assert model.license and model.license_url.startswith("https://")
    assert model.attribution


def test_revision_is_a_pinned_commit(model: ModelInfo) -> None:
    assert HEX40.match(model.revision)
    assert "/" in model.repo_id


def test_languages_are_codes(model: ModelInfo) -> None:
    assert model.languages[0] == "en"
    assert len(set(model.languages)) == len(model.languages)
    assert all(re.fullmatch(r"[a-z]{2,3}", code) for code in model.languages)


def test_language_counts() -> None:
    assert get_model("parakeet-tdt-0.6b-v2").languages == ("en",)
    assert len(get_model("parakeet-tdt-0.6b-v3").languages) == 25
    whisper = get_model("whisper-large-v3-turbo")
    assert len(whisper.languages) == 100  # Whisper's 99 languages plus Cantonese
    assert "jv" in whisper.languages and "jw" not in whisper.languages


def test_each_model_has_a_gpu_and_a_cpu_variant(model: ModelInfo) -> None:
    assert [v.key for v in model.variants] == ["gpu", "cpu"]
    assert model.variant("cpu").quantization == "int8"
    assert model.variant("gpu").quantization in (None, "fp16")
    assert model.variant("cpu").size < model.variant("gpu").size
    with pytest.raises(KeyError):
        model.variant("tpu")


def test_files_are_pinned_and_safe(model: ModelInfo) -> None:
    for variant in model.variants:
        paths = [f.path for f in variant.files]
        assert len(paths) == len(set(paths))
        for item in variant.files:
            assert item.size > 0
            assert HEX64.match(item.sha256), item.path
            path = PurePosixPath(item.path)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in item.path


def test_onnx_asr_finds_every_file_it_needs(model: ModelInfo, tmp_path) -> None:
    """onnx-asr's own file patterns for the model type must match exactly one listed file each."""
    model_type = create_asr_resolver(model.onnx_asr_name, tmp_path, offline=True).model_type
    for variant in model.variants:
        listed = [f.path for f in variant.files]
        patterns = model_type._get_model_files(variant.quantization)
        for key, pattern in patterns.items():
            matches = [p for p in listed if _glob_matches(pattern, p)]
            assert len(matches) == 1, f"{model.id} {variant.key}: {key} pattern {pattern!r} matched {matches}"
        assert "config.json" in listed, "the config sets the feature size (128 mel bins)"


def test_external_weights_are_listed_next_to_their_graphs() -> None:
    parakeet = get_model("parakeet-tdt-0.6b-v2").variant("gpu")
    assert {f.path for f in parakeet.files} >= {"encoder-model.onnx", "encoder-model.onnx.data"}


def test_attribution_credits_the_authors() -> None:
    for model_id in ("parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3"):
        info = get_model(model_id)
        assert info.license == "CC-BY-4.0"
        assert "NVIDIA" in info.attribution and "istupakov" in info.attribution
        assert "CC BY 4.0" in info.attribution
    whisper = get_model("whisper-large-v3-turbo")
    assert whisper.license == "MIT" and "OpenAI" in whisper.attribution


def _glob_matches(pattern: str, path: str) -> bool:
    """Path.glob semantics for onnx-asr's patterns: ``?`` is one character, ``**/`` any folder prefix."""
    if pattern.startswith("**/"):
        name = pattern[3:]
        return fnmatch.fnmatchcase(PurePosixPath(path).name, name)
    return "/" not in path and fnmatch.fnmatchcase(path, pattern)
