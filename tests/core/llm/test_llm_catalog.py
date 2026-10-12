"""The pinned local-model catalog and server builds."""

from __future__ import annotations

import re

import pytest

from whisp.core.llm import runtime
from whisp.core.llm.catalog import CATALOG, get_model, recommended_model

SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def test_ids_are_unique_and_one_model_is_recommended() -> None:
    ids = [m.id for m in CATALOG]
    assert len(ids) == len(set(ids))
    assert [m.id for m in CATALOG if m.recommended] == ["qwen3.5-4b"]
    assert recommended_model().id == "qwen3.5-4b"
    assert CATALOG[0].recommended  # the UI and the runtime's fallback list it first


@pytest.mark.parametrize("model", CATALOG, ids=lambda m: m.id)
def test_every_model_is_pinned_and_verified(model) -> None:
    assert model.name and model.description and len(model.description) < 140
    assert model.license == "Apache-2.0"
    assert model.license_url.startswith("https://huggingface.co/")
    assert model.vram_mb > 1000 and model.ram_mb > 1000
    for f in model.files:
        assert COMMIT.fullmatch(f.revision), f.revision
        assert SHA256.fullmatch(f.sha256), f.sha256
        assert f.size > 50_000_000
        assert f.path.endswith(".gguf")
        assert re.fullmatch(r"[\w.-]+/[\w.-]+", f.repo_id)
        assert f.url == f"https://huggingface.co/{f.repo_id}/resolve/{f.revision}/{f.path}"
    assert model.download_size == sum(f.size for f in model.files)
    needs_draft = any("{draft}" in a for a in model.server_args)
    assert needs_draft == (model.draft_file is not None)
    assert "--spec-type" in model.server_args  # every model ships with multi-token prediction


def test_get_model() -> None:
    assert get_model("qwen3.5-2b").name == "Qwen3.5 2B"
    with pytest.raises(KeyError):
        get_model("nope")


def test_server_builds_are_pinned() -> None:
    assert re.fullmatch(r"b\d+", runtime.LLAMA_BUILD)
    assert set(runtime.SERVER_FLAVORS) == {"cuda-13.4", "cuda-12.4", "vulkan", "cpu"}
    for key, flavor in runtime.SERVER_FLAVORS.items():
        assert flavor.key == key
        assert flavor.label
        assert flavor.assets
        main = flavor.assets[0]
        assert main.name == f"llama-{runtime.LLAMA_BUILD}-bin-win-{key}-x64.zip"
        for asset in flavor.assets:
            assert SHA256.fullmatch(asset.sha256)
            assert asset.size > 10_000_000
            assert asset.url == (
                f"https://github.com/ggml-org/llama.cpp/releases/download/{runtime.LLAMA_BUILD}/{asset.name}"
            )
        if key.startswith("cuda"):
            assert flavor.assets[1].name == f"cudart-llama-bin-win-{key}-x64.zip"
