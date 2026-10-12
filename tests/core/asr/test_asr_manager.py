"""ModelManager with a tiny fake model served locally: install, validate, cancel, resume, delete, in-use rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from whisp.core.asr import downloads
from whisp.core.asr.catalog import ModelFile, ModelInfo, ModelVariant
from whisp.core.asr.downloads import part_path
from whisp.core.asr.manager import MANIFEST_NAME, MSG_IN_USE, ModelInUseError, ModelManager

REVISION = "0123456789abcdef0123456789abcdef01234567"
WAIT_MS = 15_000


def payload(size: int, seed: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


@pytest.fixture
def tiny(file_server) -> ModelInfo:
    """A two-variant model whose files the local server holds."""
    contents = {
        "config.json": b'{"model_type": "nemo-conformer-tdt"}',
        "vocab.txt": b"a 0\nb 1\n<blk> 2\n",
        "weights.onnx": payload(600_000, 1),
        "weights.int8.onnx": payload(150_000, 2),
    }
    files = {}
    for name, data in contents.items():
        file_server.add(f"/org/tiny/resolve/{REVISION}/{name}", data)
        files[name] = ModelFile(name, len(data), hashlib.sha256(data).hexdigest())
    return ModelInfo(
        id="tiny",
        onnx_asr_name="nemo-conformer-tdt",
        name="Tiny",
        description="A test model.",
        languages=("en",),
        accuracy=3,
        speed=3,
        license="MIT",
        license_url="https://example.invalid/license",
        repo_id="org/tiny",
        revision=REVISION,
        variants=(
            ModelVariant("gpu", None, (files["config.json"], files["vocab.txt"], files["weights.onnx"])),
            ModelVariant("cpu", "int8", (files["config.json"], files["vocab.txt"], files["weights.int8.onnx"])),
        ),
        recommended=True,
    )


@pytest.fixture
def manager(tiny: ModelInfo, file_server, tmp_path: Path) -> ModelManager:
    return ModelManager(tmp_path / "models", catalog=(tiny,), base_url=file_server.base_url)


def install(qtbot: QtBot, manager: ModelManager, variant: str) -> None:
    manager.download("tiny", variant)
    qtbot.waitUntil(lambda: not manager.is_downloading("tiny"), timeout=WAIT_MS)
    assert manager.is_installed("tiny", variant)


def test_download_installs_the_variant_and_reports_progress(qtbot: QtBot, manager: ModelManager, tiny) -> None:
    progress: list[tuple[str, object, object]] = []
    changes: list[None] = []
    manager.progress.connect(lambda *args: progress.append(args))
    manager.changed.connect(lambda: changes.append(None))
    assert not manager.is_installed("tiny", "gpu")

    install(qtbot, manager, "gpu")

    folder = manager.model_path("tiny", "gpu")
    assert folder == manager.models_dir / "tiny" / "gpu"
    expected = ["config.json", "vocab.txt", "weights.onnx", MANIFEST_NAME]
    assert sorted(p.name for p in folder.iterdir()) == sorted(expected)
    qtbot.waitUntil(lambda: len(changes) >= 2, timeout=WAIT_MS)  # started, finished
    total = tiny.variant("gpu").size
    qtbot.waitUntil(lambda: bool(progress) and progress[-1][1] == total, timeout=WAIT_MS)
    assert all(model_id == "tiny" and type(done) is int and t == total for model_id, done, t in progress)
    assert not manager.is_installed("tiny", "cpu")
    assert manager.installed_variants("tiny") == ["gpu"]
    state = manager.install_state()["tiny"]
    assert state == {"installed": ["gpu"], "downloading": False, "variant": None, "progress": None, "in_use": False}


def test_unknown_models_and_variants_raise_key_error(manager: ModelManager) -> None:
    with pytest.raises(KeyError):
        manager.is_installed("nope", "gpu")
    with pytest.raises(KeyError):
        manager.is_installed("tiny", "tpu")
    with pytest.raises(KeyError):
        manager.download("nope", "gpu")
    with pytest.raises(KeyError):
        manager.model_info("nope")


def test_a_folder_without_its_manifest_is_not_installed(qtbot: QtBot, manager: ModelManager) -> None:
    install(qtbot, manager, "gpu")
    (manager.model_path("tiny", "gpu") / MANIFEST_NAME).unlink()
    assert not manager.is_installed("tiny", "gpu")


def test_missing_or_resized_files_make_it_not_installed(qtbot: QtBot, manager: ModelManager) -> None:
    install(qtbot, manager, "gpu")
    weights = manager.model_path("tiny", "gpu") / "weights.onnx"
    weights.write_bytes(weights.read_bytes()[:-1])
    assert not manager.is_installed("tiny", "gpu")
    weights.unlink()
    assert not manager.is_installed("tiny", "gpu")


def test_a_manifest_from_another_revision_is_not_installed(qtbot: QtBot, manager: ModelManager) -> None:
    install(qtbot, manager, "gpu")
    manifest_path = manager.model_path("tiny", "gpu") / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "f" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not manager.is_installed("tiny", "gpu")


def test_reinstalling_repairs_a_damaged_install(qtbot: QtBot, manager: ModelManager, file_server) -> None:
    install(qtbot, manager, "gpu")
    weights = manager.model_path("tiny", "gpu") / "weights.onnx"
    weights.write_bytes(b"\0" * weights.stat().st_size)  # same size, so it still looks installed
    (manager.model_path("tiny", "gpu") / MANIFEST_NAME).unlink()
    file_server.requests.clear()

    install(qtbot, manager, "gpu")

    paths = [path for path, _ in file_server.requests]
    assert paths == [f"/org/tiny/resolve/{REVISION}/weights.onnx"]  # only the damaged file was fetched again


def test_cancel_stops_and_discards_the_download(qtbot: QtBot, manager: ModelManager, file_server) -> None:
    file_server.chunk_delay = 0.005
    progress: list[int] = []
    manager.progress.connect(lambda _id, done, _total: progress.append(done))

    manager.download("tiny", "gpu")
    qtbot.waitUntil(lambda: bool(progress) and progress[-1] > 50_000, timeout=WAIT_MS)
    manager.cancel("tiny")
    qtbot.waitUntil(lambda: not manager.is_downloading("tiny"), timeout=WAIT_MS)

    assert not manager.is_installed("tiny", "gpu")
    assert not (manager.models_dir / "tiny").exists()


def test_a_failed_download_reports_a_message_and_keeps_partial_data(
    qtbot: QtBot, manager: ModelManager, file_server, fast_retries, monkeypatch
) -> None:
    monkeypatch.setattr(downloads, "MAX_ATTEMPTS", 1)
    weights_path = f"/org/tiny/resolve/{REVISION}/weights.onnx"
    file_server.drop_after[weights_path] = 200_000
    failures: list[tuple[str, str]] = []
    manager.download_failed.connect(lambda model_id, message: failures.append((model_id, message)))

    manager.download("tiny", "gpu")
    qtbot.waitUntil(lambda: bool(failures), timeout=WAIT_MS)

    assert failures == [("tiny", downloads.MSG_DROPPED)]
    assert not manager.is_installed("tiny", "gpu")
    partial = part_path(manager.model_path("tiny", "gpu") / "weights.onnx")
    assert partial.stat().st_size == 200_000

    file_server.requests.clear()
    install(qtbot, manager, "gpu")
    assert file_server.gets(weights_path) == ["bytes=200000-"]  # resumed, not restarted


def test_a_missing_file_on_the_server_fails_the_download(qtbot: QtBot, manager: ModelManager, file_server) -> None:
    del file_server.files[f"/org/tiny/resolve/{REVISION}/vocab.txt"]
    failures: list[str] = []
    manager.download_failed.connect(lambda _id, message: failures.append(message))

    manager.download("tiny", "cpu")
    qtbot.waitUntil(lambda: bool(failures), timeout=WAIT_MS)

    assert "error 404" in failures[0]
    assert not manager.is_installed("tiny", "cpu")


def test_downloading_an_installed_variant_does_nothing(qtbot: QtBot, manager: ModelManager, file_server) -> None:
    install(qtbot, manager, "cpu")
    file_server.requests.clear()
    manager.download("tiny", "cpu")
    assert not manager.is_downloading("tiny")
    assert file_server.requests == []


def test_delete_removes_every_variant(qtbot: QtBot, manager: ModelManager) -> None:
    install(qtbot, manager, "gpu")
    install(qtbot, manager, "cpu")
    changes: list[None] = []
    manager.changed.connect(lambda: changes.append(None))

    manager.delete("tiny")

    assert not (manager.models_dir / "tiny").exists()
    assert manager.installed_variants("tiny") == []
    assert changes == [None]


def test_delete_refuses_a_model_in_use(qtbot: QtBot, manager: ModelManager) -> None:
    install(qtbot, manager, "gpu")
    assert manager.retain("tiny", "gpu")
    assert manager.in_use("tiny")
    assert manager.install_state()["tiny"]["in_use"]

    with pytest.raises(ModelInUseError) as caught:
        manager.delete("tiny")
    assert isinstance(caught.value, OSError)  # callers that handle OSError handle this too
    assert str(caught.value) == MSG_IN_USE
    assert manager.is_installed("tiny", "gpu")

    manager.release("tiny")
    manager.delete("tiny")
    assert not manager.is_installed("tiny", "gpu")


def test_retain_refuses_a_model_that_is_not_installed(manager: ModelManager) -> None:
    assert not manager.retain("tiny", "gpu")
    assert not manager.in_use("tiny")


def test_delete_cancels_a_running_download(qtbot: QtBot, manager: ModelManager, file_server) -> None:
    file_server.chunk_delay = 0.005
    progress: list[int] = []
    manager.progress.connect(lambda _id, done, _total: progress.append(done))
    manager.download("tiny", "gpu")
    qtbot.waitUntil(lambda: bool(progress), timeout=WAIT_MS)

    manager.delete("tiny")

    assert not manager.is_downloading("tiny")
    assert not (manager.models_dir / "tiny").exists()


def test_install_state_reports_a_running_download(qtbot: QtBot, manager: ModelManager, file_server, tiny) -> None:
    file_server.chunk_delay = 0.005
    manager.download("tiny", "gpu")
    qtbot.waitUntil(lambda: manager.install_state()["tiny"]["progress"]["done"] > 0, timeout=WAIT_MS)
    state = manager.install_state()["tiny"]
    assert state["downloading"] and state["variant"] == "gpu"
    assert state["progress"]["total"] == tiny.variant("gpu").size
    manager.cancel("tiny")
    qtbot.waitUntil(lambda: not manager.is_downloading("tiny"), timeout=WAIT_MS)


def test_the_real_catalog_is_the_default(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path)
    assert manager.model_info("parakeet-tdt-0.6b-v2").repo_id == "istupakov/parakeet-tdt-0.6b-v2-onnx"
    assert not manager.is_installed("parakeet-tdt-0.6b-v2", "gpu")
