"""LlmRuntime: build choice, launch arguments, ports, the job object, and install/status handling."""

from __future__ import annotations

import hashlib
import io
import socket
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from whisp.core import hardware
from whisp.core.llm import catalog
from whisp.core.llm import runtime as rt
from whisp.core.llm.catalog import LlmFile, LlmModelInfo, get_model
from whisp.storage.settings import SettingsStore

# -- choosing a build ---------------------------------------------------------------------


def hw(
    *,
    nvidia: tuple[str, int] | None = None,
    gpus: tuple[tuple[str, int], ...] = (),
    cores: int = 8,
) -> hardware.Hardware:
    adapters = tuple(hardware.GpuAdapter(vendor=v, name=f"{v} gpu", vram_mb=mb) for v, mb in gpus)
    info = None
    if nvidia is not None:
        version, vram = nvidia
        info = hardware.NvidiaInfo(
            driver_version=version,
            driver_major=int(version.split(".")[0]),
            name="NVIDIA GeForce",
            vram_total_mb=vram,
            vram_free_mb=vram,
        )
        adapters = (*adapters, hardware.GpuAdapter("nvidia", "NVIDIA GeForce", vram))
    return hardware.Hardware(gpus=adapters, nvidia=info, logical_cpus=2 * cores, performance_cores=cores)


@pytest.mark.parametrize(
    ("machine", "cc", "expected"),
    [
        (hw(nvidia=("617.42", 12282)), (8, 9), "cuda-13.4"),  # RTX 4070, current driver
        (hw(nvidia=("580.88", 8192)), (12, 0), "cuda-13.4"),  # RTX 50xx needs CUDA 13
        (hw(nvidia=("617.42", 12282)), None, "cuda-13.4"),  # compute capability unknown
        (hw(nvidia=("581.15", 8192)), (6, 1), "cuda-12.4"),  # GTX 1080 on the last driver branch for Pascal
        (hw(nvidia=("560.94", 8192)), (8, 6), "cuda-12.4"),  # older driver
        (hw(nvidia=("527.41", 6144)), (7, 5), "cuda-12.4"),  # oldest driver CUDA 12 runs on
        (hw(nvidia=("527.37", 6144)), (7, 5), "vulkan"),  # too old for CUDA 12
        (hw(nvidia=("472.12", 4096)), (6, 1), "vulkan"),
        (hw(nvidia=("600.10", 2048)), (3, 5), "vulkan"),  # Kepler: no CUDA build supports it
        (hw(gpus=(("amd", 16368),)), None, "vulkan"),
        (hw(gpus=(("intel", 16384),)), None, "vulkan"),  # Arc
        (hw(gpus=(("intel", 128),)), None, "cpu"),  # integrated graphics only
        (hw(gpus=(("nvidia", 8192),)), None, "vulkan"),  # NVIDIA card but no driver tools (NVML)
        (hw(), None, "cpu"),
    ],
)
def test_choose_flavor(machine: hardware.Hardware, cc: tuple[int, int] | None, expected: str) -> None:
    assert rt.choose_flavor(machine, cc) == expected


def test_this_machine_gets_a_known_build() -> None:
    assert rt.detect_flavor() in rt.SERVER_FLAVORS


# -- launch arguments ------------------------------------------------------------------------


def test_server_args_for_a_gpu() -> None:
    model = get_model("qwen3.5-4b")
    args = rt.build_server_args(
        model=model,
        model_path=Path(r"C:\models\Qwen3.5-4B-Q4_K_M.gguf"),
        draft_path=None,
        port=51234,
        flavor="cuda-13.4",
        threads=8,
        log_file=Path(r"C:\logs\llama-server.log"),
    )
    assert args == [
        "-m", r"C:\models\Qwen3.5-4B-Q4_K_M.gguf",
        "--host", "127.0.0.1",
        "--port", "51234",
        "-c", "8192",
        "-np", "1",
        "-fa", "auto",
        "--reasoning", "off",
        "--cache-ram", "1024",
        "--no-webui",
        "--no-slots",
        "--cors-origins", "localhost",
        "--offline",
        "--log-file", r"C:\logs\llama-server.log",
        "--log-colors", "off",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "4",
    ]  # fmt: skip


def test_server_args_for_the_cpu_and_a_draft_model() -> None:
    model = get_model("gemma-4-e4b")
    args = rt.build_server_args(
        model=model,
        model_path=Path("m.gguf"),
        draft_path=Path("d.gguf"),
        port=1,
        flavor="cpu",
        threads=6,
        log_file=Path("l.log"),
    )
    assert args[args.index("--model-draft") + 1] == "d.gguf"
    assert args[-2:] == ["-t", "6"]
    with pytest.raises(ValueError):
        rt.build_server_args(
            model=model, model_path=Path("m"), draft_path=None, port=1, flavor="cpu", threads=1, log_file=Path("l")
        )


def test_server_environment_is_an_allowlist() -> None:
    engine = Path(r"C:\Whisp\runtime\llama\cpu")
    env = rt.server_environment(
        "per-launch-key",
        engine,
        {
            "SYSTEMROOT": r"C:\Windows",
            "TEMP": r"C:\Temp",
            "TMP": r"C:\Temp",
            "LOCALAPPDATA": r"C:\Users\sam\AppData\Local",
            "CUDA_VISIBLE_DEVICES": "1",
            "PATH": r"C:\Tools;C:\Windows",
            "GGML_BACKEND_PATH": r"C:\Downloads\evil.dll",  # would load a DLL into the server
            "AIP_HTTP_PORT": "80",  # would move the server's port
            "VK_INSTANCE_LAYERS": "VK_LAYER_evil",
            "LLAMA_ARG_MMPROJ": "x",
            "LLAMA_API_KEY": "old",
            "HF_TOKEN": "t",
            "Other": "1",
        },
    )
    assert env == {
        "SystemRoot": r"C:\Windows",
        "TEMP": r"C:\Temp",
        "TMP": r"C:\Temp",
        "LOCALAPPDATA": r"C:\Users\sam\AppData\Local",
        "CUDA_VISIBLE_DEVICES": "1",
        "PATH": rf"{engine};C:\Windows\System32;C:\Windows",
        "LLAMA_API_KEY": "per-launch-key",
    }


def test_find_free_port() -> None:
    port = rt.find_free_port()
    assert 1024 <= port <= 65535
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))  # really free


# -- the job object ------------------------------------------------------------------------


def test_child_runs_and_dies_with_the_job(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    script = f"import pathlib, time; pathlib.Path(r'{marker}').write_text('ran'); time.sleep(60)"
    job = rt.JobObject()
    proc = rt.spawn_in_job(job, [sys.executable, "-c", script], cwd=tmp_path, env=None)
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.read_text() == "ran"  # it was resumed after joining the job
        assert proc.poll() is None
        job.close()  # what Windows does when Whisp exits or crashes
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
        job.close()


def test_job_terminate_kills_every_process(tmp_path: Path) -> None:
    job = rt.JobObject()
    procs = [
        rt.spawn_in_job(job, [sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, env=None)
        for _ in range(2)
    ]
    try:
        job.terminate()
        for proc in procs:
            assert proc.wait(timeout=10) is not None
    finally:
        job.close()


def test_closed_job_refuses_new_processes(tmp_path: Path) -> None:
    job = rt.JobObject()
    job.close()
    job.close()  # idempotent
    with pytest.raises(OSError):
        rt.spawn_in_job(job, [sys.executable, "-c", "pass"], cwd=tmp_path, env=None)


# -- install and status (no real server) -----------------------------------------------------


class AssetServer:
    """Serves byte strings by path, like GitHub and Hugging Face downloads (with Range support)."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                data = server.files.get(self.path)
                if data is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                if self.headers.get("Range"):
                    start = int(self.headers["Range"].split("=")[1].split("-")[0])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                self.wfile.write(data[start:])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class FakeDownloads:
    server: AssetServer
    model: LlmModelInfo
    other: LlmModelInfo
    engine_size: int
    model_path: str
    """URL path of the model file on the server."""


@pytest.fixture
def fake_downloads(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDownloads]:
    """A tiny CPU 'engine' zip and two tiny models, served locally under the real URL layout."""
    engine_zip = _zip({rt.SERVER_EXE: b"not really an exe", "ggml.dll": b"x" * 1000})
    model_bytes = b"GGUF" + bytes(range(256)) * 400
    asset = rt.ServerAsset(f"llama-{rt.LLAMA_BUILD}-bin-win-cpu-x64.zip", len(engine_zip), _sha(engine_zip))
    model = LlmModelInfo(
        id="tiny-test",
        name="Tiny Test",
        description="test model",
        model_file=LlmFile("test/tiny", "0" * 40, "tiny.gguf", len(model_bytes), _sha(model_bytes)),
        draft_file=None,
        license="Apache-2.0",
        license_url="https://example.com",
        multilingual=False,
        vram_mb=100,
        ram_mb=100,
        server_args=(),
    )
    other = replace(model, id="tiny-two", name="Tiny Two")
    model_path = f"/test/tiny/resolve/{'0' * 40}/tiny.gguf"
    server = AssetServer(
        {f"/ggml-org/llama.cpp/releases/download/{rt.LLAMA_BUILD}/{asset.name}": engine_zip, model_path: model_bytes}
    )
    monkeypatch.setattr(rt, "_RELEASE_URL", server.url + "/ggml-org/llama.cpp/releases/download/{build}/{name}")
    monkeypatch.setattr(catalog, "_HF", server.url)
    flavors = dict(rt.SERVER_FLAVORS)
    flavors["cpu"] = replace(flavors["cpu"], assets=(asset,))
    monkeypatch.setattr(rt, "SERVER_FLAVORS", flavors)
    monkeypatch.setattr(rt, "CATALOG", (model, other))
    monkeypatch.setattr(rt, "get_model", {model.id: model, other.id: other}.__getitem__)
    yield FakeDownloads(server, model, other, len(engine_zip), model_path)
    server.close()


def wait_for(predicate: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the condition")
        time.sleep(0.02)


@pytest.fixture
def make_runtime(settings_store: SettingsStore, tmp_path: Path) -> Iterator[Callable[..., rt.LlmRuntime]]:
    made: list[rt.LlmRuntime] = []

    def make(**kwargs: object) -> rt.LlmRuntime:
        runtime = rt.LlmRuntime(settings_store, tmp_path / "runtime", tmp_path / "models" / "llm", **kwargs)
        made.append(runtime)
        return runtime

    yield make
    for runtime in made:
        runtime.shutdown()


def test_statuses_without_a_model(settings_store: SettingsStore, make_runtime) -> None:
    runtime = make_runtime(flavor="cpu")
    wait_for(lambda: runtime.status.message == rt.MSG_DISABLED)
    assert runtime.status.state is rt.LlmState.OFF
    settings_store.update({"polish": {"enabled": True}})
    wait_for(lambda: runtime.status.state is rt.LlmState.NOT_INSTALLED)
    assert runtime.status.message == rt.MSG_NO_MODEL
    settings_store.update({"polish": {"builtin_model": "qwen3.5-4b"}})
    wait_for(lambda: "Qwen3.5 4B" in (runtime.status.message or ""))
    settings_store.update({"polish": {"provider": "ollama"}})
    wait_for(lambda: runtime.status.message == rt.MSG_OTHER_PROVIDER)
    runtime.ensure_running()  # nothing to start
    time.sleep(0.2)
    assert runtime.base_url() is None and runtime.api_key() is None
    assert runtime.installed_models() == []
    assert [m.id for m in runtime.catalog()] == [m.id for m in catalog.CATALOG]


def test_install_unknown_model(make_runtime) -> None:
    runtime = make_runtime(flavor="cpu")
    failures: list[tuple[str, str]] = []
    runtime.install_failed.connect(lambda m, msg: failures.append((m, msg)), Qt.ConnectionType.DirectConnection)
    runtime.install("nope")
    assert failures == [("nope", rt.MSG_UNKNOWN_MODEL)]


def test_install_then_launch_failure(
    settings_store: SettingsStore,
    make_runtime,
    fake_downloads: FakeDownloads,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = fake_downloads.model
    monkeypatch.setattr(rt, "RESTART_BACKOFF_S", (0.01, 0.01, 0.01))
    runtime = make_runtime(flavor="cpu")
    finished: list[str] = []
    progress: list[tuple[int, int]] = []
    runtime.install_finished.connect(finished.append, Qt.ConnectionType.DirectConnection)
    runtime.progress.connect(lambda m, d, t: progress.append((d, t)), Qt.ConnectionType.DirectConnection)
    runtime.install(model.id)
    wait_for(lambda: finished == [model.id] and runtime.status.installing is None)
    assert runtime.is_installed(model.id)
    assert runtime.installed_models() == [model.id]
    total = fake_downloads.engine_size + model.download_size
    assert progress[-1] == (total, total)
    assert all(done <= t for done, t in progress)
    engine = tmp_path / "runtime" / "llama" / rt.LLAMA_BUILD / "cpu"
    assert (engine / rt.SERVER_EXE).read_bytes() == b"not really an exe"
    assert not list((tmp_path / "runtime" / "llama" / "downloads").glob("*"))  # zips are removed after unpacking
    # Installing again is a no-op that still reports success.
    runtime.install(model.id)
    wait_for(lambda: finished == [model.id, model.id])

    # The "engine" can't run, so starting fails, is retried, and then reported plainly.
    settings_store.update({"polish": {"enabled": True, "builtin_model": model.id}})
    wait_for(lambda: runtime.status.message == rt.MSG_WAITING)  # waits for ensure_running()
    runtime.ensure_running()
    wait_for(lambda: runtime.status.state is rt.LlmState.ERROR)
    assert runtime.status.message == rt.MSG_CANT_LAUNCH
    assert runtime.base_url() is None

    runtime.uninstall(model.id)
    wait_for(lambda: not runtime.is_installed(model.id))
    wait_for(lambda: not (tmp_path / "runtime" / "llama").exists())  # the engine goes with the last model
    wait_for(lambda: runtime.status.state is rt.LlmState.NOT_INSTALLED)


def test_corrupt_download_is_reported(make_runtime, fake_downloads: FakeDownloads) -> None:
    server, model, path = fake_downloads.server, fake_downloads.model, fake_downloads.model_path
    server.files[path] = b"X" * len(server.files[path])  # right size, wrong bytes
    runtime = make_runtime(flavor="cpu")
    failures: list[tuple[str, str]] = []
    runtime.install_failed.connect(lambda m, msg: failures.append((m, msg)), Qt.ConnectionType.DirectConnection)
    runtime.install(model.id)
    wait_for(lambda: bool(failures))
    assert failures == [(model.id, rt.MSG_CORRUPT)]
    assert not runtime.is_installed(model.id)


def test_cancel_install(make_runtime, fake_downloads: FakeDownloads, monkeypatch: pytest.MonkeyPatch) -> None:
    model = fake_downloads.model
    started = threading.Event()
    release = threading.Event()
    real = rt.download_file

    def slow_download(*args, **kwargs):
        started.set()
        release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(rt, "download_file", slow_download)
    runtime = make_runtime(flavor="cpu")
    failures: list[tuple[str, str]] = []
    runtime.install_failed.connect(lambda m, msg: failures.append((m, msg)), Qt.ConnectionType.DirectConnection)
    runtime.install(model.id)
    assert started.wait(10)
    assert runtime.status.installing == model.id
    runtime.install(model.id)  # the same download again: ignored
    runtime.install(fake_downloads.other.id)  # only one download at a time
    assert failures == [(fake_downloads.other.id, rt.MSG_BUSY)]
    runtime.cancel_install()
    release.set()
    wait_for(lambda: (model.id, rt.MSG_CANCELLED) in failures)
    wait_for(lambda: runtime.status.installing is None)
    assert not runtime.is_installed(model.id)


@pytest.mark.parametrize("model_id", [r"..\..\outside", "../../outside", "..", ".", "", "OUTSIDE"])
def test_uninstall_never_deletes_outside_the_models_folder(make_runtime, tmp_path: Path, model_id: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    models_root = tmp_path / "models" / "llm"
    (models_root / "other-model").mkdir(parents=True)
    model_id = str(outside) if model_id == "OUTSIDE" else model_id
    runtime = make_runtime(flavor="cpu")
    runtime.uninstall(model_id)
    runtime.shutdown()  # the worker has handled everything that was queued
    runtime._uninstall(model_id)  # the worker's own check holds too
    assert (outside / "keep.txt").read_text() == "keep"
    assert (models_root / "other-model").is_dir()


def test_shutdown_is_idempotent(make_runtime) -> None:
    runtime = make_runtime(flavor="cpu")
    runtime.shutdown()
    runtime.shutdown()


def test_unknown_forced_flavor(settings_store: SettingsStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        rt.LlmRuntime(settings_store, tmp_path, tmp_path, flavor="cuda-99")


def test_process_is_not_left_running_after_shutdown(tmp_path: Path) -> None:
    # The same guarantee the runtime relies on: closing the job ends the child.
    job = rt.JobObject()
    proc = rt.spawn_in_job(job, [sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, env=None)
    job.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - would be a real failure
        proc.kill()
        raise
