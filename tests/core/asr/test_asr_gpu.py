"""GPU detection messages and the GPU runtime pack installer (tiny fake wheels served locally)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import tomllib
import zipfile
from pathlib import Path

import pytest

from whisp.core import hardware
from whisp.core.asr import gpu
from whisp.core.asr.downloads import DownloadCancelled, DownloadError
from whisp.core.asr.gpu import RuntimeDll, RuntimeWheel


def dll_bytes(seed: int, size: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


def make_wheel(server, name: str, dlls: dict[str, bytes], corrupt: str | None = None) -> RuntimeWheel:
    """A zip with each DLL at a wheel-like archive path, served at /wheels/<name>.whl."""
    buffer = io.BytesIO()
    entries = []
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{name}.dist-info/RECORD", "")
        for dll_name, data in dlls.items():
            path = f"nvidia/cu13/bin/x86_64/{dll_name}"
            archive.writestr(path, data if dll_name != corrupt else data[::-1])
            entries.append(RuntimeDll(dll_name, path, len(data), hashlib.sha256(data).hexdigest()))
    url, size, sha = server.add(f"/wheels/{name}-1.0-py3-none-win_amd64.whl", buffer.getvalue())
    return RuntimeWheel(name, "1.0", url, size, sha, tuple(entries))


@pytest.fixture
def fake_pack(file_server, monkeypatch: pytest.MonkeyPatch) -> tuple[RuntimeWheel, ...]:
    wheels = (
        make_wheel(file_server, "fake-runtime", {"cudartfake.dll": dll_bytes(1, 40_000)}),
        make_wheel(file_server, "fake-blas", {"blasA.dll": dll_bytes(2, 90_000), "blasB.dll": dll_bytes(3, 7_000)}),
    )
    monkeypatch.setattr(gpu, "RUNTIME_WHEELS", wheels)
    return wheels


def test_installs_verifies_and_records_the_runtime(fake_pack, tmp_path: Path) -> None:
    target = tmp_path / "runtime" / "cuda"
    progress: list[tuple[int, int]] = []

    gpu.install_cuda_runtime(lambda done, total: progress.append((done, total)), directory=target)

    assert sorted(p.name for p in target.iterdir()) == sorted(
        ["cudartfake.dll", "blasA.dll", "blasB.dll", gpu.RUNTIME_MANIFEST]
    )
    assert (target / "blasA.dll").read_bytes() == dll_bytes(2, 90_000)
    assert gpu.runtime_installed(target)
    total = sum(w.size for w in fake_pack)
    assert progress[-1] == (total, total)
    assert not target.with_name("cuda.partial").exists()


def test_only_dlls_are_kept(fake_pack, tmp_path: Path) -> None:
    target = tmp_path / "cuda"
    gpu.install_cuda_runtime(directory=target)
    assert all(p.suffix == ".dll" or p.name == gpu.RUNTIME_MANIFEST for p in target.iterdir())


def test_an_interrupted_install_keeps_what_was_extracted(
    fake_pack, file_server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cuda"
    cancel = threading.Event()
    extract = gpu._extract_dlls

    def extract_then_cancel(*args) -> None:
        extract(*args)
        cancel.set()  # the user cancels just after the first wheel is unpacked

    monkeypatch.setattr(gpu, "_extract_dlls", extract_then_cancel)
    with pytest.raises(DownloadCancelled):
        gpu.install_cuda_runtime(cancel=cancel, directory=target)
    assert not gpu.runtime_installed(target)
    monkeypatch.setattr(gpu, "_extract_dlls", extract)
    file_server.requests.clear()

    gpu.install_cuda_runtime(directory=target)

    fetched = [path for path, _ in file_server.requests]
    assert fetched == [f"/wheels/{fake_pack[1].name}-1.0-py3-none-win_amd64.whl"]
    assert gpu.runtime_installed(target)


def test_a_dll_that_does_not_match_its_checksum_fails(file_server, tmp_path: Path, monkeypatch) -> None:
    wheel = make_wheel(file_server, "bad", {"x.dll": dll_bytes(4, 5_000)}, corrupt="x.dll")
    monkeypatch.setattr(gpu, "RUNTIME_WHEELS", (wheel,))
    with pytest.raises(DownloadError) as caught:
        gpu.install_cuda_runtime(directory=tmp_path / "cuda")
    assert "damaged" in str(caught.value)
    assert not gpu.runtime_installed(tmp_path / "cuda")


def test_a_changed_or_missing_dll_means_not_installed(fake_pack, tmp_path: Path) -> None:
    target = tmp_path / "cuda"
    gpu.install_cuda_runtime(directory=target)
    (target / "blasB.dll").write_bytes(b"short")
    assert not gpu.runtime_installed(target)
    (target / "blasB.dll").unlink()
    assert not gpu.runtime_installed(target)


def test_a_pack_for_another_onnx_runtime_is_not_installed(fake_pack, tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "cuda"
    gpu.install_cuda_runtime(directory=target)
    monkeypatch.setattr(gpu, "RUNTIME_PACK_ID", "onnxruntime-gpu-9.9/cuda-14")
    assert not gpu.runtime_installed(target)


def test_the_pinned_pack_matches_onnxruntimes_cuda_extras() -> None:
    names = {wheel.name for wheel in gpu.RUNTIME_WHEELS}
    assert names == {
        "nvidia-cuda-runtime",
        "nvidia-cublas",
        "nvidia-nvjitlink",
        "nvidia-cufft",
        "nvidia-curand",
        "nvidia-cuda-nvrtc",
        "nvidia-cudnn-cu13",
    }
    dlls = [dll.name for wheel in gpu.RUNTIME_WHEELS for dll in wheel.dlls]
    assert len(dlls) == len(set(dlls))
    assert {"cudart64_13.dll", "cublasLt64_13.dll", "cublas64_13.dll", "cufft64_12.dll", "cudnn64_9.dll"} <= set(dlls)
    for wheel in gpu.RUNTIME_WHEELS:
        assert wheel.url.startswith("https://files.pythonhosted.org/") and wheel.filename.endswith("win_amd64.whl")
        assert len(wheel.sha256) == 64 and wheel.size > 0
    assert 1_000_000_000 < gpu.runtime_download_size() < 1_300_000_000


def test_the_dev_cuda_extra_installs_exactly_the_pack() -> None:
    # `uv sync --extra cuda` must give developers the same NVIDIA files that packaged builds download.
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extra = project["project"]["optional-dependencies"]["cuda"]
    pins = dict(requirement.split("==") for requirement in extra if "==" in requirement)
    assert pins == {wheel.name: wheel.version for wheel in gpu.RUNTIME_WHEELS}
    locked = {package["name"]: package for package in tomllib.loads((root / "uv.lock").read_text("utf-8"))["package"]}
    for wheel in gpu.RUNTIME_WHEELS:
        package = locked[wheel.name]
        assert package["version"] == wheel.version
        files = {(entry["url"], entry["hash"]) for entry in package.get("wheels", [])}
        assert (wheel.url, f"sha256:{wheel.sha256}") in files, wheel.name


def test_cuda_options_use_heuristic_convolution_search() -> None:
    assert gpu.cuda_provider_options()["cudnn_conv_algo_search"] == "HEURISTIC"


# -- detection messages -------------------------------------------------------------------------------


def hardware_with(nvidia: hardware.NvidiaInfo | None, gpus: tuple[hardware.GpuAdapter, ...] = ()) -> hardware.Hardware:
    return hardware.Hardware(gpus=gpus, nvidia=nvidia, logical_cpus=8, performance_cores=4)


NVIDIA = hardware.NvidiaInfo("617.42", 617, "NVIDIA GeForce RTX 4070", 12282, 10000)


@pytest.fixture
def detection(monkeypatch: pytest.MonkeyPatch):
    """Run gpu._detect with chosen hardware, library source and probe result."""

    def run(hw: hardware.Hardware, source: str | None = "site", prepared: bool = True, probe: str | None = None):
        monkeypatch.setattr(gpu.hardware, "detect", lambda: hw)
        monkeypatch.setattr(gpu, "_library_source", lambda: source)
        monkeypatch.setattr(gpu, "prepare_cuda", lambda: prepared)
        monkeypatch.setattr(gpu, "_probe_cuda", lambda: probe)
        return gpu._detect()

    return run


def test_ready(detection) -> None:
    info = detection(hardware_with(NVIDIA))
    assert info == gpu.GpuInfo(True, "NVIDIA GeForce RTX 4070", 12282, True, "NVIDIA GeForce RTX 4070 · ready")


def test_no_nvidia_card(detection) -> None:
    info = detection(hardware_with(None, (hardware.GpuAdapter("intel", "Intel UHD Graphics", 128),)))
    assert not info.has_nvidia and not info.cuda_ready and "No NVIDIA graphics card" in info.detail


def test_nvidia_card_without_a_working_driver(detection) -> None:
    info = detection(hardware_with(None, (hardware.GpuAdapter("nvidia", "NVIDIA GeForce GTX 1060", 6144),)))
    assert info.has_nvidia and info.name == "NVIDIA GeForce GTX 1060" and not info.cuda_ready
    assert "driver" in info.detail


def test_old_driver(detection) -> None:
    old = hardware.NvidiaInfo("552.44", 552, "NVIDIA GeForce RTX 3060", 12288, 11000)
    info = detection(hardware_with(old))
    assert not info.cuda_ready
    assert info.detail.startswith("Your NVIDIA driver is too old for GPU speed (needs 580 or newer).")


def test_runtime_download_needed(detection) -> None:
    info = detection(hardware_with(NVIDIA), source=None)
    assert not info.cuda_ready
    assert info.detail == "GPU acceleration needs a one-time 1.1 GB download."
    assert info.runtime_download_size == gpu.runtime_download_size()


def test_probe_failure_is_explained(detection) -> None:
    info = detection(hardware_with(NVIDIA), probe=gpu.MSG_GPU_TOO_OLD)
    assert not info.cuda_ready and info.detail == gpu.MSG_GPU_TOO_OLD
    info = detection(hardware_with(NVIDIA), prepared=False)
    assert not info.cuda_ready and info.detail == gpu.MSG_GPU_FAILED


def test_detection_is_cached_until_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_detect() -> gpu.GpuInfo:
        calls.append(1)
        return gpu.GpuInfo(False, None, None, False, "none")

    monkeypatch.setattr(gpu, "_detect", fake_detect)
    monkeypatch.setattr(gpu, "_cached", None)
    monkeypatch.setattr(gpu.hardware, "refresh", lambda: None)
    gpu.detect_gpu()
    gpu.detect_gpu()
    assert len(calls) == 1
    gpu.refresh_gpu()
    assert len(calls) == 2


# -- this machine's GPU ---------------------------------------------------------------------------------


@pytest.mark.gpu
def test_cuda_is_ready_on_this_machine() -> None:
    info = gpu.refresh_gpu()
    assert info.has_nvidia and info.cuda_ready, info.detail
    assert info.detail.endswith("· ready")


@pytest.mark.gpu
@pytest.mark.models
def test_the_real_runtime_pack_runs_cuda_on_its_own(shared_models_dir: Path) -> None:
    """Download the pinned NVIDIA wheels (1.1 GB, once), then prove a fresh process runs CUDA from the pack alone."""
    home = shared_models_dir / "gpu-runtime-home"
    target = home / "data" / "runtime" / gpu.RUNTIME_FOLDER
    gpu.install_cuda_runtime(directory=target)
    assert gpu.runtime_installed(target)

    script = (
        "import ctypes, json, sys\n"
        "from ctypes import wintypes\n"
        "from whisp.core.asr import gpu\n"
        "info = gpu.detect_gpu()\n"
        "k = ctypes.WinDLL('kernel32')\n"
        "k.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]; k.GetModuleHandleW.restype = wintypes.HMODULE\n"
        "k.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]\n"
        "paths = {}\n"
        "for name in ('cudart64_13.dll', 'cublasLt64_13.dll', 'cudnn64_9.dll', 'cudnn_ops64_9.dll'):\n"
        "    buf = ctypes.create_unicode_buffer(1024)\n"
        "    k.GetModuleFileNameW(k.GetModuleHandleW(name), buf, 1024)\n"
        "    paths[name] = buf.value\n"
        "print(json.dumps({'ready': info.cuda_ready, 'detail': info.detail, 'paths': paths}))\n"
    )
    env = {**os.environ, "WHISP_HOME": str(home), "PYTHONPATH": str(Path(gpu.__file__).parents[3])}
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["ready"], result["detail"]
    for name, path in result["paths"].items():
        assert os.path.normcase(Path(path).parent) == os.path.normcase(target.resolve()), f"{name} came from {path}"
