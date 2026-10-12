"""NVIDIA GPU detection and the CUDA runtime that ONNX Runtime's GPU support needs.

onnxruntime-gpu 1.31 is built for CUDA 13 and cuDNN 9, which need an NVIDIA
driver from the 580 series or newer. Its CUDA support lives in
``onnxruntime_providers_cuda.dll``, which only loads once the NVIDIA runtime
DLLs (cudart, cuBLAS, cuFFT, cuDNN, ...) can be found. If they can't, ONNX
Runtime quietly runs on the CPU instead, so readiness is only ever reported
after a real CUDA session has run.

Where the NVIDIA DLLs come from:

1. The **GPU runtime pack** in ``<runtime_dir>/cuda``, downloaded on demand by
   ``install_cuda_runtime``. It holds the DLLs from the exact NVIDIA wheels that
   onnxruntime-gpu 1.31's ``[cuda,cudnn]`` extras resolve to (pinned below with
   their SHA-256), so packaged builds don't have to ship 1.7 GB of libraries.
2. In a development environment (not ``sys.frozen``), the same NVIDIA wheels
   installed in the Python environment.

To move to a newer onnxruntime-gpu, resolve its ``[cuda,cudnn]`` extras and their
NVIDIA dependencies on PyPI (``https://pypi.org/pypi/<name>/<version>/json``),
pin each ``win_amd64`` wheel's URL, size and SHA-256, and take every DLL's size
and SHA-256 from the wheel's ``RECORD`` file.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import zipfile
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from whisp.core import hardware
from whisp.core.asr.downloads import (
    DISK_MARGIN_BYTES,
    MSG_CHECKSUM,
    MSG_NO_SPACE,
    MSG_WRITE,
    DownloadCancelled,
    DownloadError,
    DownloadItem,
    ProgressCallback,
    download_files,
    format_size,
    sha256_file,
)

if TYPE_CHECKING:
    import httpx

log = logging.getLogger(__name__)

MIN_DRIVER_MAJOR = 580
"""CUDA 13 needs an NVIDIA driver from the R580 series or newer."""
CUDA_PROVIDER = "CUDAExecutionProvider"
RUNTIME_FOLDER = "cuda"
RUNTIME_MANIFEST = "whisp-runtime.json"
RUNTIME_PACK_ID = "onnxruntime-gpu-1.31.0/cuda-13.4/cudnn-9.27"
"""Identifies the pinned set below; a pack recorded with another id is downloaded again."""

MSG_NO_NVIDIA = "No NVIDIA graphics card found. Speech recognition runs on the CPU."
MSG_DRIVER_BROKEN = "The NVIDIA driver isn't responding. Reinstall it from nvidia.com for GPU speed. Using the CPU."
MSG_DRIVER_OLD = (
    "Your NVIDIA driver is too old for GPU speed (needs {minimum} or newer). Update it from nvidia.com. Using the CPU."
)
MSG_NEEDS_RUNTIME = "GPU acceleration needs a one-time {size} download."
MSG_READY = "{name} · ready"
MSG_GPU_TOO_OLD = "This graphics card is too old for Whisp's GPU acceleration. Using the CPU."
MSG_GPU_FAILED = "The graphics card couldn't run speech recognition, so Whisp is using the CPU."
MSG_RUNTIME_LOCKED = "The GPU libraries are in use. Restart Whisp, then try the download again."


@dataclass(frozen=True)
class GpuInfo:
    has_nvidia: bool
    name: str | None
    vram_mb: int | None
    cuda_ready: bool
    """The runtime libraries load and a CUDA session really runs."""
    detail: str
    """Plain-language explanation for the UI."""
    runtime_download_size: int = 0
    """Bytes ``install_cuda_runtime`` would download to enable the GPU; 0 when it isn't needed or wouldn't help."""


@dataclass(frozen=True)
class RuntimeDll:
    name: str
    """File name in the runtime folder."""
    archive_path: str
    """Path inside the wheel."""
    size: int
    sha256: str


@dataclass(frozen=True)
class RuntimeWheel:
    name: str
    version: str
    url: str
    size: int
    sha256: str
    dlls: tuple[RuntimeDll, ...] = field(default=())

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


_PYPI = "https://files.pythonhosted.org/packages/"
_CU13 = "nvidia/cu13/bin/x86_64/"
_CUDNN = "nvidia/cudnn/bin/"


def _dll(folder: str, name: str, size: int, sha256: str) -> RuntimeDll:
    return RuntimeDll(name, folder + name, size, sha256)


RUNTIME_WHEELS: tuple[RuntimeWheel, ...] = (
    RuntimeWheel(
        name="nvidia-cuda-runtime",
        version="13.4.92",
        url=_PYPI + "86/00/d5436004268f049214193659ebc36550b5ef3925c3d13b4cc980e13be6f5/"
        "nvidia_cuda_runtime-13.4.92-py3-none-win_amd64.whl",
        size=2778543,
        sha256="08dca5e4aba480c2fd5b55075c0fa71b84ef9dcf0521f2d58baa14a803a7311c",
        dlls=(
            _dll(
                _CU13,
                "cudart64_13.dll",
                551024,
                "d26fd1c2629594759af569a5252f94e4568d4ebeca67dc323834deb141aa278a",
            ),
        ),
    ),
    RuntimeWheel(
        name="nvidia-cublas",
        version="13.8.0.4",
        url=_PYPI + "a3/df/f1246959833e2c437db8be3e5b477f66b87f8817821ed40de6c7561c9a36/"
        "nvidia_cublas-13.8.0.4-py3-none-win_amd64.whl",
        size=423266897,
        sha256="8c5494423bb8a46822cb6b0cb95d7fa4be2d7b96a31155dff083839ec8297910",
        dlls=(
            _dll(
                _CU13,
                "cublasLt64_13.dll",
                493474416,
                "cad63434448e7141629e240ea093ad596a7ef6a0f67b468ba9bc1df6e1eeee33",
            ),
            _dll(
                _CU13,
                "cublas64_13.dll",
                54873200,
                "60bbba8868290311e9c1657b2193ddec667744eb555ff843f87acb7c039f9efa",
            ),
        ),
    ),
    RuntimeWheel(
        name="nvidia-nvjitlink",
        version="13.4.92",
        url=_PYPI + "72/7c/c44bd7277afafed9de262a23baec4dc8dcf7ecbb1b9ecb228821442d1e9b/"
        "nvidia_nvjitlink-13.4.92-py3-none-win_amd64.whl",
        size=39053950,
        sha256="b286f3a4f227a9363efdec263c7b91788cef1478d2b8a5fa8bab7f3e82ff82fd",
        dlls=(
            _dll(
                _CU13,
                "nvJitLink_130_0.dll",
                97119344,
                "6044faf21b4979502588ff9319521dd21a42b9068239d3291e53bc5a063b09bf",
            ),
        ),
    ),
    RuntimeWheel(
        name="nvidia-cufft",
        version="12.4.0.43",
        url=_PYPI + "7b/cc/be7fe31058127336a66c88414a2ecc6beecf680baf85994a43eaf292b202/"
        "nvidia_cufft-12.4.0.43-py3-none-win_amd64.whl",
        size=160953139,
        sha256="4ff7075f2d0b5f69291f70938d37a86ec632cbe5747184c74ba1f50f17accacc",
        dlls=(
            _dll(
                _CU13,
                "cufft64_12.dll",
                232604272,
                "d353fd42da7729709864731f35888aa177793d2350e56557df4c6ebaaf079ace",
            ),
        ),
    ),
    RuntimeWheel(
        name="nvidia-curand",
        version="10.4.4.72",
        url=_PYPI + "e2/d4/f59ab342b82ff05ff27d674acd470ba625d788bf71338f7da6c89669d6b9/"
        "nvidia_curand-10.4.4.72-py3-none-win_amd64.whl",
        size=56731059,
        sha256="e0bce83e083ef25976ee74f59e8f067c15149a74538f1be6c2462e286e7c9c68",
        dlls=(
            _dll(
                _CU13,
                "curand64_10.dll",
                60045424,
                "2ae1d341cd7da1dab91e2215c62f627e0e0288ff54cdc9003e908746584b5b2b",
            ),
        ),
    ),
    RuntimeWheel(
        name="nvidia-cuda-nvrtc",
        version="13.4.92",
        url=_PYPI + "64/5b/aa91896f64444eff1dd7f189f4db05046c7f0ba08c0f531283c1c1aac581/"
        "nvidia_cuda_nvrtc-13.4.92-py3-none-win_amd64.whl",
        size=46972283,
        sha256="6af7ac5372920f6a7a560d0699348560fe44cb52c1d722d6afd3119f8af948c4",
        dlls=(
            _dll(
                _CU13,
                "nvrtc-builtins64_134.dll",
                7243376,
                "73768e90526e8a43d5eb356f0ffad198468e8c15e95d45a84c6c4b12fb9aa02f",
            ),
            _dll(
                _CU13,
                "nvrtc64_130_0.dll",
                106038896,
                "c92027e5eae58b6f272cafb438bf59b55deb02bd326b183956e84e0a180dbd14",
            ),
        ),
    ),
    RuntimeWheel(
        name="nvidia-cudnn-cu13",
        version="9.27.0.42",
        url=_PYPI + "87/6a/e55ff0ac26a5c6e2b21f41c9d04ad096b4ed6da593fba7e25845c61b0532/"
        "nvidia_cudnn_cu13-9.27.0.42-py3-none-win_amd64.whl",
        size=436469905,
        sha256="7d96f634adafd55c72231eb0500ca77ab109ec8ebff7b33000b76e081bc4558e",
        dlls=(
            _dll(
                _CUDNN,
                "cudnn_graph64_9.dll",
                116132464,
                "ccc90435b2fd37b5ddd6a9fa0fd053ad8f46c2dc8b0a6256ef0ebea7f3efbace",
            ),
            _dll(
                _CUDNN,
                "cudnn_engines_precompiled64_9.dll",
                231660656,
                "5d9b829228ae18594dafedf718d9612606c8ce6d199aa7151bea53e863baeefe",
            ),
            _dll(
                _CUDNN,
                "cudnn_engines_runtime_compiled64_9.dll",
                39266928,
                "8e65eeb0a4a3b7afa48ea3ca71c4b88e355199f85485b6fc0badcc97ebc552ef",
            ),
            _dll(
                _CUDNN,
                "cudnn_engines_tensor_ir64_9.dll",
                156272,
                "51afb44ce4b901ccdac1434f9fc9d8d707b690f39ae1e2a6ffca98c5fd9dfa96",
            ),
            _dll(
                _CUDNN,
                "cudnn_heuristic64_9.dll",
                94795376,
                "c0439ee74a3dbec8ecf56fecbc6ac2506efcce702453e36119649545d02509c3",
            ),
            _dll(
                _CUDNN,
                "cudnn_ops64_9.dll",
                37462128,
                "2bd7f4122eaeccb85538c2a6f7923d7254ad77d7b7149fed0a8139f1e3cf0b07",
            ),
            _dll(
                _CUDNN,
                "cudnn_cnn64_9.dll",
                1533040,
                "7dd1bcc6d8f00ad635f6a8a2cd076faba21626f3412e13c61a6bece3f88eacb1",
            ),
            _dll(
                _CUDNN,
                "cudnn_adv64_9.dll",
                105226864,
                "041cab1d6439d2558b4a7073a42dd95058d0395463ffb0d37275af11ec0af8d7",
            ),
            _dll(
                _CUDNN,
                "cudnn_ext64_9.dll",
                130160,
                "a0e0e2e8e8e4678956bbef93fdccc4730bd5edf9b23d38bff49030ebba1940a6",
            ),
            _dll(
                _CUDNN,
                "cudnn64_9.dll",
                270448,
                "0016bd73ac9192537ae5af6a3f05814316726dad1d1da44f113f0a9ef00ae34c",
            ),
        ),
    ),
)
"""onnxruntime-gpu 1.31.0 ``[cuda,cudnn]`` extras plus their NVIDIA dependencies (cuBLAS for cuDNN,
nvJitLink for cuFFT), in the order their DLLs must load. Only the DLLs ONNX Runtime can use are kept
(the FFTW and BLAS drop-in libraries are skipped)."""

_CORE_DLLS = ("cudart64_13.dll", "cublasLt64_13.dll", "cublas64_13.dll", "cudnn64_9.dll")
"""Libraries that must be loaded for the CUDA provider to load at all."""


def runtime_download_size() -> int:
    """Bytes the GPU runtime pack downloads."""
    return sum(wheel.size for wheel in RUNTIME_WHEELS)


def runtime_dir() -> Path:
    """Where the GPU runtime pack is installed."""
    from whisp.storage.paths import app_paths

    return app_paths().runtime_dir / RUNTIME_FOLDER


def runtime_installed(directory: Path | None = None) -> bool:
    """True if the pinned runtime pack is complete in ``directory`` (default: ``runtime_dir()``). Cheap."""
    folder = directory if directory is not None else runtime_dir()
    try:
        manifest = json.loads((folder / RUNTIME_MANIFEST).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        log.warning("Unreadable GPU runtime manifest: %s", exc)
        return False
    if manifest != _runtime_manifest():
        log.info("The GPU runtime pack on disk is a different version; it needs downloading again")
        return False
    for dll in _all_dlls():
        try:
            if (folder / dll.name).stat().st_size != dll.size:
                return False
        except OSError:
            return False
    return True


def detect_gpu() -> GpuInfo:
    """What the GPU can do for speech recognition. Cached after the first call; any thread.

    The first call may take a second or two: it loads the CUDA libraries and runs
    a tiny model on the GPU to prove it works.
    """
    with _detect_lock:
        global _cached
        if _cached is None:
            _cached = _detect()
            log.info("GPU: %s (cuda_ready=%s): %s", _cached.name or "none", _cached.cuda_ready, _cached.detail)
        return _cached


def refresh_gpu() -> GpuInfo:
    """Forget the cached result and detect again, e.g. after installing the GPU runtime pack."""
    global _cached
    with _detect_lock:
        _cached = None
    hardware.refresh()
    return detect_gpu()


def prepare_cuda() -> bool:
    """Make the NVIDIA runtime DLLs loadable by ONNX Runtime. Idempotent; any thread.

    Returns True if the core CUDA libraries are now loaded in the process. Doesn't
    prove the GPU works; ``detect_gpu`` does that.
    """
    with _prepare_lock:
        if _core_dlls_loaded():
            return True
        source = _library_source()
        if source is None:
            return False
        try:
            from whisp.core.onnx_runtime import ort

            # Process-wide: ONNX Runtime 1.31 warns on stderr for every CUDA session that no *plugin* CUDA
            # provider is registered before it uses its built-in one. Sessions already log errors only.
            ort.set_default_logger_severity(3)
            if source == "pack":
                folder = runtime_dir()
                _add_dll_directory(folder)
                # ONNX Runtime's list leaves out cuDNN sub-libraries that cuDNN itself loads by name later,
                # so every DLL of the pack is loaded here, by full path, in dependency order.
                for dll in _all_dlls():
                    ctypes.WinDLL(str(folder / dll.name))
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=str(folder))
            else:
                for folder in _site_package_dll_dirs():
                    _add_dll_directory(folder)
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except (OSError, RuntimeError) as exc:
            log.warning("Couldn't load the NVIDIA runtime libraries: %s", exc)
            return False
        loaded = _core_dlls_loaded()
        if not loaded:
            missing = [name for name in _CORE_DLLS if not _module_loaded(name)]
            log.warning("NVIDIA runtime libraries still missing after preloading: %s", ", ".join(missing))
        else:
            origin = "the GPU runtime pack" if source == "pack" else "the Python environment"
            log.info("NVIDIA runtime libraries loaded from %s", origin)
        return loaded


def cuda_provider_options() -> dict[str, Any]:
    """CUDA execution provider options shared by every speech model session.

    ``HEURISTIC`` conv algorithm search: cuDNN's default choice was about 7x slower for
    Parakeet's convolutions, and ``EXHAUSTIVE`` benchmarks every new input shape
    (every segment length is new) for no measurable gain over the heuristics.
    """
    return {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}


def install_cuda_runtime(
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    *,
    directory: Path | None = None,
    client: httpx.Client | None = None,
) -> None:
    """Download the pinned NVIDIA wheels, keep only their DLLs in the runtime folder and verify them. Blocking.

    Resumes an interrupted install. Raises ``DownloadError`` (user-facing message)
    or ``DownloadCancelled``. ``detect_gpu`` is refreshed afterwards by the caller
    (``refresh_gpu``); ONNX Runtime picks the libraries up without a restart.
    """
    target = directory if directory is not None else runtime_dir()
    if runtime_installed(target):
        log.info("The GPU runtime pack is already installed")
        return
    cancel = cancel if cancel is not None else threading.Event()
    staging = target.with_name(target.name + ".partial")
    wheels_dir = staging / "wheels"
    try:
        wheels_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("Couldn't create %s: %s", wheels_dir, exc)
        raise DownloadError(MSG_WRITE.format(folder=staging)) from exc
    _check_runtime_space(staging, wheels_dir)
    total = runtime_download_size()
    done_before = 0
    for wheel in RUNTIME_WHEELS:
        if all(_dll_ok(staging / dll.name, dll) for dll in wheel.dlls):
            done_before += wheel.size
            if progress is not None:
                progress(done_before, total)
            continue
        item = DownloadItem(wheel.url, wheels_dir / wheel.filename, wheel.size, wheel.sha256)
        offset = done_before

        def report(done: int, _total: int, offset: int = offset) -> None:
            if progress is not None:
                progress(offset + done, total)

        download_files([item], progress=report, cancel=cancel, client=client, check_space=False)
        _extract_dlls(item.dest, wheel, staging, cancel)
        item.dest.unlink(missing_ok=True)
        done_before += wheel.size
    shutil.rmtree(wheels_dir, ignore_errors=True)
    _write_json(staging / RUNTIME_MANIFEST, _runtime_manifest())
    try:
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except OSError as exc:
        # A previous pack's DLLs are loaded in this process (Windows keeps them locked).
        log.warning("Couldn't replace the GPU runtime folder: %s", exc)
        raise DownloadError(MSG_RUNTIME_LOCKED) from exc
    log.info("GPU runtime pack installed in %s", target)


# -- detection internals ---------------------------------------------------------------------

_detect_lock = threading.Lock()
_prepare_lock = threading.Lock()
_cached: GpuInfo | None = None
_dll_directories: list[Any] = []
"""Handles from ``os.add_dll_directory``; kept so the directories stay registered."""


def _detect() -> GpuInfo:
    hw = hardware.detect()
    nvidia = hw.nvidia
    if nvidia is None:
        adapter = next((gpu for gpu in hw.gpus if gpu.vendor == "nvidia"), None)
        if adapter is None:
            return GpuInfo(False, None, None, False, MSG_NO_NVIDIA)
        # DXGI sees an NVIDIA card but its driver (NVML) doesn't answer.
        return GpuInfo(True, adapter.name, adapter.vram_mb or None, False, MSG_DRIVER_BROKEN)
    name, vram = nvidia.name, nvidia.vram_total_mb
    if nvidia.driver_major < MIN_DRIVER_MAJOR:
        return GpuInfo(True, name, vram, False, MSG_DRIVER_OLD.format(minimum=MIN_DRIVER_MAJOR))
    if _library_source() is None:
        size = runtime_download_size()
        return GpuInfo(True, name, vram, False, MSG_NEEDS_RUNTIME.format(size=format_size(size)), size)
    if not prepare_cuda():
        return GpuInfo(True, name, vram, False, MSG_GPU_FAILED)
    problem = _probe_cuda()
    if problem is not None:
        return GpuInfo(True, name, vram, False, problem)
    return GpuInfo(True, name, vram, True, MSG_READY.format(name=name))


def _probe_cuda() -> str | None:
    """Run the bundled Silero VAD model on the GPU. None if CUDA really works, else a user-facing reason."""
    try:
        from whisp.core.onnx_runtime import ort
        from whisp.storage.paths import resource_path

        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(resource_path("assets", "models", "silero_vad.onnx")),
            sess_options=options,
            providers=[(CUDA_PROVIDER, cuda_provider_options()), "CPUExecutionProvider"],
        )
        if CUDA_PROVIDER not in session.get_providers():
            log.warning("ONNX Runtime didn't start its CUDA provider; providers: %s", session.get_providers())
            return MSG_GPU_FAILED
        session.run(
            None,
            {
                "input": np.zeros((1, 512), dtype=np.float32),
                "state": np.zeros((2, 1, 128), dtype=np.float32),
                "sr": np.array(16000, dtype=np.int64),
            },
        )
        del session
    except Exception as exc:
        text = str(exc).lower()
        log.warning("The CUDA check failed: %s", exc)
        if "no kernel image" in text or "unsupported gpu architecture" in text or "compute capability" in text:
            return MSG_GPU_TOO_OLD
        return MSG_GPU_FAILED
    return None


def _library_source() -> str | None:
    """``"pack"`` (downloaded runtime), ``"site"`` (NVIDIA wheels in this Python environment) or None."""
    if runtime_installed():
        return "pack"
    if not getattr(sys, "frozen", False) and _site_package_dll_dirs():
        return "site"
    return None


def _site_package_dll_dirs() -> list[Path]:
    """Folders of NVIDIA's CUDA 13 wheels in this Python environment (development installs only)."""
    try:
        import nvidia  # namespace package installed by the NVIDIA wheels
    except ImportError:
        return []
    folders: list[Path] = []
    for root in getattr(nvidia, "__path__", []):
        for relative in ("cu13/bin/x86_64", "cudnn/bin"):
            folder = Path(root) / relative
            if folder.is_dir():
                folders.append(folder)
    has_cudart = any((folder / "cudart64_13.dll").is_file() for folder in folders)
    has_cudnn = any((folder / "cudnn64_9.dll").is_file() for folder in folders)
    return folders if has_cudart and has_cudnn else []


def _add_dll_directory(folder: Path) -> None:
    try:
        _dll_directories.append(os.add_dll_directory(str(folder)))
    except OSError as exc:
        log.warning("Couldn't add %s to the DLL search path: %s", folder, exc)
    # Libraries that are loaded by plain name later (cuDNN loads its sub-libraries itself) search PATH.
    path = os.environ.get("PATH", "")
    if str(folder).lower() not in path.lower().split(os.pathsep):
        os.environ["PATH"] = str(folder) + os.pathsep + path


def _module_loaded(name: str) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_handle = kernel32.GetModuleHandleW
    get_handle.argtypes = [wintypes.LPCWSTR]
    get_handle.restype = wintypes.HMODULE
    return bool(get_handle(name))


def _core_dlls_loaded() -> bool:
    return all(_module_loaded(name) for name in _CORE_DLLS)


# -- runtime pack internals ----------------------------------------------------------------------


def _all_dlls() -> list[RuntimeDll]:
    return [dll for wheel in RUNTIME_WHEELS for dll in wheel.dlls]


def _runtime_manifest() -> dict[str, Any]:
    return {
        "format": 1,
        "pack": RUNTIME_PACK_ID,
        "files": [{"name": dll.name, "size": dll.size, "sha256": dll.sha256} for dll in _all_dlls()],
    }


def _dll_ok(path: Path, dll: RuntimeDll) -> bool:
    """A DLL extracted by an earlier, interrupted install that can be kept."""
    try:
        return path.stat().st_size == dll.size and sha256_file(path) == dll.sha256
    except OSError:
        return False


def _check_runtime_space(staging: Path, wheels_dir: Path) -> None:
    """Room for the biggest wheel that is still missing plus every DLL not yet extracted."""
    pending_dlls = sum(dll.size for wheel in RUNTIME_WHEELS for dll in wheel.dlls if not (staging / dll.name).is_file())
    biggest_wheel = max(
        (w.size for w in RUNTIME_WHEELS if not all((staging / d.name).is_file() for d in w.dlls)), default=0
    )
    needed = pending_dlls + biggest_wheel + DISK_MARGIN_BYTES
    try:
        free = shutil.disk_usage(wheels_dir).free
    except OSError:
        log.warning("Couldn't read the free disk space for the GPU runtime; installing anyway")
        return
    if free < needed:
        log.warning("Not enough disk space for the GPU runtime: %d bytes free, %d needed", free, needed)
        drive = wheels_dir.drive or str(wheels_dir.anchor)
        raise DownloadError(MSG_NO_SPACE.format(amount=format_size(needed - free), drive=drive))


def _extract_dlls(wheel_path: Path, wheel: RuntimeWheel, staging: Path, cancel: threading.Event) -> None:
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            for dll in wheel.dlls:
                dest = staging / dll.name
                tmp = dest.with_name(dest.name + ".tmp")
                digest = hashlib.sha256()
                written = 0
                with archive.open(dll.archive_path) as src, open(tmp, "wb") as out:
                    while block := src.read(4 << 20):
                        if cancel.is_set():
                            raise DownloadCancelled
                        out.write(block)
                        digest.update(block)
                        written += len(block)
                    out.flush()
                    os.fsync(out.fileno())
                if written != dll.size or digest.hexdigest() != dll.sha256:
                    tmp.unlink(missing_ok=True)
                    log.error("%s from %s doesn't match its pinned checksum", dll.name, wheel.filename)
                    raise DownloadError(MSG_CHECKSUM)
                os.replace(tmp, dest)
    except (zipfile.BadZipFile, KeyError) as exc:
        log.error("Couldn't unpack %s: %s", wheel.filename, exc)
        wheel_path.unlink(missing_ok=True)
        raise DownloadError(MSG_CHECKSUM) from exc
    except OSError as exc:
        log.error("Couldn't unpack %s: %s", wheel.filename, exc)
        raise DownloadError(MSG_WRITE.format(folder=staging)) from exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
