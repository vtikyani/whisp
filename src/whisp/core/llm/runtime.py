"""LlmRuntime: downloads, starts and supervises the built-in local AI server.

The server is ``llama-server.exe`` from a pinned llama.cpp release, run as a
child process. Keeping it out of the Whisp process means a native crash in
the model code can't take the app down, and Ollama, LM Studio and the
built-in server all speak the same OpenAI-style HTTP API.

**Which build.** One tested llama.cpp build (``LLAMA_BUILD``) is pinned. The
flavour depends on the hardware: NVIDIA with driver 580+ and a Turing-or-newer
GPU gets the CUDA 13 build, other NVIDIA cards with driver 527.41+ the CUDA 12
build, AMD/Intel graphics cards the Vulkan build, and everything else the CPU
build (pinned to the performance cores). CUDA builds ship with their runtime
DLLs in a second zip.

**Downloads** resume after interruptions (HTTP Range), can be cancelled, check
free disk space first, verify every file's SHA-256 (from the GitHub release
``digest`` and the Hugging Face ``lfs.oid``) and only then move it into place.
Only the server and the libraries it loads are unpacked from the zips (not the
other llama.cpp tools), into a temporary folder that is renamed when complete.

**Launch.** ``127.0.0.1`` on a free port, 8K context, one slot, thinking off,
a capped prompt cache, no web UI, no ``/slots``, CORS limited to localhost,
offline, logging to a file, plus the model's own flags (MTP). A random API
key is generated for every launch (``LLAMA_API_KEY``), so no other program or
web page can use the server. Its environment is an allowlist (no variable
can load code into it or move its port). The process starts suspended, joins
a Windows Job Object with KILL_ON_JOB_CLOSE, and only then runs, so it can
never outlive Whisp — not even if Whisp crashes. Uninstalling deletes only
folders of catalog models inside the models folder.

**Supervision.** ``/health`` is polled until it answers 200. If the server
dies, it is restarted after 1, 5 and 15 seconds; after that the status turns
to ERROR with a plain explanation. Everything that waits (start, stop, health
checks) runs on the ``whisp-llm`` worker thread; downloads run on their own
short-lived thread. Public methods only queue work and return.

The server only starts after ``ensure_running()`` has been called once (so the
app can load the speech model first and the server's ``--fit`` sees the real
free VRAM), and only while ``polish.enabled`` is on, ``polish.provider`` is
``"builtin"`` and a model is installed. Turning polish off, switching provider
or choosing another model stops or restarts it.

The server log never contains dictated text at the default verbosity; it is
kept in ``<runtime_dir>/llama/logs``.
"""

from __future__ import annotations

import contextlib
import ctypes
import enum
import errno
import fnmatch
import functools
import hashlib
import json
import logging
import os
import queue
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
import zipfile
from collections.abc import Callable, Mapping
from ctypes import wintypes
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from PySide6.QtCore import QObject, Qt, Signal

from whisp import __version__
from whisp.core import hardware
from whisp.core.llm.catalog import CATALOG, LlmFile, LlmModelInfo, get_model
from whisp.core.net import is_certificate_error, tls_context

if TYPE_CHECKING:
    from whisp.storage.settings import SettingsStore

log = logging.getLogger(__name__)

# -- pinned server build --------------------------------------------------------------

LLAMA_BUILD = "b11429"
"""llama.cpp build behind release v0.6.0 (2026-10-05). Bump deliberately: re-test, then update every hash below."""

_RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{build}/{name}"


@dataclass(frozen=True)
class ServerAsset:
    name: str
    size: int
    sha256: str
    """From the GitHub release asset's ``digest``."""

    @property
    def url(self) -> str:
        return _RELEASE_URL.format(build=LLAMA_BUILD, name=self.name)


@dataclass(frozen=True)
class ServerFlavor:
    key: str
    label: str
    """Where the model runs, for status messages ("your NVIDIA graphics card")."""
    assets: tuple[ServerAsset, ...]
    uses_gpu: bool


SERVER_FLAVORS: dict[str, ServerFlavor] = {
    f.key: f
    for f in (
        ServerFlavor(
            key="cuda-13.4",
            label="NVIDIA graphics card (CUDA 13)",
            assets=(
                ServerAsset(
                    f"llama-{LLAMA_BUILD}-bin-win-cuda-13.4-x64.zip",
                    153_089_864,
                    "76ddc6eff2389570789ed608881efc6977751a722015d8e8c94f302224ff1a3a",
                ),
                ServerAsset(
                    "cudart-llama-bin-win-cuda-13.4-x64.zip",
                    423_535_356,
                    "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
                ),
            ),
            uses_gpu=True,
        ),
        ServerFlavor(
            key="cuda-12.4",
            label="NVIDIA graphics card (CUDA 12)",
            assets=(
                ServerAsset(
                    f"llama-{LLAMA_BUILD}-bin-win-cuda-12.4-x64.zip",
                    264_478_423,
                    "dd6df685c1024e6aa55ca55ad35038692d5e284d4dc895e7a646c3245d3d14ff",
                ),
                ServerAsset(
                    "cudart-llama-bin-win-cuda-12.4-x64.zip",
                    391_443_627,
                    "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
                ),
            ),
            uses_gpu=True,
        ),
        ServerFlavor(
            key="vulkan",
            label="graphics card (Vulkan)",
            assets=(
                ServerAsset(
                    f"llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip",
                    33_337_769,
                    "1bfe78ad9168b79fa02bf67f6af9f5e17a966d824d77238517f7bef12ac73b36",
                ),
            ),
            uses_gpu=True,
        ),
        ServerFlavor(
            key="cpu",
            label="processor (CPU)",
            assets=(
                ServerAsset(
                    f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip",
                    19_398_918,
                    "1283323272b04cd07905816a597a0da810918102de958f4ff6f7bbaa70ed2efe",
                ),
            ),
            uses_gpu=False,
        ),
    )
}

SERVER_EXE = "llama-server.exe"
HOST = "127.0.0.1"
CONTEXT_SIZE = 8192
CACHE_RAM_MB = 1024
"""Prompt cache in system RAM. The default (8 GB) is far more than dictation needs."""

_MIN_DRIVER_CUDA13 = 580
_MIN_DRIVER_CUDA12 = (527, 41)
_MIN_CC_CUDA13 = (7, 5)
"""CUDA 13 dropped Maxwell, Pascal and Volta; those cards still run the CUDA 12 build."""
_MIN_CC_CUDA12 = (5, 0)

START_TIMEOUT_S = 120.0
"""Longest a model may take to load (a cold read of a 5 GB file from a slow disk fits comfortably)."""
_HEALTH_POLL_S = 0.2
_RUNNING_POLL_S = 1.0
RESTART_BACKOFF_S = (1.0, 5.0, 15.0)
_STABLE_AFTER_S = 120.0
"""A server that ran this long before crashing resets the restart count."""
_STOP_WAIT_S = 5.0
_SHUTDOWN_WAIT_S = 3.0
_DOWNLOAD_RETRIES = (1.0, 3.0, 8.0)
_PROGRESS_INTERVAL_S = 0.1
_CHUNK = 1 << 20
_DISK_MARGIN = 256 * 1024 * 1024
_MARKER = "install.json"

# -- user-facing messages ---------------------------------------------------------------

MSG_DISABLED = "AI polish is off."
MSG_OTHER_PROVIDER = "AI polish is set to use another AI server."
MSG_WAITING = "The built-in AI will start in a moment."
MSG_STOPPED = "The built-in AI is stopped."
MSG_NO_MODEL = "Download a model in Styles → AI polish to use the built-in AI."
MSG_RESTARTING = "The built-in AI stopped unexpectedly. Restarting…"
MSG_GIVE_UP = "The built-in AI keeps stopping. Restart Whisp or pick a smaller model in Styles → AI polish."
MSG_OUT_OF_MEMORY = (
    "There isn't enough graphics memory for the AI model. Close other graphics-heavy apps or pick a smaller model."
)
MSG_GPU_FAILED = "Your graphics card couldn't run the AI model. Updating the graphics driver usually fixes this."
MSG_MODEL_BROKEN = "The AI model file couldn't be loaded. Remove the model and download it again."
MSG_START_TIMEOUT = "The built-in AI took too long to start."
MSG_CANT_LAUNCH = "Couldn't start the built-in AI. Your antivirus may be blocking llama-server.exe."
MSG_INTERNAL = "Something went wrong with the built-in AI. Restart Whisp to try again."
MSG_CANCELLED = "Download cancelled."
MSG_BUSY = "Another download is in progress. Wait for it to finish or cancel it."
MSG_UNKNOWN_MODEL = "That model isn't in Whisp's list of AI models."
MSG_NETWORK = "Couldn't connect to the download server. Check your internet connection and try again."
MSG_CERTIFICATE = (
    "The download server's security certificate couldn't be verified, so nothing was downloaded. "
    "On a work network, ask IT to add the network's certificate to Windows."
)
MSG_INTERRUPTED = "The download was interrupted. Try again to continue where it stopped."
MSG_CORRUPT = "The downloaded file was damaged. Please try again."
MSG_DOWNLOAD_FAILED = "The download failed unexpectedly. Please try again."


class LlmState(enum.Enum):
    OFF = "off"
    """Not wanted right now: polish is off, another provider is chosen, or the server was stopped."""
    NOT_INSTALLED = "not_installed"
    """The chosen model, or the engine for this PC's hardware, isn't downloaded."""
    STARTING = "starting"
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True)
class LlmStatus:
    state: LlmState
    model_id: str | None = None
    base_url: str | None = None
    """``http://127.0.0.1:<port>`` while READY."""
    message: str | None = None
    """Plain-language explanation for the UI."""
    backend: str | None = None
    """Where the model runs, e.g. "NVIDIA graphics card (CUDA 13)"."""
    installing: str | None = None
    """Model id whose download is in progress, if any."""


class DownloadError(Exception):
    """A download failed; the message is user-facing."""


class DownloadCancelled(Exception):
    """The user cancelled a download."""


# -- hardware -> server flavour ------------------------------------------------------------


def _driver_version(text: str) -> tuple[int, int]:
    parts = re.findall(r"\d+", text)
    if not parts:
        return (0, 0)
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def choose_flavor(hw: hardware.Hardware, compute_capability: tuple[int, int] | None) -> str:
    """The llama.cpp build to use on this machine (a key of ``SERVER_FLAVORS``).

    ``compute_capability`` is the NVIDIA GPU's CUDA compute capability, or
    None when unknown (then the driver version decides alone).
    """
    if hw.nvidia is not None:
        driver = _driver_version(hw.nvidia.driver_version)
        if driver[0] >= _MIN_DRIVER_CUDA13 and (compute_capability is None or compute_capability >= _MIN_CC_CUDA13):
            return "cuda-13.4"
        if driver >= _MIN_DRIVER_CUDA12 and (compute_capability is None or compute_capability >= _MIN_CC_CUDA12):
            return "cuda-12.4"
        return "vulkan"
    if hw.has_discrete_non_nvidia or any(g.vendor == "nvidia" and g.vram_mb >= 1024 for g in hw.gpus):
        # AMD/Intel cards, or an NVIDIA card whose driver tools are missing: Vulkan works with any recent driver.
        return "vulkan"
    return "cpu"


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


@functools.cache
def nvidia_compute_capability() -> tuple[int, int] | None:
    """CUDA compute capability of the NVIDIA GPU with the most memory (via NVML), or None if unknown."""
    nvml = None
    for path in (os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "nvml.dll"), "nvml.dll"):
        try:
            nvml = ctypes.WinDLL(path)
            break
        except OSError:
            continue
    if nvml is None:
        return None
    try:
        nvml.nvmlInit_v2.restype = ctypes.c_int
        if nvml.nvmlInit_v2() != 0:
            return None
    except AttributeError:
        return None
    try:
        nvml.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        nvml.nvmlDeviceGetCudaComputeCapability.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
        count = ctypes.c_uint(0)
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return None
        best: tuple[int, tuple[int, int]] | None = None
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                continue
            major, minor = ctypes.c_int(0), ctypes.c_int(0)
            if nvml.nvmlDeviceGetCudaComputeCapability(handle, ctypes.byref(major), ctypes.byref(minor)) != 0:
                continue
            memory = _NvmlMemory()
            total = memory.total if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0 else 0
            if best is None or total > best[0]:
                best = (total, (major.value, minor.value))
        return best[1] if best else None
    except (AttributeError, OSError):
        log.exception("Could not read the NVIDIA GPU's compute capability")
        return None
    finally:
        with contextlib.suppress(AttributeError, OSError):
            nvml.nvmlShutdown()


def detect_flavor() -> str:
    """Choose the server build for this PC. Probes the hardware: call from a worker thread."""
    hw = hardware.detect()
    cc = nvidia_compute_capability() if hw.has_nvidia else None
    flavor = choose_flavor(hw, cc)
    log.info("Local AI engine: %s build (compute capability %s)", flavor, ".".join(map(str, cc)) if cc else "n/a")
    return flavor


# -- launch arguments ---------------------------------------------------------------------


def find_free_port(host: str = HOST) -> int:
    """A TCP port that is free on ``host`` right now (the OS picks it from the ephemeral range)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def build_server_args(
    *,
    model: LlmModelInfo,
    model_path: Path,
    draft_path: Path | None,
    port: int,
    flavor: str,
    threads: int,
    log_file: Path,
) -> list[str]:
    """Command-line arguments for ``llama-server.exe`` (without the executable itself)."""
    args = [
        "-m", str(model_path),
        "--host", HOST,
        "--port", str(port),
        "-c", str(CONTEXT_SIZE),
        "-np", "1",
        "-fa", "auto",
        "--reasoning", "off",
        "--cache-ram", str(CACHE_RAM_MB),
        "--no-webui",
        "--no-slots",
        "--cors-origins", "localhost",
        "--offline",
        "--log-file", str(log_file),
        "--log-colors", "off",
    ]  # fmt: skip
    for arg in model.server_args:
        if "{draft}" in arg:
            if draft_path is None:
                raise ValueError(f"{model.id} needs a draft model file")
            arg = arg.replace("{draft}", str(draft_path))
        args.append(arg)
    if flavor == "cpu":
        args += ["-t", str(max(1, threads))]
    return args


_SERVER_ENVIRONMENT = (
    "SystemRoot",
    "windir",
    "TEMP",
    "TMP",
    "APPDATA",
    "LOCALAPPDATA",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
)
"""What the server gets from Whisp's environment: Windows' own folders, the temp folders, the user folders GPU
drivers keep their caches in, and the user's CUDA device choice. Nothing that can load code into it or change
how it runs (GGML_BACKEND_PATH, VK_*LAYER*, LLAMA_ARG_*, AIP_HTTP_PORT...)."""


def server_environment(api_key: str, server_dir: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the server: an allowlist of Whisp's own, PATH limited to the server and Windows folders,
    and the per-launch API key."""
    source = {key.upper(): value for key, value in (os.environ if base is None else base).items()}
    env = {name: source[name.upper()] for name in _SERVER_ENVIRONMENT if name.upper() in source}
    system_root = env.setdefault("SystemRoot", r"C:\Windows")
    env["PATH"] = os.pathsep.join((str(server_dir), os.path.join(system_root, "System32"), system_root))
    env["LLAMA_API_KEY"] = api_key
    return env


# -- Windows job object --------------------------------------------------------------------

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0400
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_THREAD_SUSPEND_RESUME = 0x0002
_TH32CS_SNAPTHREAD = 0x00000004
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_CREATE_SUSPENDED = 0x00000004


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


_CreateJobObjectW = _kernel32.CreateJobObjectW
_CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_CreateJobObjectW.restype = wintypes.HANDLE
_SetInformationJobObject = _kernel32.SetInformationJobObject
_SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
_SetInformationJobObject.restype = wintypes.BOOL
_AssignProcessToJobObject = _kernel32.AssignProcessToJobObject
_AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_AssignProcessToJobObject.restype = wintypes.BOOL
_TerminateJobObject = _kernel32.TerminateJobObject
_TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_TerminateJobObject.restype = wintypes.BOOL
_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL
_OpenProcess = _kernel32.OpenProcess
_OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_OpenProcess.restype = wintypes.HANDLE
_CreateToolhelp32Snapshot = _kernel32.CreateToolhelp32Snapshot
_CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_Thread32First = _kernel32.Thread32First
_Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
_Thread32First.restype = wintypes.BOOL
_Thread32Next = _kernel32.Thread32Next
_Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)]
_Thread32Next.restype = wintypes.BOOL
_OpenThread = _kernel32.OpenThread
_OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_OpenThread.restype = wintypes.HANDLE
_ResumeThread = _kernel32.ResumeThread
_ResumeThread.argtypes = [wintypes.HANDLE]
_ResumeThread.restype = wintypes.DWORD


class JobObject:
    """A Windows Job Object that kills every process in it when it is closed.

    Whisp keeps the handle open for its whole life. When Whisp exits or
    crashes, Windows closes the handle and ends the server with it.
    Processes that crash in the job end at once instead of showing an error
    dialog that would keep them alive.
    """

    def __init__(self) -> None:
        handle = _CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        )
        if not _SetInformationJobObject(
            handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = ctypes.get_last_error()
            _CloseHandle(handle)
            raise ctypes.WinError(error)
        self._handle: int | None = handle
        self._lock = threading.Lock()

    def assign(self, pid: int) -> None:
        """Put process ``pid`` into the job. Raises ``OSError`` on failure."""
        with self._lock:
            if self._handle is None:
                raise OSError("the job object is closed")
            process = _OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            if not process:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not _AssignProcessToJobObject(self._handle, process):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                _CloseHandle(process)

    def terminate(self) -> None:
        """Kill every process in the job now."""
        with self._lock:
            if self._handle is not None:
                _TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """Close the job; Windows kills every process still in it. Idempotent."""
        with self._lock:
            if self._handle is not None:
                _CloseHandle(self._handle)
                self._handle = None


def _resume_process(pid: int) -> None:
    """Resume the threads of a process created with CREATE_SUSPENDED."""
    snapshot = _CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
    if not snapshot or snapshot == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    resumed = 0
    try:
        entry = _THREADENTRY32()
        entry.dwSize = ctypes.sizeof(_THREADENTRY32)
        more = _Thread32First(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == pid:
                thread = _OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if not thread:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    if _ResumeThread(thread) == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                    resumed += 1
                finally:
                    _CloseHandle(thread)
            more = _Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        _CloseHandle(snapshot)
    if resumed == 0:
        raise OSError(f"process {pid} has no threads to resume")


def spawn_in_job(job: JobObject, args: list[str], *, cwd: Path | None, env: dict[str, str] | None) -> subprocess.Popen:
    """Start ``args`` hidden and suspended, put it into ``job``, then let it run.

    Because the process only runs once it is in the job, there is no moment
    in which it could outlive Whisp.
    """
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW | _CREATE_SUSPENDED,
    )
    try:
        job.assign(proc.pid)
        _resume_process(proc.pid)
    except OSError:
        proc.kill()
        proc.wait()
        raise
    return proc


# -- downloads -----------------------------------------------------------------------------


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    return f"{max(1, round(n / 1024**2))} MB"


def ensure_free_space(path: Path, needed: int) -> None:
    """Raise ``DownloadError`` unless ``path``'s drive has ``needed`` bytes free (plus a small margin)."""
    free = _free_bytes(path)
    if free < needed + _DISK_MARGIN:
        drive = path.drive or str(path.anchor) or str(path)
        raise DownloadError(
            f"Not enough free disk space on {drive}: {_format_size(needed + _DISK_MARGIN)} is needed "
            f"and {_format_size(free)} is free."
        )


def _http_error_message(status: int) -> str:
    if status in (401, 403, 404, 410):
        return f"The download is no longer available (HTTP {status}). Please update Whisp."
    if status == 429 or status >= 500:
        return f"The download server is busy (HTTP {status}). Try again in a few minutes."
    return f"The download failed (HTTP {status}). Please try again."


class _Retry(Exception):
    """A transient problem; the download is retried from where it stopped."""


def _replace_with_retry(src: Path, dst: Path) -> None:
    # Antivirus scanners briefly lock freshly written files; wait for the lock to go away.
    for attempt in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def download_file(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    size: int,
    sha256: str,
    cancel: threading.Event,
    on_bytes: Callable[[int], None],
) -> None:
    """Download ``url`` to ``dest``, resuming a previous partial download, and verify its SHA-256.

    Data goes to ``dest.part`` first and is moved into place only after the
    hash matches. ``on_bytes(n)`` reports progress in increments (bytes found
    in an existing partial file count too). Raises ``DownloadCancelled`` or
    ``DownloadError`` (user-facing message).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    hash_failures = 0
    attempt = 0
    while True:
        counted = 0

        def count(n: int) -> None:
            nonlocal counted
            counted += n
            on_bytes(n)

        try:
            digest = _download_attempt(client, url, part, size=size, cancel=cancel, on_bytes=count)
        except _Retry as exc:
            on_bytes(-counted)  # the next attempt reports what is on disk again
            if attempt >= len(_DOWNLOAD_RETRIES):
                raise DownloadError(str(exc) or MSG_INTERRUPTED) from exc
            delay = _DOWNLOAD_RETRIES[attempt]
            attempt += 1
            reason = type(exc.__cause__ or exc).__name__
            log.warning("Download of %s interrupted (%s); retrying in %.0f s", dest.name, reason, delay)
            if cancel.wait(delay):
                raise DownloadCancelled() from None
            continue
        if digest != sha256.lower():
            part.unlink(missing_ok=True)
            on_bytes(-counted)
            hash_failures += 1
            log.warning("Download of %s failed verification (attempt %d)", dest.name, hash_failures)
            if hash_failures >= 2:
                raise DownloadError(MSG_CORRUPT)
            continue
        _replace_with_retry(part, dest)
        return


def _part_size(part: Path) -> int:
    try:
        return part.stat().st_size
    except FileNotFoundError:
        return 0


def _download_attempt(
    client: httpx.Client,
    url: str,
    part: Path,
    *,
    size: int,
    cancel: threading.Event,
    on_bytes: Callable[[int], None],
) -> str:
    """One pass: hash what is already on disk, fetch the rest. Returns the SHA-256 of the whole file."""
    existing = _part_size(part)
    if existing > size:
        part.unlink()
        existing = 0
    hasher = hashlib.sha256()
    if existing:
        with open(part, "rb") as fh:
            while chunk := fh.read(_CHUNK):
                if cancel.is_set():
                    raise DownloadCancelled()
                hasher.update(chunk)
        on_bytes(existing)
    if existing == size:
        return hasher.hexdigest()
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    try:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416:
                part.unlink(missing_ok=True)
                raise _Retry(MSG_INTERRUPTED)
            if response.status_code == 206:
                content_range = response.headers.get("content-range", "")
                if not content_range.startswith(f"bytes {existing}-"):
                    part.unlink(missing_ok=True)
                    raise _Retry(MSG_INTERRUPTED)
                mode = "ab"
            elif response.status_code == 200:
                if existing:
                    on_bytes(-existing)
                    hasher = hashlib.sha256()
                    existing = 0
                mode = "wb"
            else:
                message = _http_error_message(response.status_code)
                if response.status_code == 429 or response.status_code >= 500:
                    raise _Retry(message)
                raise DownloadError(message)
            received = existing
            with open(part, mode) as fh:
                # Chunks as they arrive (not re-buffered to a fixed size), so an interruption loses almost nothing.
                for chunk in response.iter_bytes():
                    if cancel.is_set():
                        raise DownloadCancelled()
                    fh.write(chunk)
                    hasher.update(chunk)
                    received += len(chunk)
                    on_bytes(len(chunk))
                    if received > size:
                        break
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        if is_certificate_error(exc):
            raise DownloadError(MSG_CERTIFICATE) from exc
        raise _Retry(MSG_NETWORK) from exc
    except httpx.TransportError as exc:
        raise _Retry(MSG_INTERRUPTED) from exc
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise DownloadError(f"The disk holding {part.parent} is full. Free some space and try again.") from exc
        raise DownloadError(f"Couldn't save the download to {part.parent}: {exc.strerror or exc}.") from exc
    if received != size:
        if received > size:
            part.unlink(missing_ok=True)
        raise _Retry(MSG_INTERRUPTED)
    return hasher.hexdigest()


_SERVER_FILES = (
    SERVER_EXE,
    "llama-server-impl.dll",
    "llama.dll",
    "llama-common.dll",
    "mtmd.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu-*.dll",
    "ggml-cuda.dll",
    "ggml-vulkan.dll",
    "libomp*.dll",
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "LICENSE*",
)
"""What is unpacked from a release zip: the server, the libraries it loads (checked against build b11429: it starts,
finds the GPU and answers with only these) and their licences. The other tools in the zip (llama-cli,
ggml-rpc-server...) are left out."""


def _is_server_file(name: str) -> bool:
    """True for a top-level zip entry the server needs (see ``_SERVER_FILES``)."""
    if "/" in name or "\\" in name:
        return False
    return any(fnmatch.fnmatch(name.lower(), pattern.lower()) for pattern in _SERVER_FILES)


def _extract_zip(archive: Path, target: Path, cancel: threading.Event) -> None:
    """Unpack the server's files from ``archive`` into ``target``. A zip with any entry that would land outside
    ``target`` is refused as a whole."""
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if any(not (root / member.filename).resolve().is_relative_to(root) for member in members):
            raise DownloadError(MSG_CORRUPT)
        wanted = [member for member in members if _is_server_file(member.filename)]
        ensure_free_space(target, sum(m.file_size for m in wanted))
        for member in wanted:
            if cancel.is_set():
                raise DownloadCancelled()
            zf.extract(member, root)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _remove_tree(path: Path) -> None:
    def retry_writable(func, target, _exc) -> None:
        os.chmod(target, 0o700)
        func(target)

    # Windows won't delete a file while anyone has it open: a virus scanner, or Whisp itself checking what's
    # installed. Like renames (_replace_with_retry), wait a few seconds for that to pass.
    for attempt in range(20):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=retry_writable)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


# -- the runtime ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Config:
    model_id: str
    flavor: str


class _Phase(enum.Enum):
    IDLE = "idle"
    STARTING = "starting"
    READY = "ready"
    WAITING = "waiting"
    """Waiting to restart after a crash."""


@dataclass(frozen=True, slots=True)
class _Ensure:
    """``ensure_running()``: start the server if it should run."""


@dataclass(frozen=True, slots=True)
class _SettingsChanged:
    """The polish settings changed: start, stop or switch models to match."""


@dataclass(frozen=True, slots=True)
class _Stop:
    """``stop()``: stop the server until asked again."""


@dataclass(frozen=True, slots=True)
class _InstallDone:
    """A download finished (or failed); the server may be able to start now."""

    model_id: str


@dataclass(frozen=True, slots=True)
class _Uninstall:
    model_id: str


@dataclass(frozen=True, slots=True)
class _Shutdown:
    """Stop the server and end the worker thread."""


_Command = _Ensure | _SettingsChanged | _Stop | _InstallDone | _Uninstall | _Shutdown

_OOM_RE = re.compile(r"out of memory|failed to allocate|cudaMalloc failed|ErrorOutOfDeviceMemory", re.IGNORECASE)
_GPU_RE = re.compile(r"CUDA error|no CUDA-capable device|ggml_cuda_init: failed|vk::|Vulkan error", re.IGNORECASE)
_MODEL_RE = re.compile(r"failed to load model|error loading model|invalid magic|gguf_init", re.IGNORECASE)


class LlmRuntime(QObject):
    """Manages the built-in llama.cpp server. See the module docstring."""

    status_changed = Signal(object)
    """LlmStatus. Emitted from worker threads."""
    progress = Signal(str, object, object)
    """model_id, bytes_done, bytes_total (Python ints) for an install: engine and model files together."""
    install_finished = Signal(str)
    """model_id: the model (and the engine for this PC) is ready to use."""
    install_failed = Signal(str, str)
    """model_id, user-facing message (including "Download cancelled.")."""

    def __init__(
        self,
        settings: SettingsStore,
        runtime_dir: Path,
        models_dir: Path,
        parent: QObject | None = None,
        *,
        flavor: str | None = None,
    ) -> None:
        """Engines go to ``runtime_dir/llama/<build>/<flavor>``, models to ``models_dir/<model id>``.

        The app passes ``AppPaths.runtime_dir`` and ``AppPaths.models_dir / "llm"``. ``flavor`` forces a
        server build (a key of ``SERVER_FLAVORS``); None picks one for this PC.
        """
        super().__init__(parent)
        if flavor is not None and flavor not in SERVER_FLAVORS:
            raise ValueError(f"unknown server flavor: {flavor}")
        self._settings = settings
        self._engine_root = runtime_dir / "llama"
        self._models_root = models_dir
        self._forced_flavor = flavor
        self._flavor_cache: str | None = None
        self._lock = threading.RLock()
        self._status = LlmStatus(LlmState.OFF, message=MSG_WAITING)
        self._endpoint: tuple[str, str] | None = None
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._closed = False
        self._job = JobObject()
        self._installing: str | None = None
        self._download_thread: threading.Thread | None = None
        self._download_cancel = threading.Event()
        # Owned by the worker thread.
        self._proc: subprocess.Popen | None = None
        self._config: _Config | None = None
        self._phase = _Phase.IDLE
        self._started_at = 0.0
        self._ready_at = 0.0
        self._restart_at = 0.0
        self._crashes = 0
        self._activated = False
        self._manual_stop = False
        self._port = 0
        self._api_key = ""
        # A local connection is instant once the server listens; before that, Windows takes a while to refuse it.
        self._health = httpx.Client(trust_env=False, timeout=httpx.Timeout(1.0, connect=0.25))
        settings.changed.connect(self._on_settings_changed, Qt.ConnectionType.DirectConnection)
        self._thread = threading.Thread(target=self._run, name="whisp-llm", daemon=True)
        self._thread.start()
        # Work out the initial status (off, not installed, or waiting for ensure_running) without starting anything.
        self._commands.put(_SettingsChanged())

    # -- public API (any thread; never blocks) -------------------------------------------

    def catalog(self) -> list[LlmModelInfo]:
        return list(CATALOG)

    @property
    def status(self) -> LlmStatus:
        with self._lock:
            return self._status

    def base_url(self) -> str | None:
        """``http://127.0.0.1:<port>`` while the server is ready, else None."""
        with self._lock:
            return self._endpoint[0] if self._endpoint else None

    def api_key(self) -> str | None:
        """The per-launch API key while the server is ready, else None. Never log it."""
        with self._lock:
            return self._endpoint[1] if self._endpoint else None

    def model_dir(self, model_id: str) -> Path:
        return self._models_root / model_id

    def is_installed(self, model_id: str) -> bool:
        """True when every file of ``model_id`` is downloaded and verified. Cheap: checks sizes and a marker."""
        try:
            model = get_model(model_id)
        except KeyError:
            return False
        marker = _read_json(self.model_dir(model_id) / _MARKER)
        if not marker:
            return False
        recorded = marker.get("files", {})
        for f in model.files:
            path = self.model_dir(model_id) / f.name
            if recorded.get(f.name) != f.sha256:
                return False
            try:
                if path.stat().st_size != f.size:
                    return False
            except OSError:
                return False
        return True

    def installed_models(self) -> list[str]:
        """Ids of installed models, in catalog order (recommended first)."""
        return [m.id for m in CATALOG if self.is_installed(m.id)]

    def install(self, model_id: str) -> None:
        """Download the model and, if needed, the server build for this PC. Async; see the signals."""
        try:
            model = get_model(model_id)
        except KeyError:
            self.install_failed.emit(model_id, MSG_UNKNOWN_MODEL)
            return
        with self._lock:
            if self._closed:
                return
            if self._download_thread is not None and self._download_thread.is_alive():
                if self._installing != model_id:
                    self.install_failed.emit(model_id, MSG_BUSY)
                return
            cancel = threading.Event()
            self._download_cancel = cancel
            self._installing = model_id
            thread = threading.Thread(
                target=self._install_worker, args=(model, cancel), name="whisp-llm-download", daemon=True
            )
            self._download_thread = thread
            self._status = replace(self._status, installing=model_id)
            status = self._status
        self.status_changed.emit(status)
        thread.start()

    def cancel_install(self) -> None:
        """Stop the running download. The partial file is kept, so the next install resumes it."""
        self._download_cancel.set()

    def uninstall(self, model_id: str) -> None:
        """Delete a model (stopping the server first if it uses it). Removes the engine with the last model.

        Only models in Whisp's catalog: anything else is refused, so no other folder can be deleted.
        """
        if self._owned_model_dir(model_id) is None:
            log.warning("Not uninstalling %r: it isn't one of Whisp's AI models", model_id)
            return
        self._commands.put(_Uninstall(model_id))

    def ensure_running(self) -> None:
        """Start the server if polish is on, the provider is "builtin" and a model is installed. Async."""
        self._commands.put(_Ensure())

    def stop(self) -> None:
        """Stop the server until ``ensure_running()`` is called again or the polish settings change. Async."""
        self._commands.put(_Stop())

    def shutdown(self) -> None:
        """Stop everything: downloads, the server and the worker thread. Waits briefly; idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            download = self._download_thread
        with contextlib.suppress(RuntimeError, TypeError):
            self._settings.changed.disconnect(self._on_settings_changed)
        self._download_cancel.set()
        self._commands.put(_Shutdown())
        self._thread.join(timeout=_SHUTDOWN_WAIT_S)
        if download is not None:
            download.join(timeout=1.0)
        # Closing the job kills the server even if the worker didn't get to it.
        self._job.close()
        self._health.close()

    # -- settings ------------------------------------------------------------------------

    def _on_settings_changed(self, sections: object) -> None:
        if isinstance(sections, (set, frozenset, list, tuple)) and "polish" not in sections:
            return
        self._commands.put(_SettingsChanged())

    # -- status --------------------------------------------------------------------------

    def _set_status(self, status: LlmStatus) -> None:
        with self._lock:
            status = replace(status, installing=self._installing)
            if status == self._status:
                return
            self._status = status
        self.status_changed.emit(status)

    def _flavor(self) -> str:
        """The server build for this PC. Probes the hardware once: worker and download threads only."""
        if self._forced_flavor:
            return self._forced_flavor
        with self._lock:
            if self._flavor_cache is None:
                self._flavor_cache = detect_flavor()
            return self._flavor_cache

    def _engine_dir(self, flavor: str) -> Path:
        return self._engine_root / LLAMA_BUILD / flavor

    def _engine_installed(self, flavor: str) -> bool:
        directory = self._engine_dir(flavor)
        marker = _read_json(directory / _MARKER)
        if not marker or marker.get("build") != LLAMA_BUILD or marker.get("flavor") != flavor:
            return False
        expected = {a.name: a.sha256 for a in SERVER_FLAVORS[flavor].assets}
        return marker.get("assets") == expected and (directory / SERVER_EXE).is_file()

    # -- worker thread ---------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            timeout = self._poll_interval()
            try:
                command = self._commands.get(timeout=timeout) if timeout is not None else self._commands.get()
            except queue.Empty:
                command = None
            try:
                if isinstance(command, _Shutdown):
                    self._stop_server()
                    return
                if command is not None:
                    self._handle(command)
                self._tick()
            except Exception:
                log.exception("Local AI runtime error")
                self._stop_server()
                self._phase = _Phase.IDLE
                self._set_status(LlmStatus(LlmState.ERROR, message=MSG_INTERNAL))

    def _poll_interval(self) -> float | None:
        if self._phase is _Phase.STARTING:
            return _HEALTH_POLL_S
        if self._phase is _Phase.READY:
            return _RUNNING_POLL_S
        if self._phase is _Phase.WAITING:
            return max(0.05, self._restart_at - time.monotonic())
        return None

    def _handle(self, command: _Command) -> None:
        if isinstance(command, _Ensure):
            self._activated = True
            self._manual_stop = False
            if self._phase is _Phase.IDLE:
                self._crashes = 0
            self._reconcile()
        elif isinstance(command, _SettingsChanged):
            desired, _ = self._desired()
            if desired != self._config:
                # A real change (on/off, provider, model): forget an earlier stop and earlier crashes.
                self._manual_stop = False
                self._crashes = 0
            self._reconcile()
        elif isinstance(command, _Stop):
            self._manual_stop = True
            self._stop_server()
            self._set_status(LlmStatus(LlmState.OFF, message=MSG_STOPPED))
        elif isinstance(command, _InstallDone):
            self._crashes = 0
            self._reconcile()
        elif isinstance(command, _Uninstall):
            self._uninstall(command.model_id)

    def _desired(self) -> tuple[_Config | None, LlmStatus | None]:
        """What should be running now, or None and the status that explains why nothing should."""
        polish = self._settings.current.polish
        if not polish.enabled:
            return None, LlmStatus(LlmState.OFF, message=MSG_DISABLED)
        if polish.provider != "builtin":
            return None, LlmStatus(LlmState.OFF, message=MSG_OTHER_PROVIDER)
        model_id = polish.builtin_model
        if model_id:
            try:
                model = get_model(model_id)
            except KeyError:
                return None, LlmStatus(LlmState.NOT_INSTALLED, message=MSG_NO_MODEL)
            if not self.is_installed(model_id):
                message = f"Download {model.name} in Styles → AI polish to use it."
                return None, LlmStatus(LlmState.NOT_INSTALLED, model_id=model_id, message=message)
        else:
            installed = self.installed_models()
            if not installed:
                return None, LlmStatus(LlmState.NOT_INSTALLED, message=MSG_NO_MODEL)
            model_id = installed[0]
        flavor = self._flavor()
        if not self._engine_installed(flavor):
            label = SERVER_FLAVORS[flavor].label
            message = f"The AI engine for your {label} isn't downloaded yet. Download the model again to get it."
            return None, LlmStatus(LlmState.NOT_INSTALLED, model_id=model_id, message=message, backend=label)
        return _Config(model_id, flavor), None

    def _reconcile(self) -> None:
        desired, reason = self._desired()
        if desired is None:
            self._stop_server()
            assert reason is not None
            self._set_status(reason)
            return
        if self._config is not None and self._config != desired:
            self._stop_server()
        if self._config == desired and self._phase is not _Phase.IDLE:
            return  # already starting, running or about to restart
        if self._manual_stop:
            self._set_status(LlmStatus(LlmState.OFF, model_id=desired.model_id, message=MSG_STOPPED))
            return
        if not self._activated:
            self._set_status(LlmStatus(LlmState.OFF, model_id=desired.model_id, message=MSG_WAITING))
            return
        self._start(desired)

    def _start(self, config: _Config) -> None:
        model = get_model(config.model_id)
        flavor = SERVER_FLAVORS[config.flavor]
        engine = self._engine_dir(config.flavor)
        model_dir = self.model_dir(model.id)
        logs = self._engine_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_file = logs / "llama-server.log"
        if log_file.exists():
            with contextlib.suppress(OSError):
                os.replace(log_file, logs / "llama-server.previous.log")
        self._port = find_free_port()
        self._api_key = secrets.token_urlsafe(32)
        args = build_server_args(
            model=model,
            model_path=model_dir / model.model_file.name,
            draft_path=model_dir / model.draft_file.name if model.draft_file else None,
            port=self._port,
            flavor=config.flavor,
            threads=hardware.detect().performance_cores,
            log_file=log_file,
        )
        self._config = config
        self._phase = _Phase.STARTING
        self._started_at = time.monotonic()
        self._set_status(
            LlmStatus(
                LlmState.STARTING,
                model_id=model.id,
                message=f"Starting {model.name} on your {flavor.label}…",
                backend=flavor.label,
            )
        )
        log.info("Starting llama-server %s (%s) with %s on port %d", LLAMA_BUILD, config.flavor, model.id, self._port)
        try:
            self._proc = spawn_in_job(
                self._job,
                [str(engine / SERVER_EXE), *args],
                cwd=engine,
                env=server_environment(self._api_key, engine),
            )
        except OSError:
            log.exception("Could not launch llama-server")
            self._proc = None
            self._on_failure(MSG_CANT_LAUNCH)

    def _tick(self) -> None:
        if self._phase is _Phase.STARTING:
            assert self._proc is not None
            if self._proc.poll() is not None:
                self._on_exit()
                return
            if self._healthy():
                self._phase = _Phase.READY
                self._ready_at = time.monotonic()
                base_url = f"http://{HOST}:{self._port}"
                with self._lock:
                    self._endpoint = (base_url, self._api_key)
                assert self._config is not None
                model = get_model(self._config.model_id)
                label = SERVER_FLAVORS[self._config.flavor].label
                log.info("llama-server ready in %.1f s", self._ready_at - self._started_at)
                self._set_status(
                    LlmStatus(
                        LlmState.READY,
                        model_id=model.id,
                        base_url=base_url,
                        message=f"{model.name} is ready on your {label}.",
                        backend=label,
                    )
                )
                return
            if time.monotonic() - self._started_at > START_TIMEOUT_S:
                log.warning("llama-server did not become ready within %.0f s", START_TIMEOUT_S)
                self._kill_process()
                self._on_failure(MSG_START_TIMEOUT)
        elif self._phase is _Phase.READY:
            assert self._proc is not None
            if self._proc.poll() is not None:
                self._on_exit()
        elif self._phase is _Phase.WAITING and time.monotonic() >= self._restart_at:
            desired, reason = self._desired()
            if desired is None or self._manual_stop:
                self._phase = _Phase.IDLE
                self._config = None
                if reason is not None:
                    self._set_status(reason)
                return
            self._start(desired)

    def _healthy(self) -> bool:
        try:
            return self._health.get(f"http://{HOST}:{self._port}/health").status_code == 200
        except httpx.HTTPError:
            return False

    def _on_exit(self) -> None:
        assert self._proc is not None
        code = self._proc.returncode
        was_ready = self._phase is _Phase.READY
        tail = self._log_tail()
        log.warning("llama-server exited with code %s; last log lines:\n%s", code, tail)
        self._proc = None
        with self._lock:
            self._endpoint = None
        if was_ready and time.monotonic() - self._ready_at >= _STABLE_AFTER_S:
            self._crashes = 0
        self._on_failure(self._explain(tail))

    def _on_failure(self, message: str) -> None:
        """A start failed or the server died: restart with backoff, or give up."""
        with self._lock:
            self._endpoint = None
        self._crashes += 1
        config = self._config
        if self._crashes > len(RESTART_BACKOFF_S):
            self._phase = _Phase.IDLE
            self._config = None
            self._set_status(
                LlmStatus(LlmState.ERROR, model_id=config.model_id if config else None, message=message or MSG_GIVE_UP)
            )
            return
        self._phase = _Phase.WAITING
        self._restart_at = time.monotonic() + RESTART_BACKOFF_S[self._crashes - 1]
        self._set_status(
            LlmStatus(LlmState.STARTING, model_id=config.model_id if config else None, message=MSG_RESTARTING)
        )

    def _explain(self, tail: str) -> str:
        if _OOM_RE.search(tail):
            return MSG_OUT_OF_MEMORY
        if _MODEL_RE.search(tail):
            return MSG_MODEL_BROKEN
        if _GPU_RE.search(tail):
            return MSG_GPU_FAILED
        return MSG_GIVE_UP

    def _log_tail(self, lines: int = 15) -> str:
        path = self._engine_root / "logs" / "llama-server.log"
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - 8192))
                data = fh.read().decode("utf-8", "replace")
        except OSError:
            return "(no log)"
        return "\n".join(data.splitlines()[-lines:])

    def _kill_process(self) -> None:
        proc, self._proc = self._proc, None
        with self._lock:
            self._endpoint = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                log.warning("llama-server did not exit after being killed; ending its job")
                self._job.terminate()

    def _stop_server(self) -> None:
        if self._proc is not None:
            log.info("Stopping llama-server")
        self._kill_process()
        self._phase = _Phase.IDLE
        self._config = None

    def _owned_model_dir(self, model_id: str) -> Path | None:
        """The folder of a catalog model, if it really is a folder directly inside the models folder."""
        try:
            get_model(model_id)
        except KeyError:
            return None
        folder = self.model_dir(model_id).resolve()
        return folder if folder.parent == self._models_root.resolve() else None

    def _uninstall(self, model_id: str) -> None:
        folder = self._owned_model_dir(model_id)
        if folder is None:
            log.warning("Not uninstalling %r: it isn't one of Whisp's AI models", model_id)
            return
        with self._lock:
            download = self._download_thread if self._installing == model_id else None
        if download is not None:
            self._download_cancel.set()
            download.join(timeout=30.0)
        if self._config is not None and self._config.model_id == model_id:
            self._stop_server()
        try:
            _remove_tree(folder)
        except OSError:
            log.exception("Could not delete the %s model files", model_id)
        with self._lock:
            downloading = self._download_thread is not None and self._download_thread.is_alive()
        if not self.installed_models() and not downloading:
            # No model left: the engine (several hundred MB) has nothing to run.
            try:
                _remove_tree(self._engine_root)
            except OSError:
                log.exception("Could not delete the local AI engine")
        log.info("Uninstalled local AI model %s", model_id)
        self._reconcile()

    # -- downloads (own thread) -------------------------------------------------------------

    def _install_worker(self, model: LlmModelInfo, cancel: threading.Event) -> None:
        started = time.monotonic()
        try:
            self._install(model, cancel)
        except DownloadCancelled:
            log.info("Download of %s cancelled", model.id)
            self.install_failed.emit(model.id, MSG_CANCELLED)
        except DownloadError as exc:
            log.warning("Download of %s failed: %s", model.id, exc)
            self.install_failed.emit(model.id, str(exc))
        except Exception:
            log.exception("Download of %s failed", model.id)
            self.install_failed.emit(model.id, MSG_DOWNLOAD_FAILED)
        else:
            log.info("Installed local AI model %s in %.0f s", model.id, time.monotonic() - started)
            self.install_finished.emit(model.id)
        finally:
            with self._lock:
                self._installing = None
                self._status = replace(self._status, installing=None)
                status = self._status
            self.status_changed.emit(status)
            self._commands.put(_InstallDone(model.id))

    def _install(self, model: LlmModelInfo, cancel: threading.Event) -> None:
        flavor = SERVER_FLAVORS[self._flavor()]
        engine_needed = not self._engine_installed(flavor.key)
        model_dir = self.model_dir(model.id)
        files = [] if self.is_installed(model.id) else list(model.files)
        assets = list(flavor.assets) if engine_needed else []
        total = sum(a.size for a in assets) + sum(f.size for f in files)
        if total == 0:
            return
        staging = self._engine_root / "downloads"
        # Disk space: what is still to download, plus room to unpack the engine.
        remaining_engine = sum(a.size - _part_size(staging / (a.name + ".part")) for a in assets)
        remaining_model = sum(f.size - _part_size(model_dir / (f.name + ".part")) for f in files)
        if assets:
            ensure_free_space(self._engine_root, remaining_engine + 2 * sum(a.size for a in assets))
        if files:
            ensure_free_space(model_dir, remaining_model)
        done = 0
        last_emit = 0.0

        def on_bytes(n: int) -> None:
            nonlocal done, last_emit
            done = max(0, done + n)
            now = time.monotonic()
            if now - last_emit >= _PROGRESS_INTERVAL_S or done >= total:
                last_emit = now
                self.progress.emit(model.id, min(done, total), total)

        # Byte ranges must refer to the file itself, never to a compressed rendition of it.
        headers = {"User-Agent": f"Whisp/{__version__}", "Accept-Encoding": "identity"}
        timeout = httpx.Timeout(30.0, connect=15.0)
        with httpx.Client(verify=tls_context(), follow_redirects=True, headers=headers, timeout=timeout) as client:
            if assets:
                for asset in assets:
                    download_file(
                        client,
                        asset.url,
                        staging / asset.name,
                        size=asset.size,
                        sha256=asset.sha256,
                        cancel=cancel,
                        on_bytes=on_bytes,
                    )
                self._unpack_engine(flavor, staging, cancel)
            if files:
                for f in files:
                    self._download_model_file(client, model_dir, f, cancel, on_bytes)
                _write_json(model_dir / _MARKER, {"model": model.id, "files": {f.name: f.sha256 for f in model.files}})
        self.progress.emit(model.id, total, total)

    def _download_model_file(
        self,
        client: httpx.Client,
        model_dir: Path,
        f: LlmFile,
        cancel: threading.Event,
        on_bytes: Callable[[int], None],
    ) -> None:
        target = model_dir / f.name
        if target.exists() and target.stat().st_size == f.size and _sha256_file(target, cancel) == f.sha256:
            on_bytes(f.size)  # already here (e.g. the marker was lost)
            return
        download_file(client, f.url, target, size=f.size, sha256=f.sha256, cancel=cancel, on_bytes=on_bytes)

    def _unpack_engine(self, flavor: ServerFlavor, staging: Path, cancel: threading.Event) -> None:
        target = self._engine_dir(flavor.key)
        temporary = target.with_name(f"{target.name}.tmp-{secrets.token_hex(4)}")
        temporary.mkdir(parents=True)
        try:
            for asset in flavor.assets:
                _extract_zip(staging / asset.name, temporary, cancel)
            if not (temporary / SERVER_EXE).is_file():
                raise DownloadError(MSG_CORRUPT)
            _write_json(
                temporary / _MARKER,
                {"build": LLAMA_BUILD, "flavor": flavor.key, "assets": {a.name: a.sha256 for a in flavor.assets}},
            )
            _remove_tree(target)
            _replace_with_retry(temporary, target)
        except BaseException:
            with contextlib.suppress(OSError):
                _remove_tree(temporary)
            raise
        for asset in flavor.assets:
            with contextlib.suppress(OSError):
                (staging / asset.name).unlink()
        # Builds from older Whisp versions are no longer used.
        for old in self._engine_root.iterdir():
            if old.is_dir() and re.fullmatch(r"b\d+", old.name) and old.name != LLAMA_BUILD:
                try:
                    _remove_tree(old)
                except OSError:
                    log.warning("Could not remove the old local AI engine %s", old.name)


def _sha256_file(path: Path, cancel: threading.Event) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            if cancel.is_set():
                raise DownloadCancelled()
            hasher.update(chunk)
    return hasher.hexdigest()
