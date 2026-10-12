"""What the machine can run: graphics adapters, NVIDIA driver, CPU core layout.

Adapters come from DXGI, so the list reflects GPUs that are actually present
(the registry can hold stale entries for removed cards). NVIDIA details come
from NVML, which ships with the NVIDIA driver. Results are cached; call
``refresh()`` after hardware changes.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os
from ctypes import wintypes
from dataclasses import dataclass

log = logging.getLogger(__name__)

VENDOR_NVIDIA = 0x10DE
VENDOR_AMD = 0x1002
VENDOR_INTEL = 0x8086
_VENDOR_NAMES = {VENDOR_NVIDIA: "nvidia", VENDOR_AMD: "amd", VENDOR_INTEL: "intel"}


@dataclass(frozen=True)
class GpuAdapter:
    vendor: str
    """``"nvidia"``, ``"amd"``, ``"intel"`` or ``"other"``."""
    name: str
    vram_mb: int
    """Dedicated video memory. Integrated GPUs report little or none."""


# -- DXGI --------------------------------------------------------------------


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", wintypes.UINT),
    ]


_IID_IDXGIFactory1 = _GUID(
    0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87)
)
_DXGI_ERROR_NOT_FOUND = 0x887A0002
_DXGI_ADAPTER_FLAG_SOFTWARE = 2

# Vtable slots: IUnknown (3) + IDXGIObject (4) + IDXGIFactory (5) -> EnumAdapters1 is slot 12.
_SLOT_RELEASE = 2
_SLOT_ENUM_ADAPTERS1 = 12
# IUnknown (3) + IDXGIObject (4) + IDXGIAdapter (3) -> GetDesc1 is slot 10.
_SLOT_GET_DESC1 = 10


def _vtable_call(obj: ctypes.c_void_p, slot: int, restype, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(vtable[slot])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _vtable_call(obj, _SLOT_RELEASE, wintypes.ULONG)(obj)


def _dxgi_adapters() -> list[GpuAdapter]:
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        return []
    create = dxgi.CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    factory = ctypes.c_void_p()
    if create(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory)) != 0:
        return []
    adapters: list[GpuAdapter] = []
    try:
        enum = _vtable_call(
            factory, _SLOT_ENUM_ADAPTERS1, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)
        )
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum(factory, index, ctypes.byref(adapter))
            if (hr & 0xFFFFFFFF) == _DXGI_ERROR_NOT_FOUND or hr != 0:
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                get_desc = _vtable_call(adapter, _SLOT_GET_DESC1, ctypes.c_long, ctypes.POINTER(_DXGI_ADAPTER_DESC1))
                if get_desc(adapter, ctypes.byref(desc)) == 0 and not desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE:
                    adapters.append(
                        GpuAdapter(
                            vendor=_VENDOR_NAMES.get(desc.VendorId, "other"),
                            name=desc.Description.strip(),
                            vram_mb=int(desc.DedicatedVideoMemory // (1024 * 1024)),
                        )
                    )
            finally:
                _release(adapter)
            index += 1
    finally:
        _release(factory)
    return adapters


# -- NVML ----------------------------------------------------------------------


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


@dataclass(frozen=True)
class NvidiaInfo:
    driver_version: str
    """e.g. ``"617.42"``."""
    driver_major: int
    name: str
    vram_total_mb: int
    vram_free_mb: int


def _load_nvml() -> ctypes.WinDLL | None:
    candidates = [os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "nvml.dll"), "nvml.dll"]
    for path in candidates:
        try:
            return ctypes.WinDLL(path)
        except OSError:
            continue
    return None


def _nvidia_info() -> NvidiaInfo | None:
    nvml = _load_nvml()
    if nvml is None:
        return None
    nvml.nvmlInit_v2.restype = ctypes.c_int
    if nvml.nvmlInit_v2() != 0:
        return None
    try:
        buf = ctypes.create_string_buffer(96)
        nvml.nvmlSystemGetDriverVersion.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        if nvml.nvmlSystemGetDriverVersion(buf, len(buf)) != 0:
            return None
        driver = buf.value.decode("ascii", "replace")
        count = ctypes.c_uint(0)
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value == 0:
            return None
        best: NvidiaInfo | None = None
        for i in range(count.value):
            handle = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(handle)) != 0:
                continue
            name = ctypes.create_string_buffer(96)
            nvml.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            nvml.nvmlDeviceGetName(handle, name, len(name))
            mem = _NvmlMemory()
            nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
                continue
            info = NvidiaInfo(
                driver_version=driver,
                driver_major=int(driver.split(".")[0]) if driver.split(".")[0].isdigit() else 0,
                name=name.value.decode("utf-8", "replace"),
                vram_total_mb=int(mem.total // (1024 * 1024)),
                vram_free_mb=int(mem.free // (1024 * 1024)),
            )
            if best is None or info.vram_total_mb > best.vram_total_mb:
                best = info
        return best
    finally:
        nvml.nvmlShutdown()


def nvidia_free_vram_mb() -> int | None:
    """Live free VRAM on the largest NVIDIA GPU (not cached)."""
    info = _nvidia_info()
    return info.vram_free_mb if info else None


# -- CPU -----------------------------------------------------------------------


def _performance_core_count() -> int:
    """Physical cores of the highest efficiency class (P-cores on hybrid Intel CPUs)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetLogicalProcessorInformationEx
    get_info.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    get_info.restype = wintypes.BOOL
    relation_processor_core = 0
    size = wintypes.DWORD(0)
    get_info(relation_processor_core, None, ctypes.byref(size))
    if size.value == 0:
        return max(1, (os.cpu_count() or 2) // 2)
    buf = ctypes.create_string_buffer(size.value)
    if not get_info(relation_processor_core, buf, ctypes.byref(size)):
        return max(1, (os.cpu_count() or 2) // 2)
    classes: list[int] = []
    offset = 0
    while offset < size.value:
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship (DWORD), Size (DWORD), then PROCESSOR_RELATIONSHIP
        # whose second byte is EfficiencyClass.
        entry_size = int.from_bytes(buf.raw[offset + 4 : offset + 8], "little")
        if entry_size == 0:
            break
        classes.append(buf.raw[offset + 9])
        offset += entry_size
    if not classes:
        return max(1, (os.cpu_count() or 2) // 2)
    top = max(classes)
    return sum(1 for c in classes if c == top)


# -- public API ------------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    gpus: tuple[GpuAdapter, ...]
    nvidia: NvidiaInfo | None
    logical_cpus: int
    performance_cores: int

    @property
    def best_gpu(self) -> GpuAdapter | None:
        return max(self.gpus, key=lambda g: g.vram_mb, default=None)

    @property
    def has_nvidia(self) -> bool:
        return self.nvidia is not None

    @property
    def has_discrete_non_nvidia(self) -> bool:
        """An AMD/Intel GPU with dedicated memory (worth using via Vulkan)."""
        return any(g.vendor in ("amd", "intel") and g.vram_mb >= 1024 for g in self.gpus)


@functools.cache
def detect() -> Hardware:
    """Probe once and cache. Never raises; unknown parts are reported as absent."""
    try:
        gpus = tuple(_dxgi_adapters())
    except Exception:
        log.exception("Could not enumerate graphics adapters")
        gpus = ()
    try:
        nvidia = _nvidia_info()
    except Exception:
        log.exception("Could not query the NVIDIA driver")
        nvidia = None
    try:
        p_cores = _performance_core_count()
    except Exception:
        log.exception("Could not read the CPU core layout")
        p_cores = max(1, (os.cpu_count() or 2) // 2)
    hw = Hardware(gpus=gpus, nvidia=nvidia, logical_cpus=os.cpu_count() or 1, performance_cores=p_cores)
    log.info(
        "Hardware: %s; NVIDIA driver %s; %d logical CPUs, %d performance cores",
        ", ".join(f"{g.name} ({g.vram_mb} MB)" for g in gpus) or "no GPU",
        nvidia.driver_version if nvidia else "none",
        hw.logical_cpus,
        hw.performance_cores,
    )
    return hw


def refresh() -> Hardware:
    detect.cache_clear()
    return detect()
