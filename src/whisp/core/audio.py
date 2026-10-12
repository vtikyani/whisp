"""Microphone capture, device listing and input levels.

Audio comes from PortAudio (through sounddevice) as mono float32 at 16 kHz,
in frames of ``FRAME_SAMPLES`` samples. WASAPI shared mode with Windows'
own format conversion is preferred; MME is the fallback.

Device ids are the names Windows reports. They come from WASAPI, because MME
cuts names to 31 characters; an MME device is matched to a full name by that
prefix.

A WASAPI stream can only start on a thread with COM initialised, so
``MicrophoneStream.start()`` sets COM up for the calling thread when it has none
(Qt's GUI thread already has it; a plain worker thread such as
``whisp-pipeline`` does not).

PortAudio only discovers devices when it starts, so a fresh scan means shutting
it down and starting it again, which closes every open PortAudio stream in the
process. Scans therefore only re-initialise while no stream is open: this
module counts its own streams, and any other code that keeps a sounddevice
stream open must hold ``hold_portaudio()`` meanwhile.
"""

from __future__ import annotations

import collections
import contextlib
import ctypes
import functools
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

from whisp.core.types import FRAME_SAMPLES, SAMPLE_RATE, AudioDevice

log = logging.getLogger(__name__)

MSG_NO_DEVICE = "No microphone found. Plug one in, or pick another in Settings → Microphone."
MSG_OPEN_FAILED = (
    "Couldn't open the microphone. Check that it's plugged in and that Windows lets apps use it "
    "(Settings → Privacy & security → Microphone)."
)
MSG_DISCONNECTED = "The microphone was disconnected. Plug it back in, or pick another in Settings → Microphone."
MSG_STALLED = "The microphone stopped sending sound. Check that it's still plugged in."
MSG_SCAN_FAILED = "Couldn't list microphones. Check Windows sound settings, then try again."

LEVEL_FLOOR_DB = -60.0
"""Input level shown as 0."""
LEVEL_CEIL_DB = -12.0
"""Input level shown as 1 (loud, close speech)."""

_LEVEL_INTERVAL_S = 0.03
"""``on_level`` runs at most about 30 times a second (one 32 ms frame each)."""
_FIRST_FRAME_TIMEOUT_S = 5.0
"""Bluetooth headsets can take seconds to switch into headset mode before audio flows."""
_STALL_TIMEOUT_S = 2.0
"""A started stream that delivers nothing for this long is treated as lost."""
_STOP_GRACE_S = 0.1
"""Extra wait, beyond the device's reported latency, for audio still on its way at stop()."""
_MAX_BUFFERED_FRAMES = 60 * SAMPLE_RATE // FRAME_SAMPLES
"""About a minute of unread audio. A reader that keeps up never gets near it; a stream used
only as a level meter never reads, and this caps its memory by dropping the oldest frames."""

_WASAPI = "Windows WASAPI"
_MME = "MME"
_MME_MAPPER = "Microsoft Sound Mapper - Input"
_MME_NAME_LIMIT = 31
"""Windows cuts MME (waveIn) device names to 31 characters."""
_LOOPBACK_TAG = "[Loopback]"

_WAVE_MAPPER = 0xFFFFFFFF
_DRVM_MAPPER_PREFERRED_GET = 0x2015

_COINIT_MULTITHREADED = 0x0
_COINIT_APARTMENTTHREADED = 0x2
_S_OK = 0x0
_S_FALSE = 0x1
_RPC_E_CHANGED_MODE = 0x80010106


class AudioError(Exception):
    """A microphone problem. The message is user-facing."""


def perceptual_level(mean_square: float) -> float:
    """Map signal power (mean of squared samples) to a 0-1 meter level.

    Linear in decibels: ``LEVEL_FLOOR_DB`` (-60 dBFS) and below is 0,
    ``LEVEL_CEIL_DB`` (-12 dBFS) and above is 1.
    """
    if not mean_square > 1e-12:
        return 0.0
    db = 10.0 * math.log10(mean_square)
    return min(1.0, max(0.0, (db - LEVEL_FLOOR_DB) / (LEVEL_CEIL_DB - LEVEL_FLOOR_DB)))


# -- PortAudio bookkeeping ----------------------------------------------------

_portaudio_lock = threading.RLock()
"""Serialises PortAudio re-initialisation, device queries and stream opening."""
_portaudio_users = 0
"""Open streams of this module plus active ``hold_portaudio()`` holders."""


def _acquire_portaudio() -> None:
    global _portaudio_users
    with _portaudio_lock:
        _portaudio_users += 1


def _release_portaudio() -> None:
    global _portaudio_users
    with _portaudio_lock:
        _portaudio_users = max(0, _portaudio_users - 1)


@contextlib.contextmanager
def hold_portaudio() -> Iterator[None]:
    """Keep device scans from re-initialising PortAudio while you use it.

    Re-initialising closes every open PortAudio stream, so wrap the lifetime of
    any sounddevice stream opened outside this module (for example to play a
    sound) in this.
    """
    _acquire_portaudio()
    try:
        yield
    finally:
        _release_portaudio()


def _portaudio_idle() -> bool:
    """True when re-initialising PortAudio would not pull a stream from under anyone. Hold the lock."""
    if _portaudio_users:
        return False
    try:
        convenience = sd.get_stream()  # the stream behind sounddevice's play()/rec() helpers
    except RuntimeError:
        return True
    if convenience.closed:
        return True
    if convenience.active:
        return False
    # Finished playback/recording: close it properly now, or re-initialising would leave it dangling.
    sd.stop()
    return True


def _reinitialize_portaudio() -> None:
    """Shut PortAudio down and start it again so it sees the current devices.

    sounddevice has no public call for this; its private ``_terminate`` and
    ``_initialize`` are the documented way to re-scan.
    """
    for _ in range(sd._initialized):
        sd._terminate()
    sd._initialize()


_com_threads = threading.local()


@functools.cache
def _ole32() -> ctypes.WinDLL:
    dll = ctypes.WinDLL("ole32")
    dll.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    dll.CoInitializeEx.restype = ctypes.c_long
    dll.CoUninitialize.argtypes = []
    dll.CoUninitialize.restype = None
    return dll


def _co_initialize() -> int:
    """``CoInitializeEx`` for this thread. Returns the HRESULT as an unsigned number.

    The main thread gets a single-threaded apartment, as Qt's OLE support needs there; worker threads,
    which run no message loop, join the multithreaded apartment.
    """
    on_main_thread = threading.current_thread() is threading.main_thread()
    model = _COINIT_APARTMENTTHREADED if on_main_thread else _COINIT_MULTITHREADED
    return _ole32().CoInitializeEx(None, model) & 0xFFFFFFFF


def _ensure_com() -> None:
    """Give this thread COM if it has none: a WASAPI stream fails to start on a thread without it.

    It stays initialised for the rest of the thread's life, because the streams the thread starts
    depend on it. A thread that already uses COM (the Qt GUI thread) is left as it is.
    """
    if getattr(_com_threads, "ready", False):
        return
    result = _co_initialize()
    if result == _S_FALSE:
        _ole32().CoUninitialize()  # its owner initialised it already; don't keep a second reference
    elif result not in (_S_OK, _RPC_E_CHANGED_MODE):
        log.warning("Couldn't initialise COM on thread %r (0x%08X)", threading.current_thread().name, result)
        return
    _com_threads.ready = True


@contextlib.contextmanager
def _com_scope() -> Iterator[None]:
    """COM for the length of the block, for a short-lived thread."""
    result = _co_initialize()
    try:
        yield
    finally:
        if result in (_S_OK, _S_FALSE):
            _ole32().CoUninitialize()


def _rescan_if_idle(reason: str) -> bool:
    """Re-initialise PortAudio unless a stream is open. Hold the lock."""
    if not _portaudio_idle():
        log.debug("Not re-scanning audio devices (%s): a stream is open", reason)
        return False
    started = time.perf_counter()
    _reinitialize_portaudio()
    log.debug("Re-scanned audio devices (%s) in %.0f ms", reason, (time.perf_counter() - started) * 1000)
    return True


# -- device discovery -----------------------------------------------------------


@dataclass(frozen=True)
class _Device:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_samplerate: float


@dataclass(frozen=True)
class _DeviceTable:
    """Input devices as PortAudio currently knows them."""

    devices: tuple[_Device, ...]
    default_inputs: Mapping[str, int]
    """Host API name -> PortAudio index of its default input device (-1 if none)."""


@dataclass(frozen=True)
class _WindowsInputs:
    """What Windows reports right now, independent of PortAudio's snapshot."""

    count: int
    """Number of recording devices (waveIn)."""
    default_name: str | None
    """The default recording device's name, cut to 31 characters like every MME name."""


@dataclass(frozen=True)
class _Candidate:
    index: int
    host_api: str
    name: str
    """Reported as ``MicrophoneStream.device_name``."""


def _scan() -> _DeviceTable:
    host_apis = sd.query_hostapis()
    api_names = [str(api["name"]) for api in host_apis]
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if int(info["max_input_channels"]) <= 0 or _LOOPBACK_TAG in str(info["name"]):
            continue
        devices.append(
            _Device(
                index=index,
                name=str(info["name"]),
                host_api=api_names[int(info["hostapi"])],
                max_input_channels=int(info["max_input_channels"]),
                default_samplerate=float(info["default_samplerate"]),
            )
        )
    defaults = {str(api["name"]): int(api["default_input_device"]) for api in host_apis}
    return _DeviceTable(tuple(devices), defaults)


class _WaveInCaps(ctypes.Structure):
    _fields_ = (
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.UINT),
        ("szPname", wintypes.WCHAR * 32),
        ("dwFormats", wintypes.DWORD),
        ("wChannels", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
    )


@functools.cache
def _winmm() -> ctypes.WinDLL:
    dll = ctypes.WinDLL("winmm")
    dll.waveInGetNumDevs.argtypes = []
    dll.waveInGetNumDevs.restype = wintypes.UINT
    dll.waveInGetDevCapsW.argtypes = [ctypes.c_size_t, ctypes.POINTER(_WaveInCaps), wintypes.UINT]
    dll.waveInGetDevCapsW.restype = wintypes.UINT
    dll.waveInMessage.argtypes = [wintypes.HANDLE, wintypes.UINT, ctypes.c_size_t, ctypes.c_size_t]
    dll.waveInMessage.restype = wintypes.UINT
    return dll


def _windows_inputs() -> _WindowsInputs | None:
    """Ask Windows (not PortAudio) for the current recording devices. None if it can't be asked."""
    try:
        winmm = _winmm()
    except OSError:
        log.debug("winmm is unavailable; can't check the current default microphone", exc_info=True)
        return None
    count = int(winmm.waveInGetNumDevs())
    preferred = wintypes.DWORD(_WAVE_MAPPER)
    flags = wintypes.DWORD(0)
    result = winmm.waveInMessage(
        wintypes.HANDLE(_WAVE_MAPPER),
        _DRVM_MAPPER_PREFERRED_GET,
        ctypes.addressof(preferred),
        ctypes.addressof(flags),
    )
    name = None
    if result == 0 and preferred.value != _WAVE_MAPPER:
        caps = _WaveInCaps()
        if winmm.waveInGetDevCapsW(preferred.value, ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
            name = caps.szPname or None
    return _WindowsInputs(count, name)


def _same_device(full_name: str, mme_name: str) -> bool:
    """Whether an MME name (possibly cut to 31 characters) belongs to the device called ``full_name``."""
    if full_name == mme_name:
        return True
    return len(mme_name) >= _MME_NAME_LIMIT and full_name.startswith(mme_name)


def _looks_stale(table: _DeviceTable, windows: _WindowsInputs | None) -> bool:
    """Whether devices were added, removed or the default changed since PortAudio last scanned."""
    if windows is None:
        return False
    mme_count = sum(1 for d in table.devices if d.host_api == _MME and d.name != _MME_MAPPER)
    if mme_count != windows.count:
        return True
    if windows.default_name is None:
        return False
    return not any(_same_device(d.name, windows.default_name) for d in table.devices)


def _default_wasapi(table: _DeviceTable, windows_default: str | None) -> _Device | None:
    wasapi = [d for d in table.devices if d.host_api == _WASAPI]
    portaudio_default = table.default_inputs.get(_WASAPI, -1)
    if windows_default:
        matches = [d for d in wasapi if _same_device(d.name, windows_default)]
        if matches:
            return next((d for d in matches if d.index == portaudio_default), matches[0])
    return next((d for d in wasapi if d.index == portaudio_default), None)


def _plan(table: _DeviceTable, requested: str | None, windows_default: str | None) -> tuple[list[_Candidate], bool]:
    """Devices to try opening, best first, and whether ``requested`` was found.

    A requested device is tried through WASAPI, then MME. Otherwise (or if it
    is missing) the Windows default is used: the WASAPI device matching the
    current default, then the MME sound mapper, which always follows the
    default.
    """
    if requested:
        exact = next((d for d in table.devices if d.host_api == _WASAPI and d.name == requested), None)
        legacy = next(
            (
                d
                for d in table.devices
                if d.host_api == _MME and d.name != _MME_MAPPER and _same_device(requested, d.name)
            ),
            None,
        )
        found = [_Candidate(d.index, d.host_api, requested) for d in (exact, legacy) if d is not None]
        if found:
            return found, True

    candidates = []
    default = _default_wasapi(table, windows_default)
    if default is not None:
        candidates.append(_Candidate(default.index, _WASAPI, default.name))
    mme_default = table.default_inputs.get(_MME, -1)
    mapper = next((d for d in table.devices if d.host_api == _MME and d.name == _MME_MAPPER), None)
    fallback = mapper or next((d for d in table.devices if d.index == mme_default), None)
    if fallback is not None:
        label = default.name if default is not None else (windows_default or fallback.name)
        if label == _MME_MAPPER:
            label = "Default microphone"
        candidates.append(_Candidate(fallback.index, _MME, label))
    return candidates, not requested


def _listed_devices(table: _DeviceTable, windows_default: str | None) -> list[AudioDevice]:
    api = _WASAPI if any(d.host_api == _WASAPI for d in table.devices) else _MME
    default_index = -1
    if api == _WASAPI:
        default = _default_wasapi(table, windows_default)
        default_index = default.index if default is not None else -1
    out: list[AudioDevice] = []
    seen: set[str] = set()
    for d in table.devices:
        if d.host_api != api or d.name == _MME_MAPPER or d.name in seen:
            continue
        seen.add(d.name)
        if api == _MME:
            is_default = (
                _same_device(d.name, windows_default)
                if windows_default
                else (d.index == table.default_inputs.get(_MME, -1))
            )
        else:
            is_default = d.index == default_index
        out.append(
            AudioDevice(
                id=d.name,
                name=d.name,
                host_api=d.host_api,
                is_default=is_default,
                max_input_channels=d.max_input_channels,
                default_samplerate=d.default_samplerate,
            )
        )
    return out


def list_input_devices() -> list[AudioDevice]:
    """Fresh scan of the microphones Windows knows about.

    Re-initialises PortAudio when no stream is open; otherwise returns what
    PortAudio found last. Raises ``AudioError`` if PortAudio fails.
    """
    with _portaudio_lock:
        try:
            _rescan_if_idle("device list requested")
            table = _scan()
        except Exception as exc:
            log.exception("Couldn't list audio input devices")
            raise AudioError(MSG_SCAN_FAILED) from exc
    windows = _windows_inputs()
    devices = _listed_devices(table, windows.default_name if windows else None)
    log.debug("Found %d input device(s)", len(devices))
    return devices


# -- capture ----------------------------------------------------------------------


class MicrophoneStream:
    """One recording from a microphone, as 16 kHz mono float32 frames of ``FRAME_SAMPLES`` samples.

    ``start()`` opens the device. The PortAudio callback only queues frames and
    meters the level; ``read()`` hands the frames out on any thread. Frames
    nobody reads within about a minute are dropped, oldest first, so a stream
    used only for its level stays small. A stream records once: after
    ``stop()`` it can't be started again.
    """

    def __init__(self, device: str | None, on_level: Callable[[float], None] | None = None) -> None:
        self._requested = device or None
        self._on_level = on_level
        self._lock = threading.Lock()
        self._stream: sd.RawInputStream | None = None
        self._attempt = 0
        self._device_name = device or ""
        self._latency_s = 0.0
        self._started = False
        self._stopping = False
        self._error: str | None = None
        self._start_time = 0.0
        self._available = threading.Condition(threading.Lock())
        """Guards ``_buffer``, ``_eof`` and ``_dropped``; notified when a frame arrives or the stream ends."""
        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._eof = False
        """No more frames will arrive: stopped, unplugged or stalled."""
        self._dropped = 0
        # Written only by the audio callback; single attribute writes are atomic, so readers need no lock.
        self._received = 0
        self._last_frame_time = 0.0
        self._overflows = 0
        self._drain_target: int | None = None
        self._drained = threading.Event()
        self._level_energy = 0.0
        self._level_samples = 0
        self._level_time = 0.0
        self._level_failed = False

    @property
    def device_name(self) -> str:
        """The device being recorded (after ``start()``), or the requested one."""
        return self._device_name

    @property
    def error(self) -> str | None:
        """User-facing reason the stream ended on its own (unplugged, stalled), or None.

        Once set, ``read()`` returns the frames captured before the failure and
        then None straight away.
        """
        return self._error

    def start(self) -> None:
        """Open the microphone and start capturing. Raises ``AudioError``."""
        with self._lock:
            if self._started or self._stopping:
                raise RuntimeError("a MicrophoneStream can only be started once, and not after stop()")
            self._started = True
        _ensure_com()
        with _portaudio_lock:
            try:
                candidates = self._choose_devices()
            except Exception as exc:
                log.exception("Couldn't look up audio input devices")
                raise AudioError(MSG_OPEN_FAILED) from exc
            if not candidates:
                raise AudioError(MSG_NO_DEVICE)
            failure: Exception | None = None
            for candidate in candidates:
                try:
                    self._open(candidate)
                except Exception as exc:
                    failure = exc
                    log.warning("Couldn't open microphone %r through %s: %s", candidate.name, candidate.host_api, exc)
                    continue
                _acquire_portaudio()
                break
            else:
                raise AudioError(MSG_OPEN_FAILED) from failure
        log.info(
            "Microphone open: %r via %s (input latency %.0f ms)",
            self._device_name,
            candidate.host_api,
            self._latency_s * 1000,
        )

    def read(self, timeout: float) -> np.ndarray | None:
        """The next frame, or None on timeout and once the stream has ended. Any thread."""
        with self._available:
            if timeout > 0:
                self._available.wait_for(lambda: self._buffer or self._eof, timeout)
            if self._buffer:
                return self._buffer.popleft()
            ended = self._eof
        if not ended:
            self._check_stalled()
        return None

    def stop(self) -> None:
        """Stop capturing. Idempotent; frames already captured stay readable.

        Audio that was still on its way from the device when this was called
        (the device's reported input latency) is collected first, so everything
        spoken up to this moment can be read.
        """
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
            stream = self._stream
        if stream is None:
            self._mark_end()
            return
        if self._error is None and not self._eof:
            self._collect_in_flight_audio()
            try:
                stream.stop(ignore_errors=False)
            except Exception:
                log.exception("Stopping the microphone stream failed")
            finally:
                stream.close(ignore_errors=True)
                _release_portaudio()
        else:
            # The device is gone; closing it can block inside the driver, so never on this thread.
            threading.Thread(target=_close_lost_stream, args=(stream,), name="whisp-audio-close", daemon=True).start()
        self._mark_end()
        log.info(
            "Microphone closed after %.1f s (%d frames, %d overflow(s), %d dropped unread)",
            time.monotonic() - self._start_time,
            self._received // FRAME_SAMPLES,
            self._overflows,
            self._dropped,
        )

    # -- internals -------------------------------------------------------------

    def _choose_devices(self) -> list[_Candidate]:
        windows = _windows_inputs()
        windows_default = windows.default_name if windows else None
        rescanned = False
        table = _scan()
        if _looks_stale(table, windows):
            rescanned = _rescan_if_idle("devices changed")
            if rescanned:
                table = _scan()
        candidates, found = _plan(table, self._requested, windows_default)
        if not found and not rescanned and _rescan_if_idle("configured microphone not found"):
            table = _scan()
            candidates, found = _plan(table, self._requested, windows_default)
        if not found:
            log.warning("Microphone %r is not available; using the Windows default", self._requested)
        return candidates

    def _open(self, candidate: _Candidate) -> None:
        self._attempt += 1
        attempt = self._attempt
        extra = sd.WasapiSettings(auto_convert=True) if candidate.host_api == _WASAPI else None
        # The raw stream hands the callback PortAudio's buffer as is; sounddevice's NumPy wrapper relies on
        # array-shape assignment, which NumPy 2.5 deprecates.
        stream = sd.RawInputStream(
            device=candidate.index,
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype="float32",
            latency="high",
            extra_settings=extra,
            callback=self._callback,
            finished_callback=functools.partial(self._on_finished, attempt),
        )
        try:
            self._start_time = time.monotonic()
            stream.start()
        except BaseException:
            self._attempt += 1  # a finished-callback from this failed attempt must not end the next one
            stream.close(ignore_errors=True)
            raise
        self._stream = stream
        self._device_name = candidate.name
        self._latency_s = float(stream.latency)

    def _callback(self, indata: Any, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        # Runs on PortAudio's thread: keep it tiny, never log per frame. The buffer is only valid during this call.
        samples = np.frombuffer(indata, dtype=np.float32, count=frames).copy()
        with self._available:
            if len(self._buffer) >= _MAX_BUFFERED_FRAMES:
                self._buffer.popleft()
                self._dropped += 1
            self._buffer.append(samples)
            self._available.notify()
        received = self._received + frames
        self._received = received
        now = time.monotonic()
        self._last_frame_time = now
        if status.input_overflow:
            self._overflows += 1
        target = self._drain_target
        if target is not None and received >= target:
            self._drained.set()
        if self._on_level is not None:
            self._meter(samples, now)

    def _meter(self, samples: np.ndarray, now: float) -> None:
        self._level_energy += float(np.dot(samples, samples))
        self._level_samples += samples.size
        if now - self._level_time < _LEVEL_INTERVAL_S:
            return
        level = perceptual_level(self._level_energy / self._level_samples)
        self._level_energy = 0.0
        self._level_samples = 0
        self._level_time = now
        try:
            self._on_level(level)
        except Exception:
            if not self._level_failed:
                self._level_failed = True
                log.exception("Microphone level callback failed")

    def _on_finished(self, attempt: int) -> None:
        if attempt != self._attempt:
            return
        if not self._stopping:
            self._error = MSG_DISCONNECTED
            log.warning("Microphone %r stopped unexpectedly", self._device_name)
        self._mark_end()

    def _check_stalled(self) -> None:
        if self._stream is None or self._stopping or self._eof:
            return
        if self._received:
            silent_for, limit = time.monotonic() - self._last_frame_time, _STALL_TIMEOUT_S
        else:
            silent_for, limit = time.monotonic() - self._start_time, _FIRST_FRAME_TIMEOUT_S
        if silent_for < limit:
            return
        log.warning("Microphone %r sent no audio for %.1f s; treating it as lost", self._device_name, silent_for)
        self._error = MSG_STALLED
        self._mark_end()

    def _collect_in_flight_audio(self) -> None:
        in_flight = round(self._latency_s * SAMPLE_RATE)
        if in_flight <= 0:
            return
        self._drained.clear()
        target = self._received + in_flight
        self._drain_target = target
        if self._received < target:
            self._drained.wait(self._latency_s + _STOP_GRACE_S)
        self._drain_target = None

    def _mark_end(self) -> None:
        with self._available:
            self._eof = True
            self._available.notify_all()


def _close_lost_stream(stream: sd.RawInputStream) -> None:
    try:
        with _com_scope():
            stream.abort(ignore_errors=True)
            stream.close(ignore_errors=True)
    except Exception:
        log.exception("Closing a lost microphone stream failed")
    finally:
        _release_portaudio()
