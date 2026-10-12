"""Mute other apps while the user dictates, and always give them their sound back.

Music or a video playing during a dictation distracts the speaker and hurts
recognition. So while a dictation runs, Whisp mutes the audio *sessions* of
other apps (the per-app entries in the Windows volume mixer) and unmutes them
afterwards. It leaves alone:

* Whisp itself and the Windows system-sounds session;
* sessions that aren't playing, or that are muted already;
* call and meeting apps (``COMMUNICATION_APPS``, plus the user's own list);
* any app that is using a microphone right now. That is almost certainly a
  call, even one in a browser tab or in a softphone we don't know by name.

Browsers are not on the list (a YouTube tab should go quiet); a browser is
only left alone while one of its tabs uses the microphone.

**Exact restore.** Only sessions Whisp muted are unmuted. If someone else
(the user in the volume mixer, or the app itself) unmutes one of them during
the dictation, it's theirs from then on, even if they mute it again. Windows
reports every mute change along with a tag picked by whoever made it. Whisp
tags its own changes, so anything else is someone else's.

**Crash safety.** Windows saves an app's mute when the app's session ends
and applies it the next time the app starts. A mute that outlives Whisp
would therefore stick. Before muting, Whisp writes the sessions it's about to
mute to ``muted-sessions.json`` (atomically). After a crash,
``restore_leftovers()`` unmutes them at the next start. If an app has quit in
the meantime, or quits mid-dictation, its entry stays *pending*. Its next
session (which Windows starts muted) is unmuted as soon as it appears. Pending
entries are dropped after 30 days.

**Threads.** These calls take milliseconds to tens of milliseconds and must
never run on the Qt GUI thread. ``MuteService`` runs everything on its own
``whisp-mute`` thread, which joins COM's multithreaded apartment. A plain
``AudioMuter`` belongs to the one thread that calls it.

Nothing here logs window titles or session identifiers (they contain file
paths). Debug logs carry counts and process names only.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from contextlib import AbstractContextManager
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from whisp.storage.paths import app_paths

log = logging.getLogger(__name__)

STATE_FILE_NAME = "muted-sessions.json"
STATE_FORMAT = 1
PENDING_MAX_AGE_S = 30 * 24 * 3600.0
"""Forget a session left muted (by a crash, or by its app quitting mid-dictation) after this long."""
PENDING_CHECK_FAST_S = 5.0
PENDING_CHECK_SLOW_S = 60.0
PENDING_FAST_PERIOD_S = 300.0
"""An app left muted often starts again soon (at sign-in, or a quick relaunch), so ``MuteService`` looks for it
every 5 s for the first five minutes, then once a minute."""

COMMUNICATION_APPS: frozenset[str] = frozenset(
    {
        # Microsoft Teams: new, classic, and the consumer app that came with Windows 11.
        "ms-teams.exe",
        "msteams.exe",
        "teams.exe",
        # Zoom and its meeting/screen-share host.
        "zoom.exe",
        "cpthost.exe",
        # Webex app and Webex Meetings.
        "webex.exe",
        "ciscocollabhost.exe",
        "webexmta.exe",
        "atmgr.exe",
        "slack.exe",
        # Discord stable, PTB and Canary builds.
        "discord.exe",
        "discordptb.exe",
        "discordcanary.exe",
        # Skype and Skype for Business.
        "skype.exe",
        "lync.exe",
        "whatsapp.exe",
        "telegram.exe",
        "signal.exe",
        "viber.exe",
        # Phone Link (and its older name, Your Phone): calls from a paired phone.
        "phoneexperiencehost.exe",
        "yourphone.exe",
        # Voice chat.
        "mumble.exe",
        "ts3client_win64.exe",
        "teamspeak.exe",
    }
)
"""Executables of call and meeting apps. Never muted: hearing the other people matters more than quiet."""

EMBEDDED_BROWSER_HELPERS: frozenset[str] = frozenset(
    {"msedgewebview2.exe", "qtwebengineprocess.exe", "cefsharp.browsersubprocess.exe"}
)
"""Helper processes of embedded browsers. Apps built on one (WebView2, Qt WebEngine, CEF) may play sound through
these helpers, so exclusions are checked against the app hosting them."""


def normalize_process_name(name: str) -> str:
    """``"Spotify"``, ``"spotify.exe"`` or a full path become ``"spotify.exe"``. Blank input gives ``""``."""
    base = name.strip().replace("/", "\\").rsplit("\\", 1)[-1].strip().lower()
    if base and not base.endswith(".exe"):
        base += ".exe"
    return base


# -- which app a process belongs to ------------------------------------------------------


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = (
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    )


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_CreateToolhelp32Snapshot = _kernel32.CreateToolhelp32Snapshot
_CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_CreateToolhelp32Snapshot.restype = wintypes.HANDLE

_Process32FirstW = _kernel32.Process32FirstW
_Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
_Process32FirstW.restype = wintypes.BOOL

_Process32NextW = _kernel32.Process32NextW
_Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
_Process32NextW.restype = wintypes.BOOL

_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL

_OpenProcess = _kernel32.OpenProcess
_OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_OpenProcess.restype = wintypes.HANDLE

_GetExitCodeProcess = _kernel32.GetExitCodeProcess
_GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_GetExitCodeProcess.restype = wintypes.BOOL

_GetProcessTimes = _kernel32.GetProcessTimes
_GetProcessTimes.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
]
_GetProcessTimes.restype = wintypes.BOOL

_TH32CS_SNAPPROCESS = 0x00000002
_INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87


def process_table() -> dict[int, tuple[int, str]]:
    """Every running process: pid -> (parent pid, lower-case executable name).

    Toolhelp names every process, including elevated ones, without opening them.
    """
    snapshot = _CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == _INVALID_HANDLE_VALUE:
        log.warning("Couldn't list running processes (error %d)", ctypes.get_last_error())
        return {}
    table: dict[int, tuple[int, str]] = {}
    try:
        entry = _PROCESSENTRY32W(dwSize=ctypes.sizeof(_PROCESSENTRY32W))
        more = _Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            table[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), entry.szExeFile.lower())
            more = _Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        _CloseHandle(snapshot)
    return table


def process_start_time(pid: int) -> int | None:
    """When a running process started, in FILETIME ticks. 0 if no process with this id is running; None if Windows
    won't say.

    A pid plus its start time identifies one process, even after Windows hands the pid to a new process.
    """
    if pid <= 0:
        return None
    handle = _OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return 0 if ctypes.get_last_error() == _ERROR_INVALID_PARAMETER else None
    try:
        exit_code = wintypes.DWORD()
        if not _GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return None
        if exit_code.value != _STILL_ACTIVE:
            return 0  # it has exited; something still holds a handle to it
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not _GetProcessTimes(
            handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        _CloseHandle(handle)


def owning_app(pid: int, processes: Mapping[int, tuple[int, str]]) -> tuple[int, str]:
    """The app a process belongs to, as ``(pid, executable name)``. ``(pid, "")`` if the process is unknown.

    Chromium- and Electron-based apps run helper processes under their own executable name. Apps built on an
    embedded browser can play sound from its helpers. So the walk goes up the parent chain through same-named
    processes and embedded-browser helpers, and stops at the first other program. A parent that has exited may
    have had its pid reused by an unrelated program, which then gets the credit; helpers rarely outlive their app,
    so that is rare.
    """
    entry = processes.get(pid)
    if entry is None:
        return pid, ""
    current = pid
    parent, name = entry
    seen = {pid}
    while parent and parent not in seen:
        parent_entry = processes.get(parent)
        if parent_entry is None:
            break
        if parent_entry[1] != name and name not in EMBEDDED_BROWSER_HELPERS:
            break
        seen.add(parent)
        current = parent
        parent, name = parent_entry
    return current, name


# -- what to mute ------------------------------------------------------------------------


class MuteWatch(Protocol):
    """Watches one muted session for mute changes made by anyone but Whisp."""

    @property
    def changed_by_others(self) -> bool:
        """True once anyone but Whisp has unmuted the session."""

    def close(self) -> None:
        """Stop watching. Safe to call more than once."""


class AudioSession(Protocol):
    """One app's audio session on one output device, as seen during a snapshot."""

    instance_id: str
    session_id: str
    pid: int
    process_started: int | None
    """``process_start_time(pid)``: 0 once the process has quit (its session lingers for a moment), None if unknown."""
    process_name: str
    """Lower-case executable name; ``""`` when unknown (or not asked for)."""
    app_pid: int
    """The app the process belongs to (see ``owning_app``)."""
    app_name: str
    active: bool
    """Something is playing right now."""
    system_sounds: bool
    muted: bool

    def set_mute(self, muted: bool) -> bool:
        """Change the mute, tagged as Whisp's own change. False if Windows refused."""

    def watch(self) -> MuteWatch | None:
        """Start noticing mute changes made by others. None if Windows won't report them."""


@dataclass
class SessionSnapshot:
    sessions: list[AudioSession]
    capturing_apps: frozenset[int] = frozenset()
    """``owning_app`` pids of apps (other than Whisp) recording from a microphone right now."""


class SessionSource(Protocol):
    def snapshot(self, *, details: bool) -> AbstractContextManager[SessionSnapshot]:
        """Every session on every active output device, usable until the block ends.

        ``details`` adds process names, owning apps and ``capturing_apps``, which muting decisions need.
        """


def skip_reason(
    session: AudioSession, *, own_pid: int, excluded: frozenset[str], capturing_apps: frozenset[int]
) -> str | None:
    """Why ``session`` must not be muted, or None if it should be."""
    if session.system_sounds:
        return "system sounds"
    if session.pid == own_pid:
        return "Whisp itself"
    if session.process_started == 0:
        return "app has quit"
    if not session.active:
        return "not playing"
    if session.muted:
        return "muted already"
    if not session.process_name:
        return "unknown app"
    if session.process_name in excluded or session.app_name in excluded:
        return "call or meeting app"
    if session.app_pid in capturing_apps:
        return "using a microphone"
    return None


# -- the crash-safety record -------------------------------------------------------------


@dataclass(frozen=True)
class MutedSession:
    """A session Whisp muted, saved so it can be unmuted even after a crash."""

    instance_id: str
    """Windows' id for this one session of this process (``GetSessionInstanceIdentifier``)."""
    session_id: str
    """The app's id on this output device (``GetSessionIdentifier``). Windows saves the app's mute under it."""
    pid: int
    process_name: str
    muted_at: float
    """``time.time()`` when Whisp muted it."""
    siblings: tuple[str, ...] = ()
    """Instance ids of the app's other sessions on the device at that moment. Once this session is gone, any other
    session of the app that isn't one of these was started by Windows with the mute Whisp left behind."""
    process_started: int | None = None
    """``process_start_time(pid)`` when muted, to tell the process apart from a later one that gets its pid."""

    def to_json(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "session_id": self.session_id,
            "pid": self.pid,
            "process": self.process_name,
            "started": self.process_started,
            "muted_at": self.muted_at,
            "siblings": list(self.siblings),
        }

    @classmethod
    def from_json(cls, data: Any) -> MutedSession | None:
        """Parse one saved entry; None if it's malformed."""
        if not isinstance(data, dict):
            return None
        instance_id, session_id = data.get("instance_id"), data.get("session_id")
        pid, process_name, muted_at = data.get("pid"), data.get("process", ""), data.get("muted_at")
        siblings, started = data.get("siblings", []), data.get("started")
        if not (isinstance(instance_id, str) and instance_id and isinstance(session_id, str) and session_id):
            return None
        if not isinstance(pid, int) or isinstance(pid, bool) or not isinstance(process_name, str):
            return None
        if not isinstance(muted_at, int | float) or isinstance(muted_at, bool):
            return None
        if not isinstance(siblings, list) or not all(isinstance(s, str) for s in siblings):
            return None
        if started is not None and (not isinstance(started, int) or isinstance(started, bool)):
            return None
        return cls(instance_id, session_id, pid, process_name, float(muted_at), tuple(siblings), started)


class MuteStateFile:
    """``muted-sessions.json``: the sessions Whisp has muted, or left muted, and must unmute.

    Writes go to a temporary file that is flushed to disk and then renamed over the real one, so a crash or power
    cut leaves either the old list or the new one, never half of one.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def tmp_path(self) -> Path:
        return self.path.with_name(self.path.name + ".tmp")

    def write(self, sessions: Sequence[MutedSession]) -> bool:
        """Replace the saved list. False (and logged) if it couldn't be saved."""
        data = {"format": STATE_FORMAT, "saved_at": time.time(), "sessions": [s.to_json() for s in sessions]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tmp_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(self.tmp_path, self.path)
        except OSError:
            log.exception("Couldn't save the list of muted apps")
            with contextlib.suppress(OSError):
                self.tmp_path.unlink(missing_ok=True)
            return False
        return True

    def read(self) -> list[MutedSession]:
        """The saved list; empty if there is none. An unreadable file is logged and removed."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except OSError:
            log.exception("Couldn't read the list of muted apps")
            return []
        except ValueError:  # not UTF-8, or not JSON
            data = None
        entries = data.get("sessions") if isinstance(data, dict) and data.get("format") == STATE_FORMAT else None
        if not isinstance(entries, list):
            log.warning("The list of muted apps was unreadable; removing it")
            self.delete()
            return []
        sessions = [MutedSession.from_json(item) for item in entries]
        valid = [s for s in sessions if s is not None]
        if len(valid) != len(sessions):
            log.warning("Ignored %d malformed entries in the list of muted apps", len(sessions) - len(valid))
        return valid

    def delete(self) -> None:
        for path in (self.path, self.tmp_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("Couldn't remove %s", path.name)


# -- Core Audio through ctypes -----------------------------------------------------------


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    )


def _guid(text: str) -> _GUID:
    return _GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


_CLSID_MMDeviceEnumerator = _guid("BCDE0395-E52F-467C-8E3D-C4579291692E")
_IID_IUnknown = _guid("00000000-0000-0000-C000-000000000046")
_IID_IMMDeviceEnumerator = _guid("A95664D2-9614-4F35-A746-DE8DB63617E6")
_IID_IAudioSessionManager2 = _guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F")
_IID_IAudioSessionControl2 = _guid("BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D")
_IID_ISimpleAudioVolume = _guid("87CE5498-68D6-44E5-9215-6DA47EF883D8")
_IID_IAudioSessionEvents = _guid("24918ACC-64B3-37C1-8CA9-74A66E9957A8")
_WHISP_EVENT_CONTEXT = _guid("10F41167-3BE4-47FA-A251-A7DCCE583B35")
"""Passed with every mute change Whisp makes, so the change notifications show which ones were Whisp's."""
_WHISP_EVENT_CONTEXT_BYTES = bytes(_WHISP_EVENT_CONTEXT)
_SINK_INTERFACES = frozenset({bytes(_IID_IUnknown), bytes(_IID_IAudioSessionEvents)})


def _signed(code: int) -> int:
    """An HRESULT as the signed 32-bit value COM functions return."""
    return code - (1 << 32) if code & 0x80000000 else code


_S_OK = 0
_S_FALSE = 1
_E_NOINTERFACE = _signed(0x80004002)
_E_POINTER = _signed(0x80004003)
_RPC_E_CHANGED_MODE = 0x80010106
_COINIT_MULTITHREADED = 0x0
_CLSCTX_ALL = 0x17
_E_RENDER = 0
_E_CAPTURE = 1
_DEVICE_STATE_ACTIVE = 0x1
_SESSION_ACTIVE = 1
_SESSION_EXPIRED = 2

_ole32 = ctypes.WinDLL("ole32")

_CoInitializeEx = _ole32.CoInitializeEx
_CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
_CoInitializeEx.restype = ctypes.c_long

_CoUninitialize = _ole32.CoUninitialize
_CoUninitialize.argtypes = []
_CoUninitialize.restype = None

_CoCreateInstance = _ole32.CoCreateInstance
_CoCreateInstance.argtypes = [
    ctypes.POINTER(_GUID),
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(_GUID),
    ctypes.POINTER(ctypes.c_void_p),
]
_CoCreateInstance.restype = ctypes.c_long

_CoTaskMemFree = _ole32.CoTaskMemFree
_CoTaskMemFree.argtypes = [ctypes.c_void_p]
_CoTaskMemFree.restype = None


class _Method:
    """One COM method: its vtable slot and explicit signature (the interface pointer is the first argument)."""

    __slots__ = ("_prototype", "_slot")

    def __init__(self, slot: int, restype: Any, *argtypes: Any) -> None:
        self._slot = slot
        self._prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)

    def __call__(self, obj: ctypes.c_void_p, *args: Any) -> Any:
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return self._prototype(vtable[self._slot])(obj, *args)


_HRESULT = ctypes.c_long
_OUT_PTR = ctypes.POINTER(ctypes.c_void_p)

# IUnknown
_QueryInterface = _Method(0, _HRESULT, ctypes.POINTER(_GUID), _OUT_PTR)
_AddRef = _Method(1, wintypes.ULONG)
_Release = _Method(2, wintypes.ULONG)
# IMMDeviceEnumerator
_EnumAudioEndpoints = _Method(3, _HRESULT, ctypes.c_int, wintypes.DWORD, _OUT_PTR)
# IMMDeviceCollection
_CollectionGetCount = _Method(3, _HRESULT, ctypes.POINTER(wintypes.UINT))
_CollectionItem = _Method(4, _HRESULT, wintypes.UINT, _OUT_PTR)
# IMMDevice
_DeviceActivate = _Method(3, _HRESULT, ctypes.POINTER(_GUID), wintypes.DWORD, ctypes.c_void_p, _OUT_PTR)
# IAudioSessionManager2 (after IAudioSessionManager's GetAudioSessionControl and GetSimpleAudioVolume)
_GetSessionEnumerator = _Method(5, _HRESULT, _OUT_PTR)
# IAudioSessionEnumerator
_SessionCount = _Method(3, _HRESULT, ctypes.POINTER(ctypes.c_int))
_GetSession = _Method(4, _HRESULT, ctypes.c_int, _OUT_PTR)
# IAudioSessionControl
_GetState = _Method(3, _HRESULT, ctypes.POINTER(ctypes.c_int))
_RegisterAudioSessionNotification = _Method(10, _HRESULT, ctypes.c_void_p)
_UnregisterAudioSessionNotification = _Method(11, _HRESULT, ctypes.c_void_p)
# IAudioSessionControl2 (the strings are CoTaskMemAlloc'ed LPWSTRs, read through void**)
_GetSessionIdentifier = _Method(12, _HRESULT, _OUT_PTR)
_GetSessionInstanceIdentifier = _Method(13, _HRESULT, _OUT_PTR)
_GetProcessId = _Method(14, _HRESULT, ctypes.POINTER(wintypes.DWORD))
_IsSystemSoundsSession = _Method(15, _HRESULT)
# ISimpleAudioVolume
_SetMute = _Method(5, _HRESULT, wintypes.BOOL, ctypes.POINTER(_GUID))
_GetMute = _Method(6, _HRESULT, ctypes.POINTER(wintypes.BOOL))


def _release(obj: ctypes.c_void_p | None) -> None:
    if obj:
        _Release(obj)


def _release_all(objects: Iterable[ctypes.c_void_p]) -> None:
    for obj in objects:
        _release(obj)


def _hex(hr: int) -> str:
    return f"0x{hr & 0xFFFFFFFF:08X}"


def _query(obj: ctypes.c_void_p, iid: _GUID) -> ctypes.c_void_p | None:
    out = ctypes.c_void_p()
    if _QueryInterface(obj, ctypes.byref(iid), ctypes.byref(out)) < 0 or not out:
        return None
    return out


def _com_string(obj: ctypes.c_void_p, method: _Method) -> str | None:
    buffer = ctypes.c_void_p()
    if method(obj, ctypes.byref(buffer)) < 0:
        return None
    if not buffer.value:
        return ""
    try:
        return ctypes.wstring_at(buffer.value)
    finally:
        _CoTaskMemFree(buffer)


_com_thread = threading.local()


def _ensure_com() -> None:
    """Join COM's multithreaded apartment on this thread unless it has COM already. Kept for the thread's life.

    The sessions Whisp watches between calls stay usable only while the thread keeps COM, so this isn't undone
    after each call. ``MuteService`` undoes it when its thread ends (``_end_com``).
    """
    if getattr(_com_thread, "ready", False):
        return
    hr = _CoInitializeEx(None, _COINIT_MULTITHREADED) & 0xFFFFFFFF
    if hr in (_S_OK, _S_FALSE):
        _com_thread.owned = True
    elif hr == _RPC_E_CHANGED_MODE:
        # A single-threaded apartment is set up already (Qt does that on its thread). COM works there too.
        _com_thread.owned = False
    else:
        raise OSError(f"COM couldn't start on this thread ({_hex(hr)})")
    _com_thread.ready = True


def _end_com() -> None:
    if getattr(_com_thread, "ready", False) and getattr(_com_thread, "owned", False):
        _CoUninitialize()
    _com_thread.ready = False
    _com_thread.owned = False


# IAudioSessionEvents, implemented in Python. One vtable serves every sink, and a sink is found by its address.
# Windows calls these from its own threads, so they only set a flag. ctypes releases the GIL during the COM calls
# above, so a callback waiting for the GIL can't deadlock against a thread that is waiting inside Unregister.

_QueryInterfaceFn = ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.POINTER(_GUID), _OUT_PTR)
_RefCountFn = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)
_TextChangedFn = ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(_GUID))
_SimpleVolumeChangedFn = ctypes.WINFUNCTYPE(
    _HRESULT, ctypes.c_void_p, ctypes.c_float, wintypes.BOOL, ctypes.POINTER(_GUID)
)
_ChannelVolumeChangedFn = ctypes.WINFUNCTYPE(
    _HRESULT, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_float), wintypes.DWORD, ctypes.POINTER(_GUID)
)
_GroupingParamChangedFn = ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.POINTER(_GUID), ctypes.POINTER(_GUID))
_EnumChangedFn = ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.c_int)


class _AudioSessionEventsVtbl(ctypes.Structure):
    _fields_ = (
        ("QueryInterface", _QueryInterfaceFn),
        ("AddRef", _RefCountFn),
        ("Release", _RefCountFn),
        ("OnDisplayNameChanged", _TextChangedFn),
        ("OnIconPathChanged", _TextChangedFn),
        ("OnSimpleVolumeChanged", _SimpleVolumeChangedFn),
        ("OnChannelVolumeChanged", _ChannelVolumeChangedFn),
        ("OnGroupingParamChanged", _GroupingParamChangedFn),
        ("OnStateChanged", _EnumChangedFn),
        ("OnSessionDisconnected", _EnumChangedFn),
    )


class _ComObject(ctypes.Structure):
    _fields_ = (("vtbl", ctypes.POINTER(_AudioSessionEventsVtbl)),)


_sinks: dict[int, _SessionEventSink] = {}
_sinks_lock = threading.Lock()


class _SessionEventSink:
    """A COM object that notes when anyone but Whisp unmutes a session.

    It stays alive, in ``_sinks``, for as long as COM holds a reference to it. That's the usual COM lifetime rule,
    so a late call from Windows can never reach freed memory.
    """

    def __init__(self) -> None:
        self._object = _ComObject(ctypes.pointer(_SINK_VTBL))
        self.address = ctypes.addressof(self._object)
        self.pointer = ctypes.c_void_p(self.address)
        self.changed_by_others = False
        self._refs = 1  # the creator's reference
        with _sinks_lock:
            _sinks[self.address] = self

    def add_ref(self) -> int:
        with _sinks_lock:
            self._refs += 1
            return self._refs

    def release(self) -> int:
        with _sinks_lock:
            self._refs = max(0, self._refs - 1)
            if self._refs == 0:
                _sinks.pop(self.address, None)
            return self._refs


def _sink_at(this: int | None) -> _SessionEventSink | None:
    if not this:
        return None
    with _sinks_lock:
        return _sinks.get(this)


def _sink_query_interface(this: int | None, riid: Any, out: Any) -> int:
    try:
        if not out:
            return _E_POINTER
        sink = _sink_at(this)
        if sink is not None and riid and bytes(riid.contents) in _SINK_INTERFACES:
            sink.add_ref()
            out[0] = this
            return _S_OK
        out[0] = None
        return _E_NOINTERFACE
    except Exception:
        log.debug("IAudioSessionEvents.QueryInterface failed", exc_info=True)
        return _E_NOINTERFACE


def _sink_add_ref(this: int | None) -> int:
    sink = _sink_at(this)
    return sink.add_ref() if sink is not None else 1


def _sink_release(this: int | None) -> int:
    sink = _sink_at(this)
    return sink.release() if sink is not None else 0


def _sink_on_simple_volume_changed(this: int | None, _volume: float, muted: int, context: Any) -> int:
    try:
        ours = bool(context) and bytes(context.contents) == _WHISP_EVENT_CONTEXT_BYTES
        if not muted and not ours:
            sink = _sink_at(this)
            if sink is not None:
                sink.changed_by_others = True
    except Exception:
        log.debug("IAudioSessionEvents.OnSimpleVolumeChanged failed", exc_info=True)
    return _S_OK


def _sink_ignore(*_args: Any) -> int:
    return _S_OK


# Module-level references keep the callbacks alive for as long as the vtable can be called.
_SINK_CALLBACKS = (
    _QueryInterfaceFn(_sink_query_interface),
    _RefCountFn(_sink_add_ref),
    _RefCountFn(_sink_release),
    _TextChangedFn(_sink_ignore),
    _TextChangedFn(_sink_ignore),
    _SimpleVolumeChangedFn(_sink_on_simple_volume_changed),
    _ChannelVolumeChangedFn(_sink_ignore),
    _GroupingParamChangedFn(_sink_ignore),
    _EnumChangedFn(_sink_ignore),
    _EnumChangedFn(_sink_ignore),
)
_SINK_VTBL = _AudioSessionEventsVtbl(*_SINK_CALLBACKS)


class _CoreAudioWatch:
    """Keeps a session registered with a ``_SessionEventSink`` until closed."""

    def __init__(self, control: ctypes.c_void_p, sink: _SessionEventSink) -> None:
        self._control: ctypes.c_void_p | None = control  # a reference of our own
        self._sink = sink

    @property
    def changed_by_others(self) -> bool:
        return self._sink.changed_by_others

    def close(self) -> None:
        control, self._control = self._control, None
        if control is None:
            return
        hr = _UnregisterAudioSessionNotification(control, self._sink.pointer)
        if hr < 0:
            log.debug("Couldn't stop watching a session (%s)", _hex(hr))
        _release(control)
        self._sink.release()


@dataclass(eq=False)
class _CoreAudioSession:
    """A session read during a snapshot, with the interfaces to change it. The snapshot releases them."""

    control: ctypes.c_void_p  # IAudioSessionControl2
    volume: ctypes.c_void_p  # ISimpleAudioVolume
    instance_id: str
    session_id: str
    pid: int
    active: bool
    system_sounds: bool
    muted: bool
    process_started: int | None = None
    process_name: str = ""
    app_pid: int = 0
    app_name: str = ""

    def set_mute(self, muted: bool) -> bool:
        if not self.volume:
            return False
        hr = _SetMute(self.volume, bool(muted), ctypes.byref(_WHISP_EVENT_CONTEXT))
        if hr < 0:
            log.debug("Windows refused to change the mute of %s (%s)", self.process_name or "a session", _hex(hr))
            return False
        self.muted = bool(muted)
        return True

    def watch(self) -> MuteWatch | None:
        if not self.control:
            return None
        sink = _SessionEventSink()
        hr = _RegisterAudioSessionNotification(self.control, sink.pointer)
        if hr < 0:
            sink.release()
            log.debug("Can't watch %s for mute changes (%s)", self.process_name or "a session", _hex(hr))
            return None
        _AddRef(self.control)
        return _CoreAudioWatch(self.control, sink)

    def release(self) -> None:
        _release(self.volume)
        _release(self.control)
        self.volume = ctypes.c_void_p()
        self.control = ctypes.c_void_p()


def _active_endpoints(enumerator: ctypes.c_void_p, flow: int) -> list[ctypes.c_void_p]:
    """Every active endpoint (IMMDevice) for ``flow``; the caller releases them."""
    collection = ctypes.c_void_p()
    hr = _EnumAudioEndpoints(enumerator, flow, _DEVICE_STATE_ACTIVE, ctypes.byref(collection))
    if hr < 0 or not collection:
        log.debug("Couldn't list audio devices (%s)", _hex(hr))
        return []
    devices: list[ctypes.c_void_p] = []
    try:
        count = wintypes.UINT()
        if _CollectionGetCount(collection, ctypes.byref(count)) >= 0:
            for index in range(count.value):
                device = ctypes.c_void_p()
                if _CollectionItem(collection, index, ctypes.byref(device)) >= 0 and device:
                    devices.append(device)
    finally:
        _release(collection)
    return devices


def _endpoint_sessions(device: ctypes.c_void_p) -> list[ctypes.c_void_p]:
    """IAudioSessionControl of every session on an endpoint; the caller releases them."""
    manager = ctypes.c_void_p()
    hr = _DeviceActivate(device, ctypes.byref(_IID_IAudioSessionManager2), _CLSCTX_ALL, None, ctypes.byref(manager))
    if hr < 0 or not manager:
        # The device was unplugged or disabled after it was listed.
        log.debug("Couldn't open an audio device's sessions (%s)", _hex(hr))
        return []
    controls: list[ctypes.c_void_p] = []
    try:
        sessions = ctypes.c_void_p()
        hr = _GetSessionEnumerator(manager, ctypes.byref(sessions))
        if hr < 0 or not sessions:
            log.debug("Couldn't list an audio device's sessions (%s)", _hex(hr))
            return []
        try:
            count = ctypes.c_int()
            if _SessionCount(sessions, ctypes.byref(count)) >= 0:
                for index in range(count.value):
                    control = ctypes.c_void_p()
                    if _GetSession(sessions, index, ctypes.byref(control)) >= 0 and control:
                        controls.append(control)
        finally:
            _release(sessions)
    finally:
        _release(manager)
    return controls


def _read_session(control: ctypes.c_void_p) -> _CoreAudioSession | None:
    """Read one session. None if it ended while we looked (its calls then fail) or has expired."""
    control2 = _query(control, _IID_IAudioSessionControl2)
    volume = _query(control, _IID_ISimpleAudioVolume) if control2 is not None else None
    kept = False
    try:
        if control2 is None or volume is None:
            return None
        state, pid, muted = ctypes.c_int(), wintypes.DWORD(), wintypes.BOOL()
        # GetProcessId succeeds with AUDCLNT_S_NO_SINGLE_PROCESS for a session shared by several processes,
        # reporting the one that created it.
        if (
            _GetState(control2, ctypes.byref(state)) < 0
            or _GetProcessId(control2, ctypes.byref(pid)) < 0
            or _GetMute(volume, ctypes.byref(muted)) < 0
        ):
            return None
        system = _IsSystemSoundsSession(control2)  # S_OK: system sounds, S_FALSE: an app
        instance_id = _com_string(control2, _GetSessionInstanceIdentifier)
        session_id = _com_string(control2, _GetSessionIdentifier)
        if system < 0 or not instance_id or not session_id or state.value == _SESSION_EXPIRED:
            return None
        session = _CoreAudioSession(
            control=control2,
            volume=volume,
            instance_id=instance_id,
            session_id=session_id,
            pid=int(pid.value),
            active=state.value == _SESSION_ACTIVE,
            system_sounds=system == _S_OK,
            muted=bool(muted.value),
            app_pid=int(pid.value),
        )
        kept = True
        return session
    finally:
        if not kept:
            _release(volume)
            _release(control2)


def _active_capture_pid(control: ctypes.c_void_p) -> int:
    """The process recording through a capture session, or 0 if it isn't recording (or is system sounds)."""
    control2 = _query(control, _IID_IAudioSessionControl2)
    if control2 is None:
        return 0
    try:
        state, pid = ctypes.c_int(), wintypes.DWORD()
        if _GetState(control2, ctypes.byref(state)) < 0 or state.value != _SESSION_ACTIVE:
            return 0
        if _IsSystemSoundsSession(control2) == _S_OK or _GetProcessId(control2, ctypes.byref(pid)) < 0:
            return 0
        return int(pid.value)
    finally:
        _release(control2)


class CoreAudioSessions:
    """The real ``SessionSource``: Windows Core Audio sessions on every active output device.

    Listing every active output device covers the defaults for media (eMultimedia) and for everything else
    (eConsole), and also apps sent to another device in their own settings or Windows' per-app output setting.
    """

    @contextlib.contextmanager
    def snapshot(self, *, details: bool) -> Iterator[SessionSnapshot]:
        _ensure_com()
        enumerator = ctypes.c_void_p()
        hr = _CoCreateInstance(
            ctypes.byref(_CLSID_MMDeviceEnumerator),
            None,
            _CLSCTX_ALL,
            ctypes.byref(_IID_IMMDeviceEnumerator),
            ctypes.byref(enumerator),
        )
        if hr < 0 or not enumerator:
            raise OSError(f"Windows audio isn't available ({_hex(hr)})")
        sessions: list[_CoreAudioSession] = []
        try:
            devices = _active_endpoints(enumerator, _E_RENDER)
            try:
                for device in devices:
                    controls = _endpoint_sessions(device)
                    try:
                        for control in controls:
                            session = _read_session(control)
                            if session is not None:
                                sessions.append(session)
                    finally:
                        _release_all(controls)
            finally:
                _release_all(devices)
            for session in sessions:
                if not session.system_sounds:
                    session.process_started = process_start_time(session.pid)
            capturing: frozenset[int] = frozenset()
            if details:
                processes = process_table()
                for session in sessions:
                    session.process_name = processes.get(session.pid, (0, ""))[1]
                    session.app_pid, session.app_name = owning_app(session.pid, processes)
                capturing = _capturing_apps(enumerator, processes)
            yield SessionSnapshot(sessions=list(sessions), capturing_apps=capturing)
        finally:
            for session in sessions:
                session.release()
            _release(enumerator)


def _capturing_apps(enumerator: ctypes.c_void_p, processes: Mapping[int, tuple[int, str]]) -> frozenset[int]:
    """``owning_app`` pids of the apps (other than Whisp) recording from any microphone right now."""
    own_pid = os.getpid()
    apps: set[int] = set()
    devices = _active_endpoints(enumerator, _E_CAPTURE)
    try:
        for device in devices:
            controls = _endpoint_sessions(device)
            try:
                for control in controls:
                    pid = _active_capture_pid(control)
                    if pid and pid != own_pid:
                        apps.add(owning_app(pid, processes)[0])
            finally:
                _release_all(controls)
    finally:
        _release_all(devices)
    return frozenset(apps)


# -- muting and restoring ------------------------------------------------------------------


def _same_process(record: MutedSession, session: AudioSession) -> bool:
    """Whether ``session`` still belongs to the process that ``record`` muted (and that process still runs)."""
    if session.process_started == 0:
        return False
    if record.process_started and session.process_started:
        return record.process_started == session.process_started
    return True  # Windows wouldn't say; the session is still listed, so treat it as the same


@dataclass
class _Held:
    """A session muted in the current dictation."""

    record: MutedSession
    watch: MuteWatch | None


class AudioMuter:
    """Mutes other apps' sessions and puts them back exactly as they were.

    Not thread-safe: use it from one thread (``MuteService`` runs it on ``whisp-mute``). Never raises; failures
    are logged and reported as nothing done.
    """

    def __init__(
        self,
        extra_exclusions: Iterable[str] = (),
        *,
        state_file: Path | None = None,
        source: SessionSource | None = None,
    ) -> None:
        self._excluded = COMMUNICATION_APPS
        self.set_extra_exclusions(extra_exclusions)
        self._file = MuteStateFile(state_file if state_file is not None else app_paths().data_dir / STATE_FILE_NAME)
        self._source: SessionSource = source if source is not None else CoreAudioSessions()
        self._own_pid = os.getpid()
        self._loaded = False
        self._held: dict[str, _Held] = {}
        """Sessions muted in the current dictation, by instance id."""
        self._pending: dict[str, MutedSession] = {}
        """Sessions left muted (by a crash, a quit app or a failed unmute) that still need unmuting."""

    @property
    def is_muting(self) -> bool:
        return bool(self._held)

    @property
    def has_pending(self) -> bool:
        """Something was left muted and will be unmuted when its app is back (see ``check_pending``)."""
        return bool(self._pending)

    def set_extra_exclusions(self, names: Iterable[str]) -> None:
        """Process names (``"obs64.exe"``, ``"OBS64"``) to leave alone on top of ``COMMUNICATION_APPS``."""
        extra = {normalize_process_name(name) for name in names}
        self._excluded = COMMUNICATION_APPS | frozenset(name for name in extra if name)

    def mute_others(self) -> int:
        """Mute every other app that is playing sound and should be quiet. Returns how many sessions it muted.

        Calling it again while muted also mutes apps that started playing since.
        """
        self._load()
        try:
            with self._source.snapshot(details=True) as snap:
                return self._mute(snap)
        except Exception:
            log.exception("Couldn't mute other apps")
            return 0

    def restore(self) -> int:
        """Unmute the sessions this muter muted, unless someone else changed their mute meanwhile.

        Also unmutes anything pending whose app is back. Returns how many sessions it unmuted. Safe to call at any
        time, any number of times.
        """
        self._load()
        if not self._held and not self._pending:
            return 0
        held, self._held = self._held, {}
        count = 0
        try:
            with self._source.snapshot(details=False) as snap:
                restored, taken_over = self._restore(snap, held)
                count = restored + self._settle_pending(snap, claimed=taken_over)
        except Exception:
            log.exception("Couldn't unmute other apps")
        finally:
            for item in held.values():
                # Not settled because the snapshot failed: retry later, unless the user has taken it over.
                if item.watch is None or not item.watch.changed_by_others:
                    self._pending.setdefault(item.record.instance_id, item.record)
                if item.watch is not None:
                    item.watch.close()
            self._save()
        if count:
            log.debug("Unmuted %d app session(s)", count)
        return count

    def restore_leftovers(self) -> int:
        """Unmute what an earlier run left muted (it crashed, or an app was still away). Call once at startup.

        A listed session that still exists is unmuted if it's still muted. If its app has quit, the app's next
        session will start muted, so the entry stays pending until that session appears. Returns how many sessions
        it unmuted. The file is deleted once nothing is left to unmute.
        """
        self._load()
        if not self._pending:
            return 0
        return self._check(always_save=True)

    def check_pending(self) -> int:
        """Unmute pending sessions whose apps are back. Does nothing while muting. Returns how many it unmuted."""
        self._load()
        if self._held or not self._pending:
            return 0
        return self._check(always_save=False)

    # -- internals ---------------------------------------------------------------------

    def _load(self) -> None:
        """Pick up the saved list once, before anything here can overwrite it."""
        if self._loaded:
            return
        self._loaded = True
        records = self._file.read()
        for record in records:
            if record.instance_id not in self._held:
                self._pending.setdefault(record.instance_id, record)
        if records:
            log.debug("Found %d app session(s) left muted by an earlier run", len(records))

    def _mute(self, snap: SessionSnapshot) -> int:
        pending_before = set(self._pending)
        settled = self._settle_pending(snap, claimed=self._held.keys())
        if settled:
            log.debug("Unmuted %d app session(s) left muted earlier", settled)
        candidates: list[AudioSession] = []
        for session in snap.sessions:
            if session.instance_id in self._held:
                continue  # muted by us already, or unmuted by the user since: theirs now
            reason = skip_reason(
                session, own_pid=self._own_pid, excluded=self._excluded, capturing_apps=snap.capturing_apps
            )
            if reason is None:
                candidates.append(session)
            elif session.active and session.process_name and session.pid != self._own_pid:
                log.debug("Not muting %s: %s", session.process_name, reason)
        if not candidates:
            if set(self._pending) != pending_before:
                self._save()
            return 0
        now = time.time()
        records = {
            s.instance_id: MutedSession(
                instance_id=s.instance_id,
                session_id=s.session_id,
                pid=s.pid,
                process_name=s.process_name,
                muted_at=now,
                siblings=tuple(o.instance_id for o in snap.sessions if o.session_id == s.session_id and o is not s),
                process_started=s.process_started,
            )
            for s in candidates
        }
        # Save before muting: if Whisp dies after this point, the next start knows what to unmute.
        if not self._file.write([*self._entries(), *records.values()]):
            log.warning("Not muting other apps: couldn't save the list needed to unmute them after a crash")
            return 0
        muted: list[str] = []
        for session in candidates:
            # Watch first, so a change the user makes right after the mute can't slip past.
            watch = session.watch()
            if session.set_mute(True):
                self._held[session.instance_id] = _Held(records[session.instance_id], watch)
                muted.append(session.process_name)
            elif watch is not None:
                watch.close()
        if len(muted) != len(candidates):
            self._save()
        if muted:
            log.debug("Muted %d app session(s): %s", len(muted), ", ".join(sorted(set(muted))))
        return len(muted)

    def _restore(self, snap: SessionSnapshot, held: dict[str, _Held]) -> tuple[int, set[str]]:
        """Settle the sessions muted in this dictation, removing each from ``held`` once handled.

        Returns how many were unmuted, and the instance ids the user took over.
        """
        by_instance = {s.instance_id: s for s in snap.sessions}
        count = 0
        taken_over: set[str] = set()
        for instance_id, item in list(held.items()):
            if item.watch is not None and item.watch.changed_by_others:
                log.debug("Leaving %s as it is: its mute was changed during the dictation", item.record.process_name)
                taken_over.add(instance_id)
            else:
                unmuted, settled = self._settle(item.record, snap, by_instance, claimed=frozenset())
                count += unmuted
                if not settled:
                    self._pending[instance_id] = item.record
            if item.watch is not None:
                item.watch.close()
            del held[instance_id]
        return count, taken_over

    def _settle_pending(self, snap: SessionSnapshot, *, claimed: Set[str]) -> int:
        if not self._pending:
            return 0
        by_instance = {s.instance_id: s for s in snap.sessions}
        now = time.time()
        count = 0
        for instance_id, record in list(self._pending.items()):
            if now - record.muted_at > PENDING_MAX_AGE_S:
                log.debug("No longer waiting to unmute %s: it hasn't come back in 30 days", record.process_name)
                del self._pending[instance_id]
                continue
            unmuted, settled = self._settle(record, snap, by_instance, claimed=claimed)
            count += unmuted
            if settled:
                del self._pending[instance_id]
        return count

    @staticmethod
    def _settle(
        record: MutedSession, snap: SessionSnapshot, by_instance: Mapping[str, AudioSession], *, claimed: Set[str]
    ) -> tuple[int, bool]:
        """Unmute what ``record`` left muted, except sessions in ``claimed``. Returns (sessions unmuted, done)."""
        if record.instance_id in claimed:
            return 0, True
        session = by_instance.get(record.instance_id)
        if session is not None and _same_process(record, session):
            if not session.muted:
                return 0, True  # unmuted already, by someone else or by an earlier pass
            if session.set_mute(False):
                return 1, True
            return 0, False  # try again later
        # The process has quit (its session may linger for a moment), so Windows has saved the app as muted, and
        # unmuting the dead session wouldn't change that. The app's next session on this device starts muted:
        # unmute it once it exists. Sessions that existed when we muted aren't affected, and one the user (or this
        # dictation) has claimed is theirs: its existence alone settles the record.
        newcomers = [
            s
            for s in snap.sessions
            if s.session_id == record.session_id and s.instance_id not in record.siblings and s.process_started != 0
        ]
        if not newcomers:
            return 0, False
        count, done = 0, True
        for newcomer in newcomers:
            if newcomer.instance_id in claimed or not newcomer.muted:
                continue
            if newcomer.set_mute(False):
                count += 1
            else:
                done = False
        return count, done

    def _check(self, *, always_save: bool) -> int:
        before = set(self._pending)
        count = 0
        try:
            with self._source.snapshot(details=False) as snap:
                count = self._settle_pending(snap, claimed=self._held.keys())
        except Exception:
            log.exception("Couldn't unmute apps left muted earlier")
        if always_save or set(self._pending) != before:
            self._save()
        if count:
            log.debug("Unmuted %d app session(s) left muted earlier", count)
        return count

    def _entries(self) -> list[MutedSession]:
        held = [item.record for item in self._held.values()]
        return held + [record for instance_id, record in self._pending.items() if instance_id not in self._held]

    def _save(self) -> None:
        entries = self._entries()
        if entries:
            self._file.write(entries)
        else:
            self._file.delete()


# -- the worker thread ---------------------------------------------------------------------


class _Muter(Protocol):
    """What ``MuteService`` needs from an ``AudioMuter``."""

    @property
    def has_pending(self) -> bool:
        """Something left muted is waiting for its app to come back."""

    def set_extra_exclusions(self, names: Iterable[str]) -> None:
        """Replace the user's own list of apps to leave alone."""

    def mute_others(self) -> int:
        """Mute other apps."""

    def restore(self) -> int:
        """Unmute what was muted."""

    def restore_leftovers(self) -> int:
        """Unmute what an earlier run left muted."""

    def check_pending(self) -> int:
        """Unmute pending sessions whose apps are back."""


class MuteService:
    """Runs an ``AudioMuter`` on its own thread, ``whisp-mute``, so no COM call ever blocks the caller.

    ``mute()`` and ``restore()`` return at once and may be called from any thread. Requests are coalesced: the
    worker always moves to the most recent request, so ``mute(); restore(); mute()`` in quick succession costs one
    mute (or nothing if it is muted already), and repeated calls cost nothing.

    On start, the worker unmutes anything an earlier run left muted (``restore_leftovers``). So create the service
    at startup even when the feature is off, and only gate the ``mute()`` calls on the setting.
    """

    def __init__(self, muter: _Muter | None = None, *, extra_exclusions: Iterable[str] = ()) -> None:
        self._muter: _Muter = muter if muter is not None else AudioMuter()
        self._cond = threading.Condition()
        self._want_muted = False
        self._closing = False
        names = tuple(extra_exclusions)
        self._new_exclusions: tuple[str, ...] | None = names or None
        self._thread = threading.Thread(target=self._run, name="whisp-mute", daemon=True)
        self._thread.start()

    def mute(self) -> None:
        """Mute other apps. Returns at once."""
        self._request(True)

    def restore(self) -> None:
        """Unmute what Whisp muted. Returns at once."""
        self._request(False)

    def set_extra_exclusions(self, names: Iterable[str]) -> None:
        """Replace the user's own list of apps to leave alone, from the next mute on. Returns at once."""
        with self._cond:
            if self._closing:
                return
            self._new_exclusions = tuple(names)
            self._cond.notify()

    def shutdown(self, timeout: float = 2.0) -> None:
        """Unmute everything Whisp muted and stop the thread, waiting at most ``timeout`` seconds.

        If Windows audio doesn't answer in time, the app can still exit: the saved list lets the next start
        unmute. Later requests are ignored. Safe to call more than once.
        """
        with self._cond:
            self._closing = True
            self._want_muted = False
            self._cond.notify()
        if threading.current_thread() is self._thread:
            return
        self._thread.join(max(0.0, timeout))
        if self._thread.is_alive():
            log.warning("Unmuting other apps was still running after %.1f s; leaving it to finish", timeout)

    # -- worker thread -------------------------------------------------------------------

    def _request(self, muted: bool) -> None:
        with self._cond:
            if self._closing:
                log.debug("The mute service is shut down; ignoring a %s request", "mute" if muted else "restore")
                return
            if self._want_muted != muted:
                self._want_muted = muted
                self._cond.notify()

    def _run(self) -> None:
        try:
            _ensure_com()
        except OSError:
            log.exception("Muting other apps is unavailable")
        try:
            self._serve()
        finally:
            self._call(self._muter.restore, "unmute other apps on exit")
            _end_com()

    def _serve(self) -> None:
        self._call(self._muter.restore_leftovers, "unmute apps left muted by an earlier run")
        muted = False
        pending_since: float | None = None
        next_check: float | None = None
        while True:
            # Look for apps left muted only between dictations. One that comes back mid-dictation stays quiet until
            # the restore, which settles it.
            if muted or not self._has_pending():
                next_check = None
                if not muted:
                    pending_since = None
            elif next_check is None:
                now = time.monotonic()
                if pending_since is None:
                    pending_since = now
                fast = now - pending_since < PENDING_FAST_PERIOD_S
                next_check = now + (PENDING_CHECK_FAST_S if fast else PENDING_CHECK_SLOW_S)
            with self._cond:
                while not self._closing and self._want_muted == muted and self._new_exclusions is None:
                    timeout = None if next_check is None else next_check - time.monotonic()
                    if timeout is not None and timeout <= 0:
                        break
                    self._cond.wait(timeout)
                closing, want_muted = self._closing, self._want_muted
                exclusions, self._new_exclusions = self._new_exclusions, None
            if exclusions is not None:
                self._call(functools.partial(self._muter.set_extra_exclusions, exclusions), "update the exclusions")
            if closing:
                return
            if want_muted != muted:
                muted = want_muted  # set first: a mute that fails partway must still be restored
                if muted:
                    self._call(self._muter.mute_others, "mute other apps")
                else:
                    self._call(self._muter.restore, "unmute other apps")
                    pending_since = None  # whatever this dictation left pending gets the quick checks
            elif next_check is not None and time.monotonic() >= next_check:
                next_check = None
                self._call(self._muter.check_pending, "unmute apps left muted earlier")

    def _has_pending(self) -> bool:
        try:
            return bool(self._muter.has_pending)
        except Exception:
            log.exception("Couldn't check for apps left muted")
            return False

    @staticmethod
    def _call(action: Callable[[], object], what: str) -> None:
        try:
            action()
        except Exception:
            log.exception("Couldn't %s", what)
