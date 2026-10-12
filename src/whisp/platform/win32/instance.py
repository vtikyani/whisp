"""One Whisp per user session.

A named mutex in the session's ``Local\\`` namespace is the lock: the first
launch creates it, later launches find it. The first launch also listens on a
local socket (a named pipe) with the same name, so a second launch can ask it
to show itself before quitting.

Named pipes are machine-wide, so the name is derived from the user's SID *and*
the session id: two users, or one user signed in twice (console + remote
desktop), never collide.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import time
from ctypes import wintypes

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger(__name__)

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

_CreateMutexW = _kernel32.CreateMutexW
_CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
_CreateMutexW.restype = wintypes.HANDLE

_OpenMutexW = _kernel32.OpenMutexW
_OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_OpenMutexW.restype = wintypes.HANDLE

_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL

_GetCurrentProcess = _kernel32.GetCurrentProcess
_GetCurrentProcess.argtypes = []
_GetCurrentProcess.restype = wintypes.HANDLE

_GetCurrentProcessId = _kernel32.GetCurrentProcessId
_GetCurrentProcessId.argtypes = []
_GetCurrentProcessId.restype = wintypes.DWORD

_ProcessIdToSessionId = _kernel32.ProcessIdToSessionId
_ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
_ProcessIdToSessionId.restype = wintypes.BOOL

_LocalFree = _kernel32.LocalFree
_LocalFree.argtypes = [wintypes.HLOCAL]
_LocalFree.restype = wintypes.HLOCAL


class OVERLAPPED(ctypes.Structure):
    _fields_ = (
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    )


_CreateFileW = _kernel32.CreateFileW
_CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_CreateFileW.restype = wintypes.HANDLE

_WaitNamedPipeW = _kernel32.WaitNamedPipeW
_WaitNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
_WaitNamedPipeW.restype = wintypes.BOOL

_WriteFile = _kernel32.WriteFile
_WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD, wintypes.LPDWORD, ctypes.POINTER(OVERLAPPED)]
_WriteFile.restype = wintypes.BOOL

_GetOverlappedResult = _kernel32.GetOverlappedResult
_GetOverlappedResult.argtypes = [wintypes.HANDLE, ctypes.POINTER(OVERLAPPED), wintypes.LPDWORD, wintypes.BOOL]
_GetOverlappedResult.restype = wintypes.BOOL

_CancelIoEx = _kernel32.CancelIoEx
_CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(OVERLAPPED)]
_CancelIoEx.restype = wintypes.BOOL

_CreateEventW = _kernel32.CreateEventW
_CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
_CreateEventW.restype = wintypes.HANDLE

_WaitForSingleObject = _kernel32.WaitForSingleObject
_WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_WaitForSingleObject.restype = wintypes.DWORD

_OpenProcessToken = _advapi32.OpenProcessToken
_OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
_OpenProcessToken.restype = wintypes.BOOL

_GetTokenInformation = _advapi32.GetTokenInformation
_GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_GetTokenInformation.restype = wintypes.BOOL

_ConvertSidToStringSidW = _advapi32.ConvertSidToStringSidW
_ConvertSidToStringSidW.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
_ConvertSidToStringSidW.restype = wintypes.BOOL

TOKEN_QUERY = 0x0008
TOKEN_USER_CLASS = 1  # TokenUser
SYNCHRONIZE = 0x00100000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_FLAG_OVERLAPPED = 0x40000000
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
WAIT_OBJECT_0 = 0
ERROR_ACCESS_DENIED = 5
ERROR_ALREADY_EXISTS = 183
ERROR_PIPE_BUSY = 231
ERROR_IO_PENDING = 997

MAX_MESSAGE_BYTES = 64 * 1024
"""A connection that sends more than this without a line break is dropped."""


def _user_sid() -> str | None:
    token = wintypes.HANDLE()
    if not _OpenProcessToken(_GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        needed = wintypes.DWORD()
        _GetTokenInformation(token, TOKEN_USER_CLASS, None, 0, ctypes.byref(needed))
        if not needed.value:
            return None
        buffer = ctypes.create_string_buffer(needed.value)
        if not _GetTokenInformation(token, TOKEN_USER_CLASS, buffer, needed, ctypes.byref(needed)):
            return None
        # TOKEN_USER starts with SID_AND_ATTRIBUTES, whose first member is the SID pointer.
        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(wintypes.LPVOID))[0]
        text = wintypes.LPWSTR()
        if not _ConvertSidToStringSidW(sid_pointer, ctypes.byref(text)):
            return None
        try:
            return text.value
        finally:
            _LocalFree(ctypes.cast(text, wintypes.HLOCAL))
    finally:
        _CloseHandle(token)


def _session_id() -> int:
    session = wintypes.DWORD()
    if _ProcessIdToSessionId(_GetCurrentProcessId(), ctypes.byref(session)):
        return int(session.value)
    return 0


def _remaining_ms(deadline: float) -> int:
    return max(0, int((deadline - time.monotonic()) * 1000))


def _write_with_timeout(handle: int, payload: bytes, deadline: float) -> bool:
    """Write ``payload`` to an overlapped pipe handle, giving up (and cancelling) at ``deadline``."""
    event = _CreateEventW(None, True, False, None)
    if not event:
        log.warning("Couldn't create an event to send the message (error %d)", ctypes.get_last_error())
        return False
    try:
        overlapped = OVERLAPPED(hEvent=event)
        written = wintypes.DWORD()
        # The buffer must stay alive until the write completes or is cancelled (both are awaited below).
        buffer = ctypes.create_string_buffer(payload, len(payload))
        if not _WriteFile(handle, buffer, len(payload), None, ctypes.byref(overlapped)):
            error = ctypes.get_last_error()
            if error != ERROR_IO_PENDING:
                log.warning("Couldn't send the message to the running Whisp (error %d)", error)
                return False
            if _WaitForSingleObject(event, _remaining_ms(deadline)) != WAIT_OBJECT_0:
                _CancelIoEx(handle, ctypes.byref(overlapped))
                _GetOverlappedResult(handle, ctypes.byref(overlapped), ctypes.byref(written), True)
                log.warning("The running Whisp didn't read the message in time")
                return False
        if not _GetOverlappedResult(handle, ctypes.byref(overlapped), ctypes.byref(written), False):
            log.warning("Sending the message failed (error %d)", ctypes.get_last_error())
            return False
        return written.value == len(payload)
    finally:
        _CloseHandle(event)


def default_instance_name() -> str:
    """``Whisp-<hash>``, unique to this user in this sign-in session."""
    user = _user_sid() or os.environ.get("USERNAME", "user")
    digest = hashlib.sha256(f"{user}|{_session_id()}".encode()).hexdigest()[:16]
    return f"Whisp-{digest}"


class SingleInstance(QObject):
    """Single-instance lock plus a channel for activation messages.

    ``message`` is emitted on the thread that owns this object (the GUI thread)
    for every line another launch sends with ``send``.
    """

    message = Signal(str)

    def __init__(self, name: str | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._name = name or default_instance_name()
        self._mutex: int | None = None
        self._server: QLocalServer | None = None
        self._buffers: dict[QLocalSocket, bytearray] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def _mutex_name(self) -> str:
        return "Local\\" + self._name

    def acquire(self) -> bool:
        """True if this is the first instance (it then listens for messages); False if another one runs."""
        if self._mutex is not None:
            return True
        handle = _CreateMutexW(None, False, self._mutex_name)
        error = ctypes.get_last_error()
        if not handle:
            if error == ERROR_ACCESS_DENIED:
                # The mutex exists but belongs to a more privileged Whisp (e.g. one run as administrator).
                return False
            # Better to run twice than not at all; the hotkey conflict would then be reported to the user.
            log.error("Couldn't create the single-instance lock (error %d); continuing without it", error)
            self._listen()
            return True
        if error == ERROR_ALREADY_EXISTS:
            _CloseHandle(handle)
            return False
        self._mutex = int(handle)
        self._listen()
        return True

    def _listen(self) -> None:
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not server.listen(self._name):
            # A previous instance that crashed can leave the name behind (a socket file on some platforms).
            QLocalServer.removeServer(self._name)
            if not server.listen(self._name):
                log.warning("Couldn't listen for messages from other launches: %s", server.errorString())
                server.deleteLater()
                return
        server.newConnection.connect(self._on_new_connection)
        self._server = server

    def _primary_exists(self) -> bool:
        handle = _OpenMutexW(SYNCHRONIZE, False, self._mutex_name)
        if handle:
            _CloseHandle(handle)
            return True
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED

    def send(self, message: str, timeout_ms: int = 3000) -> bool:
        """Deliver ``message`` to the first instance. Returns False if there is none or it didn't answer.

        Blocking (up to ``timeout_ms``); meant for a second launch before it
        shows anything, and usable from any thread (it talks to the pipe
        directly rather than through Qt's event loop). Retries while the first
        instance holds the lock but is still starting up and not listening yet.
        """
        payload = (" ".join(message.splitlines()) + "\n").encode("utf-8")
        deadline = time.monotonic() + timeout_ms / 1000.0
        pipe = "\\\\.\\pipe\\" + self._name
        while self._primary_exists():
            handle = _CreateFileW(pipe, GENERIC_WRITE, 0, None, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, None)
            if handle and handle != INVALID_HANDLE_VALUE:
                try:
                    return _write_with_timeout(handle, payload, deadline)
                finally:
                    _CloseHandle(handle)
            error = ctypes.get_last_error()
            remaining = _remaining_ms(deadline)
            if remaining <= 0:
                log.warning("The running Whisp didn't answer within %d ms (error %d)", timeout_ms, error)
                return False
            if error == ERROR_PIPE_BUSY:
                # Every listening pipe instance is in use; wait for the server to offer another.
                _WaitNamedPipeW(pipe, min(remaining, 500))
            else:
                # The first instance holds the lock but isn't listening yet: it is still starting up.
                time.sleep(0.05)
        return False

    def release(self) -> None:
        """Stop listening and give up the lock. Safe to call more than once."""
        if self._server is not None:
            self._server.close()
            self._server.deleteLater()
            self._server = None
        for socket in list(self._buffers):
            socket.abort()
            socket.deleteLater()
        self._buffers.clear()
        if self._mutex is not None:
            _CloseHandle(self._mutex)
            self._mutex = None

    # -- incoming messages -----------------------------------------------------------

    def _on_new_connection(self) -> None:
        server = self._server
        if server is None:
            return
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            if socket is None:
                break
            self._buffers[socket] = bytearray()
            socket.readyRead.connect(lambda s=socket: self._read(s))
            socket.disconnected.connect(lambda s=socket: self._on_disconnected(s))
            if socket.bytesAvailable():
                self._read(socket)

    def _read(self, socket: QLocalSocket) -> None:
        buffer = self._buffers.get(socket)
        if buffer is None:
            return
        buffer += bytes(socket.readAll().data())
        while b"\n" in buffer:
            line, _, rest = bytes(buffer).partition(b"\n")
            buffer[:] = rest
            self._emit_line(line)
        if len(buffer) > MAX_MESSAGE_BYTES:
            log.warning("Dropped an oversized message from another launch")
            buffer.clear()
            socket.abort()

    def _on_disconnected(self, socket: QLocalSocket) -> None:
        if socket in self._buffers:
            self._read(socket)
            leftover = self._buffers.pop(socket, bytearray())
            if leftover:
                self._emit_line(bytes(leftover))
        socket.deleteLater()

    def _emit_line(self, line: bytes) -> None:
        text = line.decode("utf-8", errors="replace").strip()
        if text:
            log.info("Message from another launch: %s", text[:40])
            self.message.emit(text)
