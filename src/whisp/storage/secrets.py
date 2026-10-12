"""Encryption for small secrets such as API keys, using Windows DPAPI.

``protect`` encrypts text for the current Windows user and returns base64
that is safe to keep in ``settings.json``. Only the same user account can
turn it back into the original with ``unprotect`` (DPAPI keys follow the
account, so a roamed profile still works; a copied file on another account
does not).

A fixed, app-specific "entropy" value is mixed in, so another program that
asks DPAPI to decrypt the blob without knowing it gets nothing back. DPAPI is
never allowed to show UI (``CRYPTPROTECT_UI_FORBIDDEN``): this runs on worker
threads and must not block on a prompt.

Never log the plain text or the token.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import logging
from ctypes import wintypes

log = logging.getLogger(__name__)

CRYPTPROTECT_UI_FORBIDDEN = 0x1
_ENTROPY = b"Whisp.Dictation/secrets/v1"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


_crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_CryptProtectData = _crypt32.CryptProtectData
_CryptProtectData.argtypes = [
    ctypes.POINTER(_DataBlob),  # pDataIn
    wintypes.LPCWSTR,  # szDataDescr
    ctypes.POINTER(_DataBlob),  # pOptionalEntropy
    ctypes.c_void_p,  # pvReserved
    ctypes.c_void_p,  # pPromptStruct
    wintypes.DWORD,  # dwFlags
    ctypes.POINTER(_DataBlob),  # pDataOut
]
_CryptProtectData.restype = wintypes.BOOL

_CryptUnprotectData = _crypt32.CryptUnprotectData
_CryptUnprotectData.argtypes = [
    ctypes.POINTER(_DataBlob),  # pDataIn
    ctypes.POINTER(wintypes.LPWSTR),  # ppszDataDescr
    ctypes.POINTER(_DataBlob),  # pOptionalEntropy
    ctypes.c_void_p,  # pvReserved
    ctypes.c_void_p,  # pPromptStruct
    wintypes.DWORD,  # dwFlags
    ctypes.POINTER(_DataBlob),  # pDataOut
]
_CryptUnprotectData.restype = wintypes.BOOL

_LocalFree = _kernel32.LocalFree
_LocalFree.argtypes = [ctypes.c_void_p]
_LocalFree.restype = ctypes.c_void_p


def _blob(buffer: ctypes.Array[ctypes.c_char], size: int) -> _DataBlob:
    return _DataBlob(size, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))


def _take_output(blob: _DataBlob) -> bytes:
    """Copy a DPAPI output buffer, wipe it, and free it."""
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        if blob.pbData:
            ctypes.memset(blob.pbData, 0, blob.cbData)
            _LocalFree(ctypes.cast(blob.pbData, ctypes.c_void_p))


def protect(plain: str) -> str:
    """Encrypt ``plain`` for the current Windows user. Returns base64 text.

    An empty string stays empty, so "no key" is stored as "". Raises
    ``OSError`` if Windows refuses to encrypt (which should not happen for a
    normal signed-in user).
    """
    if not plain:
        return ""
    data = plain.encode("utf-8")
    data_buf = ctypes.create_string_buffer(data, len(data))
    entropy_buf = ctypes.create_string_buffer(_ENTROPY, len(_ENTROPY))
    data_in = _blob(data_buf, len(data))
    entropy = _blob(entropy_buf, len(_ENTROPY))
    data_out = _DataBlob()
    try:
        ok = _CryptProtectData(
            ctypes.byref(data_in),
            None,
            ctypes.byref(entropy),
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(data_out),
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return base64.b64encode(_take_output(data_out)).decode("ascii")
    finally:
        # Don't leave the plain text lying around in the ctypes buffer.
        ctypes.memset(data_buf, 0, len(data))


def unprotect(token: str) -> str:
    """Decrypt a value made by ``protect``. Returns "" if it can't be decrypted.

    That happens for an empty or corrupted token, or one made by a different
    Windows user. Failures are logged without the token.
    """
    if not token or not isinstance(token, str):
        return ""
    try:
        encrypted = base64.b64decode(token.strip(), validate=True)
    except (binascii.Error, ValueError):
        log.warning("A stored secret is not valid base64 (%d characters); ignoring it", len(token))
        return ""
    if not encrypted:
        return ""
    enc_buf = ctypes.create_string_buffer(encrypted, len(encrypted))
    entropy_buf = ctypes.create_string_buffer(_ENTROPY, len(_ENTROPY))
    data_in = _blob(enc_buf, len(encrypted))
    entropy = _blob(entropy_buf, len(_ENTROPY))
    data_out = _DataBlob()
    ok = _CryptUnprotectData(
        ctypes.byref(data_in),
        None,
        ctypes.byref(entropy),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(data_out),
    )
    if not ok:
        error = ctypes.get_last_error()
        log.warning("A stored secret could not be decrypted for this Windows user (error %d); ignoring it", error)
        return ""
    plain = _take_output(data_out)
    try:
        return plain.decode("utf-8")
    except UnicodeDecodeError:
        log.warning("A stored secret decrypted to invalid text; ignoring it")
        return ""
