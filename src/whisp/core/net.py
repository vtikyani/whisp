"""How Whisp checks the servers it connects to over HTTPS.

Certificates are checked against certifi's list of certificate authorities
(Mozilla's, the same on every PC) and also against the authorities in Windows'
own certificate store. Networks that inspect HTTPS with an authority of their
own, which IT departments install into Windows, then work too, as they do for
the browser.
"""

from __future__ import annotations

import functools
import logging
import ssl

import certifi

log = logging.getLogger(__name__)


@functools.cache
def tls_context() -> ssl.SSLContext:
    """The certificate check for every HTTPS connection Whisp makes. Shared: one context serves every thread."""
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        context.load_default_certs(ssl.Purpose.SERVER_AUTH)  # on Windows: its CA and ROOT certificate stores
    except (ssl.SSLError, OSError) as exc:
        log.warning("Couldn't read Windows' certificate store (%s); using certifi's list only", type(exc).__name__)
    return context


def tls_error(exc: BaseException) -> ssl.SSLError | None:
    """The TLS failure behind ``exc`` (httpx wraps the one from Python's ``ssl`` module), or None if there isn't one."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ssl.SSLError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def is_certificate_error(exc: BaseException) -> bool:
    """True if ``exc`` comes from a server certificate that couldn't be verified: retrying won't help."""
    return isinstance(tls_error(exc), ssl.SSLCertVerificationError)
