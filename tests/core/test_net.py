"""The certificate check behind every HTTPS connection, and telling TLS failures apart."""

from __future__ import annotations

import logging
import ssl
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from whisp.core import net


@pytest.fixture
def fresh_context() -> Iterator[None]:
    net.tls_context.cache_clear()
    yield
    net.tls_context.cache_clear()


def test_the_context_checks_certificates_and_is_shared(fresh_context: None) -> None:
    context = net.tls_context()
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname
    assert context.cert_store_stats()["x509_ca"] > 100  # certifi's list, plus Windows' own
    assert net.tls_context() is context


def test_an_unreadable_windows_store_leaves_certifis_list(
    fresh_context: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(self: ssl.SSLContext, purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH) -> None:
        raise ssl.SSLError("bad certificate in the store")

    monkeypatch.setattr(ssl.SSLContext, "load_default_certs", broken)
    with caplog.at_level(logging.WARNING, logger="whisp.core.net"):
        context = net.tls_context()
    assert context.cert_store_stats()["x509_ca"] > 100
    assert "certificate store" in caplog.text


def test_tls_failures_are_found_behind_httpx_errors() -> None:
    certificate = ssl.SSLCertVerificationError(1, "certificate verify failed: unable to get local issuer")
    wrapped = httpx.ConnectError("connection failed")
    wrapped.__cause__ = certificate
    assert net.tls_error(wrapped) is certificate
    assert net.is_certificate_error(wrapped)

    handshake = httpx.ConnectError("connection failed")
    handshake.__context__ = ssl.SSLError("wrong version number")
    assert isinstance(net.tls_error(handshake), ssl.SSLError)
    assert not net.is_certificate_error(handshake)

    refused = httpx.ConnectError("connection refused")
    refused.__cause__ = ConnectionRefusedError(10061, "refused")
    assert net.tls_error(refused) is None
    assert not net.is_certificate_error(refused)


class _PlainHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        pass  # a TLS handshake isn't an HTTP request; the client hanging up on the reply is expected


def test_https_to_a_plain_http_server_is_a_tls_failure() -> None:
    # What really happens when an address without http:// in front gets https and the server only speaks http.
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

    httpd = _PlainHttpServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        with (
            httpx.Client(verify=net.tls_context(), trust_env=False, timeout=5) as client,
            pytest.raises(httpx.ConnectError) as caught,
        ):
            client.get(f"https://127.0.0.1:{httpd.server_address[1]}/")
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert net.tls_error(caught.value) is not None
    assert not net.is_certificate_error(caught.value)
