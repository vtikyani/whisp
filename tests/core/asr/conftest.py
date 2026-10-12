"""Fixtures for the speech recognition tests: a local HTTP file server with misbehaviour switches."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from whisp.core.asr import downloads


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: object, client_address: object) -> None:
        """Clients hanging up mid-response (cancel tests) are expected; don't print tracebacks for them."""


@dataclass
class FileServer:
    """Serves in-memory files over HTTP on 127.0.0.1, with switches to make it misbehave.

    Supports ``Range: bytes=N-`` requests (206 responses) unless told to ignore them.
    """

    files: dict[str, bytes] = field(default_factory=dict)
    requests: list[tuple[str, str | None]] = field(default_factory=list)
    """(path, Range header) for every GET, in order."""
    fail: dict[str, list[int]] = field(default_factory=dict)
    """path -> HTTP statuses to answer with, one per request, before serving the file."""
    drop_after: dict[str, int] = field(default_factory=dict)
    """path -> close the connection after this many body bytes (once)."""
    always_drop: set[str] = field(default_factory=set)
    """Paths whose every response is cut off before the first body byte."""
    ignore_range: set[str] = field(default_factory=set)
    corrupt: set[str] = field(default_factory=set)
    """Paths served with their last byte flipped (same size, wrong checksum)."""
    chunk_delay: float = 0.0
    """Seconds to sleep between 16 KB chunks, for tests that need a download to take a while."""
    port: int = 0
    _server: ThreadingHTTPServer | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, path: str, data: bytes) -> tuple[str, int, str]:
        """Serve ``data`` at ``path``. Returns (url, size, sha256)."""
        self.files[path] = data
        return self.url(path), len(data), hashlib.sha256(data).hexdigest()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def gets(self, path: str) -> list[str | None]:
        """Range headers of the GETs for ``path``."""
        with self._lock:
            return [header for p, header in self.requests if p == path]

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                header = self.headers.get("Range")
                with owner._lock:
                    owner.requests.append((path, header))
                    statuses = owner.fail.get(path)
                    status = statuses.pop(0) if statuses else None
                    drop = 0 if path in owner.always_drop else owner.drop_after.pop(path, None)
                if status is not None:
                    self.send_response(status)
                    self.send_header("Content-Length", "0")
                    self.send_header("Retry-After", "0")
                    self.end_headers()
                    return
                data = owner.files.get(path)
                if data is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path in owner.corrupt:
                    data = data[:-1] + bytes([data[-1] ^ 0xFF])
                start = 0
                if header and header.startswith("bytes=") and path not in owner.ignore_range:
                    start = int(header[6:].split("-", 1)[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                sent = 0
                chunk = 16 * 1024
                try:
                    while sent < len(body):
                        piece = body[sent : sent + chunk]
                        if drop is not None and sent + len(piece) > drop:
                            self.wfile.write(piece[: max(0, drop - sent)])
                            self.wfile.flush()
                            self.close_connection = True
                            self.connection.close()
                            return
                        self.wfile.write(piece)
                        sent += len(piece)
                        if owner.chunk_delay:
                            time.sleep(owner.chunk_delay)
                except (ConnectionError, OSError):
                    return

        self._server = _QuietServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, name="test-file-server", daemon=True).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


@pytest.fixture
def file_server() -> Iterator[FileServer]:
    server = FileServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry immediately instead of backing off for seconds."""
    monkeypatch.setattr(downloads, "BACKOFF_BASE_S", 0.0)
