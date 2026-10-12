"""Resumable, verified downloads and safe unpacking for the local AI engine and models."""

from __future__ import annotations

import hashlib
import ssl
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from whisp.core.llm import runtime as rt

DATA = bytes(range(256)) * 4096  # 1 MiB
SHA = hashlib.sha256(DATA).hexdigest()


class RangeServer:
    """Serves ``data`` at /file. Can ignore Range, drop the connection part-way, or fail with a status."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.honour_range = True
        self.cut_after: int | None = None
        """Send only this many bytes of the first response, then close the connection."""
        self.status: int | None = None
        self.ranges: list[str | None] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                server.ranges.append(self.headers.get("Range"))
                if server.status is not None:
                    self.send_response(server.status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                if self.headers.get("Range") and server.honour_range:
                    start = int(self.headers["Range"].split("=")[1].split("-")[0])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(server.data) - 1}/{len(server.data)}")
                else:
                    self.send_response(200)
                body = server.data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if server.cut_after is not None:
                    cut, server.cut_after = server.cut_after, None
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/file"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[RangeServer]:
    s = RangeServer(DATA)
    yield s
    s.close()


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    # A short connect timeout keeps the "no connection" test quick (Windows is slow to refuse).
    with httpx.Client(timeout=httpx.Timeout(10, connect=0.5), trust_env=False) as c:
        yield c


@pytest.fixture(autouse=True)
def quick_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "_DOWNLOAD_RETRIES", (0.01, 0.01, 0.01))


class Progress:
    def __init__(self) -> None:
        self.total = 0
        self.lowest = 0

    def __call__(self, n: int) -> None:
        self.total += n
        self.lowest = min(self.lowest, self.total)


def download(client: httpx.Client, url: str, dest: Path, progress: Progress, cancel: threading.Event | None = None):
    rt.download_file(
        client, url, dest, size=len(DATA), sha256=SHA, cancel=cancel or threading.Event(), on_bytes=progress
    )


def test_a_certificate_that_cant_be_verified_stops_the_download_at_once(tmp_path: Path) -> None:
    # E.g. a work network that inspects HTTPS with an authority Windows doesn't have: retrying can't help.
    attempts: list[httpx.Request] = []

    def untrusted(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("certificate verify failed", request=request) from ssl.SSLCertVerificationError(
            1, "certificate verify failed: unable to get local issuer certificate"
        )

    with httpx.Client(transport=httpx.MockTransport(untrusted)) as client, pytest.raises(rt.DownloadError) as caught:
        download(client, "https://models.example/file", tmp_path / "model.gguf", Progress())
    assert str(caught.value) == rt.MSG_CERTIFICATE
    assert len(attempts) == 1


def test_full_download(server: RangeServer, client: httpx.Client, tmp_path: Path) -> None:
    progress = Progress()
    dest = tmp_path / "sub" / "model.gguf"
    download(client, server.url, dest, progress)
    assert dest.read_bytes() == DATA
    assert not (tmp_path / "sub" / "model.gguf.part").exists()
    assert progress.total == len(DATA) and progress.lowest >= 0
    assert server.ranges == [None]


def test_resumes_a_partial_file(server: RangeServer, client: httpx.Client, tmp_path: Path) -> None:
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(DATA[:300_000])
    progress = Progress()
    download(client, server.url, dest, progress)
    assert dest.read_bytes() == DATA
    assert server.ranges == ["bytes=300000-"]
    assert progress.total == len(DATA)


def test_server_that_ignores_range(server: RangeServer, client: httpx.Client, tmp_path: Path) -> None:
    server.honour_range = False
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(DATA[:300_000])
    progress = Progress()
    download(client, server.url, dest, progress)
    assert dest.read_bytes() == DATA
    assert progress.total == len(DATA)  # the partial bytes were counted once, then taken back


def test_dropped_connection_resumes(server: RangeServer, client: httpx.Client, tmp_path: Path) -> None:
    server.cut_after = 200_000
    dest = tmp_path / "model.gguf"
    progress = Progress()
    download(client, server.url, dest, progress)
    assert dest.read_bytes() == DATA
    assert server.ranges[0] is None
    assert server.ranges[1] == "bytes=200000-"
    assert progress.total == len(DATA)


def test_corrupt_data_is_rejected(server: RangeServer, client: httpx.Client, tmp_path: Path) -> None:
    server.data = b"X" * len(DATA)
    dest = tmp_path / "model.gguf"
    with pytest.raises(rt.DownloadError, match="damaged"):
        download(client, server.url, dest, Progress())
    assert not dest.exists()
    assert not (tmp_path / "model.gguf.part").exists()
    assert len(server.ranges) == 2  # tried once more from scratch


def test_missing_file(server: RangeServer, client: httpx.Client, tmp_path: Path) -> None:
    server.status = 404
    with pytest.raises(rt.DownloadError, match="no longer available"):
        download(client, server.url, tmp_path / "model.gguf", Progress())


def test_busy_server_is_retried_then_reported(server: RangeServer, client: httpx.Client, tmp_path: Path) -> None:
    server.status = 503
    with pytest.raises(rt.DownloadError, match="busy"):
        download(client, server.url, tmp_path / "model.gguf", Progress())
    assert len(server.ranges) == 1 + len(rt._DOWNLOAD_RETRIES)


def test_no_connection(client: httpx.Client, tmp_path: Path) -> None:
    with pytest.raises(rt.DownloadError, match="internet connection"):
        download(client, "http://127.0.0.1:9/file", tmp_path / "model.gguf", Progress())


def test_cancel_keeps_the_partial_file(server: RangeServer, client: httpx.Client, tmp_path: Path) -> None:
    cancel = threading.Event()
    seen = 0

    def progress(n: int) -> None:
        nonlocal seen
        seen += n
        if seen >= 2 * 1024 * 1024 // 4:
            cancel.set()

    server.data = DATA * 4
    with pytest.raises(rt.DownloadCancelled):
        rt.download_file(
            client,
            server.url,
            tmp_path / "model.gguf",
            size=len(server.data),
            sha256=hashlib.sha256(server.data).hexdigest(),
            cancel=cancel,
            on_bytes=progress,
        )
    assert (tmp_path / "model.gguf.part").stat().st_size > 0  # kept, so the next attempt resumes
    assert not (tmp_path / "model.gguf").exists()


def test_free_space_check(tmp_path: Path) -> None:
    rt.ensure_free_space(tmp_path, 1024)
    with pytest.raises(rt.DownloadError, match="Not enough free disk space"):
        rt.ensure_free_space(tmp_path / "not" / "yet" / "made", 10**18)


def test_only_what_the_server_needs_is_unpacked(tmp_path: Path) -> None:
    archive = tmp_path / "llama.zip"
    needed = ["llama-server.exe", "llama-server-impl.dll", "llama.dll", "ggml.dll", "ggml-cpu-haswell.dll",
              "ggml-cuda.dll", "libomp.dll", "cudart64_13.dll", "LICENSE-LLVM-OpenMP"]  # fmt: skip
    extra = ["ggml-rpc-server.exe", "ggml-rpc.dll", "llama-cli.exe", "llama-cli-impl.dll", "llama.exe",
             "tools/llama-server.exe", "readme.txt"]  # fmt: skip
    with zipfile.ZipFile(archive, "w") as zf:
        for name in needed + extra:
            zf.writestr(name, name)
    target = tmp_path / "out"
    target.mkdir()
    rt._extract_zip(archive, target, threading.Event())
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*")) == sorted(needed)


def test_a_briefly_open_file_does_not_stop_a_folder_being_deleted(tmp_path: Path) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    marker = folder / "install.json"
    marker.write_text("{}")
    reader = open(marker, "rb")  # noqa: SIM115 - someone checking what's installed, or a virus scanner
    threading.Timer(0.3, reader.close).start()
    rt._remove_tree(folder)  # Windows refuses to delete an open file until it's closed
    assert not folder.exists()


def test_zip_entries_cannot_escape(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("ok.txt", "fine")
        zf.writestr("../escape.txt", "nope")
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(rt.DownloadError):
        rt._extract_zip(archive, target, threading.Event())
    assert not (tmp_path / "escape.txt").exists()
