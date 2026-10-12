"""Resumable, verified downloads against a local HTTP server: resume, retries, checksums, cancel, disk space."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import socket
import ssl
import threading
from collections import namedtuple
from pathlib import Path

import httpx
import pytest

from whisp.core.asr import downloads
from whisp.core.asr.downloads import (
    MSG_CERTIFICATE,
    MSG_CHECKSUM,
    MSG_DISK_FILLED,
    MSG_DROPPED,
    MSG_OFFLINE,
    DownloadCancelled,
    DownloadError,
    DownloadItem,
    download_files,
    format_size,
    hf_resolve_url,
    part_path,
    remaining_bytes,
)


def payload(size: int, seed: int = 0) -> bytes:
    """Deterministic, non-repeating bytes."""
    out = bytearray()
    counter = 0
    while len(out) < size:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return bytes(out[:size])


def item_for(server, tmp_path: Path, name: str, data: bytes) -> DownloadItem:
    url, size, sha = server.add(f"/files/{name}", data)
    return DownloadItem(url=url, dest=tmp_path / name, size=size, sha256=sha)


def test_downloads_and_verifies_files_with_progress(file_server, tmp_path: Path) -> None:
    items = [
        item_for(file_server, tmp_path, "a.bin", payload(300_000, 1)),
        item_for(file_server, tmp_path, "b.bin", payload(5, 2)),
    ]
    calls: list[tuple[int, int]] = []

    download_files(items, progress=lambda done, total: calls.append((done, total)))

    for item in items:
        assert item.dest.read_bytes() == file_server.files[f"/files/{item.dest.name}"]
        assert not part_path(item.dest).exists()
    assert calls[-1] == (300_005, 300_005)
    assert all(total == 300_005 for _, total in calls)
    assert [done for done, _ in calls] == sorted(done for done, _ in calls)


def test_a_certificate_that_cant_be_verified_stops_the_download_at_once(tmp_path: Path) -> None:
    # E.g. a work network that inspects HTTPS with an authority Windows doesn't have: retrying can't help.
    attempts: list[httpx.Request] = []

    def untrusted(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("certificate verify failed", request=request) from ssl.SSLCertVerificationError(
            1, "certificate verify failed: unable to get local issuer certificate"
        )

    item = DownloadItem(url="https://models.example/a.bin", dest=tmp_path / "a.bin", size=10, sha256="0" * 64)
    with httpx.Client(transport=httpx.MockTransport(untrusted)) as client, pytest.raises(DownloadError) as caught:
        download_files([item], client=client, check_space=False)
    assert str(caught.value) == MSG_CERTIFICATE
    assert len(attempts) == 1


def test_resumes_a_partial_download_with_a_range_request(file_server, tmp_path: Path) -> None:
    data = payload(200_000, 3)
    item = item_for(file_server, tmp_path, "model.onnx", data)
    part_path(item.dest).write_bytes(data[:70_000])

    download_files([item])

    assert item.dest.read_bytes() == data
    assert file_server.gets("/files/model.onnx") == ["bytes=70000-"]


def test_restarts_when_the_server_ignores_the_range(file_server, tmp_path: Path) -> None:
    data = payload(100_000, 4)
    item = item_for(file_server, tmp_path, "x.bin", data)
    part_path(item.dest).write_bytes(data[:40_000])
    file_server.ignore_range.add("/files/x.bin")

    download_files([item])

    assert item.dest.read_bytes() == data


def test_a_damaged_partial_file_is_caught_by_the_checksum_and_fetched_again(file_server, tmp_path: Path) -> None:
    data = payload(50_000, 5)
    item = item_for(file_server, tmp_path, "x.bin", data)
    part_path(item.dest).write_bytes(b"\0" * 20_000)  # right length so far, wrong content

    download_files([item])

    assert item.dest.read_bytes() == data
    assert file_server.gets("/files/x.bin") == ["bytes=20000-", None]


def test_checksum_mismatch_fails_after_one_retry_and_leaves_nothing(file_server, tmp_path: Path) -> None:
    item = item_for(file_server, tmp_path, "x.bin", payload(30_000, 6))
    file_server.corrupt.add("/files/x.bin")

    with pytest.raises(DownloadError) as caught:
        download_files([item])

    assert str(caught.value) == MSG_CHECKSUM
    assert not item.dest.exists()
    assert not part_path(item.dest).exists()
    assert len(file_server.gets("/files/x.bin")) == 2


def test_transient_server_errors_are_retried(file_server, tmp_path: Path, fast_retries) -> None:
    item = item_for(file_server, tmp_path, "x.bin", payload(10_000, 7))
    file_server.fail["/files/x.bin"] = [503, 502, 429]

    download_files([item])

    assert item.dest.exists()
    assert len(file_server.gets("/files/x.bin")) == 4


def test_gives_up_after_repeated_failures_with_a_plain_message(file_server, tmp_path: Path, fast_retries) -> None:
    item = item_for(file_server, tmp_path, "x.bin", payload(10_000, 8))
    file_server.fail["/files/x.bin"] = [500] * 10

    with pytest.raises(DownloadError) as caught:
        download_files([item])

    assert "error 500" in str(caught.value)
    assert len(file_server.gets("/files/x.bin")) == downloads.MAX_ATTEMPTS


def test_a_dropped_connection_resumes_where_it_stopped(file_server, tmp_path: Path, fast_retries) -> None:
    data = payload(400_000, 9)
    item = item_for(file_server, tmp_path, "x.bin", data)
    file_server.drop_after["/files/x.bin"] = 150_000

    download_files([item])

    assert item.dest.read_bytes() == data
    ranges = file_server.gets("/files/x.bin")
    assert ranges[0] is None
    assert ranges[1] is not None and ranges[1].startswith("bytes=")
    assert int(ranges[1][6:-1]) > 0


def test_missing_files_fail_at_once(file_server, tmp_path: Path, fast_retries) -> None:
    item = DownloadItem(url=file_server.url("/nope.bin"), dest=tmp_path / "nope.bin", size=10, sha256="0" * 64)

    with pytest.raises(DownloadError) as caught:
        download_files([item])

    assert "error 404" in str(caught.value)
    assert len(file_server.gets("/nope.bin")) == 1


def test_offline_gives_a_connection_message(tmp_path: Path, fast_retries, monkeypatch) -> None:
    monkeypatch.setattr(downloads, "MAX_ATTEMPTS", 2)  # Windows spends ~2 s on each refused connection
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # nothing listens here once the socket closes
    item = DownloadItem(url=f"http://127.0.0.1:{port}/x.bin", dest=tmp_path / "x.bin", size=10, sha256="0" * 64)

    with pytest.raises(DownloadError) as caught:
        download_files([item])

    assert str(caught.value) == MSG_OFFLINE


def test_cancel_stops_the_download_and_keeps_the_partial_file(file_server, tmp_path: Path) -> None:
    item = item_for(file_server, tmp_path, "x.bin", payload(2_000_000, 10))
    file_server.chunk_delay = 0.002
    cancel = threading.Event()

    def progress(done: int, total: int) -> None:
        if done > 100_000:
            cancel.set()

    with pytest.raises(DownloadCancelled):
        download_files([item], progress=progress, cancel=cancel)

    assert not item.dest.exists()
    assert 0 < part_path(item.dest).stat().st_size < item.size
    assert remaining_bytes([item]) < item.size


def test_an_already_complete_file_is_not_downloaded_again(file_server, tmp_path: Path) -> None:
    data = payload(20_000, 11)
    item = item_for(file_server, tmp_path, "x.bin", data)
    item.dest.write_bytes(data)

    download_files([item])

    assert file_server.gets("/files/x.bin") == []


def test_a_wrong_file_in_the_way_is_replaced(file_server, tmp_path: Path) -> None:
    data = payload(20_000, 12)
    item = item_for(file_server, tmp_path, "x.bin", data)
    item.dest.write_bytes(payload(20_000, 99))

    download_files([item])

    assert item.dest.read_bytes() == data


def test_not_enough_disk_space_fails_before_downloading(file_server, tmp_path: Path, monkeypatch) -> None:
    item = item_for(file_server, tmp_path, "x.bin", payload(10_000, 13))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(downloads.shutil, "disk_usage", lambda _path: usage(10**9, 10**9, 1000))

    with pytest.raises(DownloadError) as caught:
        download_files([item])

    assert "There isn't enough free disk space" in str(caught.value)
    assert "512 MB" in str(caught.value)  # the safety margin is part of what must be freed
    assert file_server.gets("/files/x.bin") == []


def test_free_space_check_counts_only_what_is_left(file_server, tmp_path: Path, monkeypatch) -> None:
    data = payload(10_000, 14)
    item = item_for(file_server, tmp_path, "x.bin", data)
    part_path(item.dest).write_bytes(data[:4_000])
    seen: list[int] = []
    real = shutil.disk_usage

    def fake_usage(path):
        seen.append(remaining_bytes([item]))
        return real(path)

    monkeypatch.setattr(downloads.shutil, "disk_usage", fake_usage)
    download_files([item])

    assert seen == [6_000]


def test_disk_full_while_writing_gives_a_plain_message(file_server, tmp_path: Path, monkeypatch) -> None:
    item = item_for(file_server, tmp_path, "x.bin", payload(100_000, 15))

    class FullDisk:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def write(self, data: bytes) -> int:
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self) -> None:
            pass

    monkeypatch.setattr(downloads, "open", FullDisk, raising=False)

    with pytest.raises(DownloadError) as caught:
        download_files([item])

    assert str(caught.value) == MSG_DISK_FILLED


def test_progress_callbacks_are_throttled(file_server, tmp_path: Path) -> None:
    item = item_for(file_server, tmp_path, "x.bin", payload(3_000_000, 16))
    file_server.chunk_delay = 0.001
    calls: list[int] = []

    download_files([item], progress=lambda done, total: calls.append(done))

    # ~180 chunks of 16 KB arrive over a fraction of a second: a few updates, always including the final one.
    assert len(calls) < 30
    assert calls[-1] == item.size


def test_a_server_sending_too_much_data_is_caught(file_server, tmp_path: Path) -> None:
    data = payload(10_000, 17)
    url, _, _ = file_server.add("/files/x.bin", data + b"extra bytes that should not be here")
    item = DownloadItem(url=url, dest=tmp_path / "x.bin", size=len(data), sha256=hashlib.sha256(data).hexdigest())

    with pytest.raises(DownloadError) as caught:
        download_files([item])

    assert str(caught.value) == MSG_CHECKSUM
    assert not item.dest.exists()


def test_connections_that_keep_dropping_end_with_a_plain_message(file_server, tmp_path: Path, fast_retries) -> None:
    item = item_for(file_server, tmp_path, "x.bin", payload(50_000, 18))
    file_server.always_drop.add("/files/x.bin")

    with pytest.raises(DownloadError) as caught:
        download_files([item])

    assert str(caught.value) == MSG_DROPPED
    assert len(file_server.gets("/files/x.bin")) == downloads.MAX_ATTEMPTS
    assert not item.dest.exists()


def test_hf_resolve_url_pins_the_revision_and_quotes_paths() -> None:
    url = hf_resolve_url("org/repo", "abc123", "onnx/encoder model.onnx")
    assert url == "https://huggingface.co/org/repo/resolve/abc123/onnx/encoder%20model.onnx"
    assert hf_resolve_url("o/r", "1", "f", base_url="http://127.0.0.1:9/") == "http://127.0.0.1:9/o/r/resolve/1/f"


@pytest.mark.parametrize(
    ("size", "text"),
    [(0, "0 bytes"), (1023, "1023 bytes"), (1536, "1.5 KB"), (5 * 2**20, "5.0 MB"), (1_166_225_776, "1.1 GB")],
)
def test_format_size(size: int, text: str) -> None:
    assert format_size(size) == text


def test_cancel_before_starting(tmp_path: Path) -> None:
    cancel = threading.Event()
    cancel.set()
    item = DownloadItem(url="http://127.0.0.1:9/x", dest=tmp_path / "x", size=1, sha256="0" * 64)
    with pytest.raises(DownloadCancelled):
        download_files([item], cancel=cancel)
    assert not os.listdir(tmp_path)
