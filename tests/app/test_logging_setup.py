"""logging_setup: the log file, crash hooks, Qt messages and clean teardown."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import qWarning

from whisp.app import logging_setup


@pytest.fixture
def logs(qapp, tmp_path: Path) -> Iterator[Path]:
    before = (sys.excepthook, threading.excepthook, sys.unraisablehook, logging.getLogger().level)
    log_file = logging_setup.setup_logging(tmp_path / "logs")
    try:
        yield log_file
    finally:
        logging_setup.shutdown_logging()
        assert (sys.excepthook, threading.excepthook, sys.unraisablehook, logging.getLogger().level) == before


def read(log_file: Path) -> str:
    logging_setup.flush_logs()
    return log_file.read_text(encoding="utf-8")


def test_log_file_and_startup_summary(logs: Path) -> None:
    logging.getLogger("whisp.test").info("hello from the test")
    text = read(logs)
    assert "hello from the test" in text
    assert "Whisp " in text and "starting" in text
    assert "Python " in text and "Qt " in text
    assert "Windows" in text
    crash = (logs.parent / "crash.log").read_text(encoding="utf-8")
    assert crash.startswith("--- Whisp ")


def test_the_log_shows_the_users_folder_as_a_tilde(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    log_file = logging_setup.setup_logging(tmp_path / "logs")
    try:
        text = read(log_file)
    finally:
        logging_setup.shutdown_logging()
    assert "Logging to ~\\logs\\whisp.log" in text
    assert str(tmp_path).lower() not in text.lower()


def test_uncaught_exceptions_are_logged_with_tracebacks(logs: Path) -> None:
    try:
        raise ValueError("main thread trouble")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    def explode() -> None:
        raise ZeroDivisionError("worker trouble")

    worker = threading.Thread(target=explode, name="whisp-test-worker")
    worker.start()
    worker.join()
    text = read(logs)
    assert "ValueError: main thread trouble" in text
    assert "Unhandled exception in thread whisp-test-worker" in text
    assert "ZeroDivisionError: worker trouble" in text


def test_qt_messages_are_logged(logs: Path) -> None:
    qWarning("qt says hello")
    assert "qt says hello" in read(logs)


def test_noisy_libraries_are_quietened(logs: Path) -> None:
    logging.getLogger("httpx").info("GET http://127.0.0.1:11434/api/chat 200")
    assert "api/chat" not in read(logs)


def test_hang_dump_goes_to_the_crash_log(logs: Path) -> None:
    logging_setup.dump_all_threads("test dump")
    crash = (logs.parent / "crash.log").read_text(encoding="utf-8")
    assert "test dump" in crash
    assert "test_hang_dump_goes_to_the_crash_log" in crash


def test_windows_version_is_readable() -> None:
    assert logging_setup.windows_version().startswith("Windows")
