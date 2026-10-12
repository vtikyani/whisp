"""Logging for Whisp: a rotating log file, crash capture and a short start-up summary.

``setup_logging(logs_dir)`` runs once, first thing in ``main``. It installs:

- ``logs/whisp.log``: rotating, 2 MB × 5 files, UTF-8.
- A console handler, only when the process has a real stderr (``pythonw`` and
  windowed builds have none).
- ``sys.excepthook``, ``threading.excepthook`` and ``sys.unraisablehook``, so
  an unexpected error anywhere is logged with its traceback instead of lost.
- A Qt message handler, so Qt's own warnings land in the same log.
- ``faulthandler`` writing native crashes (access violations and the like)
  and hang dumps to ``logs/crash.log``.

Never log transcript text, selections, clipboard contents or window titles.
Log lengths and counts instead.
"""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from whisp import __version__
from whisp.storage.paths import display_path

LOG_FILE = "whisp.log"
CRASH_FILE = "crash.log"
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 5
CRASH_MAX_BYTES = 1024 * 1024
LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"

QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock")
"""Libraries that log every request at INFO; only their warnings are worth keeping."""

log = logging.getLogger(__name__)
_crash_log = logging.getLogger("whisp.crash")


@dataclass
class _Installed:
    """What ``setup_logging`` changed, so ``shutdown_logging`` can put it back."""

    handlers: list[logging.Handler]
    root_level: int
    logger_levels: dict[str, int]
    excepthook: Callable[[type[BaseException], BaseException, TracebackType | None], Any]
    threading_excepthook: Callable[[threading.ExceptHookArgs], Any]
    unraisablehook: Callable[[Any], Any]
    faulthandler_was_enabled: bool
    qt_handler: Callable[..., None] | None = None
    previous_qt_handler: Any = None
    crash_file: IO[str] | None = field(default=None)


_installed: _Installed | None = None


def setup_logging(logs_dir: Path, *, level: int = logging.INFO) -> Path:
    """Configure logging and crash capture. Calling it again replaces the previous setup. Returns the log file."""
    global _installed
    shutdown_logging()
    formatter = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = []
    log_path = logs_dir / LOG_FILE
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8", delay=True
        )
    except OSError as exc:
        _write_stderr(f"Whisp can't write its log file at {log_path}: {exc}\n")
    else:
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    if _has_console():
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        handlers.append(console)

    root = logging.getLogger()
    installed = _Installed(
        handlers=handlers,
        root_level=root.level,
        logger_levels={name: logging.getLogger(name).level for name in QUIET_LOGGERS},
        excepthook=sys.excepthook,
        threading_excepthook=threading.excepthook,
        unraisablehook=sys.unraisablehook,
        faulthandler_was_enabled=faulthandler.is_enabled(),
    )
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    sys.excepthook = _excepthook
    threading.excepthook = _threading_excepthook
    sys.unraisablehook = _unraisablehook
    _installed = installed
    _install_qt_handler(installed)
    installed.crash_file = _enable_faulthandler(logs_dir / CRASH_FILE)
    _log_startup_summary(log_path)
    return log_path


def shutdown_logging() -> None:
    """Flush and close the log files and restore the hooks ``setup_logging`` replaced."""
    global _installed
    installed = _installed
    if installed is None:
        return
    _installed = None
    if sys.excepthook is _excepthook:
        sys.excepthook = installed.excepthook
    if threading.excepthook is _threading_excepthook:
        threading.excepthook = installed.threading_excepthook
    if sys.unraisablehook is _unraisablehook:
        sys.unraisablehook = installed.unraisablehook
    if installed.qt_handler is not None:
        from PySide6.QtCore import qInstallMessageHandler

        qInstallMessageHandler(installed.previous_qt_handler)
    if installed.crash_file is not None:
        faulthandler.disable()
        with contextlib.suppress(OSError):
            installed.crash_file.close()
        if installed.faulthandler_was_enabled and sys.__stderr__ is not None:
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                faulthandler.enable(file=sys.__stderr__)
    logging.captureWarnings(False)
    root = logging.getLogger()
    for handler in installed.handlers:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(installed.root_level)
    for name, level in installed.logger_levels.items():
        logging.getLogger(name).setLevel(level)


def flush_logs() -> None:
    """Push buffered log records to disk, e.g. right before a forced exit."""
    for handler in logging.getLogger().handlers:
        with contextlib.suppress(OSError, ValueError):
            handler.flush()


def dump_all_threads(reason: str) -> None:
    """Write every thread's Python stack to ``crash.log``, to diagnose a hang."""
    installed = _installed
    if installed is None or installed.crash_file is None:
        return
    with contextlib.suppress(OSError, ValueError):
        installed.crash_file.write(f"--- {reason} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        installed.crash_file.flush()
        faulthandler.dump_traceback(file=installed.crash_file, all_threads=True)


def windows_version() -> str:
    """For example ``"Windows 11 (build 26100.2894)"``."""
    get_version = getattr(sys, "getwindowsversion", None)
    if get_version is None:
        return platform.platform()
    version = get_version()
    name = "Windows 11" if version.build >= 22000 else f"Windows {version.major}"
    revision = _update_build_revision()
    build = f"{version.build}.{revision}" if revision is not None else str(version.build)
    return f"{name} (build {build})"


# -- hooks -------------------------------------------------------------------------------


def _excepthook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    _crash_log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
    flush_logs()


def _threading_excepthook(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "an unknown thread"
    if args.exc_value is None:
        _crash_log.critical("Unhandled %s in thread %s", args.exc_type.__name__, name)
    else:
        _crash_log.critical(
            "Unhandled exception in thread %s", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )
    flush_logs()


def _unraisablehook(unraisable: Any) -> None:
    # Deliberately not logging ``unraisable.object``: its repr could contain user text.
    what = unraisable.err_msg or "Exception ignored"
    if unraisable.exc_value is None:
        _crash_log.error("%s: %s", what, unraisable.exc_type.__name__)
    else:
        _crash_log.error("%s", what, exc_info=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback))


def _install_qt_handler(installed: _Installed) -> None:
    from PySide6.QtCore import QMessageLogContext, QtMsgType, qInstallMessageHandler

    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(mode: QtMsgType, context: QMessageLogContext, message: str) -> None:
        # Qt calls this from any thread, and an exception escaping here would reach Qt's C++ code.
        try:
            level = levels.get(mode, logging.WARNING)
            category = str(context.category or "default")
            if category == "js" and level < logging.WARNING:
                return  # the web UI's console chatter can echo text from the page
            logging.getLogger("qt" if category == "default" else f"qt.{category}").log(level, "%s", message)
            if mode == QtMsgType.QtFatalMsg:
                flush_logs()
        except Exception as exc:
            _write_stderr(f"Whisp couldn't log a Qt message: {exc!r}\n")

    installed.qt_handler = handler  # keep a reference for as long as Qt may call it
    installed.previous_qt_handler = qInstallMessageHandler(handler)


def _enable_faulthandler(path: Path) -> IO[str] | None:
    try:
        if path.exists() and path.stat().st_size > CRASH_MAX_BYTES:
            os.replace(path, path.with_name(f"{path.name}.1"))
        # Stays open for the life of the process: faulthandler writes to it from a crash handler.
        crash_file = open(path, "a", encoding="utf-8")  # noqa: SIM115
    except OSError as exc:
        log.warning("Couldn't open %s (%s); native crashes won't be recorded", display_path(path), display_path(exc))
        return None
    crash_file.write(f"--- Whisp {__version__} started {time.strftime('%Y-%m-%d %H:%M:%S')} (pid {os.getpid()}) ---\n")
    crash_file.flush()
    faulthandler.enable(file=crash_file, all_threads=True)
    return crash_file


def _log_startup_summary(log_path: Path) -> None:
    from PySide6 import __version__ as pyside_version
    from PySide6.QtCore import qVersion

    packaged = " (packaged build)" if getattr(sys, "frozen", False) else ""
    log.info("Whisp %s starting, pid %d%s", __version__, os.getpid(), packaged)
    log.info(
        "Python %s %s, Qt %s, PySide6 %s",
        platform.python_version(),
        platform.architecture()[0],
        qVersion(),
        pyside_version,
    )
    log.info("%s", windows_version())
    log.info("Logging to %s", display_path(log_path))


# -- helpers -----------------------------------------------------------------------------


def _has_console() -> bool:
    return sys.stderr is not None and callable(getattr(sys.stderr, "write", None))


def _write_stderr(text: str) -> None:
    stream = sys.stderr or sys.__stderr__
    if stream is not None:
        with contextlib.suppress(OSError, ValueError, AttributeError):
            stream.write(text)


def _update_build_revision() -> int | None:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            value, _kind = winreg.QueryValueEx(key, "UBR")
        return int(value)
    except (ImportError, OSError, TypeError, ValueError):
        return None
