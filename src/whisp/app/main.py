"""Whisp's entry point: start-up, wiring and shutdown.

``main()`` reads the command line, makes sure only one Whisp runs, sets up
logging, builds every service in dependency order, wires them together, starts
the background workers and runs the Qt event loop. Quitting (tray → Quit
Whisp) stops everything in order within a few seconds; a watchdog makes sure
the process never hangs on the way out.

Command line:
    --background    start hidden in the tray (how Windows starts Whisp at sign-in)
    --show          open the Whisp window (also what happens with no option)
    --version       print the version and exit
    --diagnostics   print versions, GPU, speech engine and file locations, then exit

The main window is created the first time it's needed and destroyed when it
closes (its web view is the biggest memory user); Whisp keeps running in the
tray.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import importlib.metadata
import io
import json
import logging
import os
import platform
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, MutableMapping
from ctypes import wintypes
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Qt, QTimer, Slot

from whisp import APP_ID, APP_NAME, __version__

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

    from whisp.app.services import AppServices
    from whisp.app.tray import TrayIcon
    from whisp.core.types import EngineStatus
    from whisp.platform.win32.instance import SingleInstance
    from whisp.storage.paths import AppPaths
    from whisp.storage.settings import Settings
    from whisp.ui.main_window.window import MainWindow

log = logging.getLogger(__name__)

SHUTDOWN_BUDGET_S = 3.0
"""Time the orderly shutdown aims to finish in."""
DELIVERY_DEADLINE_S = 1.5
"""Longest delivery may take to finish text that's still on its way at quit."""
LLM_START_FALLBACK_MS = 60_000
"""Start the AI polish server after this long even if the speech engine never reported in."""
EXIT_WATCHDOG_S = 10.0
"""If the process is still alive this long after Quit, it's stuck: dump every thread's stack and exit."""
UI_STATE_FILE = "ui-state.json"
"""Small UI facts to remember between runs (config folder), e.g. that the tray tip was shown."""

_ATTACH_PARENT_PROCESS = 0xFFFFFFFF
_MB_OK = 0x0
_MB_ICONERROR = 0x10


def main(argv: list[str] | None = None) -> int:
    """Run Whisp. ``argv`` excludes the program name (default: ``sys.argv[1:]``). Returns the exit code."""
    args, unknown = _parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.version or args.diagnostics:
        _prepare_cli_output()
        print(f"{APP_NAME} {__version__}" if args.version else collect_diagnostics())
        return 0

    from whisp.app import logging_setup
    from whisp.storage.paths import app_paths

    try:
        paths = app_paths()
    except OSError as exc:
        _fatal_message(f"Whisp couldn't create its folders.\n\n{exc}")
        return 1
    # Logging stays on until the process ends (Python's exit handler flushes it), so a hang
    # while the interpreter shuts down can still be recorded by the exit watchdog.
    log_file = logging_setup.setup_logging(paths.logs_dir)
    if unknown:
        log.warning("Ignoring %d unknown command-line option(s)", len(unknown))
    try:
        return _run(args, paths)
    except Exception:
        log.exception("Whisp couldn't start")
        _fatal_message(f"Whisp couldn't start. The details are in the log:\n{log_file}")
        return 1


def _run(args: argparse.Namespace, paths: AppPaths) -> int:
    removed = disable_remote_debugging(os.environ)
    if removed:
        log.warning("Ignored settings that would open the window to debugging tools: %s", ", ".join(removed))
    _set_app_user_model_id()
    from whisp.ui.main_window.window import register_web_scheme

    register_web_scheme()  # must happen before the QApplication exists

    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    # Qt gets no options from the command line (``--webEngineArgs`` could switch on remote debugging).
    app = QApplication([sys.argv[0]])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setQuitOnLastWindowClosed(False)

    from whisp.platform.win32.instance import SingleInstance

    instance = SingleInstance()
    if not instance.acquire():
        if args.background:
            log.info("Whisp is already running; nothing to do")
        elif instance.send("show"):
            log.info("Whisp is already running; asked it to show its window")
        else:
            log.warning("Whisp is already running but didn't answer")
        return 0

    from whisp.app.tray import svg_icon
    from whisp.ui import theme

    app.setWindowIcon(svg_icon("logo.svg"))
    theme.ensure_font_loaded()
    if not QSystemTrayIcon.isSystemTrayAvailable():
        log.warning("The notification area isn't available; the tray icon will appear once it is")

    watchdog = _ExitWatchdog(EXIT_WATCHDOG_S)
    app.aboutToQuit.connect(watchdog.arm)  # connected first, so it's armed before shutdown starts
    whisp_app = WhispApp(app, paths, instance, background=args.background)
    try:
        whisp_app.start()
        exit_code = app.exec()
    finally:
        watchdog.arm()
        whisp_app.shutdown()
        # Let every Qt object go while the QApplication still exists.
        del whisp_app
        gc.collect()
    log.info("Whisp exited (code %d)", exit_code)
    return exit_code


def build_services(paths: AppPaths) -> AppServices:
    """Construct every long-lived service in dependency order. Nothing records or listens yet."""
    from whisp.app.controller import AppController
    from whisp.app.delivery import DeliveryService
    from whisp.app.services import AppServices
    from whisp.app.sounds import SoundPlayer
    from whisp.core.asr.manager import ModelManager
    from whisp.core.asr.service import AsrService
    from whisp.core.llm.runtime import LlmRuntime
    from whisp.core.session import SessionPipeline
    from whisp.core.text.polish import Polisher
    from whisp.platform.win32.audio_mute import MuteService
    from whisp.platform.win32.foreground import ForegroundWatcher, get_foreground_target
    from whisp.platform.win32.hotkeys import HotkeyService
    from whisp.platform.win32.insert import TextInserter
    from whisp.platform.win32.keys import display_chord
    from whisp.storage.history import HistoryStore
    from whisp.storage.settings import SettingsStore
    from whisp.ui.overlay import Overlay

    settings = SettingsStore(paths.settings_file)
    history = HistoryStore(paths.history_db, settings)
    models = ModelManager(paths.models_dir)
    asr = AsrService(settings, models)
    pipeline = SessionPipeline(settings, asr)
    llm = LlmRuntime(settings, paths.runtime_dir, paths.models_dir / "llm")
    polisher = Polisher(settings, llm)
    inserter = TextInserter()
    hotkeys = HotkeyService()
    foreground = ForegroundWatcher()
    overlay = Overlay(settings)
    sounds = SoundPlayer(settings)
    mute = MuteService()  # always created: it also unmutes apps left muted by a crash
    delivery = DeliveryService(
        settings,
        history,
        polisher,
        inserter,
        get_target=get_foreground_target,
        typing_activity=hotkeys.typing_activity,
    )
    controller = AppController(
        settings,
        models=models,
        asr=asr,
        pipeline=pipeline,
        polisher=polisher,
        inserter=inserter,
        hotkeys=hotkeys,
        overlay=overlay,
        sounds=sounds,
        delivery=delivery,
        mute=mute,
        get_target=get_foreground_target,
        display_chord=display_chord,
    )
    return AppServices(
        paths=paths,
        settings=settings,
        history=history,
        models=models,
        asr=asr,
        pipeline=pipeline,
        llm=llm,
        polisher=polisher,
        hotkeys=hotkeys,
        inserter=inserter,
        foreground=foreground,
        overlay=overlay,
        sounds=sounds,
        mute=mute,
        delivery=delivery,
        controller=controller,
    )


class WhispApp(QObject):
    """Owns every service, the tray icon and the main window for the life of the process."""

    def __init__(self, app: QApplication, paths: AppPaths, instance: SingleInstance, *, background: bool) -> None:
        super().__init__()
        from whisp.app.tray import TrayIcon
        from whisp.core.audio import list_input_devices

        self._app = app
        self._instance = instance
        self._background = background
        self._shut_down = False
        self._ctrl_c_timer: QTimer | None = None
        self._llm_started = False
        self.services = build_services(paths)
        self._tray = TrayIcon(self.services.controller, self.services.settings, list_input_devices)
        self._windows = _WindowHost(self.services, self._tray, paths.config_dir / UI_STATE_FILE, parent=self)

        s = self.services
        s.controller.notification.connect(self._tray.notify)
        s.controller.open_window_requested.connect(self._on_open_window_requested)
        s.foreground.changed.connect(s.controller.on_foreground_changed)
        s.settings.changed.connect(self._on_settings_changed)
        self._tray.open_requested.connect(self._on_open_requested)
        self._tray.quit_requested.connect(self._app.quit)
        self._instance.message.connect(self._on_instance_message)
        self._app.aboutToQuit.connect(self.shutdown)

    def start(self) -> None:
        """Start the background workers, show the tray icon and, when needed, the window."""
        s = self.services
        self._tray.show()
        s.history.start_maintenance()
        # The AI polish server starts once the speech model has loaded, so it sizes itself to the GPU
        # memory that's really left.
        s.asr.status_changed.connect(self._on_asr_status)
        QTimer.singleShot(LLM_START_FALLBACK_MS, self._start_llm)
        self._step("start the speech engine", s.asr.start)  # loads the model in the background if installed
        if self._step("start the keyboard shortcuts", s.hotkeys.start):
            s.controller.start()
        else:
            self._tray.notify(
                "Shortcuts aren't working",
                "Whisp couldn't listen for its keyboard shortcuts. Restart Whisp to try again.",
                "error",
            )
        self._step("watch the focused window", s.foreground.start)
        self._reconcile_autostart()
        if not s.history.persistent:
            self._tray.notify(
                "History isn't being saved",
                "Whisp couldn't open its history file, so this session's dictations won't be kept.",
                "warning",
            )
        if s.settings.save_problem:
            self._tray.notify("Settings aren't being saved", s.settings.save_problem, "warning")
        if not s.settings.current.general.onboarding_completed:
            self._windows.show("onboarding")
        elif not self._background:
            self._windows.show()
        self._ctrl_c_timer = _quit_on_ctrl_c(self._app)
        log.info("Whisp is ready")

    @Slot(object)
    def _on_asr_status(self, status: EngineStatus) -> None:
        from whisp.core.types import EngineState

        if status.state in (EngineState.READY, EngineState.NOT_INSTALLED, EngineState.ERROR):
            self._start_llm()

    @Slot()
    def _start_llm(self) -> None:
        if self._llm_started or self._shut_down:
            return
        self._llm_started = True
        self._step("start AI polish", self.services.llm.ensure_running)

    @Slot()
    def shutdown(self) -> None:
        """Stop everything in order, within a few seconds. Runs on ``aboutToQuit``; safe to call twice."""
        if self._shut_down:
            return
        self._shut_down = True
        log.info("Quitting")
        started = time.monotonic()
        deadline = started + SHUTDOWN_BUDGET_S
        s = self.services
        steps: tuple[tuple[str, Callable[[], object]], ...] = (
            ("main window", self._windows.close),
            ("tray icon", self._tray.shutdown),
            ("keyboard shortcuts", s.hotkeys.stop),
            ("recording", s.controller.shutdown),
            ("muting of other apps", s.mute.shutdown),  # puts other apps' sound back
            ("audio pipeline", s.pipeline.shutdown),
            ("delivery", lambda: s.delivery.shutdown(timeout=_time_left(deadline, DELIVERY_DEADLINE_S))),
            ("speech engine", s.asr.shutdown),
            ("AI polish", s.llm.shutdown),
            ("focus watcher", s.foreground.stop),
            ("overlay", s.overlay.shutdown),
            ("sounds", s.sounds.shutdown),
            ("history", s.history.close),
            ("single-instance lock", self._instance.release),
        )
        if self._ctrl_c_timer is not None:
            self._ctrl_c_timer.stop()
        for name, step in steps:
            step_started = time.monotonic()
            try:
                step()
            except Exception:
                log.exception("Error while stopping the %s", name)
            took = time.monotonic() - step_started
            if took > 0.5:
                log.warning("Stopping the %s took %.1f s", name, took)
        log.info("Shut down in %.2f s", time.monotonic() - started)

    # -- events ------------------------------------------------------------------------

    @Slot()
    def _on_open_requested(self) -> None:
        self._windows.show()

    @Slot(str, str)
    def _on_open_window_requested(self, page: str, section: str) -> None:
        self._windows.show(page, section or None)

    @Slot(str)
    def _on_instance_message(self, message: str) -> None:
        if message == "show":
            self._windows.show()
        else:
            log.warning("Ignoring an unknown message from another Whisp launch")

    @Slot(object)
    def _on_settings_changed(self, sections: set[str]) -> None:
        if "general" in sections:
            self._reconcile_autostart()

    # -- helpers ---------------------------------------------------------------------------

    def _reconcile_autostart(self) -> None:
        """Make "launch at sign-in" in Windows match the setting, unless the user turned it off in Windows."""
        from whisp.platform.win32 import autostart

        wanted = self.services.settings.current.general.launch_at_startup
        try:
            autostart.refresh_autostart_command()  # e.g. after Whisp was reinstalled somewhere else
            status = autostart.autostart_status()
            if status == "blocked":
                log.info("Launch at sign-in is turned off in Windows settings; leaving it alone")
                return
            if wanted != (status == "enabled"):
                autostart.set_autostart(wanted)
                log.info("Launch at sign-in turned %s", "on" if wanted else "off")
        except Exception:
            log.exception("Couldn't update launch at sign-in")

    @staticmethod
    def _step(what: str, action: Callable[[], object]) -> bool:
        try:
            action()
        except Exception:
            log.exception("Couldn't %s", what)
            return False
        return True


class _WindowHost(QObject):
    """Creates the main window on first use and lets it go when it closes, so Whisp idles in the tray."""

    def __init__(self, services: AppServices, tray: TrayIcon, state_file: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._services = services
        self._tray = tray
        self._state_file = state_file
        self._window: MainWindow | None = None
        self._closing = False

    def show(self, page: str | None = None, section: str | None = None) -> None:
        if self._closing:
            return
        window = self._window
        if window is None:
            from whisp.ui.main_window.window import MainWindow

            window = MainWindow(self._services)
            # Closing destroys the window and its web view; the next show builds a fresh one.
            window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            window.destroyed.connect(self._on_window_destroyed)
            self._window = window
            log.info("Main window opened")
        if page and section:
            window.show_page(page, section)
        elif page:
            window.show_page(page)
        else:
            if window.isMinimized():
                window.showNormal()
            else:
                window.show()
            window.raise_()
            window.activateWindow()

    def close(self) -> None:
        self._closing = True
        if self._window is not None:
            self._window.close()

    @Slot()
    def _on_window_destroyed(self) -> None:
        self._window = None
        if self._closing:
            return
        log.info("Main window closed; Whisp keeps running in the tray")
        self._explain_tray_once()

    def _explain_tray_once(self) -> None:
        state = _read_json(self._state_file)
        if state.get("tray_tip_shown"):
            return
        state["tray_tip_shown"] = True
        _write_json(self._state_file, state)
        if not self._services.settings.current.general.show_notifications:
            return
        keys = " + ".join(self._services.controller.hotkey_keys())
        if keys:
            message = f"It's in the notification area. Hold {keys} to dictate anytime."
        else:
            message = "It's in the notification area, ready when you need it."
        self._tray.notify("Whisp is still running", message, "info")


# -- diagnostics -------------------------------------------------------------------------


def collect_diagnostics() -> str:
    """Versions, GPU, speech engine and file locations as plain text, for bug reports."""
    lines = [f"{APP_NAME} diagnostics", ""]
    sections: tuple[tuple[str, Callable[[], Iterator[str]]], ...] = (
        ("Versions", _diagnose_versions),
        ("GPU", _diagnose_gpu),
        ("Speech engine", _diagnose_engine),
        ("Files", _diagnose_paths),
    )
    for title, collect in sections:
        lines.append(title)
        try:
            lines.extend(f"  {line}" for line in collect())
        except Exception as exc:
            lines.append(f"  Unavailable: {exc}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _diagnose_versions() -> Iterator[str]:
    from PySide6 import __version__ as pyside_version
    from PySide6.QtCore import qVersion

    from whisp.app.logging_setup import windows_version

    packaged = " (packaged)" if getattr(sys, "frozen", False) else ""
    yield f"Whisp {__version__}{packaged}"
    yield f"Python {platform.python_version()} {platform.architecture()[0]} ({sys.executable})"
    yield f"Qt {qVersion()}, PySide6 {pyside_version}"
    yield windows_version()
    for name, distributions in (
        ("ONNX Runtime", ("onnxruntime-gpu", "onnxruntime")),
        ("onnx-asr", ("onnx-asr",)),
        ("NumPy", ("numpy",)),
        ("sounddevice", ("sounddevice",)),
        ("HTTPX", ("httpx",)),
        ("Pydantic", ("pydantic",)),
    ):
        yield f"{name} {_distribution_version(distributions)}"
    try:
        from whisp.core.onnx_runtime import ort

        yield f"ONNX Runtime providers: {', '.join(ort.get_available_providers())}"
    except Exception as exc:
        yield f"ONNX Runtime providers: unavailable ({exc})"


def _diagnose_gpu() -> Iterator[str]:
    from whisp.core.asr.gpu import detect_gpu

    gpu = detect_gpu()
    if not gpu.has_nvidia:
        yield "No NVIDIA GPU found; speech recognition runs on the CPU"
    else:
        memory = f", {gpu.vram_mb:,} MB" if gpu.vram_mb else ""
        yield f"{gpu.name or 'NVIDIA GPU'}{memory}"
        yield f"CUDA ready: {'yes' if gpu.cuda_ready else 'no'}"
    if gpu.detail:
        yield gpu.detail


def _diagnose_engine() -> Iterator[str]:
    from whisp.core.asr.catalog import get_model
    from whisp.core.asr.manager import ModelManager
    from whisp.storage.paths import app_paths

    paths = app_paths()
    settings = _read_settings(paths.settings_file)
    transcription = settings.transcription
    yield f"Model: {transcription.model_id}"
    yield f"Device: {transcription.device}, language: {transcription.language}"
    model = get_model(transcription.model_id)
    manager = ModelManager(paths.models_dir)
    installed = [v.key for v in model.variants if manager.is_installed(transcription.model_id, v.key)]
    yield f"Downloaded: {', '.join(installed) if installed else 'not yet'}"
    polish = settings.polish
    yield f"AI polish: {'on' if polish.enabled else 'off'} ({polish.provider})"


def _diagnose_paths() -> Iterator[str]:
    from whisp.storage.paths import app_paths

    paths = app_paths()
    yield f"Settings: {paths.settings_file}"
    yield f"Data: {paths.data_dir}"
    yield f"Models: {paths.models_dir}"
    yield f"Logs: {paths.logs_dir}"
    yield f"History: {paths.history_db}"


def _distribution_version(names: tuple[str, ...]) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not installed"


def _read_settings(path: Path) -> Settings:
    """The saved settings without touching the file (no defaults written, no backups made)."""
    from whisp.storage.paths import display_path
    from whisp.storage.settings import Settings, parse_settings

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError):
        log.warning("Couldn't read %s; showing defaults", display_path(path))
        return Settings()
    return parse_settings(raw)[0]


# -- process helpers ---------------------------------------------------------------------


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="whisp", description="Fast, private, local voice dictation for Windows.")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--background", action="store_true", help="start hidden in the notification area")
    visibility.add_argument("--show", action="store_true", help="open the Whisp window (the default)")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument(
        "--diagnostics", action="store_true", help="print versions, GPU, speech engine and file locations, then exit"
    )
    # Anything else is returned so it can be logged, and otherwise ignored: Qt gets no command-line options.
    return parser.parse_known_args(argv)


_REMOTE_DEBUGGING_VARIABLE = "QTWEBENGINE_REMOTE_DEBUGGING"
_CHROMIUM_FLAGS_VARIABLE = "QTWEBENGINE_CHROMIUM_FLAGS"


def disable_remote_debugging(environ: MutableMapping[str, str]) -> list[str]:
    """Remove every outside way to switch on the web engine's remote debugging. Returns what was removed.

    Developer tools attached to the main window would have full access to its bridge to Python, so the
    environment mustn't be able to turn them on: ``QTWEBENGINE_REMOTE_DEBUGGING`` is removed, and so are
    remote-debugging switches in ``QTWEBENGINE_CHROMIUM_FLAGS`` (other switches are kept). Must run before the
    QApplication is created. The result names what was removed, never the values.
    """
    removed: list[str] = []
    if environ.pop(_REMOTE_DEBUGGING_VARIABLE, None) is not None:
        removed.append(_REMOTE_DEBUGGING_VARIABLE)
    flags = environ.get(_CHROMIUM_FLAGS_VARIABLE)
    if flags is not None:
        given = flags.split()  # the web engine splits this variable on spaces too
        kept = [flag for flag in given if not _is_remote_debugging_switch(flag)]
        if len(kept) != len(given):
            removed.append(f"remote-debugging switches in {_CHROMIUM_FLAGS_VARIABLE}")
            if kept:
                environ[_CHROMIUM_FLAGS_VARIABLE] = " ".join(kept)
            else:
                del environ[_CHROMIUM_FLAGS_VARIABLE]
    return removed


def _is_remote_debugging_switch(flag: str) -> bool:
    # Chromium accepts switches written with "--", "-" or, on Windows, "/".
    name = flag.lstrip("-/").split("=", 1)[0].lower()
    return name.startswith("remote-debugging")


def _set_app_user_model_id() -> None:
    """Group Whisp's windows and notifications under its own name in Windows, not python.exe's."""
    try:
        set_id = ctypes.WinDLL("shell32").SetCurrentProcessExplicitAppUserModelID
        set_id.argtypes = [wintypes.LPCWSTR]
        set_id.restype = ctypes.c_long
        result = set_id(APP_ID)
    except (OSError, AttributeError):
        log.exception("Couldn't set Whisp's app ID")
        return
    if result != 0:
        log.warning("SetCurrentProcessExplicitAppUserModelID failed (0x%08X)", result & 0xFFFFFFFF)


def _prepare_cli_output() -> None:
    """Let --version and --diagnostics print: to the terminal windowed Whisp was started from, or to a pipe or file.

    A pipe or file gets UTF-8. Python would otherwise use the ANSI code page there, which garbles "—" and
    graphics card names in other scripts for any reader that expects UTF-8 (PowerShell 7, editors, bug reports).
    """
    stdout = sys.stdout
    if stdout is not None:
        if isinstance(stdout, io.TextIOWrapper) and not stdout.isatty():
            stdout.reconfigure(encoding="utf-8", errors="replace")
        return
    try:
        attach = ctypes.WinDLL("kernel32", use_last_error=True).AttachConsole
        attach.argtypes = [wintypes.DWORD]
        attach.restype = wintypes.BOOL
        if not attach(_ATTACH_PARENT_PROCESS):
            return
        # Kept open until the process exits a moment later.
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    except OSError:
        return


def _fatal_message(text: str) -> None:
    """Tell the user Whisp couldn't start. Works without Qt and without a console."""
    try:
        box = ctypes.WinDLL("user32").MessageBoxW
        box.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
        box.restype = ctypes.c_int
        box(None, text, APP_NAME, _MB_OK | _MB_ICONERROR)
    except (OSError, AttributeError):
        stream = sys.stderr or sys.__stderr__
        if stream is not None:
            with contextlib.suppress(OSError, ValueError):
                stream.write(text + "\n")


def _quit_on_ctrl_c(app: QApplication) -> QTimer | None:
    """When run from a terminal, let Ctrl+C quit cleanly instead of being swallowed by Qt's event loop."""
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        return None
    signal.signal(signal.SIGINT, lambda _signum, _frame: app.quit())
    # Python runs signal handlers only between bytecodes; a periodic no-op gives it the chance.
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(250)
    return timer


class _ExitWatchdog:
    """If quitting gets stuck (a thread that won't stop, a hung driver), record every stack and leave anyway.

    Armed once the user quits. The process normally exits long before it fires;
    its thread is a daemon, so it never keeps the process alive itself.
    """

    def __init__(self, limit_s: float) -> None:
        self._limit_s = limit_s
        self._armed = False

    def arm(self) -> None:
        if self._armed:
            return
        self._armed = True
        threading.Thread(target=self._watch, name="whisp-exit-watchdog", daemon=True).start()

    def _watch(self) -> None:
        time.sleep(self._limit_s)
        from whisp.app import logging_setup

        log.critical("Whisp was still running %.0f s after Quit; closing it forcibly", self._limit_s)
        logging_setup.dump_all_threads("Quit took too long")
        logging_setup.flush_logs()
        os._exit(1)


def _time_left(deadline: float, cap: float) -> float:
    return max(0.2, min(cap, deadline - time.monotonic()))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        log.warning("Couldn't read %s; starting it afresh", path.name)
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        log.exception("Couldn't save %s", path.name)
