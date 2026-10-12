"""app.main: building services, wiring, start-up, the window's life cycle, autostart and shutdown order.

Every module owned by another part of the app is replaced with a small fake for
the duration of each test (through ``sys.modules``), so these tests check only
the wiring in ``main.py``.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QWidget

from whisp.app import main as main_module
from whisp.app import tray as tray_module
from whisp.core.types import EngineState, EngineStatus
from whisp.storage.paths import app_paths

CALLS: list[tuple] = []


def record(*entry: object) -> None:
    CALLS.append(entry)


def names() -> list[str]:
    return [str(c[0]) for c in CALLS]


# -- fakes for the other modules ---------------------------------------------------------


class FakeModelManager(QObject):
    progress = Signal(str, object, object)
    changed = Signal()
    download_failed = Signal(str, str)

    def __init__(self, models_dir: Path) -> None:
        super().__init__()
        record("ModelManager", models_dir)

    def is_installed(self, model_id: str, variant: str) -> bool:
        return True


class FakeAsrService(QObject):
    status_changed = Signal(object)

    def __init__(self, settings: object, models: object) -> None:
        super().__init__()
        record("AsrService")
        self.status = EngineStatus(EngineState.READY)

    def start(self) -> None:
        record("asr.start")

    def ensure_loaded(self) -> None:
        record("asr.ensure_loaded")

    def shutdown(self) -> None:
        record("asr.shutdown")


class FakePipeline(QObject):
    level = Signal(float)
    preview = Signal(str, str)
    finished = Signal(object)
    no_speech = Signal(str)
    failed = Signal(str, str)

    def __init__(self, settings: object, asr: object) -> None:
        super().__init__()
        record("SessionPipeline")

    def start(self, session_id: str) -> None:
        record("pipeline.start")

    def stop(self, session_id: str) -> None:
        record("pipeline.stop")

    def cancel(self, session_id: str) -> None:
        record("pipeline.cancel")

    def shutdown(self) -> None:
        record("pipeline.shutdown")


class FakeLlmRuntime(QObject):
    status_changed = Signal(object)
    progress = Signal(str, object, object)

    def __init__(self, settings: object, runtime_dir: Path, models_dir: Path) -> None:
        super().__init__()
        record("LlmRuntime", runtime_dir, models_dir)

    def ensure_running(self) -> None:
        record("llm.ensure_running")

    def shutdown(self) -> None:
        record("llm.shutdown")


class FakePolisher:
    def __init__(self, settings: object, runtime: object) -> None:
        record("Polisher")

    def is_ready(self) -> bool:
        return False


class FakeInserter:
    def __init__(self) -> None:
        record("TextInserter")

    def default_method_for(self, process_name: str) -> str | None:
        return None


class FakeHotkeys(QObject):
    pressed = Signal(str)
    released = Signal(str)
    interrupted = Signal(str)
    recorded = Signal(str)
    binding_error = Signal(str, str)
    session_locked = Signal()
    fail_start = False

    def __init__(self) -> None:
        super().__init__()
        record("HotkeyService")

    def start(self) -> None:
        record("hotkeys.start")
        if FakeHotkeys.fail_start:
            raise OSError("no message loop")

    def stop(self) -> None:
        record("hotkeys.stop")

    def set_bindings(self, bindings: dict[str, str]) -> dict[str, str]:
        record("hotkeys.set_bindings")
        return {}

    def set_cancel_active(self, active: bool) -> None:
        record("hotkeys.set_cancel_active", active)

    def set_paused(self, paused: bool) -> None:
        record("hotkeys.set_paused", paused)

    def typing_activity(self, hwnd: int) -> None:
        return None  # no keyboard information, like a service without its hook installed


class FakeForeground(QObject):
    changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        record("ForegroundWatcher")

    def start(self) -> None:
        record("foreground.start")

    def stop(self) -> None:
        record("foreground.stop")


class FakeOverlay(QObject):
    stop_clicked = Signal()
    cancel_clicked = Signal()
    idle_clicked = Signal()

    def __init__(self, settings: object) -> None:
        super().__init__()
        record("Overlay")

    def set_hotkey_hint(self, keys: list[str]) -> None:
        record("overlay.set_hotkey_hint", keys)

    def set_target(self, target: object) -> None:
        record("overlay.set_target")

    def shutdown(self) -> None:
        record("overlay.shutdown")


class FakeSounds:
    def __init__(self, settings: object) -> None:
        record("SoundPlayer")

    def play(self, name: str) -> None:
        record("sounds.play", name)

    def shutdown(self) -> None:
        record("sounds.shutdown")


class FakeSpacing:
    def __init__(self, *, activity: object = None) -> None:
        self.activity = activity

    def remember(self, target: object, text: str) -> None:
        return None

    def prefix_for(self, target: object, text: str, mode: str) -> str:
        return ""


class FakeTray(QObject):
    open_requested = Signal()
    quit_requested = Signal()
    instances: ClassVar[list[FakeTray]] = []

    def __init__(self, controller: object, settings: object, list_devices: object) -> None:
        super().__init__()
        record("TrayIcon")
        self.notes: list[tuple[str, str, str]] = []
        FakeTray.instances.append(self)

    def show(self) -> None:
        record("tray.show")

    def notify(self, title: str, message: str, kind: str = "info") -> None:
        self.notes.append((title, message, kind))

    def shutdown(self) -> None:
        record("tray.shutdown")


class FakeMainWindow(QWidget):
    instances: ClassVar[list[FakeMainWindow]] = []

    def __init__(self, services: object) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        self.pages: list[tuple[str, str | None]] = []
        FakeMainWindow.instances.append(self)
        record("MainWindow")

    def show_page(self, page: str, section: str | None = None) -> None:
        self.pages.append((page, section))
        self.show()


class FakeInstance(QObject):
    message = Signal(str)

    def acquire(self) -> bool:
        return True

    def send(self, message: str) -> bool:
        return True

    def release(self) -> None:
        record("instance.release")


@dataclass
class Autostart:
    status: str = "disabled"

    def autostart_status(self) -> str:
        return self.status

    def set_autostart(self, enabled: bool) -> None:
        record("set_autostart", enabled)
        self.status = "enabled" if enabled else "disabled"

    def refresh_autostart_command(self) -> bool:
        record("refresh_autostart_command")
        return False


def fake_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def autostart(monkeypatch: pytest.MonkeyPatch, qapp) -> Iterator[Autostart]:
    """Install a fake for every module ``main`` wires together; yields the fake autostart state."""
    CALLS.clear()
    FakeMainWindow.instances.clear()
    FakeTray.instances.clear()
    FakeHotkeys.fail_start = False
    state = Autostart()
    modules = {
        "whisp.app.sounds": fake_module("whisp.app.sounds", SoundPlayer=FakeSounds),
        "whisp.core.asr.manager": fake_module("whisp.core.asr.manager", ModelManager=FakeModelManager),
        "whisp.core.asr.service": fake_module("whisp.core.asr.service", AsrService=FakeAsrService),
        "whisp.core.llm.runtime": fake_module("whisp.core.llm.runtime", LlmRuntime=FakeLlmRuntime),
        "whisp.core.session": fake_module("whisp.core.session", SessionPipeline=FakePipeline),
        "whisp.core.audio": fake_module("whisp.core.audio", list_input_devices=lambda: []),
        "whisp.core.text.polish": fake_module("whisp.core.text.polish", Polisher=FakePolisher),
        "whisp.core.text.context": fake_module(
            "whisp.core.text.context", build_context=lambda *a: None, categorize=lambda *a: None
        ),
        "whisp.core.text.rules": fake_module("whisp.core.text.rules", apply_rules=lambda raw, ctx: raw),
        "whisp.core.text.spacing": fake_module("whisp.core.text.spacing", SpacingMemory=FakeSpacing),
        "whisp.platform.win32.foreground": fake_module(
            "whisp.platform.win32.foreground", ForegroundWatcher=FakeForeground, get_foreground_target=lambda: None
        ),
        "whisp.platform.win32.hotkeys": fake_module("whisp.platform.win32.hotkeys", HotkeyService=FakeHotkeys),
        "whisp.platform.win32.insert": fake_module("whisp.platform.win32.insert", TextInserter=FakeInserter),
        "whisp.platform.win32.keys": fake_module(
            "whisp.platform.win32.keys", display_chord=lambda chord: [p.capitalize() for p in chord.split("+")]
        ),
        "whisp.platform.win32.autostart": fake_module(
            "whisp.platform.win32.autostart",
            autostart_status=state.autostart_status,
            set_autostart=state.set_autostart,
            refresh_autostart_command=state.refresh_autostart_command,
        ),
        "whisp.ui.main_window.window": fake_module("whisp.ui.main_window.window", MainWindow=FakeMainWindow),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent_name, _, child = name.rpartition(".")
        monkeypatch.setattr(importlib.import_module(parent_name), child, module, raising=False)
    monkeypatch.setattr(importlib.import_module("whisp.ui.overlay"), "Overlay", FakeOverlay, raising=False)
    monkeypatch.setattr(tray_module, "TrayIcon", FakeTray)
    monkeypatch.setattr(main_module, "_quit_on_ctrl_c", lambda app: None)
    yield state


def make_app(
    qapp, *, onboarded: bool, background: bool = True, instance: FakeInstance | None = None
) -> main_module.WhispApp:
    whisp_app = main_module.WhispApp(qapp, app_paths(), instance or FakeInstance(), background=background)
    if onboarded:
        whisp_app.services.settings.update({"general": {"onboarding_completed": True}})
    return whisp_app


def the_tray() -> FakeTray:
    [tray] = FakeTray.instances
    return tray


def deliver_deferred_deletes() -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


# -- tests -------------------------------------------------------------------------------


def test_services_are_built_in_dependency_order(autostart, qapp) -> None:
    whisp_app = make_app(qapp, onboarded=True)
    try:
        built = [n for n in names() if n[0].isupper()]
        assert built == [
            "ModelManager",
            "AsrService",
            "SessionPipeline",
            "LlmRuntime",
            "Polisher",
            "TextInserter",
            "HotkeyService",
            "ForegroundWatcher",
            "Overlay",
            "SoundPlayer",
            "TrayIcon",
        ]
        paths = app_paths()
        assert ("ModelManager", paths.models_dir) in CALLS
        assert ("LlmRuntime", paths.runtime_dir, paths.models_dir / "llm") in CALLS
        services = whisp_app.services
        assert services.paths == paths
        assert services.controller is not None and services.delivery is not None
    finally:
        whisp_app.shutdown()


def test_start_runs_workers_then_opens_onboarding(autostart, qapp) -> None:
    whisp_app = make_app(qapp, onboarded=False)
    expected = [
        "tray.show",
        "asr.start",
        "hotkeys.start",
        "hotkeys.set_bindings",
        "foreground.start",
        "set_autostart",
        "MainWindow",
    ]
    try:
        CALLS.clear()
        whisp_app.start()
        assert [n for n in names() if n in expected] == expected
        assert ("set_autostart", True) in CALLS  # launch at sign-in is on by default
        [window] = FakeMainWindow.instances
        assert window.pages == [("onboarding", None)]
    finally:
        whisp_app.shutdown()


def test_ai_polish_starts_once_the_speech_model_is_ready(autostart, qapp) -> None:
    from whisp.core.types import EngineState, EngineStatus

    whisp_app = make_app(qapp, onboarded=True)
    try:
        whisp_app.start()
        assert "llm.ensure_running" not in names()  # waits so it sees the GPU memory really left
        whisp_app.services.asr.status_changed.emit(EngineStatus(state=EngineState.LOADING))
        assert "llm.ensure_running" not in names()
        whisp_app.services.asr.status_changed.emit(EngineStatus(state=EngineState.READY))
        whisp_app.services.asr.status_changed.emit(EngineStatus(state=EngineState.READY))
        assert names().count("llm.ensure_running") == 1
    finally:
        whisp_app.shutdown()


def test_background_start_stays_in_the_tray(autostart, qapp) -> None:
    whisp_app = make_app(qapp, onboarded=True, background=True)
    try:
        whisp_app.start()
        assert FakeMainWindow.instances == []
    finally:
        whisp_app.shutdown()


def test_normal_start_opens_the_window(autostart, qapp) -> None:
    whisp_app = make_app(qapp, onboarded=True, background=False)
    try:
        whisp_app.start()
        [window] = FakeMainWindow.instances
        assert window.isVisible()
        assert window.pages == []
    finally:
        whisp_app.shutdown()


def test_shutdown_stops_everything_in_order(autostart, qapp) -> None:
    whisp_app = make_app(qapp, onboarded=True)
    whisp_app.start()
    services = whisp_app.services
    CALLS.clear()
    whisp_app.shutdown()
    whisp_app.shutdown()  # a second call does nothing
    assert names() == [
        "tray.shutdown",
        "hotkeys.stop",
        "pipeline.shutdown",
        "asr.shutdown",
        "llm.shutdown",
        "foreground.stop",
        "overlay.shutdown",
        "sounds.shutdown",
        "instance.release",
    ]
    with pytest.raises(RuntimeError):
        services.history.last_text()  # history is closed


def test_window_is_created_on_demand_and_released_on_close(autostart, qapp) -> None:
    whisp_app = make_app(qapp, onboarded=True)
    try:
        whisp_app.start()
        tray = the_tray()
        tray.open_requested.emit()
        [first] = FakeMainWindow.instances
        assert first.isVisible()
        tray.open_requested.emit()  # already open: no second window
        assert len(FakeMainWindow.instances) == 1

        first.close()
        deliver_deferred_deletes()
        assert tray.notes == [
            ("Whisp is still running", "It's in the notification area. Hold Ctrl + Win to dictate anytime.", "info")
        ]
        state = json.loads((app_paths().config_dir / main_module.UI_STATE_FILE).read_text(encoding="utf-8"))
        assert state == {"tray_tip_shown": True}

        tray.open_requested.emit()
        assert len(FakeMainWindow.instances) == 2
        FakeMainWindow.instances[-1].close()
        deliver_deferred_deletes()
        assert len(tray.notes) == 1  # the tip is shown only once
    finally:
        whisp_app.shutdown()


def test_controller_and_second_launch_can_open_the_window(autostart, qapp) -> None:
    instance = FakeInstance()
    whisp_app = make_app(qapp, onboarded=True, instance=instance)
    try:
        whisp_app.start()
        whisp_app.services.controller.open_window_requested.emit("settings", "model")
        [window] = FakeMainWindow.instances
        assert window.pages == [("settings", "model")]
        window.hide()
        instance.message.emit("show")  # a second launch asked us to show the window
        assert len(FakeMainWindow.instances) == 1
        assert window.isVisible()
    finally:
        whisp_app.shutdown()


def test_autostart_follows_the_setting(autostart, qapp) -> None:
    whisp_app = make_app(qapp, onboarded=True)
    try:
        whisp_app.start()
        assert autostart.status == "enabled"
        whisp_app.services.settings.update({"general": {"launch_at_startup": False}})
        assert autostart.status == "disabled"
        assert CALLS[-1] == ("set_autostart", False)
    finally:
        whisp_app.shutdown()


def test_autostart_turned_off_in_windows_is_respected(autostart, qapp) -> None:
    autostart.status = "blocked"
    whisp_app = make_app(qapp, onboarded=True)
    try:
        whisp_app.start()
        assert "set_autostart" not in names()
    finally:
        whisp_app.shutdown()


def test_broken_shortcuts_are_reported_and_the_rest_still_starts(autostart, qapp) -> None:
    FakeHotkeys.fail_start = True
    whisp_app = make_app(qapp, onboarded=True)
    try:
        whisp_app.start()
        assert the_tray().notes[0][0] == "Shortcuts aren't working"
        assert "hotkeys.set_bindings" not in names()
        assert "foreground.start" in names()
    finally:
        whisp_app.shutdown()


def test_remote_debugging_is_removed_from_the_environment() -> None:
    environ = {
        "QTWEBENGINE_REMOTE_DEBUGGING": "9222",
        "QTWEBENGINE_CHROMIUM_FLAGS": (
            "--disable-gpu --remote-debugging-port=9223 -Remote-Debugging-Pipe /remote-debugging-address=0.0.0.0 "
            "--lang=en"
        ),
        "PATH": "kept",
    }
    removed = main_module.disable_remote_debugging(environ)
    assert environ == {"QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu --lang=en", "PATH": "kept"}
    assert len(removed) == 2
    assert not any("922" in item for item in removed)  # names only, never values
    only_debugging = {"QTWEBENGINE_CHROMIUM_FLAGS": " --remote-debugging-port=1 "}
    main_module.disable_remote_debugging(only_debugging)
    assert only_debugging == {}
    assert main_module.disable_remote_debugging({"OTHER": "x"}) == []


def test_qt_gets_no_command_line_options_and_no_remote_debugging(monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    from PySide6.QtWidgets import QApplication

    created: list[list[str]] = []

    class Stop(Exception):
        """Ends _run right after the QApplication would have been created."""

    class FakeApplication:
        instance = staticmethod(QApplication.instance)  # pytest-qt still finds the real one

        def __init__(self, argv: list[str]) -> None:
            created.append(list(argv))
            raise Stop

    monkeypatch.setenv("QTWEBENGINE_REMOTE_DEBUGGING", "9222")
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--remote-debugging-port=9223")
    args, unknown = main_module._parse_args(["--background", "--webEngineArgs", "--remote-debugging-port=9224"])
    assert args.background and unknown == ["--webEngineArgs", "--remote-debugging-port=9224"]
    window = fake_module("whisp.ui.main_window.window", register_web_scheme=lambda: None)
    with monkeypatch.context() as patch:  # undone before pytest-qt looks for the application again
        patch.setattr("PySide6.QtWidgets.QApplication", FakeApplication)
        patch.setitem(sys.modules, "whisp.ui.main_window.window", window)
        patch.setattr(importlib.import_module("whisp.ui.main_window"), "window", window, raising=False)
        patch.setattr(main_module, "_set_app_user_model_id", lambda: None)
        with pytest.raises(Stop):
            main_module._run(args, app_paths())
    assert created == [[sys.argv[0]]]
    assert "QTWEBENGINE_REMOTE_DEBUGGING" not in os.environ
    assert "QTWEBENGINE_CHROMIUM_FLAGS" not in os.environ


def test_diagnostics_piped_to_another_program_are_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirected output starts in the ANSI code page, which would garble "RTX 4070 — ready" for UTF-8 readers.
    piped = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", piped)
    main_module._prepare_cli_output()
    print("NVIDIA GeForce RTX 4070 — ready")
    piped.flush()
    assert piped.buffer.getvalue().decode("utf-8").splitlines() == ["NVIDIA GeForce RTX 4070 — ready"]


def test_diagnostics_never_fail(autostart) -> None:
    text = main_module.collect_diagnostics()
    assert text.startswith("Whisp diagnostics")
    for heading in ("Versions", "GPU", "Speech engine", "Files"):
        assert f"\n{heading}\n" in text
