"""Main window: the whisp:// file server, the web bridge, and the window itself."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QCoreApplication, QObject, QPoint, QSettings, Qt, QUrl, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

# Importing the window module loads Qt WebEngine, which must happen before the
# QApplication exists; so must registering the scheme. Collection runs before
# any test creates the app, so module level is the right place.
from whisp.ui.main_window import window as main_window
from whisp.ui.main_window.bridge import GENERIC_ERROR, Bridge, BridgeAdapters, BridgeError

if QCoreApplication.instance() is None:
    main_window.register_web_scheme()
SCHEME_READY = main_window._scheme_registered

WEB_ROOT = Path(main_window.__file__).resolve().parent / "web"


# -- fakes --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    id: int
    created_at: float
    kind: str = "dictation"
    app_name: str = "Slack"
    process_name: str = "slack.exe"
    category: str = "messaging"
    raw_text: str = "um hello there"
    final_text: str = "Hello there."
    word_count: int = 2
    audio_ms: int = 1500
    speech_ms: int = 900
    latency_ms: int = 240
    model_id: str = "test-model"
    polished: bool = False
    outcome: str = "inserted"


class FakeHistory(QObject):
    added = Signal(object)
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.persistent = True
        self.entries = [Entry(id=i, created_at=1_760_000_000.0 - i * 60) for i in range(1, 6)]

    def list(self, *, query: str = "", app: str = "", limit: int = 50, offset: int = 0) -> list[Entry]:
        return self.entries[offset : offset + limit]

    def get(self, entry_id: int) -> Entry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def delete(self, entry_id: int) -> None:
        self.entries = [e for e in self.entries if e.id != entry_id]
        self.changed.emit()

    def clear(self) -> None:
        self.entries = []
        self.changed.emit()

    def apps(self) -> list[str]:
        return ["Slack"]

    def stats(self) -> dict[str, Any]:
        return {
            "total_words": 10,
            "total_dictations": 5,
            "words_today": 4,
            "words_this_week": 10,
            "avg_wpm": 120,
            "time_saved_minutes": 0.1,
            "streak_days": 1,
            "best_streak_days": 2,
            "top_apps": [{"app": "Slack", "words": 10}],
            "daily": [{"date": f"2026-09-{d:02d}", "words": 0} for d in range(1, 31)],
        }


@dataclass(frozen=True)
class File:
    path: str
    size: int


@dataclass(frozen=True)
class Variant:
    key: str
    quantization: str | None
    files: tuple[File, ...]


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    description: str
    languages: tuple[str, ...]
    accuracy: int
    speed: int
    license: str
    license_url: str
    variants: tuple[Variant, ...]
    recommended: bool


CATALOG = (
    Model(
        "fast-en",
        "Fast English",
        "Quick.",
        ("en",),
        4,
        5,
        "MIT",
        "https://example.org/fast",
        (Variant("gpu", None, (File("a.onnx", 2000),)), Variant("cpu", "int8", (File("b.onnx", 500),))),
        True,
    ),
    Model(
        "multi",
        "Multilingual",
        "Many languages.",
        ("en", "de", "fr"),
        4,
        3,
        "MIT",
        "https://example.org/multi",
        (Variant("cpu", "int8", (File("c.onnx", 800),)),),
        False,
    ),
)


@dataclass(frozen=True)
class Gpu:
    has_nvidia: bool = True
    name: str | None = "NVIDIA GeForce RTX 4070"
    vram_mb: int | None = 12282
    cuda_ready: bool = True
    detail: str = "CUDA is ready."
    runtime_download_size: int = 0


class FakeModels(QObject):
    progress = Signal(str, object, object)
    changed = Signal()
    download_failed = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.installed = {("fast-en", "gpu")}
        self.downloads: list[tuple[str, str]] = []

    def is_installed(self, model_id: str, variant: str) -> bool:
        return (model_id, variant) in self.installed

    def install_state(self) -> dict[str, dict]:
        return {}

    def download(self, model_id: str, variant: str) -> None:
        self.downloads.append((model_id, variant))

    def cancel(self, model_id: str) -> None:
        return None

    def delete(self, model_id: str) -> None:
        self.installed = {item for item in self.installed if item[0] != model_id}


class FakeAsr(QObject):
    status_changed = Signal(object)
    gpu_runtime_progress = Signal(object, object)
    gpu_runtime_finished = Signal(object)
    gpu_runtime_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.gpu_runtime_installing = False
        self.gpu_install_requests = 0
        self.gpu_cancel_requests = 0

    @property
    def status(self) -> Any:
        from whisp.core.types import EngineState, EngineStatus

        return EngineStatus(EngineState.READY, "fast-en", "cuda", "NVIDIA GeForce RTX 4070")

    def install_gpu_runtime(self) -> bool:
        self.gpu_install_requests += 1
        started = not self.gpu_runtime_installing
        self.gpu_runtime_installing = True
        return started

    def cancel_gpu_runtime(self) -> None:
        self.gpu_cancel_requests += 1


class FakeLlm(QObject):
    """LlmRuntime's surface, with the real catalog and status types and nothing started."""

    status_changed = Signal(object)
    progress = Signal(str, object, object)
    install_finished = Signal(str)
    install_failed = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        from whisp.core.llm.catalog import recommended_model
        from whisp.core.llm.runtime import LlmState, LlmStatus

        self.installed = {recommended_model().id}
        self.install_requests: list[str] = []
        self.status = LlmStatus(LlmState.READY, recommended_model().id, "http://127.0.0.1:1", None, "CPU")

    def catalog(self) -> list[Any]:
        from whisp.core.llm.catalog import CATALOG

        return list(CATALOG)

    def is_installed(self, model_id: str) -> bool:
        return model_id in self.installed

    def install(self, model_id: str) -> None:
        self.install_requests.append(model_id)

    def uninstall(self, model_id: str) -> None:
        self.installed.discard(model_id)


class FakeInserter:
    """Records what Whisp's own clipboard writes would put on the clipboard."""

    def __init__(self) -> None:
        self.copied: list[str] = []

    def copy_to_clipboard(self, text: str) -> bool:
        self.copied.append(text)
        return True


class FakePolisher:
    def __init__(self) -> None:
        self.fail_listing = False

    def is_ready(self) -> bool:
        return False

    def test(self) -> tuple[bool, str]:
        return True, "It works."

    def list_models(self, provider: str) -> list[str]:
        if self.fail_listing:
            raise ConnectionError("connection refused")
        return ["b-model", "a-model"]


class FakeHotkeys(QObject):
    pressed = Signal(str)
    released = Signal(str)
    interrupted = Signal(str)
    recorded = Signal(str)
    binding_error = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.recording = False

    def start_recording(self) -> None:
        self.recording = True

    def stop_recording(self) -> None:
        self.recording = False


class FakeController(QObject):
    state_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.binding_errors: dict[str, str] = {}
        self.is_paused = False

    def problem(self) -> str | None:
        return next(iter(self.binding_errors.values()), None)

    def toggle_paused(self) -> None:
        self.is_paused = not self.is_paused
        self.state_changed.emit()


class FakeStream:
    def __init__(self, on_level: Any) -> None:
        self.error: str | None = None
        self.device_name = "Test microphone"
        self.stopped = threading.Event()
        for level in (0.2, 0.5, 0.8):
            on_level(level)

    def stop(self) -> None:
        self.stopped.set()


class FakeAdapters(BridgeAdapters):
    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.protected: list[str] = []

    def list_input_devices(self) -> list[Any]:
        from whisp.core.types import AudioDevice

        return [AudioDevice("Mic A", "Mic A", "Windows WASAPI", True, 1, 48000.0)]

    def start_microphone(self, device: str | None, on_level: Any) -> FakeStream:
        stream = FakeStream(on_level)
        self.streams.append(stream)
        return stream

    def detect_gpu(self) -> Gpu:
        return Gpu()

    def model_catalog(self) -> tuple[Model, ...]:
        return CATALOG

    def display_chord(self, chord: str) -> list[str]:
        return [part.capitalize() for part in chord.split("+")]

    def normalize_chord(self, chord: str) -> str:
        parts = [p.strip().lower() for p in chord.split("+") if p.strip()]
        if not parts:
            raise ValueError("Press at least one key.")
        return "+".join(parts)

    def autostart_status(self) -> str:
        return "disabled"

    def set_autostart(self, enabled: bool) -> None:
        return None

    def protect_secret(self, plain: str) -> str:
        token = "dpapi:" + hashlib.sha256(plain.encode()).hexdigest()[:16]
        self.protected.append(token)
        return token

    def apply_style(self, settings: Any, category: str, style: str, text: str) -> str:
        return text.upper() if style == "formal" else text

    def user_first_name(self) -> str | None:
        return "Sam"


@dataclass
class Services:
    paths: Any
    settings: Any
    history: FakeHistory
    models: FakeModels
    asr: FakeAsr
    llm: FakeLlm
    polisher: FakePolisher
    hotkeys: FakeHotkeys
    controller: FakeController
    inserter: FakeInserter


@pytest.fixture
def services(qapp, settings_store) -> Services:
    from whisp.storage.paths import app_paths

    return Services(
        paths=app_paths(),
        settings=settings_store,
        history=FakeHistory(),
        models=FakeModels(),
        asr=FakeAsr(),
        llm=FakeLlm(),
        polisher=FakePolisher(),
        hotkeys=FakeHotkeys(),
        controller=FakeController(),
        inserter=FakeInserter(),
    )


@pytest.fixture
def adapters() -> FakeAdapters:
    return FakeAdapters()


@pytest.fixture
def bridge(services: Services, adapters: FakeAdapters):
    instance = Bridge(services, adapters)  # type: ignore[arg-type]
    yield instance
    instance.shutdown()
    instance.deleteLater()


class Events:
    """Records the bridge's pushed events."""

    def __init__(self, bridge: Bridge) -> None:
        self.items: list[tuple[str, Any]] = []
        self.raw: list[str] = []
        bridge.event.connect(self._on_event)

    def _on_event(self, name: str, payload: str) -> None:
        self.raw.append(payload)
        self.items.append((name, json.loads(payload)))

    def named(self, name: str) -> list[Any]:
        return [payload for event, payload in self.items if event == name]


def call(bridge: Bridge, method: str, params: Any = None) -> dict[str, Any]:
    return json.loads(bridge.call(method, json.dumps({} if params is None else params)))


def call_and_wait(qtbot, bridge: Bridge, events: Events, method: str, params: Any = None) -> dict[str, Any]:
    """Call a method that may answer later through ``call.result``; return the final reply."""
    reply = call(bridge, method, params)
    if "pending" not in reply:
        return reply
    call_id = reply["pending"]
    qtbot.waitUntil(lambda: any(p.get("id") == call_id for p in events.named("call.result")), timeout=3000)
    result = next(p for p in events.named("call.result") if p.get("id") == call_id)
    return {k: v for k, v in result.items() if k != "id"}


# -- whisp:// resources --------------------------------------------------------------------------


@pytest.fixture
def resources() -> main_window.WebResources:
    return main_window.WebResources.bundled()


@pytest.mark.parametrize(
    ("path", "mime"),
    [
        ("/", "text/html; charset=utf-8"),
        ("/index.html", "text/html; charset=utf-8"),
        ("/js/main.js", "text/javascript; charset=utf-8"),
        ("/css/tokens.css", "text/css; charset=utf-8"),
        ("/vendor/preact/preact.module.js", "text/javascript; charset=utf-8"),
        ("/fonts/InterVariable.woff2", "font/woff2"),
        ("/brand/logo.svg", "image/svg+xml"),
        ("/qwebchannel.js", "text/javascript; charset=utf-8"),
    ],
)
def test_resources_are_served_with_the_right_type(resources, path, mime) -> None:
    response = resources.get(path)
    assert response.status == 200
    assert response.mime == mime
    assert response.body


def test_qwebchannel_comes_from_qt(resources) -> None:
    assert b"QWebChannel" in resources.get("/qwebchannel.js").body


@pytest.mark.parametrize(
    "path",
    ["/missing.js", "/js/", "/js/nope/deeper.js", "/fonts/InterVariable.ttf", "/brand/logo.png", "/fonts/", "/brand"],
)
def test_missing_or_unservable_files_are_404(resources, path) -> None:
    assert resources.get(path).status == 404


@pytest.mark.parametrize(
    "path",
    [
        "/..",
        "/../storage/settings.py",
        "/js/../../bridge.py",
        "/./index.html",
        "/fonts/../../storage/settings.py",
        "/brand/../fonts/InterVariable.woff2",
        "/C:/Windows/win.ini",
        "/css\\..\\..\\window.py",
        "//server/share/file.txt",
        "/.git/config",
        "/index.html.",
        "/index.html ",
        "/index.html\x00.png",
        "index.html",
    ],
)
def test_paths_that_could_leave_the_web_folder_are_refused(resources, path) -> None:
    assert resources.get(path).status in (403, 404)
    assert resources.get(path).status != 200


def test_dot_segments_and_drive_letters_are_403(resources) -> None:
    for path in ("/../storage/settings.py", "/C:/Windows/win.ini", "/css\\x.css", "/.hidden"):
        assert resources.get(path).status == 403


def test_only_known_file_types_are_served(tmp_path: Path) -> None:
    web = tmp_path / "web"
    web.mkdir()
    (web / "notes.py").write_text("secret = 1", encoding="utf-8")
    (web / "README").write_text("hello", encoding="utf-8")
    (web / "page.html").write_text("<p>hi</p>", encoding="utf-8")
    resources = main_window.WebResources(web, tmp_path / "assets")
    assert resources.get("/notes.py").status == 404
    assert resources.get("/README").status == 404
    assert resources.get("/page.html").status == 200


def test_index_has_a_strict_content_security_policy() -> None:
    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    csp = re.search(r'http-equiv="Content-Security-Policy"\s+content="([^"]+)"', html).group(1)
    assert "default-src 'none'" in csp
    assert "connect-src 'none'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "http" not in csp
    # The one inline script (the import map) is allowed by its exact hash.
    import_map = re.search(r'<script type="importmap">(.*?)</script>', html, re.S).group(1)
    digest = base64.b64encode(hashlib.sha256(import_map.encode("utf-8")).digest()).decode()
    assert f"'sha256-{digest}'" in csp
    for target in json.loads(import_map)["imports"].values():
        assert (WEB_ROOT / target.lstrip("/")).is_file()


def test_vendored_libraries_ship_with_their_licences() -> None:
    for folder in ("preact", "htm", "signals", "signals-core", "lucide"):
        assert (WEB_ROOT / "vendor" / folder / "LICENSE").is_file()


# -- bridge: protocol ----------------------------------------------------------------------------


def test_unknown_methods_and_bad_json_are_plain_errors(bridge) -> None:
    assert "error" in call(bridge, "no.such_method")
    assert json.loads(bridge.call("settings.get", "{not json")) == {"error": "The request was not valid JSON."}
    assert json.loads(bridge.call("settings.get", "[1, 2]")) == {"error": "The request must be a JSON object."}
    assert "ok" in json.loads(bridge.call("settings.get", ""))


def test_missing_parameters_are_reported(bridge) -> None:
    reply = call(bridge, "history.delete", {"id": "seven"})
    assert reply == {"error": "The request is missing 'id'."}


def test_unexpected_failures_are_generic_and_never_log_parameters(bridge, services, caplog) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("boom")

    services.settings.update = explode  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        reply = call(bridge, "settings.update", {"text": {"snippets": [{"trigger": "secret phrase", "text": "x"}]}})
    assert reply == {"error": GENERIC_ERROR}
    assert "settings.update" in caplog.text
    assert "secret phrase" not in caplog.text


def test_every_documented_method_is_registered(bridge) -> None:
    import whisp.ui.main_window.bridge as module

    table = module.__doc__.split("Methods (", 1)[1].split("\nEvents\n", 1)[0]
    documented = set(re.findall(r"^([a-z]+\.[a-z_]+)\s", table, re.M))
    assert documented == set(bridge.methods)


def test_slow_methods_answer_through_call_result(qtbot, bridge) -> None:
    events = Events(bridge)
    reply = call(bridge, "history.list", {"limit": 2})
    assert set(reply) == {"pending"}
    result = call_and_wait(qtbot, bridge, events, "history.list", {"limit": 2})
    entries = result["ok"]["entries"]
    assert [e["id"] for e in entries] == [1, 2]
    assert result["ok"]["has_more"] is True
    assert result["ok"]["persistent"] is True
    assert entries[0]["text"] == "Hello there."
    assert entries[0]["raw_text"] == "um hello there"


def test_slow_method_failures_are_plain_messages(qtbot, bridge, services) -> None:
    events = Events(bridge)
    services.polisher.fail_listing = True
    result = call_and_wait(qtbot, bridge, events, "polish.models", {"provider": "ollama"})
    assert result == {"error": "Couldn't reach the server. Check the address and that it's running."}
    services.polisher.fail_listing = False
    result = call_and_wait(qtbot, bridge, events, "polish.models", {"provider": "ollama"})
    assert result == {"ok": {"models": ["a-model", "b-model"]}}


def test_open_url_only_opens_web_links(bridge) -> None:
    for url in ("file:///C:/Windows/win.ini", "javascript:alert(1)", "ms-settings:privacy", "http://"):
        assert call(bridge, "app.open_url", {"url": url}) == {"error": "Only web links can be opened."}


# -- bridge: settings and secrets -------------------------------------------------------------------


def test_settings_round_trip(qtbot, bridge, services) -> None:
    events = Events(bridge)
    reply = call(bridge, "settings.update", {"audio": {"sounds_enabled": False, "max_recording_minutes": 5}})
    assert reply["ok"]["audio"]["sounds_enabled"] is False
    assert services.settings.current.audio.max_recording_minutes == 5
    assert call(bridge, "settings.get")["ok"]["audio"]["sounds_enabled"] is False
    changed = events.named("settings.changed")
    assert changed and changed[-1]["sections"] == ["audio"]

    reply = call(bridge, "settings.reset", {"section": "audio"})
    assert reply["ok"]["audio"]["sounds_enabled"] is True
    assert services.settings.current.audio.max_recording_minutes == 10


def test_invalid_settings_are_refused_and_nothing_changes(bridge, services) -> None:
    before = services.settings.current
    reply = call(bridge, "settings.update", {"audio": {"sounds_volume": 5}})
    assert reply == {"error": "That setting couldn't be saved. Check the value and try again."}
    assert call(bridge, "settings.update", {"nonsense": {"a": 1}})["error"]
    assert call(bridge, "settings.reset", {"section": "nonsense"})["error"]
    assert services.settings.current == before


def test_api_key_is_encrypted_and_never_sent_back(qtbot, bridge, services, adapters) -> None:
    events = Events(bridge)
    key = "sk-live-very-secret-123"
    reply = call(bridge, "settings.update", {"polish": {"openai_api_key": f"  {key}  ", "openai_model": "m"}})

    stored = services.settings.current.polish.openai_api_key_protected
    assert stored == adapters.protected[-1]
    assert key not in stored
    assert reply["ok"]["polish"]["openai_api_key_saved"] is True
    assert "openai_api_key_protected" not in reply["ok"]["polish"]
    everything_sent = json.dumps(reply) + json.dumps(call(bridge, "settings.get")) + "".join(events.raw)
    assert key not in everything_sent
    assert stored not in everything_sent
    assert key not in services.settings.path.read_text(encoding="utf-8")

    reply = call(bridge, "settings.update", {"polish": {"openai_api_key": ""}})
    assert reply["ok"]["polish"]["openai_api_key_saved"] is False
    assert services.settings.current.polish.openai_api_key_protected == ""


def test_page_cannot_write_the_protected_key_directly(bridge, services) -> None:
    call(bridge, "settings.update", {"polish": {"openai_api_key_protected": "forged", "openai_api_key_saved": True}})
    assert services.settings.current.polish.openai_api_key_protected == ""


def test_onboarding_complete_is_saved(bridge, services) -> None:
    assert call(bridge, "onboarding.complete")["ok"]["general"]["onboarding_completed"] is True
    assert services.settings.current.general.onboarding_completed is True


# -- bridge: hotkeys, audio, models, history -----------------------------------------------------------


def test_hotkey_validation_explains_conflicts(bridge) -> None:
    assert call(bridge, "hotkeys.validate", {"chord": "alt+shift+z", "action": "dictate"})["ok"] == {
        "valid": False,
        "message": "Already used for Paste last transcript.",
    }
    assert call(bridge, "hotkeys.validate", {"chord": "esc", "action": "dictate"})["ok"]["valid"] is False
    ok = call(bridge, "hotkeys.validate", {"chord": " Ctrl + F8 ", "action": "hands_free"})["ok"]
    assert ok == {"valid": True, "chord": "ctrl+f8", "keys": ["Ctrl", "F8"]}


def test_recorded_chords_are_forwarded_only_while_recording(qtbot, bridge, services) -> None:
    events = Events(bridge)
    services.hotkeys.recorded.emit("ctrl+f9")
    qtbot.wait(20)
    assert not events.named("hotkeys.recorded")
    assert call(bridge, "hotkeys.record_start") == {"ok": None}
    assert services.hotkeys.recording
    services.hotkeys.recorded.emit("ctrl+f9")
    qtbot.waitUntil(lambda: bool(events.named("hotkeys.recorded")))
    assert events.named("hotkeys.recorded")[-1] == {"chord": "ctrl+f9", "keys": ["Ctrl", "F9"]}


def test_binding_errors_follow_the_controller(qtbot, bridge, services) -> None:
    events = Events(bridge)
    services.controller.binding_errors = {"paste_last": "Another app uses this shortcut."}
    services.controller.state_changed.emit()
    assert events.named("hotkeys.errors") == [{"paste_last": "Another app uses this shortcut."}]
    assert call(bridge, "hotkeys.errors")["ok"] == {"paste_last": "Another app uses this shortcut."}
    services.controller.state_changed.emit()  # nothing changed, so nothing is sent
    assert len(events.named("hotkeys.errors")) == 1
    services.controller.binding_errors = {}
    services.controller.state_changed.emit()
    assert events.named("hotkeys.errors")[-1] == {}


def test_pause_status_comes_from_the_controller(qtbot, bridge, services) -> None:
    events = Events(bridge)
    assert call(bridge, "app.status")["ok"] == {"paused": False, "problem": None}
    assert call(bridge, "app.set_paused", {"paused": True})["ok"] == {"paused": True, "problem": None}
    assert services.controller.is_paused is True
    assert events.named("app.status") == [{"paused": True, "problem": None}]
    call(bridge, "app.set_paused", {"paused": True})  # already paused: no toggle
    assert services.controller.is_paused is True
    assert call(bridge, "app.set_paused", {"paused": "yes"})["error"]


def test_meter_reports_levels_and_releases_the_microphone(qtbot, bridge, adapters) -> None:
    events = Events(bridge)
    assert call(bridge, "audio.meter_start", {"device": "Mic A"}) == {"ok": None}
    qtbot.waitUntil(lambda: len(events.named("audio.level")) == 3, timeout=3000)
    assert [e["level"] for e in events.named("audio.level")] == [0.2, 0.5, 0.8]
    qtbot.waitUntil(lambda: any(e["state"] == "started" for e in events.named("audio.meter")))
    call(bridge, "audio.meter_stop")
    qtbot.waitUntil(lambda: adapters.streams[-1].stopped.is_set(), timeout=3000)


def test_meter_surfaces_device_failures(qtbot, bridge, adapters) -> None:
    events = Events(bridge)
    call(bridge, "audio.meter_start", {"device": None})
    qtbot.waitUntil(lambda: bool(adapters.streams), timeout=3000)
    adapters.streams[-1].error = "The microphone was disconnected."
    qtbot.waitUntil(lambda: any(e["state"] == "error" for e in events.named("audio.meter")), timeout=3000)
    assert events.named("audio.meter")[-1] == {"state": "error", "message": "The microphone was disconnected."}
    assert adapters.streams[-1].stopped.is_set()


def test_device_scan_failures_are_plain_messages(qtbot, bridge, adapters) -> None:
    events = Events(bridge)

    def broken() -> list[Any]:
        raise BridgeError("Windows couldn't list your microphones.")

    adapters.list_input_devices = broken  # type: ignore[method-assign]
    assert call_and_wait(qtbot, bridge, events, "audio.devices") == {"error": "Windows couldn't list your microphones."}


def test_models_list_describes_catalog_hardware_and_choice(qtbot, bridge) -> None:
    events = Events(bridge)
    data = call_and_wait(qtbot, bridge, events, "models.list")["ok"]
    fast, multi = data["models"]
    assert fast["installed"] is True and fast["preferred_variant"] == "gpu"
    assert [v["size"] for v in fast["variants"]] == [2000, 500]
    assert multi["installed"] is False and multi["preferred_variant"] == "cpu"
    assert data["recommended"] == {"id": "fast-en", "variant": "gpu"}
    assert data["gpu"]["cuda_ready"] is True


def test_models_download_select_and_delete_rules(qtbot, bridge, services) -> None:
    events = Events(bridge)
    reply = call_and_wait(qtbot, bridge, events, "models.select", {"id": "multi"})
    assert reply == {"error": "Download this model before using it."}
    assert call(bridge, "models.download", {"id": "multi", "variant": "gpu"})["error"]
    assert call(bridge, "models.download", {"id": "multi", "variant": "cpu"}) == {"ok": None}
    assert services.models.downloads == [("multi", "cpu")]
    assert call(bridge, "models.download", {"id": "nope"})["error"]
    services.models.installed.add(("multi", "cpu"))
    reply = call_and_wait(qtbot, bridge, events, "models.select", {"id": "multi"})
    assert reply["ok"]["transcription"]["model_id"] == "multi"
    assert "Choose another one" in call(bridge, "models.delete", {"id": "multi"})["error"]


def test_history_events_carry_entries(qtbot, bridge, services) -> None:
    events = Events(bridge)
    services.history.added.emit(Entry(id=99, created_at=1.0, final_text="Fresh words.", raw_text="Fresh words."))
    qtbot.waitUntil(lambda: bool(events.named("history.added")))
    entry = events.named("history.added")[-1]["entry"]
    assert entry["id"] == 99 and entry["text"] == "Fresh words." and entry["raw_text"] == ""


def test_llm_catalog_reports_sizes_and_install_state(qtbot, bridge, services) -> None:
    from whisp.core.llm.catalog import CATALOG, recommended_model

    events = Events(bridge)
    data = call_and_wait(qtbot, bridge, events, "llm.catalog")["ok"]
    assert [m["id"] for m in data["models"]] == [m.id for m in CATALOG]
    by_id = {m["id"]: m for m in data["models"]}
    best = recommended_model()
    assert by_id[best.id]["installed"] is True
    assert by_id[best.id]["size"] == best.download_size
    assert sum(m["installed"] for m in data["models"]) == 1
    assert data["status"] == {
        "state": "ready",
        "model_id": best.id,
        "message": None,
        "backend": "CPU",
        "installing": None,
    }


def test_llm_install_outcomes_are_forwarded(qtbot, bridge, services) -> None:
    events = Events(bridge)
    services.llm.install_finished.emit("some-model")
    services.llm.install_failed.emit("other-model", "Download cancelled.")
    assert events.named("llm.installed") == [{"id": "some-model"}]
    assert events.named("llm.failed") == [{"id": "other-model", "message": "Download cancelled."}]


def test_ai_model_ids_must_come_from_the_catalog(qtbot, bridge, services) -> None:
    events = Events(bridge)
    for bad in ("..\\..\\..\\victim", "C:\\Users", "not-a-model"):
        assert call(bridge, "llm.install", {"id": bad}) == {
            "error": "That AI model isn't available in this version of Whisp."
        }
        reply = call_and_wait(qtbot, bridge, events, "llm.uninstall", {"id": bad})
        assert reply == {"error": "That AI model isn't available in this version of Whisp."}
    assert services.llm.install_requests == []
    known = services.llm.catalog()[0].id
    assert call(bridge, "llm.install", {"id": known}) == {"ok": None}
    assert services.llm.install_requests == [known]


def test_history_copy_uses_whisps_own_clipboard_writes(qtbot, bridge, services) -> None:
    events = Events(bridge)
    entry = services.history.entries[0]
    assert call_and_wait(qtbot, bridge, events, "history.copy", {"id": entry.id}) == {"ok": None}
    assert services.inserter.copied == [entry.final_text]  # kept out of Windows clipboard history
    missing = call_and_wait(qtbot, bridge, events, "history.copy", {"id": 987654})
    assert missing == {"error": "That dictation no longer exists."}


def test_gpu_runtime_pack_install_cancel_and_events(qtbot, bridge, services) -> None:
    from whisp.core.asr.gpu import GpuInfo

    events = Events(bridge)
    assert call(bridge, "gpu.install_runtime") == {"ok": {"started": True}}
    assert call(bridge, "gpu.install_runtime") == {"ok": {"started": False}}  # already downloading
    assert call(bridge, "gpu.cancel_runtime") == {"ok": None}
    assert services.asr.gpu_install_requests == 2 and services.asr.gpu_cancel_requests == 1

    services.asr.gpu_runtime_progress.emit(100, 1000)
    services.asr.gpu_runtime_progress.emit(500, 1000)
    qtbot.waitUntil(lambda: bool(events.named("gpu.progress")), timeout=2000)
    assert events.named("gpu.progress")[-1] == {"done": 500, "total": 1000}  # throttled to the latest

    ready = GpuInfo(True, "NVIDIA GeForce RTX 4070", 12282, True, "NVIDIA GeForce RTX 4070 · ready")
    services.asr.gpu_runtime_finished.emit(ready)
    [installed] = events.named("gpu.installed")
    assert installed["gpu"]["cuda_ready"] is True and installed["gpu"]["installing"] is False
    services.asr.gpu_runtime_failed.emit("")
    assert events.named("gpu.failed") == [{"message": ""}]


def test_gpu_info_reports_download_size_and_progress_state(qtbot, bridge, services) -> None:
    events = Events(bridge)
    services.asr.gpu_runtime_installing = True
    info = call_and_wait(qtbot, bridge, events, "gpu.info")["ok"]
    assert info["installing"] is True
    assert "runtime_download_size" in info


def test_style_preview_runs_every_style(bridge) -> None:
    preview = call(bridge, "styles.preview", {"category": "email"})["ok"]
    assert set(preview["styles"]) == {"verbatim", "clean", "casual", "formal"}
    assert preview["styles"]["formal"] == preview["before"].upper()


def test_navigation_waits_for_the_page(qtbot, bridge) -> None:
    events = Events(bridge)
    bridge.navigate("settings", "model")
    assert not events.named("navigate")
    ready = call(bridge, "app.ready")["ok"]
    assert (ready["page"], ready["section"]) == ("settings", "model")
    assert "openai_api_key_protected" not in ready["settings"]["polish"]
    bridge.navigate("history")
    assert events.named("navigate")[-1] == {"page": "history", "section": None}
    with pytest.raises(ValueError):
        bridge.navigate("nowhere")


def test_shutdown_stops_events_and_listening(qtbot, bridge, services) -> None:
    events = Events(bridge)
    call(bridge, "hotkeys.record_start")
    bridge.shutdown()
    assert services.hotkeys.recording is False
    services.history.changed.emit()
    services.settings.update({"audio": {"sounds_enabled": False}})
    qtbot.wait(30)
    assert events.items == []


# -- the window ----------------------------------------------------------------------------------------


@pytest.mark.skipif(not SCHEME_READY, reason="the whisp:// scheme must be registered before the QApplication exists")
def test_window_boots_the_web_ui_and_is_released_on_close(qtbot, services, adapters) -> None:
    window = main_window.MainWindow(services, adapters=adapters)  # type: ignore[arg-type]
    view = window.findChild(main_window.QWebEngineView)
    with qtbot.waitSignal(view.loadFinished, timeout=15000) as loaded:
        window.show()
    assert loaded.args == [True]

    def text_of(selector: str) -> str:
        result: list[Any] = []
        view.page().runJavaScript(
            f"document.querySelector({json.dumps(selector)})?.textContent || ''", 0, result.append
        )
        qtbot.waitUntil(lambda: bool(result), timeout=3000)
        return str(result[0]).strip()

    # A first run lands on onboarding; the bridge connects and the UI renders under the strict CSP.
    qtbot.waitUntil(lambda: text_of(".ob-title") == "Welcome to Whisp", timeout=15000)
    services.settings.update({"general": {"onboarding_completed": True}})
    window.show_page("settings", "about")
    qtbot.waitUntil(lambda: text_of(".page-title") == "Settings", timeout=5000)
    qtbot.waitUntil(lambda: text_of(".about-name") == "Whisp", timeout=5000)

    # Right-click: an editing menu in text fields, nothing anywhere else.
    window.show_page("dictionary")
    qtbot.waitUntil(lambda: text_of(".page-title") == "Dictionary", timeout=5000)
    target = view.focusProxy() or view

    def right_click(selector: str) -> None:
        result: list[Any] = []
        view.page().runJavaScript(
            f"JSON.stringify(document.querySelector({json.dumps(selector)}).getBoundingClientRect())",
            0,
            result.append,
        )
        qtbot.waitUntil(lambda: bool(result), timeout=3000)
        box = json.loads(result[0])
        point = QPoint(int(box["x"] + 12), int(box["y"] + box["height"] / 2))
        QTest.mouseClick(target, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, point)

    right_click(".add-card input")
    qtbot.waitUntil(lambda: isinstance(QApplication.activePopupWidget(), QMenu), timeout=5000)
    menu = QApplication.activePopupWidget()
    assert len([action for action in menu.actions() if not action.isSeparator()]) == 6
    menu.close()
    qtbot.waitUntil(lambda: QApplication.activePopupWidget() is None, timeout=3000)
    right_click(".page-title")
    qtbot.wait(400)
    assert QApplication.activePopupWidget() is None

    # End to end: the dictionary form saves through the bridge into SettingsStore. The click
    # waits a moment, as a person would, so Preact has rendered the typed values.
    view.page().runJavaScript(
        "const [word, variants] = document.querySelectorAll('.add-card input');"
        "word.value = 'Kubernetes'; word.dispatchEvent(new Event('input', {bubbles: true}));"
        "variants.value = 'cooper netties, kubernetties'; variants.dispatchEvent(new Event('input', {bubbles: true}));"
        "setTimeout(() => document.querySelector('.add-card button[type=submit]').click(), 50);"
    )
    qtbot.waitUntil(lambda: len(services.settings.current.text.dictionary) == 1, timeout=5000)
    entry = services.settings.current.text.dictionary[0]
    assert (entry.term, entry.sounds_like) == ("Kubernetes", ("cooper netties", "kubernetties"))
    qtbot.waitUntil(lambda: text_of(".entry-term") == "Kubernetes", timeout=5000)

    # And a settings toggle flips the stored value.
    before = services.settings.current.general.show_notifications
    window.show_page("settings", "general")
    qtbot.waitUntil(lambda: text_of(".tab[aria-selected='true']") == "General", timeout=5000)
    view.page().runJavaScript("document.querySelector('button[aria-label=\"Notifications\"]').click()")
    qtbot.waitUntil(lambda: services.settings.current.general.show_notifications is not before, timeout=5000)

    with qtbot.waitSignal(window.destroyed, timeout=5000):
        window.close()
    store = QSettings(str(services.paths.config_dir / "window.ini"), QSettings.Format.IniFormat)
    assert store.value("main/geometry") is not None


@pytest.mark.skipif(not SCHEME_READY, reason="the whisp:// scheme must be registered before the QApplication exists")
@pytest.mark.qt_log_ignore("couldn't start")  # this bare page has no web channel, so the UI reports that
def test_scheme_handler_serves_pages_and_404s(qtbot, qapp) -> None:
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile

    profile = QWebEngineProfile()
    handler = main_window.WebSchemeHandler(main_window.WebResources.bundled(), profile)
    profile.installUrlSchemeHandler(main_window.SCHEME, handler)
    page = QWebEnginePage(profile)
    try:
        with qtbot.waitSignal(page.loadFinished, timeout=15000) as loaded:
            page.load(QUrl("whisp://app/index.html"))
        assert loaded.args == [True]
        assert page.title() == "Whisp"
        with qtbot.waitSignal(page.loadFinished, timeout=15000) as missing:
            page.load(QUrl("whisp://app/missing.html"))
        assert missing.args == [False]
    finally:
        page.deleteLater()
        qtbot.wait(50)
        profile.deleteLater()
        qtbot.wait(50)
