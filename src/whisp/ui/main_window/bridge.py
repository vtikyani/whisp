"""The bridge between the web UI and Whisp's services.

The page reaches Python through one QWebChannel object named ``bridge``.

Protocol
========

``call(method, params_json) -> str``
    ``params_json`` is a JSON object. The reply is ``{"ok": value}`` or
    ``{"error": "plain message"}``. Methods that touch the disk, the network or
    audio devices run on a worker thread so the GUI thread never waits. They
    reply ``{"pending": id}`` at once, and the outcome follows as a
    ``call.result`` event: ``{"id": id, "ok": value}`` or ``{"id": id, "error": message}``.
    ``web/js/bridge.js`` turns both shapes into one Promise.

``event(name, payload_json)``
    Pushes from Python to the page.

Methods (``*`` = answered through ``call.result``)
==================================================

=========================  ================================  ==========================================
Method                     Params                            Result
=========================  ================================  ==========================================
app.ready                  {}                                {info, settings, theme, reduced_motion,
                                                             page, section, engine, status}
app.info *                 {}                                {name, version, qt, chromium, python,
                                                             windows, first_name}
app.status                 {}                                {paused, problem} from AppController
app.set_paused             {paused}                          {paused, problem}
app.open_folder            {name: logs|data|config|models}   null
app.open_url               {url: http(s) only}               null
app.open_system_settings   {page: microphone|sound|startup}  null
app.licenses *             {}                                {components: [...], models: [...]}
settings.get               {}                                settings (API key replaced by a flag)
settings.update            patch, e.g. {"audio": {...}}      settings
settings.reset             {section?: name}                  settings (no section = everything)
hotkeys.display            {chord}                           {keys: [...]}
hotkeys.validate           {chord, action}                   {valid, chord?, keys?, message?}
hotkeys.record_start       {}                                null (then event hotkeys.recorded)
hotkeys.record_cancel      {}                                null
hotkeys.errors             {}                                {action: message} (AppController.binding_errors)
audio.devices *            {}                                {devices: [...], selected}
audio.meter_start          {device: name|null}               null (then audio.meter / audio.level)
audio.meter_stop           {}                                null
models.list *              {}                                {models, selected, recommended, gpu, device}
models.download            {id, variant?}                    null (then models.progress / .changed)
models.cancel              {id}                              null
models.delete *            {id}                              null
models.select *            {id}                              settings
engine.status              {}                                engine status
gpu.info *                 {}                                {has_nvidia, name, vram_mb, cuda_ready, detail,
                                                             runtime_download_size, installing}
gpu.install_runtime        {}                                {started} (then gpu.progress / .installed / .failed)
gpu.cancel_runtime         {}                                null
history.list *             {query?, app?, limit?, offset?}   {entries: [...], has_more, persistent}
history.delete *           {id}                              null
history.clear *            {}                                null
history.copy *             {id}                              null (text goes to the clipboard)
history.apps *             {}                                {apps: [...]}
history.processes *        {}                                {apps: [{process, app}]} recent apps
stats.get *                {}                                HistoryStore.stats()
styles.preview             {category}                        {before, styles: {style: text}}
polish.status              {}                                {ready, llm}
polish.test *              {}                                {ok, message}
polish.models *            {provider: ollama|openai}         {models: [...]}
llm.catalog *              {}                                {models: [...], status}
llm.install                {id}                              null (then llm.progress, llm.installed/failed)
llm.cancel                 {}                                null
llm.uninstall *            {id}                              null
autostart.status           {}                                {status: enabled|disabled|blocked}
autostart.set              {enabled}                         {status}
onboarding.complete        {}                                settings
=========================  ================================  ==========================================

Events
======

=================  ===========================================================
Event              Payload
=================  ===========================================================
call.result        {id, ok} or {id, error}
navigate           {page, section}
settings.changed   {settings, sections}
theme.changed      {theme: dark|light}
motion.changed     {reduced}
engine.status      {state, model_id, model_name, device, device_name, message}
app.status         {paused, problem}
hotkeys.recorded   {chord, keys} (chord "" means the user pressed Esc)
hotkeys.errors     {action: message}
audio.meter        {state: started|stopped|error, device_name?, message?}
audio.level        {level} about 30 times a second while the meter runs
models.progress    {id, done, total} at most 10 times a second
models.changed     {}
models.failed      {id, message}
history.added      {entry}
history.changed    {}
gpu.progress       {done, total} at most 10 times a second while the GPU runtime pack downloads
gpu.installed      {gpu} (gpu.info shape) once it's installed; the speech model then moves to the GPU
gpu.failed         {message} ("" when the user cancelled)
llm.status         {state, model_id, message, backend, installing}
llm.progress       {id, done, total} at most 10 times a second
llm.installed      {id}
llm.failed         {id, message}
=================  ===========================================================

Privacy: the stored API key never leaves Python. The page sends a new key as
``polish.openai_api_key`` inside ``settings.update``; it is encrypted with
``storage.secrets.protect`` before it is saved, and the page only ever sees
``polish.openai_api_key_saved``. Parameters are never logged.
"""

from __future__ import annotations

import ctypes
import dataclasses
import enum
import itertools
import json
import logging
import math
import platform
import queue
import sys
import threading
from collections.abc import Callable, Iterable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from PySide6.QtCore import QObject, QTimer, QUrl, Signal, Slot, qVersion
from PySide6.QtGui import QDesktopServices

from whisp import APP_NAME, __version__
from whisp.storage.paths import app_paths, resource_path
from whisp.storage.settings import SECTIONS, Settings

if TYPE_CHECKING:
    from whisp.app.services import AppServices
    from whisp.core.asr.catalog import ModelInfo
    from whisp.core.asr.gpu import GpuInfo
    from whisp.core.llm.catalog import LlmModelInfo
    from whisp.core.llm.runtime import LlmStatus
    from whisp.core.types import AudioDevice, EngineStatus

log = logging.getLogger(__name__)

PAGES = ("home", "history", "dictionary", "snippets", "styles", "settings", "onboarding")
GENERIC_ERROR = "Something went wrong. The details are in the log."

# Same names as app.controller.ACTION_LABELS, so the window and tray notifications agree.
ACTION_LABELS = {
    "dictate": "Dictation",
    "command": "Command mode",
    "paste_last": "Paste last transcript",
    "copy_last": "Copy last transcript",
    "hands_free": "Hands-free",
}

STYLE_IDS = ("verbatim", "clean", "casual", "formal")

# What a speech model typically hands back for each kind of app (fillers, and
# punctuation that isn't always finished), before any style is applied. The
# Styles page runs these through the real text rules, so each style's effect
# shows: clean drops fillers, casual drops a lone final period, formal adds a
# missing one.
STYLE_SAMPLES = {
    "messaging": "Um, sounds good, I'll see you at seven.",
    "email": "Hi Maya, thanks for sending the contract over. Uh, I'll sign it today",
    "code": "um add a retry loop around the upload call",
    "document": "So, uh, the main finding is that response times dropped by about a third",
    "other": "Um, remind me to pick up the dry cleaning before six.",
}

SYSTEM_SETTINGS_URIS = {
    "microphone": "ms-settings:privacy-microphone",
    "sound": "ms-settings:sound",
    "startup": "ms-settings:startupapps",
}

# Components that ship inside Whisp, with the licence text bundled next to them.
BUNDLED_NOTICES = (
    ("Inter typeface", "OFL-1.1", "https://github.com/rsms/inter", ("assets", "fonts", "Inter-LICENSE.txt")),
    (
        "Silero VAD",
        "MIT",
        "https://github.com/snakers4/silero-vad",
        ("assets", "models", "SILERO_VAD_LICENSE.txt"),
    ),
    ("Preact", "MIT", "https://preactjs.com", ("ui", "main_window", "web", "vendor", "preact", "LICENSE")),
    (
        "Preact Signals",
        "MIT",
        "https://github.com/preactjs/signals",
        ("ui", "main_window", "web", "vendor", "signals", "LICENSE"),
    ),
    ("htm", "Apache-2.0", "https://github.com/developit/htm", ("ui", "main_window", "web", "vendor", "htm", "LICENSE")),
    ("Lucide icons", "ISC", "https://lucide.dev", ("ui", "main_window", "web", "vendor", "lucide", "LICENSE")),
)

# Python libraries Whisp runs on: (distribution, display name, licence, project URL).
RUNTIME_LIBRARIES = (
    # The distribution Whisp depends on (not the "PySide6" meta-package, which packaged builds don't carry).
    ("PySide6-Essentials", "Qt for Python (PySide6)", "LGPL-3.0", "https://www.qt.io/qt-for-python"),
    ("onnxruntime-gpu", "ONNX Runtime", "MIT", "https://onnxruntime.ai"),
    ("onnx-asr", "onnx-asr", "MIT", "https://github.com/istupakov/onnx-asr"),
    ("numpy", "NumPy", "BSD-3-Clause", "https://numpy.org"),
    ("sounddevice", "python-sounddevice", "MIT", "https://github.com/spatialaudio/python-sounddevice"),
    ("pydantic", "Pydantic", "MIT", "https://docs.pydantic.dev"),
    ("httpx", "HTTPX", "BSD-3-Clause", "https://www.python-httpx.org"),
)


class BridgeError(Exception):
    """A failure whose message is shown to the user as is."""


class _Stream(Protocol):
    @property
    def device_name(self) -> str: ...

    @property
    def error(self) -> str | None: ...

    def stop(self) -> None: ...


class BridgeAdapters:
    """Module-level functions the bridge uses, behind one seam.

    Everything the bridge needs that is not an ``AppServices`` member lives here,
    imported on first use so the window opens without loading audio, GPU or
    speech libraries. Tests and the render script pass a subclass.
    """

    def list_input_devices(self) -> list[AudioDevice]:
        """Raises BridgeError with a user-facing message if Windows can't list them."""
        from whisp.core.audio import AudioError, list_input_devices

        try:
            return list_input_devices()
        except AudioError as exc:
            raise BridgeError(str(exc)) from exc

    def start_microphone(self, device: str | None, on_level: Callable[[float], None]) -> _Stream:
        """Open and start a stream. Raises BridgeError with a user-facing message."""
        from whisp.core.audio import AudioError, MicrophoneStream

        stream = MicrophoneStream(device, on_level=on_level)
        try:
            stream.start()
        except AudioError as exc:
            raise BridgeError(str(exc)) from exc
        return stream

    def detect_gpu(self) -> GpuInfo:
        from whisp.core.asr.gpu import detect_gpu

        return detect_gpu()

    def model_catalog(self) -> tuple[ModelInfo, ...]:
        from whisp.core.asr.catalog import CATALOG

        return CATALOG

    def display_chord(self, chord: str) -> list[str]:
        from whisp.platform.win32.keys import display_chord

        return display_chord(chord)

    def normalize_chord(self, chord: str) -> str:
        """Canonical form of ``chord``. Raises ValueError with a friendly message."""
        from whisp.platform.win32.keys import format_chord, parse_chord

        return format_chord(parse_chord(chord))

    def autostart_status(self) -> str:
        from whisp.platform.win32.autostart import autostart_status

        return autostart_status()

    def set_autostart(self, enabled: bool) -> None:
        from whisp.platform.win32.autostart import set_autostart

        set_autostart(enabled)

    def protect_secret(self, plain: str) -> str:
        from whisp.storage.secrets import protect

        return protect(plain)

    def apply_style(self, settings: Settings, category: str, style: str, text: str) -> str:
        from whisp.core.text.context import TextContext
        from whisp.core.text.rules import apply_rules
        from whisp.core.types import AppCategory

        ctx = TextContext(
            category=AppCategory(category),
            style=style,
            app_name="",
            language=None,
            dictionary=settings.text.dictionary,
            snippets=(),
            remove_fillers=settings.text.remove_fillers,
            voice_commands=False,
            custom_instructions="",
        )
        return apply_rules(text, ctx)

    def user_first_name(self) -> str | None:
        """The signed-in user's first name, from their Windows display name."""
        return _windows_first_name()


@dataclass(frozen=True)
class _Deferred:
    """A method result computed off the GUI thread.

    ``work`` runs on a worker thread. ``finish``, if given, then runs on the GUI
    thread with ``work``'s result and returns the reply value.
    """

    work: Callable[[], Any]
    finish: Callable[[Any], Any] | None = None


class _Workers:
    """A few daemon threads for bridge calls that must not block the GUI thread.

    Daemon threads rather than a ThreadPoolExecutor, so a slow network check can
    never hold up the app when it exits.
    """

    def __init__(self, size: int) -> None:
        self._size = size
        self._jobs: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, job: Callable[[], None]) -> None:
        with self._lock:
            if self._closed:
                raise BridgeError("The window is closing.")
            if not self._threads:
                for index in range(self._size):
                    thread = threading.Thread(target=self._run, name=f"whisp-ui-{index}", daemon=True)
                    thread.start()
                    self._threads.append(thread)
            self._jobs.put(job)

    def shutdown(self) -> None:
        """Let queued jobs finish, then end the threads. Never waits."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for _ in self._threads:
                self._jobs.put(None)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                job()
            except Exception:
                log.exception("A background task for the main window failed")


class _LevelMeter:
    """Runs the microphone level meter on its own thread.

    The level arrives through ``on_level`` on the audio thread. The meter never
    reads frames: MicrophoneStream drops audio nobody reads after about a
    minute, so a meter-only stream stays small. The thread just watches for
    the stop request and for the device failing (``stream.error``).
    """

    def __init__(
        self,
        start_stream: Callable[[str | None, Callable[[float], None]], _Stream],
        publish_level: Callable[[int, float], None],
        publish_state: Callable[[int, str, dict[str, Any]], None],
    ) -> None:
        self._start_stream = start_stream
        self._publish_level = publish_level
        self._publish_state = publish_state
        self._generation = 0
        self._stop: threading.Event | None = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def running(self) -> bool:
        return self._stop is not None

    def start(self, device: str | None) -> None:
        self.stop()
        self._generation += 1
        stop = threading.Event()
        self._stop = stop
        thread = threading.Thread(
            target=self._run, args=(self._generation, device, stop), name="whisp-ui-meter", daemon=True
        )
        thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
            self._stop = None

    def _run(self, generation: int, device: str | None, stop: threading.Event) -> None:
        try:
            stream = self._start_stream(device, lambda level: self._publish_level(generation, level))
        except BridgeError as exc:
            self._publish_state(generation, "error", {"message": str(exc)})
            return
        except Exception:
            log.exception("Could not open the microphone for the level meter")
            self._publish_state(generation, "error", {"message": "Couldn't open the microphone."})
            return
        failure: str | None = None
        try:
            if not stop.is_set():
                self._publish_state(generation, "started", {"device_name": stream.device_name})
            while not stop.wait(0.2):
                failure = stream.error
                if failure:
                    break
        finally:
            try:
                stream.stop()
            except Exception:
                log.exception("Could not close the level meter's microphone stream")
        if failure:
            self._publish_state(generation, "error", {"message": failure})
        else:
            self._publish_state(generation, "stopped", {})


class Bridge(QObject):
    """RPC endpoint for the web UI. Lives on the GUI thread."""

    event = Signal(str, str)
    _posted = Signal(object)  # a callable to run on the GUI thread
    _meter_level = Signal(int, float)
    _meter_state = Signal(int, str, object)

    def __init__(
        self,
        services: AppServices,
        adapters: BridgeAdapters | None = None,
        *,
        theme: str = "dark",
        reduced_motion: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._services = services
        self._adapters = adapters or BridgeAdapters()
        self._theme = theme
        self._reduced_motion = reduced_motion
        self._ready = False
        self._page: tuple[str, str | None] = ("home", None)
        self._closed = False
        self._post_lock = threading.Lock()
        self._call_ids = itertools.count(1)
        self._workers = _Workers(3)
        self._recording_hotkey = False
        self._hotkey_errors: dict[str, str] = self._controller_errors()
        self._status: dict[str, Any] = self._controller_status()
        self._model_progress: dict[str, dict[str, int]] = {}
        self._pending_progress: dict[tuple[str, str], dict[str, Any]] = {}
        self._progress_timer = QTimer(self)
        self._progress_timer.setSingleShot(True)
        self._progress_timer.setInterval(100)
        self._progress_timer.timeout.connect(self._flush_progress)
        self._meter = _LevelMeter(self._adapters.start_microphone, self._post_meter_level, self._post_meter_state)
        self._posted.connect(self._run_posted)
        self._meter_level.connect(self._on_meter_level)
        self._meter_state.connect(self._on_meter_state)
        self._connections: list[tuple[Any, Callable[..., Any]]] = []
        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "app.ready": self._app_ready,
            "app.info": self._app_info_deferred,
            "app.status": lambda _p: self._controller_status(),
            "app.set_paused": self._set_paused,
            "app.open_folder": self._open_folder,
            "app.open_url": self._open_url,
            "app.open_system_settings": self._open_system_settings,
            "app.licenses": self._licenses,
            "settings.get": lambda _p: self._public_settings(),
            "settings.update": self._settings_update,
            "settings.reset": self._settings_reset,
            "hotkeys.display": self._hotkeys_display,
            "hotkeys.validate": self._hotkeys_validate,
            "hotkeys.record_start": self._hotkeys_record_start,
            "hotkeys.record_cancel": self._hotkeys_record_cancel,
            "hotkeys.errors": lambda _p: self._controller_errors(),
            "audio.devices": self._audio_devices,
            "audio.meter_start": self._audio_meter_start,
            "audio.meter_stop": self._audio_meter_stop,
            "models.list": self._models_list,
            "models.download": self._models_download,
            "models.cancel": self._models_cancel,
            "models.delete": self._models_delete,
            "models.select": self._models_select,
            "engine.status": lambda _p: self._engine_json(self._services.asr.status),
            "gpu.info": lambda _p: _Deferred(self._gpu_info),
            "gpu.install_runtime": lambda _p: {"started": self._services.asr.install_gpu_runtime()},
            "gpu.cancel_runtime": lambda _p: self._services.asr.cancel_gpu_runtime(),
            "history.list": self._history_list,
            "history.delete": self._history_delete,
            "history.clear": lambda _p: _Deferred(self._services.history.clear, lambda _r: None),
            "history.copy": self._history_copy,
            "history.apps": lambda _p: _Deferred(lambda: {"apps": list(self._services.history.apps())}),
            "history.processes": self._history_processes,
            "stats.get": lambda _p: _Deferred(lambda: _jsonable(self._services.history.stats())),
            "styles.preview": self._styles_preview,
            "polish.status": self._polish_status,
            "polish.test": self._polish_test,
            "polish.models": self._polish_models,
            "llm.catalog": self._llm_catalog,
            "llm.install": self._llm_install,
            "llm.cancel": self._llm_cancel,
            "llm.uninstall": self._llm_uninstall,
            "autostart.status": lambda _p: {"status": self._adapters.autostart_status()},
            "autostart.set": self._autostart_set,
            "onboarding.complete": self._onboarding_complete,
        }
        self._connect_services()

    # -- public API (GUI thread) ----------------------------------------------

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self._methods)

    def navigate(self, page: str, section: str | None = None) -> None:
        if page not in PAGES:
            raise ValueError(f"unknown page: {page}")
        self._page = (page, section)
        if self._ready:
            self._emit("navigate", {"page": page, "section": section})

    def set_theme(self, theme: str) -> None:
        if theme != self._theme:
            self._theme = theme
            self._emit("theme.changed", {"theme": theme})

    def set_reduced_motion(self, reduced: bool) -> None:
        if reduced != self._reduced_motion:
            self._reduced_motion = reduced
            self._emit("motion.changed", {"reduced": reduced})

    def page_reloaded(self) -> None:
        """The page is loading again (e.g. after a renderer crash): wait for app.ready."""
        self._ready = False
        self._meter.stop()
        self._cancel_hotkey_recording()

    def shutdown(self) -> None:
        """Stop everything this window started. Safe to call more than once."""
        with self._post_lock:
            if self._closed:
                return
            self._closed = True
        self._meter.stop()
        self._cancel_hotkey_recording()
        self._progress_timer.stop()
        self._workers.shutdown()
        for signal, slot in self._connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                log.debug("Service signal was already disconnected")
        self._connections.clear()

    # -- the channel's slot ---------------------------------------------------

    @Slot(str, str, result=str)
    def call(self, method: str, params_json: str) -> str:
        handler = self._methods.get(method)
        if handler is None:
            return _dumps({"error": f"Unknown method: {method}"})
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError):
            return _dumps({"error": "The request was not valid JSON."})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _dumps({"error": "The request must be a JSON object."})
        try:
            result = handler(params)
        except BridgeError as exc:
            return _dumps({"error": str(exc)})
        except Exception:
            log.exception("Bridge method %s failed", method)
            return _dumps({"error": GENERIC_ERROR})
        if isinstance(result, _Deferred):
            try:
                call_id = self._start_deferred(method, result)
            except BridgeError as exc:
                return _dumps({"error": str(exc)})
            return _dumps({"pending": call_id})
        try:
            return _dumps({"ok": result})
        except (TypeError, ValueError):
            log.exception("Bridge method %s returned a value that is not JSON", method)
            return _dumps({"error": GENERIC_ERROR})

    # -- plumbing -----------------------------------------------------------------

    def _emit(self, name: str, payload: Any) -> None:
        if self._closed:
            return
        try:
            text = _dumps(payload)
        except (TypeError, ValueError):
            log.exception("Could not encode the %s event", name)
            return
        self.event.emit(name, text)

    def _post(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the GUI thread. Any thread; dropped once the window is closing."""
        with self._post_lock:
            if not self._closed:
                self._posted.emit(fn)

    def _run_posted(self, fn: Callable[[], None]) -> None:
        if not self._closed:
            fn()

    def _start_deferred(self, method: str, deferred: _Deferred) -> int:
        call_id = next(self._call_ids)

        def job() -> None:
            try:
                outcome: tuple[str, Any] = ("ok", deferred.work())
            except BridgeError as exc:
                outcome = ("error", str(exc))
            except Exception:
                log.exception("Bridge method %s failed", method)
                outcome = ("error", GENERIC_ERROR)
            self._post(lambda: self._finish_deferred(call_id, method, deferred, outcome))

        self._workers.submit(job)
        return call_id

    def _finish_deferred(self, call_id: int, method: str, deferred: _Deferred, outcome: tuple[str, Any]) -> None:
        kind, value = outcome
        if kind == "ok" and deferred.finish is not None:
            try:
                value = deferred.finish(value)
            except BridgeError as exc:
                kind, value = "error", str(exc)
            except Exception:
                log.exception("Bridge method %s failed", method)
                kind, value = "error", GENERIC_ERROR
        try:
            _dumps(value)
        except (TypeError, ValueError):
            log.exception("Bridge method %s returned a value that is not JSON", method)
            kind, value = "error", GENERIC_ERROR
        self._emit("call.result", {"id": call_id, kind: value})

    def _listen(self, signal: Any, slot: Callable[..., Any]) -> None:
        signal.connect(slot)
        self._connections.append((signal, slot))

    def _connect_services(self) -> None:
        s = self._services
        self._listen(s.settings.changed, self._on_settings_changed)
        self._listen(s.asr.status_changed, self._on_engine_status)
        self._listen(s.models.progress, self._on_model_progress)
        self._listen(s.models.changed, self._on_models_changed)
        self._listen(s.models.download_failed, self._on_model_failed)
        self._listen(s.history.added, self._on_history_added)
        self._listen(s.history.changed, self._on_history_changed)
        self._listen(s.hotkeys.recorded, self._on_hotkey_recorded)
        self._listen(s.controller.state_changed, self._on_controller_changed)
        self._listen(s.llm.status_changed, self._on_llm_status)
        self._listen(s.llm.progress, self._on_llm_progress)
        self._listen(s.llm.install_finished, self._on_llm_installed)
        self._listen(s.llm.install_failed, self._on_llm_failed)
        self._listen(s.asr.gpu_runtime_progress, self._on_gpu_progress)
        self._listen(s.asr.gpu_runtime_finished, self._on_gpu_installed)
        self._listen(s.asr.gpu_runtime_failed, self._on_gpu_failed)

    def _controller_errors(self) -> dict[str, str]:
        """Shortcuts that couldn't be registered, as AppController reports them."""
        errors = self._services.controller.binding_errors
        return {str(action): str(message) for action, message in errors.items() if message}

    def _controller_status(self) -> dict[str, Any]:
        controller = self._services.controller
        return {"paused": bool(controller.is_paused), "problem": controller.problem()}

    # -- service events -------------------------------------------------------------

    def _on_settings_changed(self, sections: object) -> None:
        names = sorted(str(s) for s in sections) if isinstance(sections, Iterable) else []
        self._emit("settings.changed", {"settings": self._public_settings(), "sections": names})

    def _on_controller_changed(self) -> None:
        # state_changed fires on every recording too; only pass on what the window shows.
        errors = self._controller_errors()
        if errors != self._hotkey_errors:
            self._hotkey_errors = errors
            self._emit("hotkeys.errors", dict(errors))
        status = self._controller_status()
        if status != self._status:
            self._status = status
            self._emit("app.status", status)

    def _on_engine_status(self, status: EngineStatus) -> None:
        self._emit("engine.status", self._engine_json(status))

    def _on_model_progress(self, model_id: str, done: object, total: object) -> None:
        done_i, total_i = _as_int(done), _as_int(total)
        self._model_progress[model_id] = {"done": done_i, "total": total_i}
        self._queue_progress("models.progress", model_id, {"id": model_id, "done": done_i, "total": total_i})

    def _on_models_changed(self) -> None:
        for model_id in list(self._model_progress):
            if not self._services.models.is_downloading(model_id):
                self._forget_model_progress(model_id)
        self._emit("models.changed", {})

    def _on_model_failed(self, model_id: str, message: str) -> None:
        self._forget_model_progress(model_id)
        self._emit("models.failed", {"id": model_id, "message": message})

    def _forget_model_progress(self, model_id: str) -> None:
        """A download ended: drop its progress, including an update still waiting to be sent.

        Updates go out in batches, so the last one would otherwise reach the page after the download's end and
        show it downloading again, at 100%.
        """
        self._model_progress.pop(model_id, None)
        self._pending_progress.pop(("models.progress", model_id), None)

    def _on_history_added(self, entry: Any) -> None:
        self._emit("history.added", {"entry": _entry_json(entry)})

    def _on_history_changed(self) -> None:
        self._emit("history.changed", {})

    def _on_hotkey_recorded(self, chord: str) -> None:
        if not self._recording_hotkey:
            return
        self._recording_hotkey = False
        try:
            keys = self._adapters.display_chord(chord)
        except ValueError:
            keys = []
        self._emit("hotkeys.recorded", {"chord": chord, "keys": keys})

    def _on_llm_status(self, status: Any) -> None:
        self._emit("llm.status", _llm_status_json(status))

    def _on_llm_progress(self, model_id: str, done: object, total: object) -> None:
        payload = {"id": model_id, "done": _as_int(done), "total": _as_int(total)}
        self._queue_progress("llm.progress", model_id, payload)

    def _gpu_info(self) -> dict[str, Any]:
        """Worker thread: GPU status plus whether the GPU runtime pack is downloading."""
        return {**_gpu_json(self._adapters.detect_gpu()), "installing": self._services.asr.gpu_runtime_installing}

    def _on_gpu_progress(self, done: object, total: object) -> None:
        self._queue_progress("gpu.progress", "runtime", {"done": _as_int(done), "total": _as_int(total)})

    def _on_gpu_installed(self, gpu: GpuInfo) -> None:
        self._pending_progress.pop(("gpu.progress", "runtime"), None)
        self._emit("gpu.installed", {"gpu": {**_gpu_json(gpu), "installing": False}})

    def _on_gpu_failed(self, message: str) -> None:
        self._pending_progress.pop(("gpu.progress", "runtime"), None)
        self._emit("gpu.failed", {"message": message})

    def _on_llm_installed(self, model_id: str) -> None:
        self._pending_progress.pop(("llm.progress", model_id), None)
        self._emit("llm.installed", {"id": model_id})

    def _on_llm_failed(self, model_id: str, message: str) -> None:
        self._pending_progress.pop(("llm.progress", model_id), None)
        self._emit("llm.failed", {"id": model_id, "message": message})

    def _queue_progress(self, event: str, key: str, payload: dict[str, Any]) -> None:
        """Send progress at most ten times a second per item, always ending on the latest value."""
        self._pending_progress[(event, key)] = payload
        if not self._progress_timer.isActive():
            self._progress_timer.start()

    def _flush_progress(self) -> None:
        pending, self._pending_progress = self._pending_progress, {}
        for (event, _key), payload in pending.items():
            self._emit(event, payload)

    def _post_meter_level(self, generation: int, level: float) -> None:
        with self._post_lock:
            if not self._closed:
                self._meter_level.emit(generation, float(level))

    def _post_meter_state(self, generation: int, state: str, info: dict[str, Any]) -> None:
        with self._post_lock:
            if not self._closed:
                self._meter_state.emit(generation, state, info)

    def _on_meter_level(self, generation: int, level: float) -> None:
        if generation == self._meter.generation and self._meter.running:
            self._emit("audio.level", {"level": round(max(0.0, min(1.0, level)), 3)})

    def _on_meter_state(self, generation: int, state: str, info: object) -> None:
        if generation != self._meter.generation:
            return
        if state in ("error", "stopped") and self._meter.running:
            self._meter.stop()
        payload = {"state": state}
        if isinstance(info, dict):
            payload.update(info)
        self._emit("audio.meter", payload)

    # -- app ------------------------------------------------------------------------

    def _app_ready(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._ready = True
        page, section = self._page
        return {
            "info": _basic_info(),
            "settings": self._public_settings(),
            "theme": self._theme,
            "reduced_motion": self._reduced_motion,
            "page": page,
            "section": section,
            "engine": self._engine_json(self._services.asr.status),
            "status": self._controller_status(),
        }

    def _set_paused(self, params: dict[str, Any]) -> dict[str, Any]:
        paused = _req_bool(params, "paused")
        if bool(self._services.controller.is_paused) != paused:
            self._services.controller.toggle_paused()
        return self._controller_status()

    def _app_info_deferred(self, _params: dict[str, Any]) -> _Deferred:
        def work() -> dict[str, Any]:
            info = _basic_info()
            info["first_name"] = self._adapters.user_first_name()
            return info

        return _Deferred(work)

    def _open_folder(self, params: dict[str, Any]) -> None:
        paths = self._services.paths
        folders = {
            "logs": paths.logs_dir,
            "data": paths.data_dir,
            "config": paths.config_dir,
            "models": paths.models_dir,
        }
        name = _req_str(params, "name")
        if name not in folders:
            raise BridgeError(f"Unknown folder: {name}")
        folder = folders[name]
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BridgeError("Couldn't open that folder.") from exc
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            raise BridgeError("Couldn't open that folder.")

    def _open_url(self, params: dict[str, Any]) -> None:
        url = QUrl(_req_str(params, "url"), QUrl.ParsingMode.StrictMode)
        if not url.isValid() or url.scheme() not in ("http", "https") or not url.host():
            raise BridgeError("Only web links can be opened.")
        if not QDesktopServices.openUrl(url):
            raise BridgeError("Couldn't open your web browser.")

    def _open_system_settings(self, params: dict[str, Any]) -> None:
        uri = SYSTEM_SETTINGS_URIS.get(_req_str(params, "page"))
        if uri is None:
            raise BridgeError("Unknown Windows settings page.")
        if not QDesktopServices.openUrl(QUrl(uri)):
            raise BridgeError("Couldn't open Windows Settings.")

    def _licenses(self, _params: dict[str, Any]) -> _Deferred:
        def work() -> dict[str, Any]:
            from importlib import metadata

            components = []
            for name, licence, url, parts in BUNDLED_NOTICES:
                try:
                    text = resource_path(*parts).read_text(encoding="utf-8")
                except OSError:
                    log.warning("Licence file for %s is missing", name)
                    continue
                components.append({"name": name, "license": licence, "url": url, "text": text})
            for dist, name, licence, url in RUNTIME_LIBRARIES:
                try:
                    version = metadata.version(dist)
                except metadata.PackageNotFoundError:
                    version = None
                components.append({"name": name, "license": licence, "url": url, "version": version})
            models = []
            try:
                for info in self._adapters.model_catalog():
                    models.append(
                        {
                            "name": info.name,
                            "license": info.license,
                            "url": info.license_url,
                            "attribution": getattr(info, "attribution", ""),
                        }
                    )
            except ImportError:
                log.warning("Speech model catalog is unavailable")
            ai_models = [
                {"name": info.name, "license": info.license, "url": info.license_url}
                for info in self._services.llm.catalog()
            ]
            ai_models.append(
                {
                    "name": "llama.cpp server",
                    "license": "MIT",
                    "url": "https://github.com/ggml-org/llama.cpp",
                    "attribution": "Runs the AI polish models. Downloaded only if you turn on built-in AI polish.",
                }
            )
            return {"components": components, "models": models, "ai_models": ai_models}

        return _Deferred(work)

    # -- settings -------------------------------------------------------------------

    def _public_settings(self) -> dict[str, Any]:
        data = self._services.settings.as_dict()
        polish = data.get("polish", {})
        polish["openai_api_key_saved"] = bool(polish.pop("openai_api_key_protected", ""))
        return data

    def _settings_update(self, patch: dict[str, Any]) -> dict[str, Any]:
        patch = json.loads(json.dumps(patch))  # a private deep copy
        polish = patch.get("polish")
        if polish is not None and not isinstance(polish, dict):
            raise BridgeError("The request was not valid.")
        if isinstance(polish, dict):
            polish.pop("openai_api_key_protected", None)
            polish.pop("openai_api_key_saved", None)
            if "openai_api_key" in polish:
                key = polish.pop("openai_api_key")
                if not isinstance(key, str):
                    raise BridgeError("The API key must be text.")
                key = key.strip()
                if len(key) > 4096:
                    raise BridgeError("That API key is too long.")
                try:
                    polish["openai_api_key_protected"] = self._adapters.protect_secret(key) if key else ""
                except Exception as exc:
                    log.exception("Could not encrypt the API key")
                    raise BridgeError("Couldn't save the API key securely.") from exc
            if not polish:
                patch.pop("polish")
        try:
            self._services.settings.update(patch)
        except ValueError as exc:
            log.warning("Rejected a settings change: %s", exc)
            raise BridgeError("That setting couldn't be saved. Check the value and try again.") from exc
        return self._public_settings()

    def _settings_reset(self, params: dict[str, Any]) -> dict[str, Any]:
        section = params.get("section")
        if section is not None and section not in SECTIONS:
            raise BridgeError(f"Unknown settings section: {section}")
        self._services.settings.reset(section)
        return self._public_settings()

    def _onboarding_complete(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._services.settings.update({"general": {"onboarding_completed": True}})
        return self._public_settings()

    def _autostart_set(self, params: dict[str, Any]) -> dict[str, Any]:
        enabled = _req_bool(params, "enabled")
        try:
            self._adapters.set_autostart(enabled)
        except OSError as exc:
            log.exception("Could not change the startup setting")
            raise BridgeError("Windows didn't let Whisp change its startup setting.") from exc
        self._services.settings.update({"general": {"launch_at_startup": enabled}})
        return {"status": self._adapters.autostart_status()}

    # -- hotkeys --------------------------------------------------------------------

    def _hotkeys_display(self, params: dict[str, Any]) -> dict[str, Any]:
        chord = _req_str(params, "chord", allow_empty=True)
        if not chord:
            return {"keys": []}
        try:
            return {"keys": self._adapters.display_chord(chord)}
        except ValueError as exc:
            raise BridgeError(str(exc)) from exc

    def _hotkeys_validate(self, params: dict[str, Any]) -> dict[str, Any]:
        chord = _req_str(params, "chord")
        action = _req_str(params, "action")
        if action not in ACTION_LABELS:
            raise BridgeError(f"Unknown shortcut: {action}")
        try:
            canonical = self._adapters.normalize_chord(chord)
        except ValueError as exc:
            return {"valid": False, "message": str(exc)}
        if canonical in ("esc", "escape"):
            return {"valid": False, "message": "Esc is reserved for cancelling a dictation."}
        bindings = self._services.settings.current.hotkeys
        for other, label in ACTION_LABELS.items():
            value = getattr(bindings, other)
            if other == action or not value:
                continue
            try:
                taken = self._adapters.normalize_chord(value)
            except ValueError:
                continue
            if taken == canonical:
                return {"valid": False, "message": f"Already used for {label}."}
        return {"valid": True, "chord": canonical, "keys": self._adapters.display_chord(canonical)}

    def _hotkeys_record_start(self, _params: dict[str, Any]) -> None:
        self._services.hotkeys.start_recording()
        self._recording_hotkey = True

    def _hotkeys_record_cancel(self, _params: dict[str, Any]) -> None:
        self._cancel_hotkey_recording()

    def _cancel_hotkey_recording(self) -> None:
        """Hand the keyboard back to HotkeyService if this window was recording a shortcut."""
        if not self._recording_hotkey:
            return
        self._recording_hotkey = False
        try:
            self._services.hotkeys.stop_recording()
        except Exception:
            log.exception("Could not stop recording a shortcut")

    # -- audio ----------------------------------------------------------------------

    def _audio_devices(self, _params: dict[str, Any]) -> _Deferred:
        def work() -> dict[str, Any]:
            devices = self._adapters.list_input_devices()
            return {
                "devices": [
                    {"id": d.id, "name": d.name, "host_api": d.host_api, "is_default": d.is_default} for d in devices
                ],
                "selected": self._services.settings.current.audio.input_device,
            }

        return _Deferred(work)

    def _audio_meter_start(self, params: dict[str, Any]) -> None:
        device = params.get("device")
        if device is not None and not isinstance(device, str):
            raise BridgeError("The request was not valid.")
        self._meter.start(device or None)

    def _audio_meter_stop(self, _params: dict[str, Any]) -> None:
        self._meter.stop()

    # -- speech models ------------------------------------------------------------

    def _catalog(self) -> tuple[ModelInfo, ...]:
        try:
            return tuple(self._adapters.model_catalog())
        except ImportError as exc:
            raise BridgeError("The speech model list isn't available.") from exc

    def _model_info(self, model_id: str) -> ModelInfo:
        for info in self._catalog():
            if info.id == model_id:
                return info
        raise BridgeError("That speech model isn't in Whisp's catalog.")

    def _models_list(self, _params: dict[str, Any]) -> _Deferred:
        catalog = self._catalog()
        manager = self._services.models

        def work() -> dict[str, Any]:
            gpu = self._adapters.detect_gpu()
            current = self._services.settings.current.transcription
            models = []
            for info in catalog:
                variants = [
                    {
                        "key": v.key,
                        "quantization": v.quantization,
                        "size": sum(f.size for f in v.files),
                        "installed": bool(manager.is_installed(info.id, v.key)),
                    }
                    for v in info.variants
                ]
                models.append(
                    {
                        "id": info.id,
                        "name": info.name,
                        "description": info.description,
                        "languages": list(info.languages),
                        "accuracy": info.accuracy,
                        "speed": info.speed,
                        "license": info.license,
                        "license_url": info.license_url,
                        "recommended": info.recommended,
                        "variants": variants,
                        "installed": any(v["installed"] for v in variants),
                        "preferred_variant": _preferred_variant(info, current.device, gpu),
                    }
                )
            return {
                "models": models,
                "selected": current.model_id,
                "device": current.device,
                "recommended": _recommendation(catalog, gpu),
                "gpu": {**_gpu_json(gpu), "installing": self._services.asr.gpu_runtime_installing},
            }

        def finish(result: dict[str, Any]) -> dict[str, Any]:
            state = manager.install_state()
            for model in result["models"]:
                entry = state.get(model["id"], {}) if isinstance(state, dict) else {}
                progress = self._model_progress.get(model["id"])
                downloading = bool(entry.get("downloading")) if isinstance(entry, dict) else False
                model["downloading"] = downloading or progress is not None
                model["progress"] = progress
            return result

        return _Deferred(work, finish)

    def _models_download(self, params: dict[str, Any]) -> None:
        info = self._model_info(_req_str(params, "id"))
        variant = params.get("variant")
        keys = [v.key for v in info.variants]
        if variant is None:
            variant = _preferred_variant(info, self._services.settings.current.transcription.device, None)
        if variant not in keys:
            raise BridgeError("That version of the model isn't available.")
        size = next(sum(f.size for f in v.files) for v in info.variants if v.key == variant)
        self._model_progress[info.id] = {"done": 0, "total": size}
        self._services.models.download(info.id, variant)

    def _models_cancel(self, params: dict[str, Any]) -> None:
        model_id = _req_str(params, "id")
        self._forget_model_progress(model_id)
        self._services.models.cancel(model_id)

    def _models_delete(self, params: dict[str, Any]) -> _Deferred:
        info = self._model_info(_req_str(params, "id"))
        if info.id == self._services.settings.current.transcription.model_id:
            raise BridgeError("Whisp is using this model. Choose another one before deleting it.")

        def work() -> None:
            try:
                self._services.models.delete(info.id)
            except OSError as exc:
                log.exception("Could not delete speech model %s", info.id)
                raise BridgeError("Couldn't delete the model files. Close anything using them and try again.") from exc

        return _Deferred(work)

    def _models_select(self, params: dict[str, Any]) -> _Deferred:
        info = self._model_info(_req_str(params, "id"))
        manager = self._services.models

        def work() -> bool:
            # Checking the files touches the disk, so it runs off the GUI thread.
            return any(manager.is_installed(info.id, v.key) for v in info.variants)

        def finish(installed: bool) -> dict[str, Any]:
            if not installed:
                raise BridgeError("Download this model before using it.")
            self._services.settings.update({"transcription": {"model_id": info.id}})
            return self._public_settings()

        return _Deferred(work, finish)

    def _engine_json(self, status: EngineStatus) -> dict[str, Any]:
        name = None
        if status.model_id:
            try:
                name = next((m.name for m in self._adapters.model_catalog() if m.id == status.model_id), None)
            except ImportError:
                name = None
        return {
            "state": status.state.value,
            "model_id": status.model_id,
            "model_name": name,
            "device": status.device,
            "device_name": status.device_name,
            "message": status.message,
        }

    # -- history --------------------------------------------------------------------

    def _history_list(self, params: dict[str, Any]) -> _Deferred:
        query = _opt_str(params, "query", max_len=200)
        app = _opt_str(params, "app", max_len=200)
        limit = _opt_int(params, "limit", default=50, low=1, high=200)
        offset = _opt_int(params, "offset", default=0, low=0, high=10_000_000)
        history = self._services.history

        def work() -> dict[str, Any]:
            entries = history.list(query=query, app=app, limit=limit + 1, offset=offset)
            return {
                "entries": [_entry_json(e) for e in entries[:limit]],
                "has_more": len(entries) > limit,
                # False when the history file couldn't be opened and this session's history lives in memory.
                "persistent": bool(history.persistent),
            }

        return _Deferred(work)

    def _history_delete(self, params: dict[str, Any]) -> _Deferred:
        entry_id = _req_int(params, "id")
        return _Deferred(lambda: self._services.history.delete(entry_id), lambda _r: None)

    def _history_copy(self, params: dict[str, Any]) -> _Deferred:
        entry_id = _req_int(params, "id")

        def work() -> None:
            # On the worker thread: the clipboard may be busy for a moment. Whisp's own clipboard writes
            # keep dictations out of Windows clipboard history and cloud clipboard sync.
            entry = self._services.history.get(entry_id)
            if entry is None:
                raise BridgeError("That dictation no longer exists.")
            if not self._services.inserter.copy_to_clipboard(entry.final_text):
                raise BridgeError("Couldn't copy to the clipboard. Please try again.")

        return _Deferred(work)

    def _history_processes(self, _params: dict[str, Any]) -> _Deferred:
        def work() -> dict[str, Any]:
            seen: dict[str, str] = {}
            for entry in self._services.history.list(limit=500):
                process = (entry.process_name or "").strip().lower()
                if process and process not in seen:
                    seen[process] = entry.app_name or process
                if len(seen) >= 30:
                    break
            return {"apps": [{"process": p, "app": a} for p, a in seen.items()]}

        return _Deferred(work)

    # -- styles & polish -------------------------------------------------------------

    def _styles_preview(self, params: dict[str, Any]) -> dict[str, Any]:
        category = _req_str(params, "category")
        if category not in STYLE_SAMPLES:
            raise BridgeError(f"Unknown category: {category}")
        sample = STYLE_SAMPLES[category]
        settings = self._services.settings.current
        try:
            styles = {style: self._adapters.apply_style(settings, category, style, sample) for style in STYLE_IDS}
        except ImportError as exc:
            raise BridgeError("Style previews aren't available.") from exc
        return {"before": sample, "styles": styles}

    def _polish_status(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"ready": bool(self._services.polisher.is_ready()), "llm": self._llm_status()}

    def _polish_test(self, _params: dict[str, Any]) -> _Deferred:
        def work() -> dict[str, Any]:
            ok, message = self._services.polisher.test()
            return {"ok": bool(ok), "message": str(message)}

        return _Deferred(work)

    def _polish_models(self, params: dict[str, Any]) -> _Deferred:
        provider = _req_str(params, "provider")
        if provider not in ("ollama", "openai"):
            raise BridgeError(f"Unknown provider: {provider}")

        def work() -> dict[str, Any]:
            try:
                models = self._services.polisher.list_models(provider)
            except Exception as exc:
                # Usually the server isn't running or the address is wrong.
                log.warning("Could not list %s models (%s)", provider, type(exc).__name__)
                raise BridgeError("Couldn't reach the server. Check the address and that it's running.") from exc
            return {"models": sorted({str(m) for m in models}, key=str.lower)}

        return _Deferred(work)

    def _llm_status(self) -> dict[str, Any]:
        return _llm_status_json(self._services.llm.status)

    def _llm_catalog(self, _params: dict[str, Any]) -> _Deferred:
        llm = self._services.llm

        def work() -> list[dict[str, Any]]:
            # is_installed reads a marker file and checks sizes, so it stays off the GUI thread.
            return [_llm_model_json(info, installed=llm.is_installed(info.id)) for info in llm.catalog()]

        return _Deferred(work, lambda models: {"models": models, "status": self._llm_status()})

    def _llm_model_id(self, params: dict[str, Any]) -> str:
        """The requested AI model id, only if it's one Whisp offers (it becomes part of a folder path)."""
        model_id = _req_str(params, "id")
        if model_id not in {info.id for info in self._services.llm.catalog()}:
            raise BridgeError("That AI model isn't available in this version of Whisp.")
        return model_id

    def _llm_install(self, params: dict[str, Any]) -> None:
        self._services.llm.install(self._llm_model_id(params))

    def _llm_cancel(self, _params: dict[str, Any]) -> None:
        self._services.llm.cancel_install()

    def _llm_uninstall(self, params: dict[str, Any]) -> _Deferred:
        model_id = self._llm_model_id(params)
        return _Deferred(lambda: self._services.llm.uninstall(model_id), lambda _r: None)


# -- helpers ---------------------------------------------------------------------------


def _dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _as_int(value: object) -> int:
    try:
        return max(0, int(value))  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


def _req_str(params: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = params.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise BridgeError(f"The request is missing '{key}'.")
    return value


def _opt_str(params: dict[str, Any], key: str, *, max_len: int) -> str:
    value = params.get(key) or ""
    if not isinstance(value, str):
        raise BridgeError(f"'{key}' must be text.")
    return value.strip()[:max_len]


def _req_int(params: dict[str, Any], key: str) -> int:
    value = params.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgeError(f"The request is missing '{key}'.")
    return value


def _opt_int(params: dict[str, Any], key: str, *, default: int, low: int, high: int) -> int:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgeError(f"'{key}' must be a whole number.")
    return max(low, min(high, value))


def _req_bool(params: dict[str, Any], key: str) -> bool:
    value = params.get(key)
    if not isinstance(value, bool):
        raise BridgeError(f"The request is missing '{key}'.")
    return value


def _basic_info() -> dict[str, Any]:
    from PySide6.QtWebEngineCore import qWebEngineChromiumVersion

    version = sys.getwindowsversion()
    return {
        "name": APP_NAME,
        "version": __version__,
        "qt": qVersion(),
        "chromium": qWebEngineChromiumVersion(),
        "python": platform.python_version(),
        "windows": f"{version.major}.{version.minor}.{version.build}",
        "frozen": bool(getattr(sys, "frozen", False)),
        "data_dir": str(app_paths().data_dir),
    }


def _entry_json(entry: Any) -> dict[str, Any]:
    raw = entry.raw_text or ""
    return {
        "id": entry.id,
        "created_at": entry.created_at,
        "kind": entry.kind,
        "app_name": entry.app_name,
        "process_name": entry.process_name,
        "category": entry.category,
        "text": entry.final_text,
        "raw_text": raw if raw != entry.final_text else "",
        "word_count": entry.word_count,
        "audio_ms": entry.audio_ms,
        "speech_ms": entry.speech_ms,
        "latency_ms": entry.latency_ms,
        "polished": entry.polished,
        "outcome": entry.outcome,
    }


def _gpu_json(gpu: GpuInfo) -> dict[str, Any]:
    return {
        "has_nvidia": gpu.has_nvidia,
        "name": gpu.name,
        "vram_mb": gpu.vram_mb,
        "cuda_ready": gpu.cuda_ready,
        "detail": gpu.detail,
        "runtime_download_size": gpu.runtime_download_size,
    }


def _preferred_variant(info: ModelInfo, device: str, gpu: GpuInfo | None) -> str:
    """The variant to download for the user's device setting (``auto`` follows the GPU check)."""
    keys = [v.key for v in info.variants]
    wants_gpu = device == "gpu" or (device == "auto" and gpu is not None and gpu.cuda_ready)
    if wants_gpu and "gpu" in keys:
        return "gpu"
    if "cpu" in keys:
        return "cpu"
    return keys[0]


def _recommendation(catalog: tuple[ModelInfo, ...], gpu: GpuInfo) -> dict[str, str] | None:
    """The model to suggest first: a recommended one that suits this PC's hardware."""
    if not catalog:
        return None
    wanted = "gpu" if gpu.cuda_ready else "cpu"
    candidates = [m for m in catalog if m.recommended] or list(catalog)
    for info in candidates:
        if any(v.key == wanted for v in info.variants):
            return {"id": info.id, "variant": wanted}
    first = candidates[0]
    return {"id": first.id, "variant": _preferred_variant(first, "auto", gpu)}


def _llm_status_json(status: LlmStatus) -> dict[str, Any]:
    return {
        "state": status.state.value,
        "model_id": status.model_id,
        "message": status.message,
        "backend": status.backend,
        "installing": status.installing,
    }


def _llm_model_json(info: LlmModelInfo, *, installed: bool) -> dict[str, Any]:
    return {
        "id": info.id,
        "name": info.name,
        "description": info.description,
        "size": info.download_size,
        "license": info.license,
        "license_url": info.license_url,
        "multilingual": info.multilingual,
        "vram_mb": info.vram_mb,
        "ram_mb": info.ram_mb,
        "recommended": info.recommended,
        "installed": installed,
    }


_NAME_DISPLAY = 3


def _windows_first_name() -> str | None:
    """First name from GetUserNameExW(NameDisplay), or None when Windows doesn't know it."""
    try:
        secur32 = ctypes.WinDLL("secur32", use_last_error=True)
        get_name = secur32.GetUserNameExW
        get_name.argtypes = [ctypes.c_int, wintypes.LPWSTR, ctypes.POINTER(wintypes.ULONG)]
        get_name.restype = wintypes.BOOLEAN
        size = wintypes.ULONG(256)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not get_name(_NAME_DISPLAY, buffer, ctypes.byref(size)):
            return None
        display = buffer.value.strip()
    except OSError:
        return None
    if "," in display:  # "Surname, Given"
        display = display.split(",", 1)[1].strip()
    first = display.split()[0] if display.split() else ""
    if not first or len(first) > 32 or not any(ch.isalpha() for ch in first) or "\\" in first or "@" in first:
        return None
    return first
