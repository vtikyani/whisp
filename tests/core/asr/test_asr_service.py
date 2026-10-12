"""AsrService with a fake engine: ordering, priorities, coalescing, exactly one callback, reloads, shutdown."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from pytestqt.qtbot import QtBot

from whisp.core.asr import service as service_module
from whisp.core.asr.catalog import PARAKEET_V2, PARAKEET_V3, ModelInfo
from whisp.core.asr.engine import EngineError, EngineSpec
from whisp.core.asr.gpu import GpuInfo
from whisp.core.asr.service import (
    MSG_CANCELLED,
    MSG_CLOSING,
    MSG_GPU_LOST,
    MSG_RECOGNITION_FAILED,
    MSG_UNKNOWN_MODEL,
    AsrService,
)
from whisp.core.types import AsrResult, EngineState, Segment
from whisp.storage.settings import SettingsStore

WAIT_MS = 10_000
GPU_READY = GpuInfo(True, "NVIDIA Test GPU", 12000, True, "ready")
GPU_MISSING = GpuInfo(False, None, None, False, "No NVIDIA graphics card found.")


class FakeModels(QObject):
    """The parts of ModelManager the service uses, with installs switched on and off by the test."""

    changed = Signal()

    def __init__(self, installed: dict[str, set[str]]) -> None:
        super().__init__()
        self.catalog = {info.id: info for info in (PARAKEET_V2, PARAKEET_V3)}
        self.installed = installed
        self.retained: Counter[str] = Counter()
        self._lock = threading.Lock()

    def model_info(self, model_id: str) -> ModelInfo:
        return self.catalog[model_id]

    def installed_variants(self, model_id: str) -> list[str]:
        info = self.catalog[model_id]
        return [v.key for v in info.variants if v.key in self.installed.get(model_id, set())]

    def model_path(self, model_id: str, variant: str) -> Path:
        return Path("/models") / model_id / variant

    def retain(self, model_id: str, variant: str) -> bool:
        with self._lock:
            if variant not in self.installed.get(model_id, set()):
                return False
            self.retained[model_id] += 1
            return True

    def release(self, model_id: str) -> None:
        with self._lock:
            self.retained[model_id] -= 1


@dataclass
class Script:
    """Controls and records the fake engines."""

    specs: list[EngineSpec] = field(default_factory=list)
    engines: list[FakeEngine] = field(default_factory=list)
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    """(kind, label, language) per transcription: kind is "seg" or "preview"."""
    load_gate: threading.Event | None = None
    load_error: Exception | None = None
    gate: threading.Event | None = None
    """When set to an Event, the next transcription waits for it (then the gate is cleared)."""
    entered: threading.Event = field(default_factory=threading.Event)
    fail_next: int = 0
    """How many of the next transcriptions fail (``True`` counts as one)."""
    preview_text: str = "preview"
    notice: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def factory(self, spec: EngineSpec) -> FakeEngine:
        self.specs.append(spec)
        if self.load_gate is not None:
            assert self.load_gate.wait(10)
        if self.load_error is not None:
            error, self.load_error = self.load_error, None
            raise error
        engine = FakeEngine(self, spec)
        self.engines.append(engine)
        return engine


class FakeEngine:
    def __init__(self, script: Script, spec: EngineSpec) -> None:
        self._script = script
        self.spec = spec
        self.device = "cuda" if spec.use_cuda else "cpu"
        self.device_name = "NVIDIA Test GPU" if spec.use_cuda else "CPU"
        self.variant = "gpu" if spec.use_cuda else ("cpu" if "cpu" in spec.folders else "gpu")
        self.notice = script.notice
        self.closed = False

    def transcribe(self, samples: np.ndarray, language: str) -> tuple[str, str | None]:
        script = self._script
        label = _label(samples)
        kind = "preview" if label.startswith("p") else "seg"
        with script.lock:
            script.calls.append((kind, label, language))
            gate, script.gate = script.gate, None
            fail = script.fail_next > 0
            if fail:
                script.fail_next -= 1
        script.entered.set()
        if gate is not None:
            assert gate.wait(10)
        if fail:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        if kind == "preview":
            return script.preview_text, language
        return f"text {label}", "en"

    def close(self) -> None:
        self.closed = True


def _label(samples: np.ndarray) -> str:
    code = round(float(samples[0]))
    return f"p{-code}" if code < 0 else f"s{code}"


def segment(session: str, index: int, *, final: bool = False, code: int | None = None) -> Segment:
    marker = float(code if code is not None else index)
    return Segment(session, index, np.full(8000, marker, dtype=np.float32), 400, final)


def preview_audio(code: int) -> np.ndarray:
    return np.full(8000, -float(code), dtype=np.float32)


class Results:
    """Collects callbacks with the thread they ran on; counts calls per job."""

    def __init__(self) -> None:
        self.items: list[AsrResult] = []
        self.threads: list[str] = []
        self.previews: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def callback(self) -> Callable[[AsrResult], None]:
        def receive(result: AsrResult) -> None:
            with self._lock:
                self.items.append(result)
                self.threads.append(threading.current_thread().name)

        return receive

    def preview(self, session: str) -> Callable[[str], None]:
        def receive(text: str) -> None:
            with self._lock:
                self.previews.append((session, text))

        return receive

    def keys(self) -> list[tuple[str, int]]:
        with self._lock:
            return [(r.session_id, r.index) for r in self.items]

    def by_key(self) -> dict[tuple[str, int], AsrResult]:
        with self._lock:
            return {(r.session_id, r.index): r for r in self.items}


@pytest.fixture
def script() -> Script:
    return Script()


@pytest.fixture
def results() -> Results:
    return Results()


@pytest.fixture
def make_service(settings_store: SettingsStore, script: Script, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service_module, "SHUTDOWN_WAIT_S", 0.5)
    monkeypatch.setattr(service_module, "IDLE_CHECK_S", 0.02)
    services: list[AsrService] = []

    def make(
        installed: dict[str, set[str]] | None = None,
        gpu: GpuInfo = GPU_READY,
        clock: Callable[[], float] = time.monotonic,
        start: bool = True,
    ) -> tuple[AsrService, FakeModels]:
        models = FakeModels({"parakeet-tdt-0.6b-v2": {"gpu", "cpu"}} if installed is None else installed)
        service = AsrService(settings_store, models, engine_factory=script.factory, gpu_info=lambda: gpu, clock=clock)
        services.append(service)
        if start:
            service.start()
        return service, models

    yield make
    for service in services:
        service.shutdown()


def wait_ready(qtbot: QtBot, service: AsrService, model_id: str = "parakeet-tdt-0.6b-v2") -> None:
    qtbot.waitUntil(
        lambda: service.status.state is EngineState.READY and service.status.model_id == model_id, timeout=WAIT_MS
    )


# -- loading -------------------------------------------------------------------------------------------------


def test_start_loads_the_configured_model_on_the_gpu(qtbot: QtBot, make_service, script: Script) -> None:
    statuses = []
    service, models = make_service(start=False)
    service.status_changed.connect(statuses.append)
    assert service.status.state is EngineState.UNLOADED
    service.start()
    wait_ready(qtbot, service)

    assert len(script.specs) == 1
    spec = script.specs[0]
    assert spec.model is PARAKEET_V2 and spec.use_cuda and set(spec.folders) == {"gpu", "cpu"}
    status = service.status
    assert (status.device, status.device_name, status.message) == ("cuda", "NVIDIA Test GPU", None)
    qtbot.waitUntil(lambda: len(statuses) >= 2, timeout=WAIT_MS)
    assert [s.state for s in statuses] == [EngineState.LOADING, EngineState.READY]
    assert models.retained["parakeet-tdt-0.6b-v2"] == 1


@pytest.mark.parametrize(
    ("device", "installed", "gpu", "use_cuda"),
    [
        ("auto", {"gpu", "cpu"}, GPU_READY, True),
        ("auto", {"gpu", "cpu"}, GPU_MISSING, False),
        ("auto", {"cpu"}, GPU_READY, False),
        ("auto", {"gpu"}, GPU_MISSING, False),
        ("gpu", {"cpu"}, GPU_MISSING, True),
        ("cpu", {"gpu", "cpu"}, GPU_READY, False),
    ],
)
def test_device_and_variant_choice(
    qtbot: QtBot, make_service, script: Script, settings_store: SettingsStore, device, installed, gpu, use_cuda
) -> None:
    settings_store.update({"transcription": {"device": device}})
    service, _ = make_service({"parakeet-tdt-0.6b-v2": installed}, gpu=gpu)
    wait_ready(qtbot, service)
    spec = script.specs[0]
    assert spec.use_cuda is use_cuda
    assert set(spec.folders) == installed


def test_nothing_installed_answers_every_job_with_an_error(qtbot: QtBot, make_service, script, results) -> None:
    service, _ = make_service({})
    assert service.status.state is EngineState.NOT_INSTALLED
    assert "Parakeet TDT 0.6B v2" in (service.status.message or "")

    for index in range(3):
        service.submit(segment("a", index), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 3, timeout=WAIT_MS)

    assert script.specs == []
    assert all(r.error and "Download it in Settings" in r.error and r.text == "" for r in results.items)
    assert set(results.threads) == {"whisp-asr"}


def test_an_unknown_model_id_is_reported(qtbot: QtBot, make_service, settings_store, results) -> None:
    settings_store.update({"transcription": {"model_id": "no-such-model"}})
    service, _ = make_service()
    assert service.status.state is EngineState.NOT_INSTALLED
    assert service.status.message == MSG_UNKNOWN_MODEL
    service.submit(segment("a", 0), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 1, timeout=WAIT_MS)
    assert results.items[0].error == MSG_UNKNOWN_MODEL


def test_jobs_wait_while_the_model_loads(qtbot: QtBot, make_service, script: Script, results) -> None:
    script.load_gate = threading.Event()
    service, _ = make_service()
    qtbot.waitUntil(lambda: service.status.state is EngineState.LOADING, timeout=WAIT_MS)
    for index in range(3):
        service.submit(segment("a", index), results.callback())
    time.sleep(0.1)
    assert results.items == []

    script.load_gate.set()
    qtbot.waitUntil(lambda: len(results.items) == 3, timeout=WAIT_MS)

    assert results.keys() == [("a", 0), ("a", 1), ("a", 2)]
    assert [r.text for r in results.items] == ["text s0", "text s1", "text s2"]
    assert all(r.error is None and r.audio_ms == 500 and r.language == "en" for r in results.items)
    assert set(results.threads) == {"whisp-asr"}


# -- ordering, priorities, coalescing -----------------------------------------------------------------------


def block_worker(qtbot: QtBot, service: AsrService, script: Script, results: Results) -> threading.Event:
    """Occupy the worker with a segment that waits for the returned event."""
    gate = threading.Event()
    script.gate = gate
    script.entered.clear()
    service.submit(segment("busy", 0, code=99), results.callback())
    qtbot.waitUntil(script.entered.is_set, timeout=WAIT_MS)
    return gate


def test_segments_go_first_in_order_and_only_the_newest_preview_runs(
    qtbot: QtBot, make_service, script: Script, results: Results
) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    gate = block_worker(qtbot, service, script, results)

    service.submit_preview("a", preview_audio(1), results.preview("a"))
    service.submit(segment("c", 0, code=10), results.callback())
    service.submit_preview("a", preview_audio(2), results.preview("a"))
    service.submit_preview("b", preview_audio(3), results.preview("b"))
    service.submit(segment("c", 1, code=11), results.callback())
    gate.set()
    qtbot.waitUntil(lambda: len(results.items) == 3 and len(results.previews) == 2, timeout=WAIT_MS)

    assert [label for _, label, _ in script.calls] == ["s99", "s10", "s11", "p2", "p3"]
    assert results.previews == [("a", "preview"), ("b", "preview")]


def test_a_segment_drops_its_sessions_queued_preview(qtbot: QtBot, make_service, script: Script, results) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    gate = block_worker(qtbot, service, script, results)

    service.submit_preview("a", preview_audio(1), results.preview("a"))  # the speech in progress...
    service.submit(segment("a", 0, code=5), results.callback())  # ...became a segment, which runs first
    service.submit_preview("b", preview_audio(2), results.preview("b"))
    gate.set()
    qtbot.waitUntil(lambda: len(results.items) == 2 and len(results.previews) == 1, timeout=WAIT_MS)
    time.sleep(0.1)

    assert [label for _, label, _ in script.calls] == ["s99", "s5", "p2"]  # no time spent on the stale preview
    assert results.previews == [("b", "preview")]


def test_a_final_segment_drops_the_sessions_preview(qtbot: QtBot, make_service, script: Script, results) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    gate = block_worker(qtbot, service, script, results)

    service.submit_preview("a", preview_audio(1), results.preview("a"))
    service.submit(segment("a", 0, final=True, code=5), results.callback())
    service.submit_preview("a", preview_audio(2), results.preview("a"))  # too late: the session has finished
    gate.set()
    qtbot.waitUntil(lambda: len(results.items) == 2, timeout=WAIT_MS)
    time.sleep(0.1)

    assert [label for _, label, _ in script.calls] == ["s99", "s5"]
    assert results.previews == []


def test_discard_answers_queued_segments_and_drops_previews(
    qtbot: QtBot, make_service, script: Script, results: Results
) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    gate = block_worker(qtbot, service, script, results)

    service.submit(segment("a", 0, code=1), results.callback())
    service.submit(segment("b", 0, code=2), results.callback())
    service.submit(segment("a", 1, code=3), results.callback())
    service.submit_preview("a", preview_audio(4), results.preview("a"))
    service.discard("a")
    service.submit_preview("a", preview_audio(5), results.preview("a"))  # ignored after the discard
    gate.set()
    qtbot.waitUntil(lambda: len(results.items) == 4, timeout=WAIT_MS)
    time.sleep(0.1)

    answers = results.by_key()
    assert answers[("a", 0)].error == MSG_CANCELLED and answers[("a", 1)].error == MSG_CANCELLED
    assert answers[("b", 0)].text == "text s2" and answers[("b", 0)].error is None
    assert [label for _, label, _ in script.calls] == ["s99", "s2"]
    assert results.previews == []
    assert Counter(results.keys()) == Counter({("busy", 0): 1, ("a", 0): 1, ("a", 1): 1, ("b", 0): 1})


def test_preview_callbacks_only_carry_text(qtbot: QtBot, make_service, script: Script, results) -> None:
    script.preview_text = ""
    service, _ = make_service()
    wait_ready(qtbot, service)
    service.submit_preview("a", preview_audio(1), results.preview("a"))
    qtbot.waitUntil(lambda: len(script.calls) == 1, timeout=WAIT_MS)  # the preview ran (a segment would drop it)
    service.submit(segment("a", 0), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 1, timeout=WAIT_MS)
    assert [kind for kind, _, _ in script.calls] == ["preview", "seg"]
    assert results.previews == []


# -- exactly one callback ----------------------------------------------------------------------------------------


def test_shutdown_answers_queued_jobs_at_once_and_the_running_one_later(
    qtbot: QtBot, make_service, script: Script, results: Results
) -> None:
    service, models = make_service()
    wait_ready(qtbot, service)
    gate = block_worker(qtbot, service, script, results)
    for index in range(5):
        service.submit(segment("a", index, code=index), results.callback())

    service.shutdown()  # the worker is stuck in a recognition: shutdown waits briefly, then returns

    keys = results.keys()
    assert sorted(keys) == [("a", i) for i in range(5)]
    assert all(r.error == MSG_CLOSING for r in results.items)
    gate.set()
    qtbot.waitUntil(lambda: len(results.items) == 6, timeout=WAIT_MS)
    assert results.by_key()[("busy", 0)].text == "text s99"
    qtbot.waitUntil(lambda: models.retained["parakeet-tdt-0.6b-v2"] == 0, timeout=WAIT_MS)
    assert script.engines[0].closed
    time.sleep(0.1)
    assert Counter(results.keys()).most_common(1)[0][1] == 1


def test_submit_after_shutdown_is_answered_immediately(make_service, results: Results) -> None:
    service, _ = make_service()
    service.shutdown()
    service.submit(segment("a", 0), results.callback())
    service.submit_preview("a", preview_audio(1), results.preview("a"))
    assert [r.error for r in results.items] == [MSG_CLOSING]
    assert results.threads == [threading.current_thread().name]
    assert results.previews == []
    service.shutdown()  # twice is fine


def test_a_failing_callback_does_not_stop_the_worker(qtbot: QtBot, make_service, results: Results) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)

    def explode(_result: AsrResult) -> None:
        raise ValueError("callback bug")

    service.submit(segment("a", 0), explode)
    service.submit(segment("a", 1), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 1, timeout=WAIT_MS)
    assert results.items[0].index == 1


def test_a_gpu_failure_moves_to_the_cpu_and_recognises_the_segment_again(
    qtbot: QtBot, make_service, script: Script, results: Results
) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    script.fail_next = 1
    service.submit(segment("a", 0), results.callback())
    service.submit(segment("a", 1), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 2, timeout=WAIT_MS)

    first, second = results.items
    assert (first.index, first.error, first.text) == (0, None, "text s0")  # the words that hit the failure survive
    assert second.error is None and second.text == "text s1"
    assert [label for _, label, _ in script.calls] == ["s0", "s0", "s1"]
    assert script.engines[0].closed
    assert [spec.use_cuda for spec in script.specs] == [True, False]
    assert service.status.device == "cpu" and service.status.message == MSG_GPU_LOST


def test_a_segment_that_also_fails_on_the_cpu_is_answered_once_with_an_error(
    qtbot: QtBot, make_service, script: Script, results: Results
) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    script.fail_next = 2
    service.submit(segment("a", 0), results.callback())
    service.submit(segment("a", 1), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 2, timeout=WAIT_MS)
    time.sleep(0.1)

    first, second = results.items
    assert first.error == MSG_RECOGNITION_FAILED and first.text == ""
    assert second.error is None and second.text == "text s1"
    assert [label for _, label, _ in script.calls] == ["s0", "s0", "s1"]  # one retry on the CPU, no more
    assert Counter(results.keys()) == Counter({("a", 0): 1, ("a", 1): 1})


def test_a_failure_on_the_cpu_is_answered_without_a_retry(
    qtbot: QtBot, make_service, script: Script, settings_store: SettingsStore, results: Results
) -> None:
    settings_store.update({"transcription": {"device": "cpu"}})
    service, _ = make_service()
    wait_ready(qtbot, service)
    script.fail_next = 1
    service.submit(segment("a", 0), results.callback())
    service.submit(segment("a", 1), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 2, timeout=WAIT_MS)

    first, second = results.items
    assert first.error == MSG_RECOGNITION_FAILED and second.text == "text s1"
    assert [label for _, label, _ in script.calls] == ["s0", "s1"]
    assert len(script.engines) == 1 and not script.engines[0].closed


# -- reloading -------------------------------------------------------------------------------------------------------


def test_changing_the_model_or_device_reloads(
    qtbot: QtBot, make_service, script: Script, settings_store: SettingsStore
) -> None:
    service, models = make_service({"parakeet-tdt-0.6b-v2": {"gpu", "cpu"}, "parakeet-tdt-0.6b-v3": {"gpu"}})
    wait_ready(qtbot, service)

    settings_store.update({"transcription": {"model_id": "parakeet-tdt-0.6b-v3"}})
    wait_ready(qtbot, service, "parakeet-tdt-0.6b-v3")
    assert script.engines[0].closed
    assert models.retained == Counter({"parakeet-tdt-0.6b-v2": 0, "parakeet-tdt-0.6b-v3": 1})

    settings_store.update({"transcription": {"device": "cpu"}})
    qtbot.waitUntil(lambda: len(script.engines) == 3 and service.status.state is EngineState.READY, timeout=WAIT_MS)
    assert script.engines[1].closed
    assert service.status.device == "cpu" and not script.specs[2].use_cuda


def test_changing_the_language_needs_no_reload(
    qtbot: QtBot, make_service, script: Script, settings_store: SettingsStore, results: Results
) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    service.submit(segment("a", 0), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 1, timeout=WAIT_MS)
    settings_store.update({"transcription": {"language": "de"}})
    service.submit(segment("a", 1), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 2, timeout=WAIT_MS)
    assert [language for _, _, language in script.calls] == ["auto", "de"]
    assert len(script.engines) == 1


def test_a_freshly_installed_model_loads_by_itself(qtbot: QtBot, make_service, script: Script) -> None:
    service, models = make_service({})
    qtbot.waitUntil(lambda: service.status.state is EngineState.NOT_INSTALLED, timeout=WAIT_MS)
    models.installed["parakeet-tdt-0.6b-v2"] = {"cpu"}
    models.changed.emit()
    wait_ready(qtbot, service)
    assert service.status.device == "cpu"


def test_a_newly_installed_gpu_variant_is_switched_to(qtbot: QtBot, make_service, script: Script) -> None:
    service, models = make_service({"parakeet-tdt-0.6b-v2": {"cpu"}})
    wait_ready(qtbot, service)
    assert service.status.device == "cpu"
    models.installed["parakeet-tdt-0.6b-v2"] = {"cpu", "gpu"}
    models.changed.emit()
    qtbot.waitUntil(
        lambda: service.status.device == "cuda" and service.status.state is EngineState.READY, timeout=WAIT_MS
    )
    assert script.engines[0].closed


def test_a_load_failure_is_reported_and_retried_by_ensure_loaded(
    qtbot: QtBot, make_service, script: Script, results: Results
) -> None:
    script.load_error = EngineError("The speech model couldn't be loaded.")
    service, models = make_service()
    qtbot.waitUntil(lambda: service.status.state is EngineState.ERROR, timeout=WAIT_MS)
    assert service.status.message == "The speech model couldn't be loaded."
    assert models.retained["parakeet-tdt-0.6b-v2"] == 0

    service.submit(segment("a", 0), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 1, timeout=WAIT_MS)
    assert results.items[0].error == "The speech model couldn't be loaded."
    assert len(script.specs) == 1  # not retried for every job

    service.ensure_loaded()
    wait_ready(qtbot, service)
    assert len(script.specs) == 2


def test_ensure_loaded_does_nothing_when_ready(qtbot: QtBot, make_service, script: Script) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    service.ensure_loaded()
    time.sleep(0.1)
    assert len(script.specs) == 1


def test_reload_loads_again(qtbot: QtBot, make_service, script: Script) -> None:
    service, _ = make_service()
    wait_ready(qtbot, service)
    service.reload()
    qtbot.waitUntil(lambda: len(script.engines) == 2 and service.status.state is EngineState.READY, timeout=WAIT_MS)
    assert script.engines[0].closed


def test_the_engine_notice_is_shown(qtbot: QtBot, make_service, script: Script) -> None:
    script.notice = "The graphics card couldn't load the model, so it runs on the CPU."
    service, _ = make_service()
    wait_ready(qtbot, service)
    assert service.status.message == script.notice


# -- idle unload -----------------------------------------------------------------------------------------------------


# -- GPU runtime pack --------------------------------------------------------------------------


GPU_NEEDS_PACK = GpuInfo(True, "NVIDIA Test GPU", 12000, False, "Needs a one-time download.", 1_000_000)


def _gpu_service(settings_store: SettingsStore, script: Script, installer) -> tuple[AsrService, dict[str, GpuInfo]]:
    """A service whose GPU check reports whatever the fake installer leaves behind, like the real cache."""
    gpu = {"info": GPU_NEEDS_PACK}

    def refresh() -> GpuInfo:
        gpu["info"] = GPU_READY
        return GPU_READY

    models = FakeModels({"parakeet-tdt-0.6b-v2": {"gpu", "cpu"}})
    service = AsrService(
        settings_store,
        models,
        engine_factory=script.factory,
        gpu_info=lambda: gpu["info"],
        gpu_installer=installer,
        gpu_refresh=refresh,
    )
    return service, gpu


def test_installing_the_gpu_pack_moves_the_model_to_the_gpu(
    qtbot: QtBot, settings_store: SettingsStore, script: Script
) -> None:
    def installer(progress, cancel: threading.Event) -> None:
        progress(500_000, 1_000_000)
        progress(1_000_000, 1_000_000)

    service, _gpu = _gpu_service(settings_store, script, installer)
    progress: list[tuple[object, object]] = []
    service.gpu_runtime_progress.connect(lambda done, total: progress.append((done, total)))
    try:
        service.start()
        qtbot.waitUntil(lambda: service.status.state is EngineState.READY, timeout=WAIT_MS)
        assert service.status.device == "cpu"
        with qtbot.waitSignal(service.gpu_runtime_finished, timeout=WAIT_MS) as finished:
            assert service.install_gpu_runtime() is True
        assert finished.args[0].cuda_ready is True
        assert progress[-1] == (1_000_000, 1_000_000)
        qtbot.waitUntil(lambda: service.status.device == "cuda", timeout=WAIT_MS)
        assert service.gpu_runtime_installing is False
    finally:
        service.shutdown()


def test_cancelling_or_failing_the_gpu_pack_is_reported(
    qtbot: QtBot, settings_store: SettingsStore, script: Script
) -> None:
    from whisp.core.asr.downloads import DownloadCancelled, DownloadError

    release = threading.Event()
    outcomes = iter([DownloadCancelled(), DownloadError("Not enough disk space.")])

    def installer(progress, cancel: threading.Event) -> None:
        release.wait(5)
        raise next(outcomes)

    service, gpu = _gpu_service(settings_store, script, installer)
    try:
        assert service.install_gpu_runtime() is True
        assert service.install_gpu_runtime() is False  # one download at a time
        with qtbot.waitSignal(service.gpu_runtime_failed, timeout=WAIT_MS) as cancelled:
            release.set()
        assert cancelled.args == [""]
        qtbot.waitUntil(lambda: not service.gpu_runtime_installing, timeout=WAIT_MS)
        with qtbot.waitSignal(service.gpu_runtime_failed, timeout=WAIT_MS) as failed:
            assert service.install_gpu_runtime() is True
        assert failed.args == ["Not enough disk space."]
        assert gpu["info"] is GPU_NEEDS_PACK  # nothing changed
    finally:
        service.shutdown()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_an_idle_model_is_unloaded_and_loads_again_when_needed(
    qtbot: QtBot, make_service, script: Script, settings_store: SettingsStore, results: Results
) -> None:
    clock = FakeClock()
    settings_store.update({"transcription": {"unload_after_minutes": 2}})
    service, models = make_service(clock=clock)
    wait_ready(qtbot, service)

    clock.now += 119
    time.sleep(0.2)
    assert service.status.state is EngineState.READY
    clock.now += 2
    qtbot.waitUntil(lambda: service.status.state is EngineState.UNLOADED, timeout=WAIT_MS)
    assert script.engines[0].closed and models.retained["parakeet-tdt-0.6b-v2"] == 0

    service.submit(segment("a", 0), results.callback())
    qtbot.waitUntil(lambda: len(results.items) == 1, timeout=WAIT_MS)
    assert results.items[0].error is None
    assert len(script.engines) == 2 and service.status.state is EngineState.READY


def test_zero_minutes_means_never_unload(qtbot: QtBot, make_service, script: Script) -> None:
    clock = FakeClock()
    service, _ = make_service(clock=clock)
    wait_ready(qtbot, service)
    clock.now += 10**6
    time.sleep(0.2)
    assert service.status.state is EngineState.READY
    assert not script.engines[0].closed
