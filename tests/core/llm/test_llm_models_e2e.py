"""End to end with the real pinned llama.cpp server and models.

Marked ``models``: the first run downloads the server build for this PC and
the models into the shared test cache (``shared_models_dir``); later runs
reuse them. The latency test is also marked ``gpu``. Run with::

    .venv\\Scripts\\python.exe -m pytest tests/core/llm/test_llm_models_e2e.py -m models -s
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from PySide6.QtCore import Qt

from whisp.core.llm.runtime import LlmRuntime, LlmState
from whisp.core.text.context import TextContext
from whisp.core.text.polish import Polisher, validate_cleanup
from whisp.core.text.rules import apply_rules
from whisp.core.types import AppCategory
from whisp.storage.settings import SettingsStore

pytestmark = pytest.mark.models

INSTALL_TIMEOUT_S = 3600
READY_TIMEOUT_S = 180

MESSY = (
    "um so basically i wanted to uh tell you that the the meeting tomorrow got moved to three thirty and we "
    "should probably bring the budget numbers and also uh the vendor list"
)

# Four different ~150-word dictations, so every request after the first has a cached prompt prefix but new text.
DICTATIONS = (
    "okay so um i just got off the call with the client and uh they want to move the launch from march fifteenth "
    "to april second which honestly gives us a bit more breathing room um the main things they asked for are first "
    "a cleaner onboarding flow second better error messages when the payment fails and third they want the "
    "dashboard to load faster on mobile i told them we'd send a revised timeline by friday so can you uh pull "
    "together the estimates for the onboarding work and loop in sarah on the payment stuff because she built most "
    "of that last year also the the budget is still twelve thousand dollars so we need to be careful about adding "
    "scope oh and they mentioned they might want a demo next tuesday at two thirty so keep that slot free if you "
    "can thanks",
    "hey so quick update on the hiring front um we interviewed four candidates this week for the backend role and "
    "uh two of them were really strong the first one has about six years of experience with python and postgres "
    "and she did really well on the system design part the second one is more junior but he picked up the take "
    "home assignment super quickly and his code was honestly cleaner than what we usually see um i think we should "
    "move both of them to the final round if the budget allows it i'd like to schedule the final interviews for "
    "next wednesday and thursday afternoon could you check with david whether he can join at least one of them "
    "and also we still need someone from product to do the culture interview so maybe ask priya let me know what "
    "you think and whether we should send an offer before the end of the month",
    "so for the quarterly review i'm thinking we structure it like this um we start with the numbers revenue was "
    "up about eighteen percent compared to last quarter mostly because of the enterprise deals that closed in "
    "august then we talk about churn which uh went down a little bit from four point two percent to three point "
    "eight percent after that i want to spend some time on the roadmap specifically the analytics dashboard and "
    "the new permissions model because those are the two things customers keep asking about and then we finish "
    "with hiring and budget for next quarter i'll put together the slides by thursday but i need the final revenue "
    "numbers from finance before then so if you could ping them today that would be great oh and let's keep the "
    "whole thing under forty five minutes because people get restless after that",
    "um hi everyone just a reminder that the office will be closed on monday for the public holiday so uh please "
    "make sure your laptops are charged and you have everything you need to work from home tuesday morning also "
    "the it team is going to update the vpn software over the long weekend so the first time you log in after the "
    "holiday it might ask you to install an update just follow the prompts and it should only take a couple of "
    "minutes if anything goes wrong reach out on the help desk channel rather than emailing people directly "
    "because that way whoever is on call can pick it up faster finally the team lunch has been moved from friday "
    "to the following wednesday at twelve thirty at the italian place around the corner so please update your "
    "calendars and let me know if you have any dietary requirements",
)


def _connect(signal_obj, fn: Callable) -> None:
    signal_obj.connect(fn, Qt.ConnectionType.DirectConnection)


def install(runtime: LlmRuntime, model_id: str) -> None:
    done = threading.Event()
    errors: list[str] = []

    def finished(mid: str) -> None:
        if mid == model_id:
            done.set()

    def failed(mid: str, message: str) -> None:
        if mid == model_id:
            errors.append(message)
            done.set()

    _connect(runtime.install_finished, finished)
    _connect(runtime.install_failed, failed)
    started = time.monotonic()
    runtime.install(model_id)
    assert done.wait(INSTALL_TIMEOUT_S), "install timed out"
    assert not errors, errors
    assert runtime.is_installed(model_id)
    print(f"\n{model_id} installed (or already present) in {time.monotonic() - started:.1f} s")


def wait_until(predicate: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def wait_ready(runtime: LlmRuntime, *, not_url: str | None = None) -> float:
    started = time.monotonic()

    def ready() -> bool:
        status = runtime.status
        assert status.state is not LlmState.ERROR, status.message
        return status.state is LlmState.READY and runtime.base_url() is not None and runtime.base_url() != not_url

    wait_until(ready, READY_TIMEOUT_S, "the server to become ready")
    return time.monotonic() - started


def make_ctx(category: AppCategory = AppCategory.DOCUMENT, style: str = "clean") -> TextContext:
    return TextContext(category, style, "", "en", (), (), True, True, "")


class WithoutWarmup:
    """The runtime minus its status signal, so the Polisher doesn't pre-load the prompt (for cold timings)."""

    def __init__(self, runtime: LlmRuntime) -> None:
        self._runtime = runtime

    def base_url(self) -> str | None:
        return self._runtime.base_url()

    def api_key(self) -> str | None:
        return self._runtime.api_key()

    def installed_models(self) -> list[str]:
        return self._runtime.installed_models()


def new_runtime(settings: SettingsStore, shared: Path, *, flavor: str | None = None) -> LlmRuntime:
    return LlmRuntime(settings, shared / "runtime", shared / "llm", flavor=flavor)


def test_install_start_polish_recover_and_stop(settings_store: SettingsStore, shared_models_dir: Path) -> None:
    runtime = new_runtime(settings_store, shared_models_dir)
    try:
        install(runtime, "qwen3.5-2b")
        settings_store.update(
            {"polish": {"enabled": True, "provider": "builtin", "builtin_model": "qwen3.5-2b", "timeout_seconds": 20}}
        )
        runtime.ensure_running()
        print(f"server ready in {wait_ready(runtime):.1f} s on {runtime.status.backend}")
        base, key = runtime.base_url(), runtime.api_key()
        assert base and key and base.startswith("http://127.0.0.1:")

        # Locked down: the key is required, /slots is off, other web pages are not let in, the key isn't logged.
        assert httpx.post(f"{base}/v1/chat/completions", json={"messages": []}).status_code == 401
        auth = {"Authorization": f"Bearer {key}"}
        assert httpx.get(f"{base}/slots", headers=auth).status_code in (404, 501)
        preflight = httpx.options(
            f"{base}/v1/chat/completions",
            headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
        )
        assert preflight.headers.get("access-control-allow-origin") is None
        log_file = shared_models_dir / "runtime" / "llama" / "logs" / "llama-server.log"
        assert key not in log_file.read_text(encoding="utf-8", errors="replace")

        polisher = Polisher(settings_store, runtime)
        ctx = make_ctx()
        text = apply_rules(MESSY, ctx)
        result = polisher.polish(text, ctx)
        assert result.used, result.error
        assert validate_cleanup(text, result.text, "stop") is None
        assert "budget" in result.text and "vendor list" in result.text
        print(f"polished {len(text.split())} words in {result.elapsed_ms} ms")
        assert MESSY not in log_file.read_text(encoding="utf-8", errors="replace")  # no dictated text in the log

        rewrite = polisher.rewrite("hey can u send me the file tmrw", "make this more formal", ctx)
        assert rewrite.used, rewrite.error

        # A crash is noticed and the server comes back on a new port with a new key.
        proc = runtime._proc  # the test reaches in to simulate a crash
        assert proc is not None
        os.kill(proc.pid, signal.SIGTERM)
        print(f"restarted in {wait_ready(runtime, not_url=base):.1f} s")
        assert runtime.api_key() != key
        assert polisher.polish(text, ctx).used

        # Turning polish off stops the server.
        restarted = runtime._proc
        settings_store.update({"polish": {"enabled": False}})
        wait_until(lambda: runtime.status.state is LlmState.OFF, 30, "the server to stop")
        assert runtime.base_url() is None
        assert restarted is not None and restarted.poll() is not None
    finally:
        runtime.shutdown()


def test_cpu_build_with_qwen35_2b(settings_store: SettingsStore, shared_models_dir: Path) -> None:
    runtime = new_runtime(settings_store, shared_models_dir, flavor="cpu")
    try:
        install(runtime, "qwen3.5-2b")
        settings_store.update(
            {"polish": {"enabled": True, "provider": "builtin", "builtin_model": "qwen3.5-2b", "timeout_seconds": 60}}
        )
        runtime.ensure_running()
        start_s = wait_ready(runtime)
        polisher = Polisher(settings_store, WithoutWarmup(runtime))
        ctx = make_ctx()
        timings = []
        for raw in (MESSY, DICTATIONS[0], DICTATIONS[1]):
            text = apply_rules(raw, ctx)
            result = polisher.polish(text, ctx)
            assert result.used, result.error
            timings.append((len(text.split()), result.elapsed_ms))
        print(f"\nCPU, Qwen3.5-2B: ready in {start_s:.1f} s; (words, ms): {timings}")
        _record(shared_models_dir, "cpu-qwen3.5-2b", {"ready_s": round(start_s, 1), "words_ms": timings})
    finally:
        runtime.shutdown()


@pytest.mark.gpu
def test_latency_qwen35_4b_on_the_gpu(settings_store: SettingsStore, shared_models_dir: Path) -> None:
    runtime = new_runtime(settings_store, shared_models_dir)
    try:
        install(runtime, "qwen3.5-4b")
        settings_store.update(
            {"polish": {"enabled": True, "provider": "builtin", "builtin_model": "qwen3.5-4b", "timeout_seconds": 30}}
        )
        runtime.ensure_running()
        start_s = wait_ready(runtime)
        assert runtime.status.backend and "NVIDIA" in runtime.status.backend
        polisher = Polisher(settings_store, WithoutWarmup(runtime))
        ctx = make_ctx(AppCategory.MESSAGING, "casual")
        texts = [apply_rules(raw, ctx) for raw in DICTATIONS]
        results = []
        for text in texts:
            result = polisher.polish(text, ctx)
            assert result.used, result.error
            results.append(result)
        words = [len(t.split()) for t in texts]
        cold = results[0].elapsed_ms
        warm = [r.elapsed_ms for r in results[1:]]
        print(
            f"\nGPU ({runtime.status.backend}), Qwen3.5-4B: server ready in {start_s:.1f} s; words {words}; "
            f"cold {cold} ms; warm {warm} ms (median {statistics.median(warm):.0f} ms)"
        )
        _record(
            shared_models_dir,
            "gpu-qwen3.5-4b",
            {"ready_s": round(start_s, 1), "words": words, "cold_ms": cold, "warm_ms": warm},
        )
        assert statistics.median(warm) < 3000
    finally:
        runtime.shutdown()


def _record(shared: Path, key: str, value: dict) -> None:
    path = shared / "llm-latency.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data[key] = {**value, "at": time.strftime("%Y-%m-%d %H:%M")}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
