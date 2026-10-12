"""Polisher against a small local OpenAI/Ollama-compatible HTTP server started by the test."""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from PySide6.QtCore import QObject, Signal

from whisp.core.text import polish as polish_module
from whisp.core.text.context import TextContext
from whisp.core.text.polish import (
    MSG_ADDED,
    MSG_ANSWERED,
    MSG_BUILTIN_NOT_READY,
    MSG_COMMAND_NEEDS_AI,
    MSG_CUT_OFF,
    MSG_EMPTY,
    MSG_NO_SELECTION,
    MSG_QUESTION,
    MSG_REPEATED,
    MSG_TOO_LONG,
    MSG_TOO_SLOW,
    MSG_UNEXPECTED,
    Polisher,
)
from whisp.core.text.prompts import CLEANUP_STOP, COMMAND_STOP, COMMAND_SYSTEM_PROMPT
from whisp.core.types import AppCategory
from whisp.storage.secrets import protect
from whisp.storage.settings import SettingsStore

Responder = Callable[[str, dict[str, Any]], tuple[int, dict[str, Any], float]]


@dataclass
class Request:
    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any] = field(default_factory=dict)


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        pass  # e.g. an https handshake sent to this plain-http server; the test checks what the client saw


class FakeServer:
    """Records every request and answers with ``responder(path, body) -> (status, json, delay_s)``."""

    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.responder: Responder = lambda path, body: (200, chat_reply("Hello there."), 0.0)
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else {}
                server.requests.append(Request(method, self.path, dict(self.headers), body))
                status, payload, delay = server.responder(self.path, body)
                if delay:
                    time.sleep(delay)
                data = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # the client gave up (timeout test)

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

        self.httpd = _QuietServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def posts(self) -> list[Request]:
        return [r for r in self.requests if r.method == "POST"]


def chat_reply(
    content: str, finish: str = "stop", completion_tokens: int | None = None, timings: dict[str, Any] | None = None
) -> dict[str, Any]:
    usage = {"completion_tokens": completion_tokens} if completion_tokens is not None else {}
    reply: dict[str, Any] = {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": usage,
    }
    if timings is not None:
        reply["timings"] = timings  # what llama-server adds
    return reply


def ollama_reply(content: str, **fields: Any) -> dict[str, Any]:
    reply = {"message": {"role": "assistant", "content": content}, "done": True, "done_reason": "stop", "eval_count": 9}
    return {**reply, **fields}


class FakeRuntime:
    def __init__(self, base_url: str | None, key: str | None = "secret-key") -> None:
        self._base_url = base_url
        self._key = key

    def base_url(self) -> str | None:
        return self._base_url

    def api_key(self) -> str | None:
        return self._key

    def installed_models(self) -> list[str]:
        return ["qwen3.5-4b"]


@pytest.fixture
def server() -> Iterator[FakeServer]:
    s = FakeServer()
    yield s
    s.close()


def enable(store: SettingsStore, **polish: Any) -> None:
    store.update({"polish": {"enabled": True, "provider": "builtin", **polish}})


def make_ctx(style: str = "clean", category: AppCategory = AppCategory.DOCUMENT) -> TextContext:
    return TextContext(category, style, "", None, (), (), True, True, "")


TEXT = "So the meeting tomorrow is moved to three thirty and we should bring the budget numbers"
CLEANED = "So the meeting tomorrow is moved to 3:30, and we should bring the budget numbers."
LONG_TEXT = " ".join(["the budget numbers are ready and the meeting moved"] * 30)
"""About 440 tokens: about 11 s at 40 tokens/s, over a 3-second limit."""


# -- built-in provider ----------------------------------------------------------------------


def test_builtin_request_shape(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store)
    server.responder = lambda path, body: (200, chat_reply(CLEANED), 0.0)
    result = Polisher(settings_store, FakeRuntime(server.url)).polish(TEXT, make_ctx())
    assert result.used and result.error is None
    assert result.text == CLEANED
    request = server.posts()[0]
    assert request.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret-key"
    body = request.body
    assert body["temperature"] == 0
    assert body["top_k"] == 1
    assert body["presence_penalty"] == 0
    assert body["frequency_penalty"] == 0
    assert body["repeat_penalty"] == 1.0
    assert body["stream"] is False
    assert body["stop"] == list(CLEANUP_STOP)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert 64 < body["max_tokens"] < 200  # bounded by the input length
    assert body["messages"][-1]["content"].endswith(f"<transcript>{TEXT}</transcript>")


def test_thinking_is_stripped(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store)
    server.responder = lambda path, body: (200, chat_reply(f"<think>Let me clean this.</think>\n{CLEANED}"), 0.0)
    result = Polisher(settings_store, FakeRuntime(server.url)).polish(TEXT, make_ctx())
    assert result.used and result.text == CLEANED


def test_timeout_falls_back_to_the_input(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, timeout_seconds=1.0)
    server.responder = lambda path, body: (200, chat_reply(CLEANED), 2.5)
    started = time.monotonic()
    result = Polisher(settings_store, FakeRuntime(server.url)).polish(TEXT, make_ctx())
    elapsed = time.monotonic() - started
    assert not result.used
    assert result.text == TEXT
    assert "longer than 1 seconds" in (result.error or "")
    assert elapsed < 2.0


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (chat_reply("The capital of France is Paris."), MSG_ANSWERED),
        (chat_reply("Paris."), MSG_ADDED),
        (chat_reply("What's the capital of France? What's the capital of France?"), MSG_REPEATED),
        (chat_reply("What's the capital", finish="length"), MSG_CUT_OFF),
        (chat_reply(""), MSG_EMPTY),
    ],
)
def test_invalid_output_falls_back(
    settings_store: SettingsStore, server: FakeServer, reply: dict[str, Any], error: str
) -> None:
    enable(settings_store)
    server.responder = lambda path, body: (200, reply, 0.0)
    source = "what's the capital of france"
    result = Polisher(settings_store, FakeRuntime(server.url)).polish(source, make_ctx())
    assert not result.used
    assert result.text == source
    assert result.error == error


def test_skips_without_a_request(settings_store: SettingsStore, server: FakeServer) -> None:
    polisher = Polisher(settings_store, FakeRuntime(server.url))
    off = polisher.polish(TEXT, make_ctx())  # polish is off by default
    assert (off.text, off.used, off.error) == (TEXT, False, None)
    enable(settings_store)
    verbatim = polisher.polish(TEXT, make_ctx("verbatim", AppCategory.CODE))
    assert (verbatim.text, verbatim.used, verbatim.error) == (TEXT, False, None)
    empty = polisher.polish("   ", make_ctx())
    assert not empty.used and empty.error is None
    too_long = polisher.polish("word " * 3000, make_ctx())
    assert too_long.error == MSG_TOO_LONG
    assert server.posts() == []


def test_builtin_not_ready(settings_store: SettingsStore) -> None:
    enable(settings_store)
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert not polisher.is_ready()
    result = polisher.polish(TEXT, make_ctx())
    assert (result.used, result.error, result.text) == (False, MSG_BUILTIN_NOT_READY, TEXT)


def test_never_raises(settings_store: SettingsStore) -> None:
    class Broken(FakeRuntime):
        def base_url(self) -> str | None:
            raise RuntimeError("boom")

    enable(settings_store)
    polisher = Polisher(settings_store, Broken("http://127.0.0.1:9"))
    assert polisher.polish(TEXT, make_ctx()).error == MSG_UNEXPECTED
    assert polisher.rewrite("Some text.", "make it formal", make_ctx()).error == MSG_UNEXPECTED


def test_unreachable_server_is_skipped_for_a_while(settings_store: SettingsStore) -> None:
    dead = FakeServer()
    url = dead.url
    dead.close()  # nothing listens on this port any more
    enable(settings_store, provider="ollama", ollama_url=url, ollama_model="m", timeout_seconds=5.0)
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert polisher.is_ready()
    first = polisher.polish(TEXT, make_ctx())
    assert not first.used
    assert "Couldn't reach the AI server" in (first.error or "")
    # Windows takes a moment to refuse a connection; later dictations don't wait for it again.
    assert not polisher.is_ready()
    started = time.monotonic()
    second = polisher.polish(TEXT, make_ctx())
    assert second.error == first.error
    assert time.monotonic() - started < 0.2


def test_slow_server_is_skipped_for_long_texts(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, timeout_seconds=3.0)
    slow = {"predicted_n": 40, "predicted_ms": 1000.0}  # llama-server's own timings: 40 tokens/s
    server.responder = lambda path, body: (200, chat_reply(CLEANED, completion_tokens=40, timings=slow), 1.0)
    polisher = Polisher(settings_store, FakeRuntime(server.url))
    assert polisher.polish(TEXT, make_ctx()).used
    result = polisher.polish(LONG_TEXT, make_ctx())
    assert result.error == MSG_TOO_SLOW
    assert len(server.posts()) == 1


def test_builtin_speed_is_the_servers_own_generation_time(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, timeout_seconds=3.0)
    fast = {"predicted_n": 40, "predicted_ms": 100.0}  # writing took 0.1 s of the 1 s answer
    server.responder = lambda path, body: (200, chat_reply(CLEANED, completion_tokens=40, timings=fast), 1.0)
    polisher = Polisher(settings_store, FakeRuntime(server.url))
    assert polisher.polish(TEXT, make_ctx()).used
    assert polisher.polish(LONG_TEXT, make_ctx()).error != MSG_TOO_SLOW
    assert len(server.posts()) == 2


def test_ollama_speed_leaves_out_loading_the_model(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, provider="ollama", ollama_url=server.url, ollama_model="m", timeout_seconds=3.0)
    cold = ollama_reply(CLEANED, load_duration=900_000_000, eval_count=40, eval_duration=100_000_000)
    server.responder = lambda path, body: (200, cold, 1.0)  # most of the first answer was loading the model
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert polisher.polish(TEXT, make_ctx()).used
    assert polisher.polish(LONG_TEXT, make_ctx()).error != MSG_TOO_SLOW
    assert len(server.posts()) == 2


def test_slow_ollama_generation_is_skipped_for_long_texts(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, provider="ollama", ollama_url=server.url, ollama_model="m", timeout_seconds=3.0)
    slow = ollama_reply(CLEANED, eval_count=40, eval_duration=1_000_000_000)  # 40 tokens/s of real writing
    server.responder = lambda path, body: (200, slow, 0.0)
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert polisher.polish(TEXT, make_ctx()).used
    assert polisher.polish(LONG_TEXT, make_ctx()).error == MSG_TOO_SLOW
    assert len(server.posts()) == 1


def test_one_slow_answer_from_a_server_without_timings_does_not_stop_polish(
    settings_store: SettingsStore, server: FakeServer
) -> None:
    enable(settings_store, provider="openai", openai_url=server.url, openai_model="m", timeout_seconds=3.0)
    delays = iter([1.0, 0.0, 0.0])  # the first answer included loading the model
    server.responder = lambda path, body: (200, chat_reply(CLEANED, completion_tokens=40), next(delays))
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert polisher.polish(TEXT, make_ctx()).used  # 40 tokens/s by the clock
    assert polisher.polish(LONG_TEXT, make_ctx()).error != MSG_TOO_SLOW  # one slow answer could be a cold start
    assert polisher.polish(LONG_TEXT, make_ctx()).error != MSG_TOO_SLOW  # and the fast answers since outvote it
    assert len(server.posts()) == 3


def test_a_server_without_timings_that_stays_slow_is_skipped_for_long_texts(
    settings_store: SettingsStore, server: FakeServer
) -> None:
    enable(settings_store, provider="openai", openai_url=server.url, openai_model="m", timeout_seconds=3.0)
    server.responder = lambda path, body: (200, chat_reply(CLEANED, completion_tokens=40), 1.0)
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert polisher.polish(TEXT, make_ctx()).used
    assert polisher.polish(TEXT, make_ctx()).used  # slow again: it really is this slow
    assert polisher.polish(LONG_TEXT, make_ctx()).error == MSG_TOO_SLOW
    assert len(server.posts()) == 2


def test_warm_up_when_the_builtin_server_becomes_ready(settings_store: SettingsStore, server: FakeServer) -> None:
    class Runtime(QObject):
        status_changed = Signal(object)

        def __init__(self, url: str) -> None:
            super().__init__()
            self._url = url

        def base_url(self) -> str | None:
            return self._url

        def api_key(self) -> str | None:
            return "secret-key"

        def installed_models(self) -> list[str]:
            return []

    @dataclass
    class State:
        value: str

    @dataclass
    class Status:
        state: State
        base_url: str

    runtime = Runtime(server.url)
    polisher = Polisher(settings_store, runtime)  # the app keeps it alive; so must the test
    runtime.status_changed.emit(Status(State("ready"), server.url))
    runtime.status_changed.emit(Status(State("ready"), server.url))  # warmed once per server launch
    deadline = time.monotonic() + 5
    while len(server.posts()) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.1)
    posts = server.posts()
    assert len(posts) == 2
    assert all(p.body["max_tokens"] == 1 for p in posts)
    assert posts[1].body["messages"][0]["content"] == COMMAND_SYSTEM_PROMPT
    assert polisher.is_ready() is False  # polish itself is still switched off


# -- command mode -----------------------------------------------------------------------------


def test_rewrite(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store)
    formal = "Hello everyone, I can't attend tomorrow. Could we move it to Friday?"
    server.responder = lambda path, body: (200, chat_reply(formal), 0.0)
    selection = "hey guys cant make it tmrw. push to fri?"
    result = Polisher(settings_store, FakeRuntime(server.url)).rewrite(selection, "make this more formal", make_ctx())
    assert result.used and result.text == formal
    body = server.posts()[0].body
    assert body["stop"] == list(COMMAND_STOP)
    assert body["messages"][0]["content"] == COMMAND_SYSTEM_PROMPT
    assert body["messages"][-1]["content"].startswith("<instruction>make this more formal</instruction>")


def test_rewrite_refusals(settings_store: SettingsStore, server: FakeServer) -> None:
    polisher = Polisher(settings_store, FakeRuntime(server.url))
    assert polisher.rewrite("Text.", "make it formal", make_ctx()).error == MSG_COMMAND_NEEDS_AI
    enable(settings_store)
    assert polisher.rewrite("   ", "make it formal", make_ctx()).error == MSG_NO_SELECTION
    question = polisher.rewrite("Our revenue grew.", "what does this mean", make_ctx())
    assert (question.text, question.used, question.error) == ("Our revenue grew.", False, MSG_QUESTION)
    assert server.posts() == []


def test_rewrite_keeps_edge_whitespace(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store)
    server.responder = lambda path, body: (200, chat_reply("A shorter line."), 0.0)
    result = Polisher(settings_store, FakeRuntime(server.url)).rewrite("A long line here.\n", "shorten it", make_ctx())
    assert result.text == "A shorter line.\n"


# -- Ollama ------------------------------------------------------------------------------------


def test_ollama_request_shape(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, provider="ollama", ollama_url=server.url + "/", ollama_model="qwen3.5:4b")
    server.responder = lambda path, body: (200, ollama_reply(CLEANED), 0.0)
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert polisher.is_ready()
    result = polisher.polish(TEXT, make_ctx())
    assert result.used and result.text == CLEANED
    request = server.posts()[0]
    assert request.path == "/api/chat"
    body = request.body
    assert body["model"] == "qwen3.5:4b"
    assert body["think"] is False
    assert body["keep_alive"] == "30m"
    assert body["stream"] is False
    options = body["options"]
    assert options["temperature"] == 0
    assert options["presence_penalty"] == 0
    assert options["repeat_penalty"] == 1.0
    assert options["num_ctx"] == 8192
    assert options["stop"] == list(CLEANUP_STOP)
    assert options["num_predict"] > 0


def test_ollama_without_think_support(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, provider="ollama", ollama_url=server.url, ollama_model="old-model")

    def responder(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        if "think" in body:
            return 400, {"error": "unknown field think"}, 0.0
        return 200, ollama_reply(CLEANED), 0.0

    server.responder = responder
    result = Polisher(settings_store, FakeRuntime(None)).polish(TEXT, make_ctx())
    assert result.used
    assert [("think" in r.body) for r in server.posts()] == [True, False]


def test_ollama_models(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, provider="ollama", ollama_url=server.url)
    server.responder = lambda path, body: (200, {"models": [{"name": "qwen3.5:4b"}, {"name": "gemma4:e4b"}]}, 0.0)
    assert Polisher(settings_store, FakeRuntime(None)).list_models("ollama") == ["gemma4:e4b", "qwen3.5:4b"]
    assert server.requests[0].path == "/api/tags"


# -- OpenAI-compatible -------------------------------------------------------------------------


def test_openai_compatible(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(
        settings_store,
        provider="openai",
        openai_url=server.url,  # no "/v1": Whisp adds it
        openai_model="local-model",
        openai_api_key_protected=protect("sk-test-123"),
    )
    calls: list[bool] = []

    def responder(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        calls.append("chat_template_kwargs" in body)
        if "chat_template_kwargs" in body:
            return 400, {"error": {"message": "Unrecognized request argument supplied: chat_template_kwargs"}}, 0.0
        return 200, chat_reply(f"<think>hidden</think>{CLEANED}"), 0.0

    server.responder = responder
    polisher = Polisher(settings_store, FakeRuntime(None))
    first = polisher.polish(TEXT, make_ctx())
    second = polisher.polish(TEXT, make_ctx())
    assert first.used and second.used and second.text == CLEANED
    assert calls == [True, False, False]  # the server's refusal is remembered
    request = server.posts()[0]
    assert request.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-test-123"
    assert request.body["model"] == "local-model"
    assert "top_k" not in request.body and "repeat_penalty" not in request.body  # standard fields only


def test_openai_uses_the_servers_model_when_none_is_chosen(settings_store: SettingsStore, server: FakeServer) -> None:
    enable(settings_store, provider="openai", openai_url=server.url + "/v1")

    def responder(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        if path == "/v1/models":
            return 200, {"data": [{"id": "loaded-model"}, {"id": "other"}]}, 0.0
        return 200, chat_reply(CLEANED), 0.0

    server.responder = responder
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert polisher.polish(TEXT, make_ctx()).used
    assert server.posts()[0].body["model"] == "loaded-model"
    assert polisher.list_models("openai") == ["loaded-model", "other"]


def test_credentials_in_a_server_address_stay_out_of_messages_and_logs(
    settings_store: SettingsStore, caplog: pytest.LogCaptureFixture
) -> None:
    dead = FakeServer()
    port = dead.httpd.server_address[1]
    dead.close()  # nothing listens there any more
    enable(settings_store, provider="openai", openai_url=f"http://sam:hunter2@127.0.0.1:{port}", openai_model="m")
    with caplog.at_level(logging.DEBUG):
        result = Polisher(settings_store, FakeRuntime(None)).polish(TEXT, make_ctx())
    assert result.error == f"Couldn't reach the AI server at 127.0.0.1:{port}. Is it running?"
    assert "hunter2" not in caplog.text and "sam:" not in caplog.text


def test_https_to_a_server_that_only_speaks_http_says_how_to_fix_it(
    settings_store: SettingsStore, server: FakeServer
) -> None:
    # An address typed without http:// in front gets https, which plain-http servers (most home Ollamas) can't do.
    enable(settings_store, provider="openai", openai_url=server.url.replace("http://", "https://"), openai_model="m")
    polisher = Polisher(settings_store, FakeRuntime(None))
    result = polisher.polish(TEXT, make_ctx())
    assert not result.used
    assert result.error is not None and "type http:// in front" in result.error
    assert polisher.polish(TEXT, make_ctx()).error == result.error  # skipped for a while, with the same advice
    assert server.posts() == []


def test_a_certificate_that_cant_be_verified_is_named() -> None:
    error = httpx.ConnectError("certificate verify failed")
    error.__cause__ = ssl.SSLCertVerificationError(1, "certificate verify failed: self-signed certificate")
    message = polish_module._connection_problem("https://ai.example.com/v1", error)
    assert "security certificate" in message and "nothing was sent" in message
    refused = httpx.ConnectError("connection refused")
    assert polish_module._connection_problem("https://ai.example.com/v1", refused).endswith("Is it running?")


def test_test_warns_when_a_remote_server_is_not_encrypted(
    settings_store: SettingsStore, server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def responder(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        transcript = body["messages"][-1]["content"].split("<transcript>")[1].split("</transcript>")[0]
        return 200, chat_reply(transcript.rstrip(".") + "."), 0.0

    server.responder = responder
    settings_store.update({"polish": {"provider": "openai", "openai_url": server.url, "openai_model": "m"}})
    polisher = Polisher(settings_store, FakeRuntime(None))
    ok, message = polisher.test()
    assert ok and "encrypted" not in message  # a server on this PC
    monkeypatch.setattr(polish_module, "_is_local", lambda url: False)  # the same server, as if on another PC
    ok, message = polisher.test()
    assert ok and "isn't encrypted" in message


def test_list_models_builtin_and_unreachable(settings_store: SettingsStore) -> None:
    enable(settings_store, provider="ollama", ollama_url="http://127.0.0.1:9")
    polisher = Polisher(settings_store, FakeRuntime(None))
    assert polisher.list_models("builtin") == ["qwen3.5-4b"]
    assert polisher.list_models("ollama") == []


def test_test_round_trip(settings_store: SettingsStore, server: FakeServer) -> None:
    def responder(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
        transcript = body["messages"][-1]["content"].split("<transcript>")[1].split("</transcript>")[0]
        return 200, chat_reply(transcript.rstrip(".") + "."), 0.0

    server.responder = responder
    polisher = Polisher(settings_store, FakeRuntime(server.url))
    ok, message = polisher.test()  # works before polish is switched on
    assert ok, message
    assert "answered in" in message
    server.responder = lambda path, body: (200, chat_reply("Sure! Here is a poem about the sea instead."), 0.0)
    ok, message = polisher.test()
    assert not ok
    assert "didn't pass Whisp's checks" in message
