"""DeliveryService: text pipeline, insertion outcomes, command mode, ordering, paste-last and shutdown.

The text tools, polisher and inserter are small fakes that follow the documented
interfaces; history is the real HistoryStore on a temporary database.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from whisp.app import delivery as delivery_module
from whisp.app.delivery import DeliveryService, TextTools
from whisp.core.types import (
    AppCategory,
    DeliveryResult,
    InsertionResult,
    Outcome,
    RecordingMode,
    SessionInfo,
    SessionKind,
    TargetWindow,
    Transcript,
)
from whisp.storage.history import HistoryStore
from whisp.storage.settings import Settings, SettingsStore

SLACK = TargetWindow(hwnd=101, pid=11, process_name="slack.exe", app_name="Slack", title="general")
CODE = TargetWindow(hwnd=202, pid=22, process_name="code.exe", app_name="Visual Studio Code", title="main.py")
NOTES = TargetWindow(hwnd=303, pid=33, process_name="notes.exe", app_name="Notes", title="todo")


# -- fakes -------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeContext:
    category: AppCategory
    style: str
    app_name: str
    language: str | None


@dataclass(frozen=True)
class FakePolishResult:
    text: str
    used: bool
    error: str | None = None
    elapsed_ms: int = 5


def fake_build_context(settings: Settings, target: TargetWindow | None, language: str | None) -> FakeContext:
    category = target.category if target is not None else AppCategory.OTHER
    return FakeContext(category, settings.text.styles[category.value], target.app_name if target else "", language)


def fake_categorize(process_name: str, title: str, overrides: Mapping[str, str]) -> AppCategory:
    if process_name in overrides:
        return AppCategory(overrides[process_name])
    known = {"slack.exe": AppCategory.MESSAGING, "code.exe": AppCategory.CODE}
    return known.get(process_name, AppCategory.OTHER)


def fake_apply_rules(raw: str, ctx: FakeContext) -> str:
    """Drops filler words, like the real rules do."""
    return " ".join(w for w in raw.split() if w.lower().strip(",.") not in {"um", "uh"})


def raising(exc: Exception) -> Callable[..., FakePolishResult]:
    def fail(*_args: object) -> FakePolishResult:
        raise exc

    return fail


class FakeSpacing:
    """Adds a space after earlier text in the same window, unless spacing is off."""

    def __init__(self) -> None:
        self.memory: dict[int | None, str] = {}

    def prefix_for(self, target: TargetWindow | None, text: str, mode: str) -> str:
        last = self.memory.get(target.hwnd if target else None, "")
        if mode == "never" or not last or last[-1].isspace():
            return ""
        return " "

    def remember(self, target: TargetWindow | None, text: str) -> None:
        self.memory[target.hwnd if target else None] = text


class FakePolisher:
    def __init__(self) -> None:
        self.ready = True
        self.polish_calls: list[tuple[str, FakeContext]] = []
        self.rewrite_calls: list[tuple[str, str, FakeContext]] = []
        self.polish_with: Callable[[str], FakePolishResult] = lambda t: FakePolishResult(f"{t.capitalize()}!", True)
        self.rewrite_with: Callable[[str, str], FakePolishResult] = lambda sel, _: FakePolishResult(sel.upper(), True)

    def is_ready(self) -> bool:
        return self.ready

    def polish(self, text: str, ctx: FakeContext) -> FakePolishResult:
        self.polish_calls.append((text, ctx))
        return self.polish_with(text)

    def rewrite(self, selection: str, instruction: str, ctx: FakeContext) -> FakePolishResult:
        self.rewrite_calls.append((selection, instruction, ctx))
        return self.rewrite_with(selection, instruction)


@dataclass
class Insert:
    text: str
    target: TargetWindow | None
    method: str
    restore_clipboard: bool
    on_focus_change: str
    thread: str


@dataclass
class FakeInserter:
    inserted: list[Insert] = field(default_factory=list)
    clipboard: list[str] = field(default_factory=list)
    result: Callable[[str, str], InsertionResult] | None = None
    defaults: dict[str, str] = field(default_factory=dict)
    copy_ok: bool = True
    delay_s: float = 0.0
    fail_with: Exception | None = None

    def insert(
        self, text: str, *, target: TargetWindow | None, method: str, restore_clipboard: bool, on_focus_change: str
    ) -> InsertionResult:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with
        self.inserted.append(
            Insert(text, target, method, restore_clipboard, on_focus_change, threading.current_thread().name)
        )
        return self.result(text, method) if self.result else InsertionResult(ok=True, method=method)

    def copy_to_clipboard(self, text: str) -> bool:
        self.clipboard.append(text)
        return self.copy_ok

    def capture_selection(self, timeout_ms: int = 500) -> str | None:
        return None

    def default_method_for(self, process_name: str) -> str | None:
        return self.defaults.get(process_name)


# -- harness -----------------------------------------------------------------------------


@dataclass
class Harness:
    service: DeliveryService
    settings: SettingsStore
    history: HistoryStore
    polisher: FakePolisher
    inserter: FakeInserter
    spacing: FakeSpacing
    focus: dict[str, TargetWindow | None]
    results: list[DeliveryResult]

    def deliver(self, qtbot, info: SessionInfo, text: str, language: str | None = "en") -> DeliveryResult:
        expected = len(self.results) + 1
        self.service.submit(info, transcript(info, text, language))
        qtbot.waitUntil(lambda: len(self.results) >= expected, timeout=5000)
        return self.results[expected - 1]

    def paste_last(self, qtbot) -> DeliveryResult:
        expected = len(self.results) + 1
        self.service.paste_last()
        qtbot.waitUntil(lambda: len(self.results) >= expected, timeout=5000)
        return self.results[expected - 1]


def make_service(
    settings: SettingsStore,
    history: HistoryStore,
    polisher: FakePolisher,
    inserter: FakeInserter,
    spacing: FakeSpacing,
    focus: dict[str, TargetWindow | None],
    *,
    apply_rules: Callable[[str, FakeContext], str] = fake_apply_rules,
) -> DeliveryService:
    tools = TextTools(
        build_context=fake_build_context, categorize=fake_categorize, apply_rules=apply_rules, spacing=spacing
    )
    return DeliveryService(settings, history, polisher, inserter, text_tools=tools, get_target=lambda: focus["target"])


@pytest.fixture
def history(qapp, settings_store: SettingsStore, tmp_path: Path) -> Iterator[HistoryStore]:
    store = HistoryStore(tmp_path / "history.db", settings_store)
    yield store
    store.close()


@pytest.fixture
def h(qapp, settings_store: SettingsStore, history: HistoryStore) -> Iterator[Harness]:
    polisher, inserter, spacing = FakePolisher(), FakeInserter(), FakeSpacing()
    focus: dict[str, TargetWindow | None] = {"target": SLACK}
    service = make_service(settings_store, history, polisher, inserter, spacing, focus)
    results: list[DeliveryResult] = []
    service.delivered.connect(lambda result: results.append(result))
    yield Harness(service, settings_store, history, polisher, inserter, spacing, focus, results)
    service.shutdown()


def session(
    kind: SessionKind = SessionKind.DICTATION, target: TargetWindow | None = SLACK, selection: str | None = None
) -> SessionInfo:
    now = time.time()
    return SessionInfo(
        session_id=uuid.uuid4().hex,
        kind=kind,
        mode=RecordingMode.HOLD,
        target=target,
        started_at=now - 2.0,
        stopped_at=now,
        stopped_monotonic=time.monotonic(),
        selection=selection,
    )


def transcript(info: SessionInfo, text: str, language: str | None = "en") -> Transcript:
    return Transcript(
        session_id=info.session_id, text=text, audio_ms=2_000, speech_ms=1_500, inference_ms=80, language=language
    )


# -- dictation ---------------------------------------------------------------------------


def test_dictation_runs_rules_inserts_and_records_history(h: Harness, qtbot) -> None:
    info = session()
    result = h.deliver(qtbot, info, "um hello there")
    assert result.outcome is Outcome.INSERTED
    assert result.session_id == info.session_id
    assert result.final_text == "hello there"
    assert result.word_count == 2
    assert result.latency_ms >= 0
    assert result.polished is False
    assert result.message is None
    assert result.target is not None and result.target.category is AppCategory.MESSAGING

    [insert] = h.inserter.inserted
    assert insert.text == "hello there"
    assert insert.target == result.target
    assert insert.method == "paste"
    assert insert.restore_clipboard is True
    assert insert.on_focus_change == "clipboard"
    assert insert.thread == "whisp-delivery"
    assert h.polisher.polish_calls == []

    [entry] = h.history.list()
    assert entry.kind == "dictation"
    assert entry.raw_text == "um hello there"
    assert entry.final_text == "hello there"
    assert entry.word_count == 2
    assert (entry.app_name, entry.process_name, entry.category) == ("Slack", "slack.exe", "messaging")
    assert entry.outcome == "inserted"
    assert entry.created_at == pytest.approx(info.started_at)
    assert entry.model_id == h.settings.current.transcription.model_id
    assert (entry.audio_ms, entry.speech_ms) == (2_000, 1_500)


def test_smart_spacing_between_dictations(h: Harness, qtbot) -> None:
    h.deliver(qtbot, session(), "first part.")
    h.deliver(qtbot, session(), "second part")
    assert [i.text for i in h.inserter.inserted] == ["first part.", " second part"]
    assert h.results[-1].final_text == "second part"  # the space isn't part of the dictation
    h.settings.update({"text": {"smart_spacing": "never"}})
    h.deliver(qtbot, session(), "third part")
    assert h.inserter.inserted[-1].text == "third part"


def test_polish_runs_when_enabled_ready_and_not_verbatim(h: Harness, qtbot) -> None:
    h.settings.update({"polish": {"enabled": True}})
    result = h.deliver(qtbot, session(), "um hello there")
    assert result.polished is True
    assert result.final_text == "Hello there!"
    assert h.inserter.inserted[-1].text == "Hello there!"
    [(text, ctx)] = h.polisher.polish_calls
    assert text == "hello there"
    assert ctx.style == "casual"
    assert h.history.list()[0].polished is True


def test_polish_is_skipped_for_verbatim_apps(h: Harness, qtbot) -> None:
    h.settings.update({"polish": {"enabled": True}})
    result = h.deliver(qtbot, session(target=CODE), "um fix the bug")
    assert h.polisher.polish_calls == []
    assert result.final_text == "fix the bug"
    assert result.polished is False


def test_polish_is_skipped_when_disabled_or_not_ready(h: Harness, qtbot) -> None:
    h.deliver(qtbot, session(), "not enabled")
    h.settings.update({"polish": {"enabled": True}})
    h.polisher.ready = False
    h.deliver(qtbot, session(), "not ready")
    assert h.polisher.polish_calls == []


@pytest.mark.parametrize(
    "polish_with",
    [
        lambda text: FakePolishResult(text, False, "timed out"),
        lambda text: FakePolishResult("   ", True),
        raising(RuntimeError("server crashed")),
    ],
    ids=["fell-back", "empty", "raised"],
)
def test_polish_problems_fall_back_to_the_rule_output(h: Harness, qtbot, polish_with) -> None:
    h.settings.update({"polish": {"enabled": True}})
    h.polisher.polish_with = polish_with
    result = h.deliver(qtbot, session(), "um keep my words")
    assert result.outcome is Outcome.INSERTED
    assert result.final_text == "keep my words"
    assert result.polished is False


def test_insertion_method_resolution(h: Harness, qtbot) -> None:
    h.inserter.defaults = {"code.exe": "paste_shift_insert", "slack.exe": "paste_shift_insert"}
    h.settings.update({"insertion": {"method": "type", "app_methods": {"slack.exe": "type"}}})
    h.deliver(qtbot, session(target=SLACK), "user override wins")
    h.deliver(qtbot, session(target=CODE), "app default next")
    h.deliver(qtbot, session(target=NOTES), "general setting last")
    h.deliver(qtbot, session(target=None), "no target")
    assert [i.method for i in h.inserter.inserted] == ["type", "paste_shift_insert", "type", "type"]


@pytest.mark.parametrize(
    ("insertion", "copy_ok", "outcome", "message"),
    [
        (InsertionResult(ok=True, method="paste"), True, Outcome.INSERTED, None),
        (InsertionResult(ok=True, method="type", focus_changed=True), True, Outcome.INSERTED, None),
        (
            InsertionResult(ok=True, method="clipboard", focus_changed=True),
            True,
            Outcome.COPIED,
            delivery_module.MSG_COPIED_FOCUS,
        ),
        (InsertionResult(ok=True, method="clipboard"), True, Outcome.COPIED, delivery_module.MSG_COPIED),
        (
            InsertionResult(ok=False, method="paste", error="SendInput failed"),
            True,
            Outcome.COPIED,
            delivery_module.MSG_COPIED_FALLBACK,
        ),
        (
            InsertionResult(ok=False, method="paste", error="SendInput failed"),
            False,
            Outcome.FAILED,
            delivery_module.MSG_INSERT_FAILED,
        ),
        (
            InsertionResult(ok=False, method="clipboard", error="clipboard busy"),
            True,
            Outcome.FAILED,
            delivery_module.MSG_INSERT_FAILED,
        ),
    ],
    ids=["inserted", "inserted-anyway", "copied-focus", "copied", "fallback-copy", "failed", "clipboard-failed"],
)
def test_insertion_outcome_mapping(
    h: Harness, qtbot, insertion: InsertionResult, copy_ok: bool, outcome: Outcome, message: str | None
) -> None:
    h.inserter.result = lambda text, method: insertion
    h.inserter.copy_ok = copy_ok
    result = h.deliver(qtbot, session(), "where did my words go")
    assert result.outcome is outcome
    assert result.message == message
    assert result.final_text == "where did my words go"
    [entry] = h.history.list()
    assert entry.outcome == outcome.value
    assert entry.final_text == "where did my words go"  # kept so the user can recover it


def test_failure_message_without_history(h: Harness, qtbot) -> None:
    h.settings.update({"privacy": {"history_enabled": False}})
    h.inserter.result = lambda text, method: InsertionResult(ok=False, method="paste")
    h.inserter.copy_ok = False
    assert h.deliver(qtbot, session(), "lost words").message == delivery_module.MSG_INSERT_FAILED_NO_HISTORY


def test_copied_text_leaves_out_the_spacing_space(h: Harness, qtbot) -> None:
    h.deliver(qtbot, session(), "first.")
    h.inserter.result = lambda text, method: InsertionResult(ok=True, method="clipboard", focus_changed=True)
    result = h.deliver(qtbot, session(), "second")
    assert result.outcome is Outcome.COPIED
    assert h.inserter.inserted[-1].text == " second"
    assert h.inserter.clipboard[-1] == "second"


def test_inserter_exception_falls_back_to_the_clipboard(h: Harness, qtbot) -> None:
    h.inserter.fail_with = RuntimeError("driver exploded")
    result = h.deliver(qtbot, session(), "save me")
    assert result.outcome is Outcome.COPIED
    assert h.inserter.clipboard == ["save me"]


@pytest.mark.parametrize("text", ["", "   ", "um uh"])
def test_nothing_to_insert_is_no_speech(h: Harness, qtbot, text: str) -> None:
    result = h.deliver(qtbot, session(), text)
    assert result.outcome is Outcome.NO_SPEECH
    assert h.inserter.inserted == []
    assert h.history.list() == []


def test_broken_text_rules_still_insert_the_words(qapp, settings_store, history, qtbot) -> None:
    def broken_rules(raw: str, ctx: FakeContext) -> str:
        raise ValueError("bad regex")

    inserter, focus = FakeInserter(), {"target": SLACK}
    service = make_service(
        settings_store, history, FakePolisher(), inserter, FakeSpacing(), focus, apply_rules=broken_rules
    )
    results: list[DeliveryResult] = []
    service.delivered.connect(lambda r: results.append(r))
    try:
        info = session()
        service.submit(info, transcript(info, "  um raw words  "))
        qtbot.waitUntil(lambda: len(results) == 1, timeout=5000)
        assert results[0].outcome is Outcome.INSERTED
        assert inserter.inserted[0].text == "um raw words"
    finally:
        service.shutdown()


def test_unexpected_error_fails_one_session_and_the_thread_keeps_going(h: Harness, qtbot) -> None:
    broken, good = session(), session()
    h.service.submit(broken, Transcript(broken.session_id, None, 0, 0, 0))  # type: ignore[arg-type]
    h.service.submit(good, transcript(good, "still working"))
    qtbot.waitUntil(lambda: len(h.results) == 2, timeout=5000)
    first, second = h.results
    assert first.session_id == broken.session_id
    assert first.outcome is Outcome.FAILED
    assert first.message == delivery_module.MSG_UNEXPECTED
    assert second.session_id == good.session_id
    assert second.outcome is Outcome.INSERTED


def test_history_failure_does_not_change_the_outcome(h: Harness, qtbot, monkeypatch) -> None:
    def refuse(entry):
        raise RuntimeError("disk full")

    monkeypatch.setattr(h.history, "add", refuse)
    assert h.deliver(qtbot, session(), "inserted anyway").outcome is Outcome.INSERTED


def test_sessions_are_delivered_in_submission_order(h: Harness, qtbot) -> None:
    h.inserter.delay_s = 0.03
    infos = [session() for _ in range(6)]
    for n, info in enumerate(infos):
        h.service.submit(info, transcript(info, f"part {n}"))
    qtbot.waitUntil(lambda: len(h.results) == len(infos), timeout=5000)
    assert [r.session_id for r in h.results] == [i.session_id for i in infos]
    assert [i.text.strip() for i in h.inserter.inserted] == [f"part {n}" for n in range(6)]


# -- command mode ------------------------------------------------------------------------


def test_command_without_selection_asks_for_one(h: Harness, qtbot) -> None:
    for selection in (None, "", "   "):
        result = h.deliver(qtbot, session(SessionKind.COMMAND, selection=selection), "make it formal")
        assert result.outcome is Outcome.FAILED
        assert result.message == delivery_module.MSG_NO_SELECTION
    assert h.polisher.rewrite_calls == []
    assert h.inserter.inserted == []


def test_command_needs_ai_polish(h: Harness, qtbot) -> None:
    h.polisher.ready = False
    result = h.deliver(qtbot, session(SessionKind.COMMAND, selection="hello"), "make it formal")
    assert result.outcome is Outcome.FAILED
    assert result.message == delivery_module.MSG_POLISH_NOT_READY
    assert h.inserter.inserted == []


def test_command_rewrites_the_selection_and_pastes_over_it(h: Harness, qtbot) -> None:
    h.deliver(qtbot, session(), "earlier text.")  # spacing memory must not prefix a replacement
    info = session(SessionKind.COMMAND, selection="hello world")
    result = h.deliver(qtbot, info, "make it loud")
    assert result.outcome is Outcome.INSERTED
    assert result.kind is SessionKind.COMMAND
    assert result.final_text == "HELLO WORLD"
    assert result.polished is True
    [(selection, instruction, ctx)] = h.polisher.rewrite_calls
    assert (selection, instruction) == ("hello world", "make it loud")
    assert ctx.category is AppCategory.MESSAGING
    assert h.inserter.inserted[-1].text == "HELLO WORLD"
    entry = h.history.list()[0]
    assert entry.kind == "command"
    assert (entry.raw_text, entry.final_text, entry.word_count) == ("make it loud", "HELLO WORLD", 3)


@pytest.mark.parametrize(
    "rewrite_with",
    [
        lambda sel, ins: FakePolishResult(sel, False, "timed out"),
        lambda sel, ins: FakePolishResult("", True),
        raising(RuntimeError("server crashed")),
    ],
    ids=["fell-back", "empty", "raised"],
)
def test_failed_rewrite_leaves_the_selection_alone(h: Harness, qtbot, rewrite_with) -> None:
    h.polisher.rewrite_with = rewrite_with
    result = h.deliver(qtbot, session(SessionKind.COMMAND, selection="keep me"), "translate it")
    assert result.outcome is Outcome.FAILED
    assert result.message == delivery_module.MSG_REWRITE_FAILED
    assert h.inserter.inserted == []


def test_empty_command_is_no_speech(h: Harness, qtbot) -> None:
    result = h.deliver(qtbot, session(SessionKind.COMMAND, selection="text"), "  ")
    assert result.outcome is Outcome.NO_SPEECH
    assert h.polisher.rewrite_calls == []


# -- paste last / copy last ---------------------------------------------------------------


def test_paste_last_inserts_into_the_window_focused_now(h: Harness, qtbot) -> None:
    h.deliver(qtbot, session(target=SLACK), "um send this again")
    h.focus["target"] = CODE
    result = h.paste_last(qtbot)
    assert result.outcome is Outcome.INSERTED
    assert result.final_text == "send this again"
    last = h.inserter.inserted[-1]
    assert last.text == "send this again"
    assert last.target is not None and last.target.process_name == "code.exe"
    assert len(h.history.list()) == 1  # pasting again isn't a new dictation


def test_paste_last_works_with_history_off(h: Harness, qtbot) -> None:
    h.settings.update({"privacy": {"history_enabled": False}})
    h.deliver(qtbot, session(), "private words")
    assert h.history.last_text() is None
    h.focus["target"] = NOTES
    assert h.paste_last(qtbot).final_text == "private words"


def test_paste_last_falls_back_to_history(h: Harness, qtbot) -> None:
    h.deliver(qtbot, session(), "from an earlier run")
    fresh_inserter, focus = FakeInserter(), {"target": NOTES}
    fresh = make_service(h.settings, h.history, FakePolisher(), fresh_inserter, FakeSpacing(), focus)
    results: list[DeliveryResult] = []
    fresh.delivered.connect(lambda r: results.append(r))
    try:
        fresh.paste_last()
        qtbot.waitUntil(lambda: len(results) == 1, timeout=5000)
        assert results[0].outcome is Outcome.INSERTED
        assert fresh_inserter.inserted[0].text == "from an earlier run"
    finally:
        fresh.shutdown()


def test_paste_last_with_nothing_to_paste(h: Harness, qtbot) -> None:
    result = h.paste_last(qtbot)
    assert result.outcome is Outcome.FAILED
    assert result.message == delivery_module.MSG_NOTHING_YET
    assert h.inserter.inserted == []


@pytest.mark.parametrize("history_on", [True, False])
def test_clearing_history_forgets_the_last_text(h: Harness, qtbot, history_on: bool) -> None:
    h.settings.update({"privacy": {"history_enabled": history_on}})
    h.deliver(qtbot, session(), "words to forget")
    h.history.clear()
    result = h.paste_last(qtbot)
    assert result.outcome is Outcome.FAILED
    assert result.message == delivery_module.MSG_NOTHING_YET
    assert [i.text for i in h.inserter.inserted] == ["words to forget"]  # only the dictation itself
    h.service.copy_last()
    qtbot.waitUntil(lambda: len(h.results) == 3, timeout=5000)
    assert h.results[-1].message == delivery_module.MSG_NOTHING_YET
    assert h.inserter.clipboard == []
    # What's dictated after the clear can be pasted again as usual.
    h.deliver(qtbot, session(), "new words")
    assert h.paste_last(qtbot).final_text == "new words"


def test_failed_text_can_be_pasted_again(h: Harness, qtbot) -> None:
    h.inserter.result = lambda text, method: InsertionResult(ok=False, method="paste")
    h.inserter.copy_ok = False
    assert h.deliver(qtbot, session(), "try me again").outcome is Outcome.FAILED
    h.inserter.result = None
    result = h.paste_last(qtbot)
    assert result.outcome is Outcome.INSERTED
    assert h.inserter.inserted[-1].text == "try me again"


def test_copy_last(h: Harness, qtbot) -> None:
    h.deliver(qtbot, session(), "copy me")
    h.service.copy_last()
    qtbot.waitUntil(lambda: len(h.results) == 2, timeout=5000)
    assert h.results[-1].outcome is Outcome.COPIED
    assert h.results[-1].message == delivery_module.MSG_COPIED
    assert h.inserter.clipboard[-1] == "copy me"
    h.inserter.copy_ok = False
    h.service.copy_last()
    qtbot.waitUntil(lambda: len(h.results) == 3, timeout=5000)
    assert h.results[-1].outcome is Outcome.FAILED


# -- shutdown ----------------------------------------------------------------------------


def test_shutdown_finishes_queued_work_then_ignores_new_work(h: Harness) -> None:
    h.inserter.delay_s = 0.05
    for n in range(3):
        info = session()
        h.service.submit(info, transcript(info, f"queued {n}"))
    h.service.shutdown(timeout=5.0)
    assert [i.text.strip() for i in h.inserter.inserted] == ["queued 0", "queued 1", "queued 2"]
    late = session()
    h.service.submit(late, transcript(late, "too late"))
    h.service.paste_last()
    time.sleep(0.1)
    assert len(h.inserter.inserted) == 3


def test_shutdown_waits_no_longer_than_its_deadline(h: Harness) -> None:
    h.inserter.delay_s = 2.0
    info = session()
    h.service.submit(info, transcript(info, "slow app"))
    time.sleep(0.05)
    started = time.monotonic()
    h.service.shutdown(timeout=0.2)
    assert time.monotonic() - started < 1.0
