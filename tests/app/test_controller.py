"""AppController: the recording state machine, ordering, outcomes, command mode, pausing and shortcuts.

Every service the controller talks to is a small fake that follows the documented
interface and records what it was asked to do, in order, in one shared call log.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import pytest
from PySide6.QtCore import QObject, Signal

from whisp.app import controller as controller_module
from whisp.app.controller import AppController, AppState
from whisp.core.types import (
    AppCategory,
    DeliveryResult,
    EngineState,
    EngineStatus,
    Outcome,
    RecordingMode,
    SessionInfo,
    SessionKind,
    TargetWindow,
    Transcript,
)
from whisp.storage.settings import SettingsStore

SLACK = TargetWindow(hwnd=101, pid=11, process_name="slack.exe", app_name="Slack", title="general")
NOTES = TargetWindow(hwnd=303, pid=33, process_name="notes.exe", app_name="Notes", title="todo")

Calls = list[tuple]


# -- fakes -------------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeHotkeys(QObject):
    pressed = Signal(str)
    released = Signal(str)
    interrupted = Signal(str)
    recorded = Signal(str)
    binding_error = Signal(str, str)
    session_locked = Signal()

    def __init__(self, calls: Calls) -> None:
        super().__init__()
        self.calls = calls
        self.bindings: dict[str, str] | None = None
        self.errors: dict[str, str] = {}
        self.cancel_active = False
        self.paused = False

    def start(self) -> None:
        self.calls.append(("hotkeys.start",))

    def stop(self) -> None:
        self.calls.append(("hotkeys.stop",))

    def set_bindings(self, bindings: dict[str, str]) -> dict[str, str]:
        self.bindings = dict(bindings)
        self.calls.append(("hotkeys.set_bindings", dict(bindings)))
        return dict(self.errors)

    def set_cancel_active(self, active: bool) -> None:
        self.cancel_active = active
        self.calls.append(("hotkeys.set_cancel_active", active))

    def start_recording(self) -> None:
        self.calls.append(("hotkeys.start_recording",))

    def stop_recording(self) -> None:
        self.calls.append(("hotkeys.stop_recording",))

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.calls.append(("hotkeys.set_paused", paused))


class FakePipeline(QObject):
    level = Signal(float)
    preview = Signal(str, str)
    finished = Signal(object)
    no_speech = Signal(str)
    failed = Signal(str, str)

    def __init__(self, calls: Calls) -> None:
        super().__init__()
        self.calls = calls
        self.started: list[str] = []

    def start(self, session_id: str) -> None:
        self.started.append(session_id)
        self.calls.append(("pipeline.start", session_id))

    def stop(self, session_id: str) -> None:
        self.calls.append(("pipeline.stop", session_id))

    def cancel(self, session_id: str) -> None:
        self.calls.append(("pipeline.cancel", session_id))

    def shutdown(self) -> None:
        self.calls.append(("pipeline.shutdown",))

    def end_on_its_own(self, session_id: str, how: str) -> None:
        """What the real pipeline does when the mic is unplugged, stalls or sends only silence mid-recording."""
        if how == "finished":
            self.finished.emit(Transcript(session_id, "words before the mic went away", 4_000, 3_000, 90))
        elif how == "no_speech":
            self.no_speech.emit(session_id)
        else:
            self.failed.emit(session_id, SILENT_MIC)


SILENT_MIC = "The microphone is sending only silence. Check that it isn't muted."


class FakeOverlay(QObject):
    stop_clicked = Signal()
    cancel_clicked = Signal()
    idle_clicked = Signal()

    def __init__(self, calls: Calls) -> None:
        super().__init__()
        self.calls = calls

    def show_recording(self, kind: SessionKind, mode: RecordingMode) -> None:
        self.calls.append(("overlay.show_recording", kind, mode))

    def set_mode(self, mode: RecordingMode) -> None:
        self.calls.append(("overlay.set_mode", mode))

    def set_level(self, level: float) -> None:
        self.calls.append(("overlay.set_level", level))

    def set_preview(self, text: str) -> None:
        self.calls.append(("overlay.set_preview", text))

    def set_note(self, text: str | None) -> None:
        self.calls.append(("overlay.set_note", text))

    def show_processing(self) -> None:
        self.calls.append(("overlay.show_processing",))

    def show_outcome(self, outcome: Outcome, message: str | None = None) -> None:
        self.calls.append(("overlay.show_outcome", outcome, message))

    def show_notice(self, text: str, kind: str = "info") -> None:
        self.calls.append(("overlay.show_notice", text, kind))

    def set_target(self, target: TargetWindow | None) -> None:
        self.calls.append(("overlay.set_target", target))

    def set_hotkey_hint(self, keys: list[str]) -> None:
        self.calls.append(("overlay.set_hotkey_hint", list(keys)))

    def shutdown(self) -> None:
        self.calls.append(("overlay.shutdown",))


class FakeSounds:
    def __init__(self, calls: Calls) -> None:
        self.calls = calls

    def play(self, name: str) -> None:
        self.calls.append(("sounds.play", name))

    def shutdown(self) -> None:
        self.calls.append(("sounds.shutdown",))


class FakeDelivery(QObject):
    delivered = Signal(object)

    def __init__(self, calls: Calls) -> None:
        super().__init__()
        self.calls = calls
        self.submitted: list[tuple[SessionInfo, Transcript]] = []

    def submit(self, info: SessionInfo, transcript: Transcript) -> None:
        self.submitted.append((info, transcript))
        self.calls.append(("delivery.submit", info.session_id))

    def paste_last(self) -> None:
        self.calls.append(("delivery.paste_last",))

    def copy_last(self) -> None:
        self.calls.append(("delivery.copy_last",))

    def shutdown(self) -> None:
        self.calls.append(("delivery.shutdown",))


class FakeAsr(QObject):
    status_changed = Signal(object)

    def __init__(self, calls: Calls) -> None:
        super().__init__()
        self.calls = calls
        self._status = EngineStatus(EngineState.READY, model_id="parakeet-tdt-0.6b-v2")

    @property
    def status(self) -> EngineStatus:
        return self._status

    def set_state(self, state: EngineState, message: str | None = None, *, emit: bool = True) -> None:
        self._status = EngineStatus(state, model_id="parakeet-tdt-0.6b-v2", message=message)
        if emit:
            self.status_changed.emit(self._status)

    def ensure_loaded(self) -> None:
        self.calls.append(("asr.ensure_loaded",))


@dataclass
class FakeModels:
    installed: set[tuple[str, str]] = field(default_factory=set)

    def is_installed(self, model_id: str, variant: str) -> bool:
        if variant not in ("gpu", "cpu"):
            raise KeyError(variant)
        return (model_id, variant) in self.installed


class FakePolisher:
    def __init__(self) -> None:
        self.ready = True

    def is_ready(self) -> bool:
        return self.ready


class FakeMute:
    def __init__(self, calls: Calls) -> None:
        self.calls = calls

    def mute(self) -> None:
        self.calls.append(("mute.mute",))

    def restore(self) -> None:
        self.calls.append(("mute.restore",))


class FakeInserter:
    def __init__(self) -> None:
        self.selection: str | None = "selected words"
        self.capture_threads: list[str] = []
        self.release = threading.Event()
        self.release.set()

    def capture_selection(self, timeout_ms: int = 500) -> str | None:
        self.capture_threads.append(threading.current_thread().name)
        self.release.wait(5)
        return self.selection


def fake_categorize(process_name: str, title: str, overrides: Mapping[str, str]) -> AppCategory:
    if process_name in overrides:
        return AppCategory(overrides[process_name])
    return {"slack.exe": AppCategory.MESSAGING}.get(process_name, AppCategory.OTHER)


def fake_display_chord(chord: str) -> list[str]:
    if "!" in chord:
        raise ValueError("not a chord")
    return [part.capitalize() for part in chord.split("+")]


# -- harness -----------------------------------------------------------------------------


@dataclass
class Env:
    controller: AppController
    settings: SettingsStore
    calls: Calls
    hotkeys: FakeHotkeys
    pipeline: FakePipeline
    overlay: FakeOverlay
    delivery: FakeDelivery
    asr: FakeAsr
    models: FakeModels
    polisher: FakePolisher
    inserter: FakeInserter
    mute: FakeMute
    clock: FakeClock
    wall: FakeClock
    focus: dict[str, TargetWindow | Exception | None]
    notifications: list[tuple[str, str, str]]
    windows: list[tuple[str, str]]

    def called(self, name: str) -> list[tuple]:
        return [c[1:] for c in self.calls if c[0] == name]

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def press(self, action: str = "dictate") -> None:
        self.hotkeys.pressed.emit(action)

    def release(self, action: str = "dictate", *, after: float = 0.0) -> None:
        self.clock.advance(after)
        self.hotkeys.released.emit(action)

    def hold(self, action: str = "dictate", seconds: float = 1.5) -> str:
        """Press, hold and release a recording shortcut; returns the session id."""
        self.press(action)
        session_id = self.pipeline.started[-1]
        self.release(action, after=seconds)
        return session_id

    def finish(self, session_id: str, text: str = "hello there") -> None:
        self.pipeline.finished.emit(Transcript(session_id, text, 1_500, 1_200, 90))

    def submitted_ids(self) -> list[str]:
        return [info.session_id for info, _ in self.delivery.submitted]

    def deliver(self, session_id: str, outcome: Outcome = Outcome.INSERTED, message: str | None = None) -> None:
        self.delivery.delivered.emit(
            DeliveryResult(
                session_id, SessionKind.DICTATION, outcome, "hello there", 2, 300, target=SLACK, message=message
            )
        )


def make_env(settings: SettingsStore) -> Env:
    calls: Calls = []
    focus: dict[str, TargetWindow | Exception | None] = {"target": SLACK}

    def get_target() -> TargetWindow | None:
        calls.append(("foreground.get_target",))
        current = focus["target"]
        if isinstance(current, Exception):
            raise current
        return current

    env_parts = {
        "hotkeys": FakeHotkeys(calls),
        "pipeline": FakePipeline(calls),
        "overlay": FakeOverlay(calls),
        "delivery": FakeDelivery(calls),
        "asr": FakeAsr(calls),
        "models": FakeModels(),
        "polisher": FakePolisher(),
        "inserter": FakeInserter(),
        "mute": FakeMute(calls),
    }
    clock, wall = FakeClock(1_000.0), FakeClock(1_760_000_000.0)
    controller = AppController(
        settings,
        sounds=FakeSounds(calls),
        get_target=get_target,
        categorize=fake_categorize,
        display_chord=fake_display_chord,
        clock=clock,
        wall_clock=wall,
        **env_parts,
    )
    notifications: list[tuple[str, str, str]] = []
    windows: list[tuple[str, str]] = []
    controller.notification.connect(lambda title, message, kind: notifications.append((title, message, kind)))
    controller.open_window_requested.connect(lambda page, section: windows.append((page, section)))
    controller.start()
    return Env(
        controller=controller,
        settings=settings,
        calls=calls,
        clock=clock,
        wall=wall,
        focus=focus,
        notifications=notifications,
        windows=windows,
        **env_parts,
    )


@pytest.fixture
def env(qapp, settings_store: SettingsStore) -> Iterator[Env]:
    e = make_env(settings_store)
    yield e
    e.controller.shutdown()


# -- muting other apps -------------------------------------------------------------------


def test_other_apps_are_muted_shortly_after_recording_starts_and_restored_on_stop(env: Env, qtbot) -> None:
    env.press("hands_free")
    assert env.called("mute.mute") == []  # not straight away
    qtbot.waitUntil(lambda: env.called("mute.mute") == [()], timeout=2000)
    env.press("hands_free")  # stop
    assert env.called("mute.restore") == [()]


def test_an_accidental_shortcut_never_mutes(env: Env, qtbot) -> None:
    env.press("dictate")
    env.clock.advance(0.05)
    env.hotkeys.interrupted.emit("dictate")  # it was Ctrl+Win+D, not dictation
    qtbot.waitUntil(lambda: not env.controller.is_recording, timeout=1000)
    qtbot.wait(controller_module.MUTE_DELAY_MS + 100)
    assert env.called("mute.mute") == []


def test_cancel_restores_other_apps(env: Env, qtbot) -> None:
    env.press("hands_free")
    qtbot.waitUntil(lambda: env.called("mute.mute") == [()], timeout=2000)
    env.press("cancel")
    assert env.called("mute.restore") == [()]


def test_no_muting_when_the_setting_is_off(env: Env, qtbot) -> None:
    env.settings.update({"audio": {"mute_other_apps": False}})
    env.press("hands_free")
    qtbot.wait(controller_module.MUTE_DELAY_MS + 100)
    assert env.called("mute.mute") == []


# -- starting ----------------------------------------------------------------------------


def test_start_captures_the_window_first_then_starts_everything_in_order(env: Env) -> None:
    env.calls.clear()
    env.press()
    assert env.names() == [
        "foreground.get_target",
        "pipeline.start",
        "hotkeys.set_cancel_active",
        "overlay.set_target",
        "overlay.show_recording",
        "sounds.play",
    ]
    session_id = env.pipeline.started[0]
    assert len(session_id) == 32 and int(session_id, 16) >= 0  # uuid4 hex
    [target] = [c[0] for c in env.called("overlay.set_target")]
    assert target.category is AppCategory.MESSAGING  # categorised on capture
    assert env.called("overlay.show_recording") == [(SessionKind.DICTATION, RecordingMode.HOLD_PENDING)]
    assert env.called("sounds.play") == [("start",)]
    assert env.hotkeys.cancel_active is True
    assert env.controller.is_recording
    assert env.controller.state is AppState.RECORDING


def test_quick_tap_locks_hands_free_and_next_press_stops(env: Env) -> None:
    env.press()
    session_id = env.pipeline.started[0]
    env.release(after=0.1)
    assert env.controller.recording_mode is RecordingMode.HANDS_FREE
    assert env.called("overlay.set_mode") == [(RecordingMode.HANDS_FREE,)]
    assert ("lock",) in env.called("sounds.play")
    assert env.called("pipeline.stop") == []

    env.clock.advance(5)
    env.press()
    assert env.called("pipeline.stop") == [(session_id,)]
    assert env.hotkeys.cancel_active is False
    assert ("overlay.show_processing",) in env.calls
    assert env.called("sounds.play")[-1] == ("stop",)
    assert not env.controller.is_recording
    env.release(after=0.05)  # the release of the stopping press does nothing
    assert len(env.pipeline.started) == 1


def test_hold_then_release_stops(env: Env) -> None:
    session_id = env.hold(seconds=1.5)
    assert env.called("pipeline.stop") == [(session_id,)]
    assert not env.controller.is_recording
    assert env.controller.state is AppState.PROCESSING
    env.finish(session_id)
    [(info, transcript)] = env.delivery.submitted
    assert info.mode is RecordingMode.HOLD
    assert info.kind is SessionKind.DICTATION
    assert info.started_at == 1_760_000_000.0
    assert info.stopped_monotonic == 1_001.5
    assert info.target is not None and info.target.process_name == "slack.exe"
    assert transcript.text == "hello there"


def test_releasing_a_different_shortcut_is_ignored(env: Env) -> None:
    env.press("dictate")
    env.release("command", after=2.0)
    assert env.controller.is_recording
    env.release("dictate")
    assert not env.controller.is_recording


def test_hands_free_shortcut_toggles_and_any_recording_shortcut_stops(env: Env) -> None:
    env.press("hands_free")
    assert env.called("overlay.show_recording") == [(SessionKind.DICTATION, RecordingMode.HANDS_FREE)]
    env.release("hands_free", after=0.1)
    assert env.controller.is_recording
    env.press("hands_free")
    assert not env.controller.is_recording
    env.press("hands_free")
    env.press("command")  # stops instead of starting a command
    assert not env.controller.is_recording
    assert len(env.pipeline.started) == 2


def test_esc_cancels(env: Env) -> None:
    env.press()
    session_id = env.pipeline.started[0]
    env.press("cancel")
    assert env.called("pipeline.cancel") == [(session_id,)]
    assert env.hotkeys.cancel_active is False
    assert env.called("overlay.show_outcome") == [(Outcome.CANCELLED, None)]
    assert env.called("sounds.play")[-1] == ("cancel",)
    assert env.controller.state is AppState.IDLE
    env.finish(session_id)  # nothing from a cancelled session is delivered
    assert env.delivery.submitted == []


def test_esc_does_nothing_when_not_recording(env: Env) -> None:
    env.press("cancel")
    assert env.called("pipeline.cancel") == []


def test_overlay_buttons(env: Env) -> None:
    env.press()  # holding
    env.overlay.stop_clicked.emit()
    assert env.controller.is_recording  # the stop button is only for hands-free
    env.overlay.cancel_clicked.emit()
    assert not env.controller.is_recording
    assert env.called("sounds.play")[-1] == ("cancel",)

    env.overlay.idle_clicked.emit()
    assert env.controller.recording_mode is RecordingMode.HANDS_FREE
    env.overlay.stop_clicked.emit()
    assert not env.controller.is_recording
    assert len(env.called("pipeline.stop")) == 1


def test_interrupted_early_cancels_silently_after_a_short_grace(env: Env, qtbot) -> None:
    env.press()
    env.clock.advance(0.4)
    env.hotkeys.interrupted.emit("dictate")
    assert env.controller.is_recording  # waiting to see if a longer Whisp shortcut follows
    qtbot.waitUntil(lambda: not env.controller.is_recording, timeout=1000)
    assert len(env.called("pipeline.cancel")) == 1
    assert ("cancel",) not in env.called("sounds.play")


def test_ctrl_win_then_space_keeps_the_recording_as_hands_free(env: Env, qtbot) -> None:
    env.press("dictate")
    session_id = env.pipeline.started[0]
    env.clock.advance(0.1)
    env.hotkeys.interrupted.emit("dictate")
    env.press("hands_free")
    qtbot.wait(250)  # past the grace period: nothing gets cancelled
    assert env.controller.is_recording
    assert env.controller.recording_mode is RecordingMode.HANDS_FREE
    assert env.pipeline.started == [session_id]  # same recording, no restart
    assert env.called("pipeline.cancel") == []
    assert ("lock",) in env.called("sounds.play")
    env.release("dictate")  # letting go of Ctrl+Win doesn't stop hands-free
    assert env.controller.is_recording
    env.press("hands_free")
    assert not env.controller.is_recording
    assert env.called("pipeline.stop") == [(session_id,)]


def test_letting_go_during_the_interruption_grace_still_cancels_silently(env: Env, qtbot) -> None:
    # Ctrl+Win+D tapped quickly: D interrupts, and Ctrl+Win are let go before the grace period ends.
    env.press("dictate")
    [session_id] = env.pipeline.started
    env.clock.advance(0.1)
    env.hotkeys.interrupted.emit("dictate")
    env.release("dictate", after=0.05)
    assert env.controller.recording_mode is not RecordingMode.HANDS_FREE  # not a tap that locks hands-free
    qtbot.waitUntil(lambda: not env.controller.is_recording, timeout=1000)
    assert env.called("pipeline.cancel") == [(session_id,)]
    assert env.called("pipeline.stop") == []
    assert env.called("sounds.play") == [("start",)]  # no lock, stop or cancel sound


def test_letting_go_of_a_longer_hold_during_the_interruption_grace_delivers_nothing(env: Env, qtbot) -> None:
    # Held past the tap threshold, then Ctrl+Win+Right to switch desktops, let go at once.
    env.press("dictate")
    [session_id] = env.pipeline.started
    env.clock.advance(0.5)
    env.hotkeys.interrupted.emit("dictate")
    env.release("dictate", after=0.05)
    assert env.called("pipeline.stop") == []
    qtbot.waitUntil(lambda: not env.controller.is_recording, timeout=1000)
    assert env.called("pipeline.cancel") == [(session_id,)]
    env.finish(session_id)
    assert env.delivery.submitted == []
    assert ("stop",) not in env.called("sounds.play")


@pytest.mark.parametrize("longer", ["hands_free", "command"])
def test_a_longer_shortcut_after_the_tap_threshold_keeps_the_recording(env: Env, qtbot, longer: str) -> None:
    # Ctrl+Win held a little longer than a tap before Space (or Alt) joins: still one keypress.
    env.press("dictate")
    [session_id] = env.pipeline.started
    env.clock.advance(0.5)
    qtbot.waitUntil(lambda: env.controller.recording_mode is RecordingMode.HOLD, timeout=2000)
    env.hotkeys.interrupted.emit("dictate")
    env.press(longer)
    qtbot.wait(250)  # past the grace period: nothing gets cancelled
    assert env.controller.is_recording
    assert env.pipeline.started == [session_id]
    assert env.called("pipeline.cancel") == []
    if longer == "hands_free":
        assert env.controller.recording_mode is RecordingMode.HANDS_FREE
    else:
        assert env.called("overlay.show_recording")[-1] == (SessionKind.COMMAND, RecordingMode.HOLD)
        env.release("command", after=1.0)
        assert env.called("pipeline.stop") == [(session_id,)]


def test_other_apps_stay_unmuted_while_an_interrupted_shortcut_is_decided(
    qapp, settings_store, monkeypatch, qtbot
) -> None:
    # The mute delay can run out during the grace period (Ctrl+Win held a moment, then D).
    monkeypatch.setattr(controller_module, "MUTE_DELAY_MS", 50)
    monkeypatch.setattr(controller_module, "INTERRUPT_GRACE_MS", 400)
    e = make_env(settings_store)
    try:
        e.press("dictate")
        e.clock.advance(0.2)
        e.hotkeys.interrupted.emit("dictate")
        qtbot.waitUntil(lambda: not e.controller.is_recording, timeout=3000)
        assert e.called("mute.mute") == []
        # When the longer Whisp shortcut does follow, the recording gets its quiet after all.
        e.press("dictate")
        e.clock.advance(0.1)
        e.hotkeys.interrupted.emit("dictate")
        e.press("hands_free")
        qtbot.waitUntil(lambda: e.called("mute.mute") == [()], timeout=3000)
    finally:
        e.controller.shutdown()


def test_hands_free_press_arriving_before_the_interruption_also_converts(env: Env, qtbot) -> None:
    env.press("dictate")
    env.clock.advance(0.1)
    env.press("hands_free")
    env.hotkeys.interrupted.emit("dictate")  # now ignored: the session belongs to hands-free
    qtbot.wait(250)
    assert env.controller.recording_mode is RecordingMode.HANDS_FREE
    assert env.called("pipeline.cancel") == []


def test_pressing_ctrl_win_space_to_stop_hands_free_doesnt_start_a_new_recording(env: Env, qtbot) -> None:
    env.press("hands_free")
    [session_id] = env.pipeline.started
    env.clock.advance(5.0)
    # Ctrl+Win+Space arrives as: dictate pressed (stops), interrupted, then hands_free pressed.
    env.press("dictate")
    env.hotkeys.interrupted.emit("dictate")
    env.clock.advance(0.05)
    env.press("hands_free")
    qtbot.wait(250)
    assert not env.controller.is_recording
    assert env.pipeline.started == [session_id]
    assert env.called("pipeline.stop") == [(session_id,)]


def test_a_new_hands_free_press_after_the_gesture_window_starts_recording(env: Env) -> None:
    env.press("hands_free")
    env.clock.advance(3.0)
    env.press("dictate")  # stops hands-free
    env.clock.advance(controller_module.STOP_GESTURE_S + 0.1)
    env.press("hands_free")
    assert env.controller.is_recording
    assert len(env.pipeline.started) == 2


def test_an_unrelated_shortcut_right_after_stopping_still_works(env: Env) -> None:
    env.settings.update({"hotkeys": {"hands_free": "f8"}})
    env.press("hands_free")
    env.clock.advance(2.0)
    env.press("dictate")  # stops hands-free
    env.clock.advance(0.1)
    env.press("hands_free")  # F8 doesn't extend Ctrl+Win, so this is a deliberate new recording
    assert env.controller.is_recording
    assert len(env.pipeline.started) == 2


def test_ctrl_win_then_alt_turns_the_recording_into_a_command(env: Env, qtbot) -> None:
    env.press("dictate")
    session_id = env.pipeline.started[0]
    env.clock.advance(0.05)
    env.hotkeys.interrupted.emit("dictate")
    env.press("command")
    qtbot.wait(250)
    assert env.controller.is_recording
    assert env.called("overlay.show_recording")[-1] == (SessionKind.COMMAND, RecordingMode.HOLD_PENDING)
    env.release("command", after=1.5)
    env.finish(session_id, "make it shorter")
    qtbot.waitUntil(lambda: len(env.delivery.submitted) == 1, timeout=3000)
    info, _ = env.delivery.submitted[0]
    assert info.kind is SessionKind.COMMAND
    assert info.selection == "selected words"


def test_one_minute_warning_before_the_length_limit(qapp, settings_store, monkeypatch, qtbot) -> None:
    monkeypatch.setattr(controller_module, "MINUTE_MS", 200)  # a "minute" lasts 0.2 s here
    settings_store.update({"audio": {"max_recording_minutes": 5}})
    e = make_env(settings_store)
    try:
        e.press("hands_free")
        warning = (controller_module.MSG_ONE_MINUTE_LEFT, "info")
        qtbot.waitUntil(lambda: warning in e.called("overlay.show_notice"), timeout=3000)
        assert e.controller.is_recording  # warned, still recording
        assert ("lock",) in e.called("sounds.play")
    finally:
        e.controller.shutdown()


def test_no_warning_when_the_limit_is_one_minute(qapp, settings_store, monkeypatch, qtbot) -> None:
    monkeypatch.setattr(controller_module, "MINUTE_MS", 50)
    settings_store.update({"audio": {"max_recording_minutes": 1}})
    e = make_env(settings_store)
    try:
        e.press("hands_free")
        qtbot.waitUntil(lambda: not e.controller.is_recording, timeout=3000)
        assert (controller_module.MSG_ONE_MINUTE_LEFT, "info") not in e.called("overlay.show_notice")
    finally:
        e.controller.shutdown()


def test_interrupted_later_or_hands_free_is_ignored(env: Env) -> None:
    env.press()
    env.clock.advance(1.2)
    env.hotkeys.interrupted.emit("dictate")
    assert env.controller.is_recording
    env.release()
    env.press("hands_free")
    env.hotkeys.interrupted.emit("hands_free")
    assert env.controller.is_recording
    assert env.called("pipeline.cancel") == []


# -- several sessions --------------------------------------------------------------------


def test_a_new_recording_can_start_while_the_last_one_is_processing(env: Env) -> None:
    first = env.hold()
    env.press()
    assert env.controller.is_recording
    assert env.pipeline.started == [first, env.pipeline.started[1]]


def test_overlapping_sessions_are_delivered_in_start_order(env: Env) -> None:
    a = env.hold()
    b = env.hold()
    c = env.hold()
    env.finish(c, "third")
    env.finish(b, "second")
    assert env.delivery.submitted == []  # both wait for the first one
    env.finish(a, "first")
    assert env.submitted_ids() == [a, b, c]
    assert [t.text for _, t in env.delivery.submitted] == ["first", "second", "third"]


def test_a_session_without_speech_doesnt_hold_up_later_ones(env: Env) -> None:
    a = env.hold()
    b = env.hold()
    env.finish(b)
    env.pipeline.no_speech.emit(a)
    assert env.submitted_ids() == [b]


def test_no_speech_shows_on_the_overlay_unless_recording(env: Env) -> None:
    a = env.hold()
    env.pipeline.no_speech.emit(a)
    assert env.called("overlay.show_outcome") == [(Outcome.NO_SPEECH, None)]
    b = env.hold()
    env.press()  # recording again
    env.pipeline.no_speech.emit(b)
    assert env.called("overlay.show_outcome") == [(Outcome.NO_SPEECH, None)]


def test_pipeline_failure_is_shown_and_notified(env: Env) -> None:
    a = env.hold()
    env.pipeline.failed.emit(a, "The microphone stopped.")
    assert env.called("overlay.show_outcome") == [(Outcome.FAILED, "The microphone stopped.")]
    assert env.called("sounds.play")[-1] == ("error",)
    assert env.notifications == [("Dictation didn't work", "The microphone stopped.", "error")]
    assert env.controller.state is AppState.IDLE


def test_no_notifications_when_they_are_turned_off(env: Env) -> None:
    env.settings.update({"general": {"show_notifications": False}})
    a = env.hold()
    env.pipeline.failed.emit(a, "The microphone stopped.")
    assert env.notifications == []


def test_failure_at_start_ends_the_recording(env: Env) -> None:
    env.press()
    session_id = env.pipeline.started[0]
    env.pipeline.failed.emit(session_id, "Couldn't start the microphone.")
    assert not env.controller.is_recording
    assert env.hotkeys.cancel_active is False
    assert env.called("pipeline.cancel") == [(session_id,)]
    assert env.called("overlay.show_outcome") == [(Outcome.FAILED, "Couldn't start the microphone.")]
    env.release(after=2.0)  # releasing the key afterwards changes nothing
    assert env.called("pipeline.stop") == []


@pytest.mark.parametrize("how", ["finished", "no_speech", "failed"])
@pytest.mark.parametrize("action", ["dictate", "hands_free"])
def test_pipeline_can_end_a_recording_on_its_own(qapp, settings_store, monkeypatch, qtbot, how, action) -> None:
    # The max-length timer is armed for 30 ms here: it must be stopped when the pipeline ends the session.
    monkeypatch.setattr(controller_module, "MINUTE_MS", 30)
    settings_store.update({"audio": {"max_recording_minutes": 1}})
    e = make_env(settings_store)
    states: list[AppState] = []
    e.controller.state_changed.connect(lambda: states.append(e.controller.state))
    try:
        e.press(action)
        session_id = e.pipeline.started[0]
        e.clock.advance(2.0)
        e.pipeline.end_on_its_own(session_id, how)

        assert not e.controller.is_recording
        assert e.hotkeys.cancel_active is False
        assert e.controller.state is not AppState.RECORDING
        assert states[-1] is e.controller.state
        if how == "finished":
            assert ("overlay.show_processing",) in e.calls
            [(info, _)] = e.delivery.submitted
            assert info.session_id == session_id
            assert info.stopped_monotonic == 1_002.0
            assert info.mode in (RecordingMode.HOLD, RecordingMode.HANDS_FREE)
            e.deliver(session_id)
            assert e.called("overlay.show_outcome") == [(Outcome.INSERTED, None)]
        elif how == "no_speech":
            assert e.called("overlay.show_outcome") == [(Outcome.NO_SPEECH, None)]
        else:
            assert e.called("overlay.show_outcome") == [(Outcome.FAILED, SILENT_MIC)]
            assert e.notifications == [("Dictation didn't work", SILENT_MIC, "error")]
        assert e.controller.state is AppState.IDLE

        qtbot.wait(120)  # long enough for a forgotten max-length timer to fire
        e.release(action, after=0.5)  # letting go of the key afterwards changes nothing
        assert e.called("pipeline.stop") == []
        assert not e.controller.is_recording
        assert e.called("overlay.show_notice") == []  # no "stopped at the limit" notice

        e.press("hands_free")  # and the next recording starts normally
        assert e.controller.is_recording
        assert len(e.pipeline.started) == 2
    finally:
        e.controller.shutdown()


def test_level_and_preview_go_to_the_overlay_only_while_recording(env: Env) -> None:
    env.press()
    session_id = env.pipeline.started[0]
    env.pipeline.level.emit(0.5)
    env.pipeline.preview.emit(session_id, "hello")
    env.pipeline.preview.emit("another-session", "stale")
    assert env.called("overlay.set_level") == [(0.5,)]
    assert env.called("overlay.set_preview") == [("hello",)]
    env.release(after=2)
    env.pipeline.level.emit(0.7)
    env.pipeline.preview.emit(session_id, "late")
    assert env.called("overlay.set_level") == [(0.5,)]
    assert env.called("overlay.set_preview") == [("hello",)]


# -- outcomes ----------------------------------------------------------------------------


def test_delivered_outcome_is_shown_and_summarised(env: Env) -> None:
    a = env.hold()
    env.finish(a)
    assert env.controller.state is AppState.PROCESSING
    env.deliver(a)
    assert env.called("overlay.show_outcome") == [(Outcome.INSERTED, None)]
    assert env.controller.state is AppState.IDLE
    assert env.controller.status_message() == "Ready. Hold Ctrl + Win to dictate.\nLast: 2 words into Slack."


def test_delivery_doesnt_disturb_a_new_recording(env: Env) -> None:
    a = env.hold()
    env.finish(a)
    env.press()
    env.deliver(a, Outcome.INSERTED)
    assert env.called("overlay.show_outcome") == []
    assert env.called("overlay.show_notice") == []


def test_clipboard_and_failures_are_mentioned_on_a_live_recording(env: Env) -> None:
    a = env.hold()
    b = env.hold()
    env.finish(a)
    env.finish(b)
    env.press()  # recording again
    env.deliver(a, Outcome.COPIED, "The window changed. Copied — press Ctrl + V to paste.")
    env.deliver(b, Outcome.FAILED, "Couldn't insert your text.")
    assert env.called("overlay.show_outcome") == []
    assert env.called("overlay.show_notice") == [
        ("The window changed. Copied — press Ctrl + V to paste.", "info"),
        ("Couldn't insert your text.", "error"),
    ]
    assert ("error",) not in env.called("sounds.play")  # no sounds while recording
    assert env.notifications == [("Dictation didn't work", "Couldn't insert your text.", "error")]


def test_an_earlier_failure_doesnt_hide_work_still_in_progress(env: Env) -> None:
    a = env.hold()
    b = env.hold()
    env.finish(a)
    env.finish(b)
    env.deliver(a, Outcome.FAILED, "Couldn't insert your text.")
    assert env.called("overlay.show_notice") == [("Couldn't insert your text.", "error")]
    assert env.called("overlay.show_outcome") == []
    assert env.called("sounds.play")[-1] == ("error",)
    env.deliver(b)
    assert env.called("overlay.show_outcome") == [(Outcome.INSERTED, None)]


def test_microphone_failure_ends_the_recording_view_even_while_others_process(env: Env) -> None:
    a = env.hold()
    env.finish(a)
    env.press()
    b = env.pipeline.started[-1]
    env.pipeline.failed.emit(b, "Couldn't start the microphone.")
    assert env.called("overlay.show_outcome") == [(Outcome.FAILED, "Couldn't start the microphone.")]
    assert not env.controller.is_recording


def test_inserted_outcome_waits_while_more_text_is_coming(env: Env) -> None:
    a = env.hold()
    b = env.hold()
    env.finish(a)
    env.deliver(a)
    assert env.called("overlay.show_outcome") == []  # still processing b
    env.finish(b)
    env.deliver(b)
    assert env.called("overlay.show_outcome") == [(Outcome.INSERTED, None)]


def test_copied_and_failed_outcomes(env: Env) -> None:
    a = env.hold()
    env.finish(a)
    env.deliver(a, Outcome.COPIED, "Copied — press Ctrl + V to paste.")
    assert env.called("overlay.show_outcome") == [(Outcome.COPIED, "Copied — press Ctrl + V to paste.")]
    assert env.notifications == []
    b = env.hold()
    env.finish(b)
    env.deliver(b, Outcome.FAILED, "Couldn't insert your text.")
    assert env.called("overlay.show_outcome")[-1] == (Outcome.FAILED, "Couldn't insert your text.")
    assert env.called("sounds.play")[-1] == ("error",)
    assert env.notifications == [("Dictation didn't work", "Couldn't insert your text.", "error")]


def test_max_length_stops_with_a_notice(qapp, settings_store, monkeypatch, qtbot) -> None:
    monkeypatch.setattr(controller_module, "MINUTE_MS", 20)
    settings_store.update({"audio": {"max_recording_minutes": 3}})
    e = make_env(settings_store)
    try:
        e.press("hands_free")
        qtbot.waitUntil(lambda: not e.controller.is_recording, timeout=3000)
        assert len(e.called("pipeline.stop")) == 1
        message = "Recording stopped at the 3-minute limit. You can change it in Settings."
        assert e.called("overlay.show_notice") == [(controller_module.MSG_ONE_MINUTE_LEFT, "info"), (message, "info")]
        assert e.notifications == [("Recording stopped", message, "info")]
    finally:
        e.controller.shutdown()


# -- pausing, paste last -----------------------------------------------------------------


def test_paused_ignores_shortcuts(env: Env) -> None:
    env.controller.toggle_paused()
    assert env.hotkeys.paused is True
    assert env.controller.state is AppState.PAUSED
    env.press()
    env.press("hands_free")
    env.press("paste_last")
    env.controller.start_hands_free()
    assert env.pipeline.started == []
    assert env.called("delivery.paste_last") == []
    env.controller.toggle_paused()
    assert env.hotkeys.paused is False
    env.press()
    assert env.controller.is_recording


def test_pausing_finishes_the_recording_in_progress(env: Env) -> None:
    env.press("hands_free")
    session_id = env.pipeline.started[0]
    env.controller.toggle_paused()
    assert env.called("pipeline.stop") == [(session_id,)]
    assert env.called("pipeline.cancel") == []


def test_locking_the_pc_stops_a_hands_free_recording_and_keeps_its_words(env: Env) -> None:
    env.press("hands_free")
    [session_id] = env.pipeline.started
    env.hotkeys.session_locked.emit()  # Win+L, or the PC locked itself while nobody was there
    assert not env.controller.is_recording
    assert env.called("pipeline.stop") == [(session_id,)]
    assert env.called("pipeline.cancel") == []
    assert env.hotkeys.cancel_active is False
    env.finish(session_id)
    assert env.submitted_ids() == [session_id]
    env.hotkeys.session_locked.emit()  # not recording: nothing to do
    assert len(env.called("pipeline.stop")) == 1


def test_paste_and_copy_last(env: Env) -> None:
    env.press("paste_last")
    env.controller.copy_last()
    assert ("delivery.paste_last",) in env.calls
    assert ("delivery.copy_last",) in env.calls


# -- command mode ------------------------------------------------------------------------


def test_command_is_refused_when_polish_isnt_ready(env: Env) -> None:
    env.polisher.ready = False
    env.press("command")
    assert env.pipeline.started == []
    assert env.called("overlay.show_notice") == [(controller_module.MSG_COMMAND_NEEDS_POLISH, "warning")]
    assert env.inserter.capture_threads == []


def test_command_reads_the_selection_on_a_worker_thread(env: Env, qtbot) -> None:
    env.inserter.selection = "make this formal please"
    env.press("command")
    assert env.called("overlay.show_recording") == [(SessionKind.COMMAND, RecordingMode.HOLD_PENDING)]
    session_id = env.pipeline.started[0]
    env.release("command", after=2.0)
    env.finish(session_id, "make it formal")
    qtbot.waitUntil(lambda: len(env.delivery.submitted) == 1, timeout=3000)
    [(info, transcript)] = env.delivery.submitted
    assert info.kind is SessionKind.COMMAND
    assert info.selection == "make this formal please"
    assert transcript.text == "make it formal"
    assert env.inserter.capture_threads == ["whisp-selection"]


def test_command_reads_the_selection_only_after_the_shortcut_is_released(env: Env, qtbot) -> None:
    env.press("command")
    qtbot.wait(50)
    assert env.inserter.capture_threads == []  # Ctrl+C under a held Ctrl+Win+Alt wouldn't copy
    env.release("command", after=1.0)
    qtbot.waitUntil(lambda: env.inserter.capture_threads == ["whisp-selection"], timeout=3000)


def test_command_waits_for_a_slow_selection(env: Env, qtbot) -> None:
    env.inserter.release.clear()
    env.press("command")
    session_id = env.pipeline.started[0]
    env.release("command", after=1.0)
    env.finish(session_id, "shorter")
    qtbot.wait(50)
    assert env.delivery.submitted == []
    env.inserter.release.set()
    qtbot.waitUntil(lambda: len(env.delivery.submitted) == 1, timeout=3000)
    assert env.delivery.submitted[0][0].selection == "selected words"


def test_command_without_a_selection_still_goes_to_delivery(env: Env, qtbot) -> None:
    env.inserter.selection = None
    env.press("command")
    session_id = env.pipeline.started[0]
    env.release("command", after=1.0)
    env.finish(session_id, "shorter")
    qtbot.waitUntil(lambda: len(env.delivery.submitted) == 1, timeout=3000)
    assert env.delivery.submitted[0][0].selection is None  # delivery explains what to do


# -- speech model ------------------------------------------------------------------------


def test_no_speech_model_opens_onboarding_instead_of_recording(env: Env) -> None:
    env.asr.set_state(EngineState.NOT_INSTALLED)
    assert env.controller.state is AppState.ERROR
    env.press()
    assert env.pipeline.started == []
    assert env.called("overlay.show_notice") == [(controller_module.MSG_NO_MODEL, "warning")]
    assert env.windows == [("onboarding", "")]
    env.settings.update({"general": {"onboarding_completed": True}})
    env.press()
    assert env.windows[-1] == ("settings", "model")


def test_a_just_downloaded_model_is_used_even_if_the_engine_hasnt_noticed(env: Env) -> None:
    env.asr.set_state(EngineState.NOT_INSTALLED)
    env.models.installed.add((env.settings.current.transcription.model_id, "cpu"))
    env.press()
    assert env.controller.is_recording
    assert ("asr.ensure_loaded",) in env.calls


def test_loading_note_while_the_model_loads(env: Env) -> None:
    env.asr.set_state(EngineState.UNLOADED, emit=False)
    env.press()
    assert ("asr.ensure_loaded",) in env.calls
    assert env.called("overlay.set_note") == [(controller_module.MSG_LOADING_MODEL,)]
    env.asr.set_state(EngineState.LOADING)
    env.asr.set_state(EngineState.READY)
    assert env.called("overlay.set_note") == [(controller_module.MSG_LOADING_MODEL,), (None,)]


def test_tray_status_messages(env: Env) -> None:
    c = env.controller
    assert c.status_message() == "Ready. Hold Ctrl + Win to dictate."
    env.asr.set_state(EngineState.LOADING)
    assert c.status_message() == controller_module.MSG_LOADING_MODEL
    env.asr.set_state(EngineState.ERROR, "The model files are damaged. Download it again in Settings.")
    assert c.state is AppState.ERROR
    assert c.status_message() == "The model files are damaged. Download it again in Settings."
    env.asr.set_state(EngineState.READY)
    env.press()
    assert c.status_message() == "Listening…"
    env.release(after=2)
    assert c.status_message() == "Working on your text…"


# -- shortcuts ---------------------------------------------------------------------------


def test_bindings_are_applied_with_unset_ones_left_out(env: Env) -> None:
    assert env.hotkeys.bindings == {
        "dictate": "ctrl+win",
        "command": "ctrl+win+alt",
        "paste_last": "alt+shift+z",
        "copy_last": "alt+shift+x",
        "hands_free": "ctrl+win+space",
    }
    env.settings.update({"hotkeys": {"hands_free": "", "copy_last": ""}})
    assert env.hotkeys.bindings is not None
    assert "hands_free" not in env.hotkeys.bindings and "copy_last" not in env.hotkeys.bindings
    assert env.called("overlay.set_hotkey_hint")[-1] == (["Ctrl", "Win"],)


def test_copy_last_shortcut_copies_the_last_transcript(env: Env) -> None:
    env.press("copy_last")
    assert env.called("delivery.copy_last") == [()]


def test_binding_errors_are_reported_once_and_kept_for_the_ui(env: Env) -> None:
    env.hotkeys.errors = {"command": "Another app already uses this shortcut."}
    env.settings.update({"hotkeys": {"command": "ctrl+shift+space"}})
    assert env.hotkeys.bindings is not None and env.hotkeys.bindings["command"] == "ctrl+shift+space"
    assert env.controller.binding_errors == {"command": "Another app already uses this shortcut."}
    assert env.controller.state is AppState.ERROR
    assert env.controller.status_message() == "Command mode shortcut: Another app already uses this shortcut."
    assert env.notifications == [
        ("Shortcut not available", "Command mode: Another app already uses this shortcut.", "warning")
    ]
    env.settings.update({"hotkeys": {"tap_to_lock_ms": 400}})  # same error again: no repeat notification
    assert len(env.notifications) == 1

    env.hotkeys.errors = {}
    env.settings.update({"hotkeys": {"dictate": "f8"}})
    assert env.controller.binding_errors == {}
    assert env.controller.state is AppState.IDLE
    assert env.called("overlay.set_hotkey_hint")[-1] == (["F8"],)

    env.hotkeys.binding_error.emit("dictate", "Windows refused the shortcut.")
    assert env.controller.binding_errors == {"dictate": "Windows refused the shortcut."}
    assert env.notifications[-1] == ("Shortcut not available", "Dictation: Windows refused the shortcut.", "warning")


def test_esc_unavailable_for_one_recording_is_mentioned_once_and_doesnt_stick(env: Env) -> None:
    message = "Esc is already used by another app, so it can't cancel a dictation."
    env.press("hands_free")
    env.hotkeys.binding_error.emit("cancel", message)  # Esc is registered only while recording
    env.press("hands_free")
    assert env.notifications == [("Esc can't cancel", message, "warning")]
    assert env.controller.binding_errors == {}
    assert env.controller.problem() is None  # Whisp itself works: no error badge in the tray
    env.press("hands_free")
    env.hotkeys.binding_error.emit("cancel", message)  # the next recording: no repeat notification
    assert len(env.notifications) == 1


# -- targets -----------------------------------------------------------------------------


def test_starting_from_the_tray_uses_the_window_focused_at_stop(env: Env) -> None:
    env.focus["target"] = None  # the tray menu has focus
    env.controller.start_hands_free(from_menu=True)
    session_id = env.pipeline.started[0]
    env.focus["target"] = NOTES
    env.overlay.stop_clicked.emit()
    env.finish(session_id)
    [(info, _)] = env.delivery.submitted
    assert info.target is not None and info.target.process_name == "notes.exe"


def test_category_rules(env: Env) -> None:
    env.focus["target"] = TargetWindow(1, 2, "mailer.exe", "Mailer", "Inbox", category=AppCategory.EMAIL)
    env.press()
    [target] = [c[0] for c in env.called("overlay.set_target")]
    assert target.category is AppCategory.EMAIL  # kept: our rules don't know this app
    env.press("cancel")
    env.settings.update({"text": {"app_categories": {"mailer.exe": "other"}}})
    env.press()
    assert env.called("overlay.set_target")[-1][0].category is AppCategory.OTHER  # the user's choice wins


def test_foreground_changes_move_the_idle_overlay(env: Env) -> None:
    env.controller.on_foreground_changed(NOTES)
    assert env.called("overlay.set_target") == [(NOTES,)]
    env.press()
    env.controller.on_foreground_changed(SLACK)
    assert env.called("overlay.set_target")[-1][0].process_name == "slack.exe"
    assert len(env.called("overlay.set_target")) == 2  # the one from starting, not the focus change


# -- robustness --------------------------------------------------------------------------


def test_finished_sessions_stop_waiting_for_one_that_never_reports(qapp, settings_store, monkeypatch, qtbot) -> None:
    monkeypatch.setattr(controller_module, "RESULT_WAIT_MS", 30)
    e = make_env(settings_store)
    try:
        a = e.hold()
        b = e.hold()
        e.finish(b)
        qtbot.waitUntil(lambda: e.submitted_ids() == [b], timeout=3000)
        e.finish(a)  # turns up late: still delivered, never dropped
        assert e.submitted_ids() == [b, a]
    finally:
        e.controller.shutdown()


def test_shutdown_cancels_the_recording_and_ignores_later_events(env: Env) -> None:
    env.press()
    session_id = env.pipeline.started[0]
    env.controller.shutdown()
    assert env.called("pipeline.cancel") == [(session_id,)]
    env.press()
    env.finish(session_id)
    assert len(env.pipeline.started) == 1
    assert env.delivery.submitted == []


def test_start_survives_a_broken_foreground_query(env: Env) -> None:
    env.focus["target"] = OSError("access denied")
    env.press()
    assert env.controller.is_recording
    assert env.called("overlay.set_target") == [(None,)]
