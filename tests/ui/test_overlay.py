"""Overlay tests: scene states and morphs, window geometry and timers, clicks, positioning, reduced motion."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtTest import QTest

from whisp.core.types import Outcome, RecordingMode, SessionKind, TargetWindow
from whisp.ui.overlay import Overlay, style
from whisp.ui.overlay import overlay as overlay_module
from whisp.ui.overlay.motion import Fade, LevelFollower, Spring
from whisp.ui.overlay.scene import COPIED_TEXT, Display, OverlayScene
from whisp.ui.overlay.text import parse_runs

MARGIN = style.GLOW_MARGIN


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def run(scene: OverlayScene, clock: Clock, seconds: float, fps: int = 60) -> None:
    for _ in range(max(1, round(seconds * fps))):
        clock.now += 1.0 / fps
        scene.advance()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def scene(qapp, clock: Clock) -> OverlayScene:
    s = OverlayScene(clock=clock)
    s.set_hint_keys(["Ctrl", "Win"])
    return s


@pytest.fixture
def overlay(qtbot, settings_store, monkeypatch):
    monkeypatch.setattr(overlay_module, "animations_enabled", lambda: True)
    # Most tests here exercise the resting pill, which is off by default.
    settings_store.update({"overlay": {"show_on": "primary", "idle_pill": True}})
    ov = Overlay(settings_store)
    ov.set_hotkey_hint(["Ctrl", "Win"])
    yield ov
    ov.shutdown()


def target_size(scene: OverlayScene) -> tuple[float, float]:
    return scene.width.target, scene.height.target


# -- motion primitives -------------------------------------------------------------


def test_spring_is_critically_damped_and_retargets_without_jumps() -> None:
    spring = Spring(0.0, omega=20.0, precision=0.001)
    spring.target = 100.0
    values = []
    for _ in range(60):
        spring.step(1 / 60)
        values.append(spring.value)
    assert max(values) <= 100.0 + 1e-6  # no overshoot from rest
    assert spring.settled and spring.value == 100.0
    spring.target = 0.0
    spring.step(1 / 60)
    before = spring.value
    spring.target = 50.0  # retarget mid-flight
    assert spring.value == before  # nothing jumps when the target moves


def test_fade_reverses_smoothly_and_honours_delay() -> None:
    fade = Fade(False, fade_in=0.2, fade_out=0.1, delay=0.05)
    fade.show(True)
    fade.step(0.05)
    assert fade.progress == 0.0  # still waiting
    fade.step(0.1)
    assert fade.progress == pytest.approx(0.5)
    fade.show(False)
    fade.step(0.025)
    assert fade.progress == pytest.approx(0.25)


def test_level_follower_attacks_fast_and_releases_slowly() -> None:
    follower = LevelFollower(attack=0.06, release=0.25)
    follower.set(1.0)
    for _ in range(6):  # 100 ms
        follower.step(1 / 60)
    assert follower.value > 0.75
    follower.set(0.0)
    for _ in range(6):
        follower.step(1 / 60)
    assert 0.4 < follower.value < 0.75  # still decaying gently
    follower.set(float("nan"))
    assert follower.target == 0.0


# -- scene: states and sizes -----------------------------------------------------------


def test_rest_state_is_idle_hairline_or_hidden(scene: OverlayScene, clock: Clock) -> None:
    scene.set_rest(True)
    run(scene, clock, 0.6)
    assert scene.display is Display.IDLE
    assert target_size(scene) == (style.IDLE_WIDTH, style.IDLE_HEIGHT)
    assert scene.alpha.value == pytest.approx(style.IDLE_OPACITY)
    assert scene.interactive
    scene.set_rest(False)
    run(scene, clock, 0.6)
    assert scene.display is Display.HIDDEN
    assert not scene.visible
    assert not scene.animating


def test_recording_sizes_follow_design(scene: OverlayScene, clock: Clock) -> None:
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD_PENDING)
    assert target_size(scene) == (style.HOLD_WIDTH, style.ROW_HEIGHT)
    assert not scene.interactive
    scene.set_mode(RecordingMode.HANDS_FREE)
    assert target_size(scene) == (style.HANDS_FREE_WIDTH, style.ROW_HEIGHT)
    assert scene.interactive
    scene.set_preview("hello there")
    assert target_size(scene) == (style.HANDS_FREE_WIDTH, style.ROW_HEIGHT + style.PREVIEW_HEIGHT)
    scene.show_processing()
    assert target_size(scene) == (style.PROCESSING_WIDTH, style.ROW_HEIGHT)
    scene.show_outcome(Outcome.INSERTED)
    assert target_size(scene) == (style.DONE_SIZE, style.DONE_SIZE)
    assert scene.display is Display.DONE


def test_command_mode_adds_the_edit_chip(scene: OverlayScene) -> None:
    scene.show_recording(SessionKind.COMMAND, RecordingMode.HOLD)
    assert scene.chip_fade.visible
    assert scene.width.target > style.HOLD_WIDTH
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    assert not scene.chip_fade.visible


def test_any_state_can_interrupt_any_other_without_jumps(scene: OverlayScene, clock: Clock) -> None:
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HANDS_FREE)
    run(scene, clock, 0.5)
    scene.show_processing()
    run(scene, clock, 0.08)  # mid-morph
    width, velocity = scene.width.value, scene.width.velocity
    assert style.PROCESSING_WIDTH < width < style.HANDS_FREE_WIDTH
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)  # a new dictation starts abruptly
    assert scene.width.value == width
    assert scene.width.velocity == velocity
    run(scene, clock, 0.8)
    assert scene.width.value == pytest.approx(style.HOLD_WIDTH)


def test_motion_starts_smoothly_after_a_long_rest(scene: OverlayScene, clock: Clock) -> None:
    scene.set_rest(True)
    run(scene, clock, 1.0)
    assert not scene.animating  # the window's frame timer would be stopped now
    clock.now += 30.0  # nothing happens for a while
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    clock.now += 0.2  # the first frame arrives late
    scene.advance()
    progress = (scene.width.value - style.IDLE_WIDTH) / (style.HOLD_WIDTH - style.IDLE_WIDTH)
    assert 0.0 < progress < 0.1  # one frame of motion, not a jump


def test_inserted_draws_check_then_retracts_to_rest(scene: OverlayScene, clock: Clock) -> None:
    scene.set_rest(True)
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    run(scene, clock, 0.5)
    scene.show_processing()
    run(scene, clock, 0.3)
    scene.show_outcome(Outcome.INSERTED)
    run(scene, clock, 0.3)
    assert scene.display is Display.DONE
    assert scene.check_fade.visible
    run(scene, clock, 0.6)
    assert scene.display is Display.IDLE
    run(scene, clock, 0.6)
    assert scene.width.value == pytest.approx(style.IDLE_WIDTH)


def test_messages_expire_and_cancel_retracts_at_once(scene: OverlayScene, clock: Clock) -> None:
    scene.show_outcome(Outcome.COPIED)
    assert scene.display is Display.MESSAGE
    assert scene.message.current is not None and scene.message.current.text == COPIED_TEXT
    run(scene, clock, style.COPIED_SECONDS - 0.2)
    assert scene.display is Display.MESSAGE
    run(scene, clock, 0.4)
    assert scene.display is Display.HIDDEN

    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    run(scene, clock, 0.3)
    scene.show_outcome(Outcome.CANCELLED)
    assert scene.display is Display.HIDDEN
    assert scene.alpha.target == 0.0


def test_failed_tints_border_and_no_speech_shakes(scene: OverlayScene, clock: Clock) -> None:
    scene.show_outcome(Outcome.FAILED, "Microphone not found")
    run(scene, clock, 0.5)
    assert scene.tint.target == 1.0
    assert scene.tint.value > 0.95
    assert scene.message.current is not None and scene.message.current.tint == "error"

    scene.show_outcome(Outcome.NO_SPEECH)
    anchor = QPointF(400, 400)
    offsets = []
    for _ in range(20):
        run(scene, clock, 1 / 60)
        offsets.append(scene.pill_rect(anchor).center().x() - anchor.x())
    assert max(offsets) > 1.0 and min(offsets) < -1.0  # it moves both ways
    run(scene, clock, 0.6)
    assert scene.pill_rect(anchor).center().x() == pytest.approx(anchor.x())


def test_outcomes_of_earlier_sessions_never_cover_a_live_recording(scene: OverlayScene, clock: Clock) -> None:
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HANDS_FREE)
    run(scene, clock, 0.3)
    scene.show_outcome(Outcome.INSERTED)
    scene.show_outcome(Outcome.NO_SPEECH)
    assert scene.display is Display.LISTENING
    scene.show_outcome(Outcome.COPIED)
    assert scene.display is Display.LISTENING
    assert scene.line.current is not None and scene.line.current.kind == "message"
    scene.show_outcome(Outcome.FAILED, "Microphone disconnected")
    assert scene.display is Display.MESSAGE


def test_notice_is_a_message_at_rest_and_a_note_while_recording(scene: OverlayScene, clock: Clock) -> None:
    scene.show_notice("Model ready", "info")
    assert scene.display is Display.MESSAGE
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    scene.show_notice("Microphone is very quiet", "warning")
    assert scene.display is Display.LISTENING
    line = scene.line.current
    assert line is not None and line.kind == "message" and line.tone == "warning"
    run(scene, clock, style.MESSAGE_MAX_SECONDS + 0.1)
    assert scene.line.current is None


def test_note_and_preview_share_the_line_beneath_the_wisp(scene: OverlayScene) -> None:
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    scene.set_note("Loading model…")
    assert scene.line.current is not None and scene.line.current.kind == "note"
    scene.set_note(None)
    assert scene.line.current is None
    scene.set_preview("word " * 400)
    assert scene.line.current is not None and scene.line.current.kind == "preview"
    assert len(scene._preview) <= 240  # only the visible tail is kept


def test_hit_testing_finds_buttons_and_a_forgiving_idle_target(scene: OverlayScene, clock: Clock) -> None:
    anchor = QPointF(300, 300)
    scene.set_rest(True)
    run(scene, clock, 0.6)
    pill = scene.pill_rect(anchor)
    near = QPointF(pill.right() + 12, pill.center().y() + 6)  # outside the 40 x 6 hairline, inside the target
    assert scene.hit_test(near, anchor) == "idle"
    assert scene.hit_test(QPointF(pill.center().x(), pill.center().y() - 40), anchor) is None

    scene.show_recording(SessionKind.DICTATION, RecordingMode.HANDS_FREE)
    run(scene, clock, 0.8)
    pill = scene.pill_rect(anchor)
    cy = pill.top() + style.ROW_HEIGHT / 2
    left = QPointF(pill.left() + style.BUTTON_INSET + style.BUTTON_SIZE / 2, cy)
    right = QPointF(pill.right() - style.BUTTON_INSET - style.BUTTON_SIZE / 2, cy)
    assert scene.hit_test(left, anchor) == "cancel"
    assert scene.hit_test(right, anchor) == "stop"
    assert scene.hit_test(pill.center(), anchor) is None
    scene.press(right, anchor)
    assert scene.release(right, anchor) == "stop"
    scene.press(right, anchor)
    assert scene.release(left, anchor) is None  # released elsewhere: no click


def test_idle_hover_expands_after_a_short_delay(scene: OverlayScene, clock: Clock) -> None:
    anchor = QPointF(300, 300)
    scene.set_rest(True)
    run(scene, clock, 0.6)
    scene.pointer_moved(scene.pill_rect(anchor).center(), anchor)
    assert scene.next_deadline() is not None
    run(scene, clock, 0.6)
    assert scene.hint_fade.visible
    assert scene.height.target == style.IDLE_HOVER_HEIGHT
    scene.pointer_moved(None, anchor)
    run(scene, clock, 0.8)
    assert not scene.hint_fade.visible
    assert target_size(scene) == (style.IDLE_WIDTH, style.IDLE_HEIGHT)

    # Clicking the expanded pill starts a recording; afterwards the pill rests collapsed again.
    scene.pointer_moved(scene.pill_rect(anchor).center(), anchor)
    run(scene, clock, 0.6)
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HANDS_FREE)
    run(scene, clock, 0.3)
    scene.show_outcome(Outcome.CANCELLED)
    run(scene, clock, 0.6)
    assert scene.display is Display.IDLE
    assert target_size(scene) == (style.IDLE_WIDTH, style.IDLE_HEIGHT)


def test_reduced_motion_snaps_sizes_and_never_shakes(scene: OverlayScene, clock: Clock) -> None:
    scene.reduced_motion = True
    scene.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    assert scene.width.value == style.HOLD_WIDTH  # no spring
    assert scene.alpha.target == 1.0  # but it still fades in
    scene.set_level(1.0)
    run(scene, clock, 0.5)
    assert scene.wisp.energy > 0.5  # the wisp still reacts to the voice
    scene.show_outcome(Outcome.NO_SPEECH)
    assert scene.geometry_settled  # sizes snapped and no shake


def test_scene_paints_every_state_without_errors(scene: OverlayScene, clock: Clock) -> None:
    image = QImage(600, 200, QImage.Format.Format_ARGB32_Premultiplied)
    anchor = QPointF(300, 170)

    def paint() -> None:
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        scene.paint(painter, anchor)
        painter.end()

    scene.set_rest(True)
    for step in (
        lambda: scene.show_recording(SessionKind.COMMAND, RecordingMode.HANDS_FREE),
        lambda: scene.set_preview("move the launch to Thursday"),
        lambda: scene.set_note("Loading model…"),
        scene.show_processing,
        lambda: scene.show_outcome(Outcome.INSERTED),
        lambda: scene.show_outcome(Outcome.COPIED),
        lambda: scene.show_notice("Press Ctrl + Shift + Space to try again", "warning"),
    ):
        step()
        scene.set_level(0.6)
        run(scene, clock, 0.2)
        paint()
    # Something was drawn: the pill is not fully transparent.
    assert image.pixelColor(300, 170 - int(scene.height.value / 2)).alpha() > 0


def test_key_combinations_become_key_caps() -> None:
    runs = parse_runs("Copied — press Ctrl + V to paste")
    assert [r.keys for r in runs if r.keys] == [("Ctrl", "V")]
    assert parse_runs("Hold Ctrl+Win to dictate")[1].keys == ("Ctrl", "Win")
    assert all(not r.keys for r in parse_runs("Press Space to pause"))  # a single key in prose stays text


# -- the window ----------------------------------------------------------------------


def expected_anchor(screen, position: str = "bottom") -> QPointF:
    area = screen.availableGeometry()
    x = area.x() + area.width() // 2
    y = area.y() + style.SCREEN_MARGIN if position == "top" else area.y() + area.height() - style.SCREEN_MARGIN
    return QPointF(x, y)


def settle(qtbot, overlay: Overlay, width: float, height: float) -> None:
    qtbot.waitUntil(
        lambda: (
            overlay.scene.geometry_settled
            and overlay.window.geometry().width() == width + 2 * MARGIN
            and overlay.window.geometry().height() == height + 2 * MARGIN
        ),
        timeout=3000,
    )


def test_window_never_takes_focus(overlay: Overlay, qtbot) -> None:
    window = overlay.window
    flags = window.windowFlags()
    for flag in (
        Qt.WindowType.FramelessWindowHint,
        Qt.WindowType.Tool,
        Qt.WindowType.WindowStaysOnTopHint,
        Qt.WindowType.WindowDoesNotAcceptFocus,
    ):
        assert flags & flag
    assert window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert window.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert window.focusPolicy() == Qt.FocusPolicy.NoFocus
    overlay.show_recording(SessionKind.DICTATION, RecordingMode.HANDS_FREE)
    qtbot.waitUntil(window.isVisible, timeout=2000)
    qtbot.wait(50)
    assert QGuiApplication.focusWindow() is not window.windowHandle()


def test_idle_pill_is_tight_and_static(overlay: Overlay, qtbot) -> None:
    window = overlay.window
    settle(qtbot, overlay, style.IDLE_WIDTH, style.IDLE_HEIGHT)
    assert window.isVisible()
    qtbot.waitUntil(lambda: not window.animating, timeout=2000)
    assert not window.waiting  # nothing scheduled: no CPU spent while idle
    assert not window.click_through  # the idle pill can be hovered and clicked
    anchor = expected_anchor(QGuiApplication.primaryScreen())
    assert window.geometry().bottom() + 1 == anchor.y() + MARGIN
    assert window.geometry().center().x() in (anchor.x(), anchor.x() - 1)


def test_recording_geometry_and_click_through(overlay: Overlay, qtbot) -> None:
    window = overlay.window
    overlay.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    for _ in range(5):
        overlay.set_level(0.5)
    settle(qtbot, overlay, style.HOLD_WIDTH, style.ROW_HEIGHT)
    assert window.animating  # the wisp is alive while listening
    assert window.click_through  # hold mode has nothing to click
    anchor = expected_anchor(QGuiApplication.primaryScreen())
    assert window.geometry() == QRect(
        int(anchor.x() - style.HOLD_WIDTH / 2 - MARGIN),
        int(anchor.y() - style.ROW_HEIGHT - MARGIN),
        int(style.HOLD_WIDTH + 2 * MARGIN),
        int(style.ROW_HEIGHT + 2 * MARGIN),
    )
    overlay.set_mode(RecordingMode.HANDS_FREE)
    settle(qtbot, overlay, style.HANDS_FREE_WIDTH, style.ROW_HEIGHT)
    assert not window.click_through


def test_timers_stop_once_hidden(overlay: Overlay, settings_store, qtbot) -> None:
    window = overlay.window
    settings_store.update({"overlay": {"idle_pill": False}})
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=2000)
    overlay.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    qtbot.waitUntil(window.isVisible, timeout=2000)
    assert window.animating
    overlay.show_outcome(Outcome.CANCELLED)
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=2000)
    assert not window.animating
    assert not window.waiting


def test_static_message_waits_instead_of_animating(overlay: Overlay, qtbot) -> None:
    window = overlay.window
    overlay.show_outcome(Outcome.COPIED)
    qtbot.waitUntil(lambda: not window.animating, timeout=2000)
    assert window.isVisible()
    assert window.waiting  # one wake-up scheduled for when the message expires


def test_buttons_emit_signals(overlay: Overlay, qtbot) -> None:
    window = overlay.window
    overlay.show_recording(SessionKind.DICTATION, RecordingMode.HANDS_FREE)
    settle(qtbot, overlay, style.HANDS_FREE_WIDTH, style.ROW_HEIGHT)
    cy = MARGIN + style.ROW_HEIGHT // 2
    stop = QPoint(int(window.width() - MARGIN - style.BUTTON_INSET - style.BUTTON_SIZE / 2), int(cy))
    cancel = QPoint(int(MARGIN + style.BUTTON_INSET + style.BUTTON_SIZE / 2), int(cy))
    with qtbot.waitSignal(overlay.stop_clicked, timeout=1000):
        QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, stop)
    with qtbot.waitSignal(overlay.cancel_clicked, timeout=1000):
        QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, cancel)


def test_nothing_shows_at_rest_by_default(qtbot, settings_store, monkeypatch) -> None:
    monkeypatch.setattr(overlay_module, "animations_enabled", lambda: True)
    ov = Overlay(settings_store)
    try:
        qtbot.wait(300)
        assert not ov.window.isVisible()
        ov.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
        qtbot.waitUntil(ov.window.isVisible, timeout=2000)
        ov.show_outcome(Outcome.CANCELLED)
        qtbot.waitUntil(lambda: not ov.window.isVisible(), timeout=2000)
    finally:
        ov.shutdown()


def test_idle_click_emits_idle_clicked(overlay: Overlay, qtbot) -> None:
    window = overlay.window
    settle(qtbot, overlay, style.IDLE_WIDTH, style.IDLE_HEIGHT)
    centre = QPoint(window.width() // 2, window.height() - MARGIN - int(style.IDLE_HEIGHT) // 2)
    with qtbot.waitSignal(overlay.idle_clicked, timeout=1000):
        QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)


def test_position_follows_settings(overlay: Overlay, settings_store, qtbot) -> None:
    primary = QGuiApplication.primaryScreen()
    assert overlay.window.anchor == expected_anchor(primary)
    settings_store.update({"overlay": {"position": "top"}})
    assert overlay.window.anchor == expected_anchor(primary, "top")
    overlay.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    settle(qtbot, overlay, style.HOLD_WIDTH, style.ROW_HEIGHT)
    assert overlay.window.geometry().top() == expected_anchor(primary, "top").y() - MARGIN


def test_target_window_picks_its_monitor(overlay: Overlay, settings_store) -> None:
    settings_store.update({"overlay": {"show_on": "active"}})
    for screen in QGuiApplication.screens():
        native = overlay_module.native_geometry(screen)
        rect = (native.left(), native.top(), native.left() + native.width(), native.top() + native.height())
        overlay.set_target(TargetWindow(hwnd=1, pid=1, process_name="a.exe", app_name="A", title="", monitor_rect=rect))
        assert overlay.screen is screen
        assert overlay.window.anchor == expected_anchor(screen)


def test_fullscreen_target_hides_only_the_idle_pill(overlay: Overlay, settings_store, qtbot) -> None:
    settings_store.update({"overlay": {"show_on": "active"}})
    primary = QGuiApplication.primaryScreen()
    native = overlay_module.native_geometry(primary)
    rect = (native.left(), native.top(), native.left() + native.width(), native.top() + native.height())
    target = TargetWindow(
        hwnd=1, pid=1, process_name="game.exe", app_name="Game", title="", is_fullscreen=True, monitor_rect=rect
    )
    overlay.set_target(target)
    qtbot.waitUntil(lambda: not overlay.window.isVisible(), timeout=2000)
    overlay.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    qtbot.waitUntil(overlay.window.isVisible, timeout=2000)  # dictating still shows the pill
    overlay.show_outcome(Outcome.CANCELLED)
    qtbot.waitUntil(lambda: not overlay.window.isVisible(), timeout=2000)
    settings_store.update({"overlay": {"hide_in_fullscreen": False}})
    qtbot.waitUntil(overlay.window.isVisible, timeout=2000)


def test_reduced_motion_is_read_per_session(overlay: Overlay, monkeypatch) -> None:
    monkeypatch.setattr(overlay_module, "animations_enabled", lambda: False)
    overlay.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    assert overlay.scene.reduced_motion
    assert overlay.scene.width.value == style.HOLD_WIDTH


def test_shutdown_is_final_and_idempotent(overlay: Overlay, qtbot) -> None:
    overlay.show_recording(SessionKind.DICTATION, RecordingMode.HOLD)
    overlay.shutdown()
    overlay.shutdown()
    overlay.show_outcome(Outcome.INSERTED)  # ignored, never raises
    overlay.set_level(0.5)
    assert not overlay.window.isVisible()
    assert not overlay.window.animating
