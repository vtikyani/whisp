"""Tray icon: rendering for every state, tooltip, menu state and the microphone choice."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QIcon, QImage

from whisp.app import tray as tray_module
from whisp.app.controller import AppState
from whisp.app.tray import TrayIcon, render_tray_icon, svg_icon
from whisp.core.types import AudioDevice, RecordingMode
from whisp.storage.settings import SettingsStore


def image(icon: QIcon, size: int) -> QImage:
    return icon.pixmap(size, size).toImage().convertToFormat(QImage.Format.Format_ARGB32)


def pixels(img: QImage) -> list[QColor]:
    return [img.pixelColor(x, y) for y in range(img.height()) for x in range(img.width())]


def max_alpha(img: QImage) -> int:
    return max(c.alpha() for c in pixels(img))


@pytest.mark.parametrize("state", list(AppState))
@pytest.mark.parametrize("light_taskbar", [False, True])
def test_every_state_renders_at_every_size(qapp, state: AppState, light_taskbar: bool) -> None:
    icon = render_tray_icon(state, light_taskbar=light_taskbar)
    assert not icon.isNull()
    for size in tray_module.TRAY_ICON_SIZES:
        assert icon.pixmap(size, size).size().width() == size
        assert max_alpha(image(icon, size)) > 0


def test_idle_icon_contrasts_with_the_taskbar(qapp) -> None:
    on_dark = [c for c in pixels(image(render_tray_icon(AppState.IDLE, light_taskbar=False), 32)) if c.alpha() > 200]
    on_light = [c for c in pixels(image(render_tray_icon(AppState.IDLE, light_taskbar=True), 32)) if c.alpha() > 200]
    assert on_dark and min(c.lightness() for c in on_dark) > 200
    assert on_light and max(c.lightness() for c in on_light) < 60


def test_recording_is_colourful_and_processing_is_dimmer(qapp) -> None:
    recording = image(render_tray_icon(AppState.RECORDING, light_taskbar=False), 32)
    processing = image(render_tray_icon(AppState.PROCESSING, light_taskbar=False), 32)
    assert any(c.saturation() > 100 and c.alpha() > 200 for c in pixels(recording))
    assert max_alpha(processing) < max_alpha(recording)
    paused = image(render_tray_icon(AppState.PAUSED, light_taskbar=False), 32)
    assert max_alpha(paused) < max_alpha(image(render_tray_icon(AppState.IDLE, light_taskbar=False), 32))


def test_error_icon_has_a_red_dot(qapp) -> None:
    img = image(render_tray_icon(AppState.ERROR, light_taskbar=False), 32)
    corner = img.pixelColor(26, 26)
    assert corner.red() > 200 and corner.green() < 150 and corner.alpha() == 255
    idle = image(render_tray_icon(AppState.IDLE, light_taskbar=False), 32)
    assert idle.pixelColor(26, 26).alpha() == 0


def test_app_icon_renders_the_logo(qapp) -> None:
    icon = svg_icon("logo.svg")
    assert max_alpha(image(icon, 256)) == 255


# -- the tray icon object ----------------------------------------------------------------


class FakeController(QObject):
    state_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState.IDLE
        self.message = "Ready. Hold Ctrl + Win to dictate."
        self.is_recording = False
        self.recording_mode: RecordingMode | None = None
        self.is_paused = False
        self.calls: list[str] = []

    def status_message(self) -> str:
        return self.message

    def start_hands_free(self, *, from_menu: bool = False) -> None:
        self.calls.append(f"start_hands_free(from_menu={from_menu})")

    def copy_last(self) -> None:
        self.calls.append("copy_last")

    def toggle_paused(self) -> None:
        self.is_paused = not self.is_paused
        self.calls.append("toggle_paused")


MICS = [
    AudioDevice("USB Mic", "USB Mic", "Windows WASAPI", True, 1, 48_000.0),
    AudioDevice("Headset", "Headset", "Windows WASAPI", False, 1, 16_000.0),
]


@pytest.fixture
def tray(qapp, settings_store: SettingsStore) -> Iterator[tuple[TrayIcon, FakeController]]:
    controller = FakeController()
    icon = TrayIcon(controller, settings_store, lambda: list(MICS))  # type: ignore[arg-type]
    yield icon, controller
    icon.shutdown()


def actions(tray: TrayIcon) -> dict[str, object]:
    return {a.text(): a for a in tray.contextMenu().actions() if a.text()}


def test_tooltip_and_menu_follow_the_controller(tray) -> None:
    icon, controller = tray
    assert icon.toolTip() == "Whisp — Ready. Hold Ctrl + Win to dictate."
    labels = [a.text() for a in icon.contextMenu().actions()]
    assert labels == [
        "Open Whisp",
        "Start dictation",
        "Copy last transcript",
        "",
        "Microphone",
        "Pause Whisp",
        "",
        "Quit Whisp",
    ]
    controller.is_recording, controller.recording_mode = True, RecordingMode.HANDS_FREE
    controller.state, controller.message = AppState.RECORDING, "Listening…"
    controller.state_changed.emit()
    assert icon.toolTip() == "Whisp — Listening…"
    assert "Stop dictation" in actions(icon)
    controller.recording_mode = RecordingMode.HOLD
    controller.state_changed.emit()
    assert not actions(icon)["Stop dictation"].isEnabled()


def test_tooltip_is_kept_within_the_windows_limit(tray) -> None:
    icon, controller = tray
    controller.message = "word " * 60
    controller.state_changed.emit()
    assert len(icon.toolTip()) <= tray_module.TOOLTIP_LIMIT
    assert icon.toolTip().endswith("…")


def test_menu_actions_reach_the_controller(tray, qtbot) -> None:
    icon, controller = tray
    found = actions(icon)
    found["Start dictation"].trigger()
    found["Copy last transcript"].trigger()
    found["Pause Whisp"].trigger()
    assert controller.calls == ["start_hands_free(from_menu=True)", "copy_last", "toggle_paused"]
    assert found["Pause Whisp"].isChecked()
    with qtbot.waitSignal(icon.open_requested, timeout=1000):
        found["Open Whisp"].trigger()
    with qtbot.waitSignal(icon.quit_requested, timeout=1000):
        found["Quit Whisp"].trigger()


def test_microphone_menu_lists_devices_and_saves_the_choice(tray, settings_store, qtbot) -> None:
    icon, _controller = tray
    icon._start_device_scan()
    mic_menu = actions(icon)["Microphone"].menu()
    qtbot.waitUntil(lambda: any(a.text() == "Headset" for a in mic_menu.actions()), timeout=3000)
    choices = {a.text(): a for a in mic_menu.actions() if a.isCheckable()}
    assert list(choices) == ["System default", "USB Mic", "Headset"]
    assert choices["System default"].isChecked()
    choices["Headset"].trigger()
    assert settings_store.current.audio.input_device == "Headset"
    choices = {a.text(): a for a in mic_menu.actions() if a.isCheckable()}
    assert choices["Headset"].isChecked()


def test_a_microphone_choice_that_couldnt_be_saved_is_logged_and_not_shown(tray, settings_store, qtbot, caplog) -> None:
    icon, _controller = tray
    icon._start_device_scan()
    mic_menu = actions(icon)["Microphone"].menu()
    qtbot.waitUntil(lambda: any(a.text() == "Headset" for a in mic_menu.actions()), timeout=3000)

    def locked(_settings: object) -> None:
        raise PermissionError(13, "The file is in use by another program")

    settings_store._save = locked  # settings.json can't be written right now
    try:
        [headset] = [a for a in mic_menu.actions() if a.text() == "Headset"]
        headset.trigger()
    finally:
        del settings_store._save
    assert settings_store.current.audio.input_device is None
    assert "Couldn't switch microphones" in caplog.text
    checked = [a.text() for a in mic_menu.actions() if a.isCheckable() and a.isChecked()]
    assert checked == ["System default"]  # the menu shows the microphone really in use


def test_an_unplugged_microphone_stays_visible(tray, settings_store) -> None:
    icon, _controller = tray
    settings_store.update({"audio": {"input_device": "Old Webcam"}})
    mic_menu = actions(icon)["Microphone"].menu()
    labels = [a.text() for a in mic_menu.actions()]
    assert "Old Webcam (not connected)" in labels
