"""Whisp's icon in the notification area: status at a glance, a small menu and notifications.

The icon is the wisp from ``assets/brand``, rendered with QSvgRenderer at
every common tray size so Windows picks a sharp one at any display scale:

- idle: white on a dark taskbar, ink on a light one
- recording: the opal wisp
- processing: the opal wisp, dimmed
- paused: the idle wisp, dimmed
- error: the idle wisp with a small red dot

Menu: Open Whisp · Start dictation · Copy last transcript · Microphone ▸ ·
Pause Whisp · Quit Whisp. Double-click opens the window.

"Copy last transcript" copies rather than pastes: choosing a tray menu item
moves keyboard focus to the taskbar, so there is nowhere sensible to paste.
The paste-last shortcut still pastes straight into the focused app.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable, Sequence

from PySide6.QtCore import QByteArray, QObject, QPointF, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from whisp import APP_NAME
from whisp.app.controller import AppController, AppState
from whisp.core.types import AudioDevice, RecordingMode
from whisp.platform.win32 import system
from whisp.storage.paths import resource_path
from whisp.storage.settings import SettingsStore
from whisp.ui import theme

log = logging.getLogger(__name__)

TRAY_ICON_SIZES = (16, 20, 24, 28, 32, 36, 40, 48, 64)
"""The small-icon size at 100% to 400% display scaling."""
APP_ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
PROCESSING_OPACITY = 0.55
PAUSED_OPACITY = 0.4
DANGER_ON_LIGHT = "#DC2626"
"""``danger`` on light surfaces (docs/DESIGN.md); ``theme`` only carries the dark palette."""
THEME_POLL_MS = 3_000
TOOLTIP_LIMIT = 127
"""Windows truncates tray tooltips beyond this many characters."""
NOTIFICATION_MS = 6_000

_ACTIVE_SVG = "tray-active.svg"
_IDLE_ON_DARK_SVG = "tray-idle-light.svg"
_IDLE_ON_LIGHT_SVG = "tray-idle-dark.svg"


@functools.cache
def _svg_bytes(name: str) -> bytes:
    return resource_path("assets", "brand", name).read_bytes()


def svg_icon(name: str, sizes: Sequence[int] = APP_ICON_SIZES, *, opacity: float = 1.0) -> QIcon:
    """An icon from ``assets/brand/<name>`` with a crisp pixmap at each size."""
    return _render_icon(name, sizes, opacity=opacity, dot=None)


def render_tray_icon(state: AppState, *, light_taskbar: bool, sizes: Sequence[int] = TRAY_ICON_SIZES) -> QIcon:
    """The tray icon for ``state``, drawn for a light or dark taskbar."""
    if state in (AppState.RECORDING, AppState.PROCESSING):
        name = _ACTIVE_SVG
    else:
        name = _IDLE_ON_LIGHT_SVG if light_taskbar else _IDLE_ON_DARK_SVG
    opacity = {AppState.PROCESSING: PROCESSING_OPACITY, AppState.PAUSED: PAUSED_OPACITY}.get(state, 1.0)
    dot = None
    if state is AppState.ERROR:
        dot = QColor(DANGER_ON_LIGHT) if light_taskbar else theme.color("danger")
    return _render_icon(name, sizes, opacity=opacity, dot=dot)


def _render_icon(name: str, sizes: Sequence[int], *, opacity: float, dot: QColor | None) -> QIcon:
    try:
        renderer = QSvgRenderer(QByteArray(_svg_bytes(name)))
    except OSError:
        log.exception("Brand icon %s is missing; the installation looks damaged", name)
        renderer = None
    if renderer is not None and not renderer.isValid():
        log.error("Brand icon %s couldn't be read; the installation looks damaged", name)
        renderer = None
    icon = QIcon()
    for size in sizes:
        icon.addPixmap(_render_pixmap(renderer, size, opacity, dot))
    return icon


def _render_pixmap(renderer: QSvgRenderer | None, size: int, opacity: float, dot: QColor | None) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if renderer is not None:
            painter.setOpacity(opacity)
            renderer.render(painter, QRectF(0, 0, size, size))
        if dot is not None:
            painter.setOpacity(1.0)
            painter.setPen(Qt.PenStyle.NoPen)
            radius = max(2.0, size * 0.2)
            gap = max(1.0, size / 16)
            center = QPointF(size - radius - gap / 2, size - radius - gap / 2)
            # Clear a thin ring first so the dot stays readable where it overlaps the wisp.
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.setBrush(Qt.GlobalColor.black)
            painter.drawEllipse(center, radius + gap, radius + gap)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setBrush(dot)
            painter.drawEllipse(center, radius, radius)
    finally:
        painter.end()
    return pixmap


class TrayIcon(QSystemTrayIcon):
    """The notification-area icon. Call ``show()`` once the app is running and ``shutdown()`` when quitting."""

    open_requested = Signal()
    """Open Whisp, or the icon was double-clicked."""
    quit_requested = Signal()

    _devices_scanned = Signal(object)
    """list[AudioDevice] or None, from the device-scan thread."""

    def __init__(
        self,
        controller: AppController,
        settings: SettingsStore,
        list_devices: Callable[[], list[AudioDevice]],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._settings = settings
        self._list_devices = list_devices
        self._devices: list[AudioDevice] = []
        self._scanning = False
        self._light_taskbar = system.taskbar_uses_light_theme()
        self._icon_key: tuple[AppState, bool] | None = None
        self._closed = False

        # A QSystemTrayIcon doesn't own its menu; keep it alive here.
        self._menu = QMenu()
        self._open_action = self._menu.addAction("Open Whisp")
        self._open_action.triggered.connect(self._on_open_clicked)
        self._menu.setDefaultAction(self._open_action)
        self._dictate_action = self._menu.addAction("Start dictation")
        self._dictate_action.triggered.connect(self._on_dictate_clicked)
        self._copy_action = self._menu.addAction("Copy last transcript")
        self._copy_action.triggered.connect(self._on_copy_clicked)
        self._menu.addSeparator()
        self._mic_menu = self._menu.addMenu("Microphone")
        self._mic_group: QActionGroup | None = None
        self._pause_action = self._menu.addAction("Pause Whisp")
        self._pause_action.setCheckable(True)
        self._pause_action.triggered.connect(self._on_pause_clicked)
        self._menu.addSeparator()
        quit_action = self._menu.addAction("Quit Whisp")
        quit_action.triggered.connect(self._on_quit_clicked)
        self._menu.aboutToShow.connect(self._on_menu_about_to_show)
        self.setContextMenu(self._menu)

        self.activated.connect(self._on_activated)
        controller.state_changed.connect(self.refresh)
        settings.changed.connect(self._on_settings_changed)
        self._devices_scanned.connect(self._on_devices_scanned)

        # The taskbar's light/dark mode can change on its own (Settings → Personalisation → Colours),
        # without the app theme Qt watches; checking the registry every few seconds costs nothing.
        self._theme_timer = QTimer(self)
        self._theme_timer.setInterval(THEME_POLL_MS)
        self._theme_timer.timeout.connect(self._check_taskbar_theme)
        self._theme_timer.start()
        app = QGuiApplication.instance()
        if isinstance(app, QGuiApplication):
            app.styleHints().colorSchemeChanged.connect(self._check_taskbar_theme)
            app.primaryScreenChanged.connect(self._redraw_icon)

        self._rebuild_mic_menu()
        self.refresh()

    # -- public ----------------------------------------------------------------------------

    @Slot()
    def refresh(self) -> None:
        """Bring the icon, tooltip and menu in line with the controller."""
        if self._closed:
            return
        controller = self._controller
        self._apply_icon(controller.state)
        self.setToolTip(_clip(f"{APP_NAME} — {controller.status_message()}", TOOLTIP_LIMIT))
        recording = controller.is_recording
        self._dictate_action.setText("Stop dictation" if recording else "Start dictation")
        self._dictate_action.setEnabled(
            not controller.is_paused and (not recording or controller.recording_mode is RecordingMode.HANDS_FREE)
        )
        self._pause_action.setChecked(controller.is_paused)

    @Slot(str, str, str)
    def notify(self, title: str, message: str, kind: str = "info") -> None:
        """Show a Windows notification from the tray icon."""
        if self._closed:
            return
        if not self.isVisible() or not QSystemTrayIcon.supportsMessages():
            log.info("Notification not shown (tray unavailable): %s", title)
            return
        icon = {
            "warning": QSystemTrayIcon.MessageIcon.Warning,
            "error": QSystemTrayIcon.MessageIcon.Critical,
        }.get(kind, QSystemTrayIcon.MessageIcon.Information)
        self.showMessage(title, message, icon, NOTIFICATION_MS)

    def shutdown(self) -> None:
        """Remove the icon from the taskbar right away (otherwise it lingers until hovered)."""
        if self._closed:
            return
        self._closed = True
        self._theme_timer.stop()
        self.hide()

    # -- menu ------------------------------------------------------------------------------

    @Slot()
    def _on_menu_about_to_show(self) -> None:
        self.refresh()
        self._start_device_scan()

    @Slot()
    def _on_open_clicked(self) -> None:
        self.open_requested.emit()

    @Slot()
    def _on_quit_clicked(self) -> None:
        self.quit_requested.emit()

    @Slot()
    def _on_dictate_clicked(self) -> None:
        self._controller.start_hands_free(from_menu=True)

    @Slot()
    def _on_copy_clicked(self) -> None:
        self._controller.copy_last()

    @Slot()
    def _on_pause_clicked(self) -> None:
        self._controller.toggle_paused()
        self.refresh()

    @Slot(QSystemTrayIcon.ActivationReason)
    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.open_requested.emit()

    # -- microphones -------------------------------------------------------------------------

    def _start_device_scan(self) -> None:
        # Listing devices can take a moment (PortAudio may re-scan), so it never runs on the GUI thread.
        if self._scanning or self._closed:
            return
        self._scanning = True
        self._rebuild_mic_menu()
        threading.Thread(target=self._scan_devices, name="whisp-device-scan", daemon=True).start()

    def _scan_devices(self) -> None:
        devices: list[AudioDevice] | None
        try:
            devices = list(self._list_devices())
        except Exception:
            log.exception("Couldn't list microphones")
            devices = None
        try:
            self._devices_scanned.emit(devices)
        except RuntimeError:
            log.debug("Microphone list arrived after the tray icon was gone")

    @Slot(object)
    def _on_devices_scanned(self, devices: list[AudioDevice] | None) -> None:
        self._scanning = False
        if self._closed:
            return
        if devices is not None:
            self._devices = devices
        self._rebuild_mic_menu()

    def _rebuild_mic_menu(self) -> None:
        menu = self._mic_menu
        menu.clear()
        if self._mic_group is not None:
            self._mic_group.deleteLater()
        group = QActionGroup(menu)
        group.setExclusive(True)
        self._mic_group = group
        current = self._settings.current.audio.input_device

        def add_choice(label: str, device_id: str | None, *, checked: bool, enabled: bool = True) -> None:
            action = QAction(label, menu)
            action.setCheckable(True)
            action.setChecked(checked)
            action.setEnabled(enabled)
            action.triggered.connect(functools.partial(self._select_device, device_id))
            group.addAction(action)
            menu.addAction(action)

        add_choice("System default", None, checked=current is None)
        listed: set[str] = set()
        if self._devices:
            menu.addSeparator()
        for device in self._devices:
            if device.id in listed:
                continue
            listed.add(device.id)
            add_choice(device.name, device.id, checked=device.id == current)
        if current is not None and current not in listed and not self._scanning:
            # Keep showing the chosen microphone while it's unplugged, so the choice isn't a mystery.
            add_choice(f"{current} (not connected)", current, checked=True, enabled=False)
        if self._scanning:
            menu.addAction("Looking for microphones…").setEnabled(False)
        elif not self._devices:
            menu.addAction("No microphones found").setEnabled(False)

    def _select_device(self, device_id: str | None, *_checked: object) -> None:
        if self._settings.current.audio.input_device == device_id:
            return
        try:
            self._settings.update({"audio": {"input_device": device_id}})
        except (ValueError, OSError):  # OSError: settings.json couldn't be saved, so nothing changed
            log.exception("Couldn't switch microphones")
            self._rebuild_mic_menu()  # the click already moved the check mark; put it back

    @Slot(object)
    def _on_settings_changed(self, sections: set[str]) -> None:
        if not self._closed and "audio" in sections:
            self._rebuild_mic_menu()

    # -- icon ------------------------------------------------------------------------------

    def _apply_icon(self, state: AppState, *, force: bool = False) -> None:
        key = (state, self._light_taskbar)
        if key == self._icon_key and not force:
            return
        self._icon_key = key
        self.setIcon(render_tray_icon(state, light_taskbar=self._light_taskbar, sizes=_tray_sizes()))

    @Slot()
    def _check_taskbar_theme(self) -> None:
        light = system.taskbar_uses_light_theme()
        if light != self._light_taskbar:
            self._light_taskbar = light
            self._redraw_icon()

    @Slot()
    def _redraw_icon(self) -> None:
        if not self._closed:
            self._apply_icon(self._controller.state, force=True)


def _tray_sizes() -> tuple[int, ...]:
    """The standard sizes, plus the exact small-icon size for the primary screen's scale."""
    screen = QGuiApplication.primaryScreen()
    ratio = screen.devicePixelRatio() if screen is not None else 1.0
    exact = round(16 * ratio)
    return tuple(sorted({*TRAY_ICON_SIZES, exact}))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
