"""The dictation overlay (the pill): the public API described in docs/ARCHITECTURE.md.

``Overlay`` turns controller calls into scene states, keeps the pill on the
right monitor (the one holding the dictation target, or the primary one), and
follows settings, screen and reduced-motion changes. All methods must be
called on the GUI thread.

Call order the overlay expects from the controller:

* ``show_recording`` when a session starts, ``set_mode`` / ``set_level`` /
  ``set_preview`` / ``set_note`` while it runs;
* ``show_processing`` as soon as recording stops;
* ``show_outcome`` once the session is delivered (or cancelled).

Outcomes that arrive while a newer recording is on screen belong to an earlier
session; INSERTED / COPIED / NO_SPEECH then never cover the live recording
(COPIED is mentioned on the note line). CANCELLED and FAILED always end the
recording view, so use ``show_notice(..., "error")`` to report an earlier
session's failure without interrupting a recording.
"""

from __future__ import annotations

import contextlib
import logging

from PySide6.QtCore import QObject, QPointF, QRect, QSize, Signal
from PySide6.QtGui import QCursor, QGuiApplication, QScreen

from whisp.core.types import Outcome, RecordingMode, SessionKind, TargetWindow
from whisp.platform.win32.system import animations_enabled
from whisp.storage.settings import SettingsStore
from whisp.ui.overlay import style
from whisp.ui.overlay.scene import OverlayScene
from whisp.ui.overlay.window import OverlayWindow

log = logging.getLogger(__name__)


def native_geometry(screen: QScreen) -> QRect:
    """A screen's rectangle in physical pixels.

    Qt 6 keeps each screen's native top-left corner in logical coordinates and
    scales only the size, so the native rectangle can be rebuilt exactly.
    """
    geometry = screen.geometry()
    ratio = screen.devicePixelRatio()
    return QRect(geometry.topLeft(), QSize(round(geometry.width() * ratio), round(geometry.height() * ratio)))


def screen_for_monitor(monitor_rect: tuple[int, int, int, int]) -> QScreen | None:
    """The screen that best overlaps a monitor rectangle given in physical pixels (left, top, right, bottom)."""
    left, top, right, bottom = monitor_rect
    monitor = QRect(left, top, max(0, right - left), max(0, bottom - top))
    best: QScreen | None = None
    best_area = 0
    for screen in QGuiApplication.screens():
        overlap = native_geometry(screen).intersected(monitor)
        area = 0 if overlap.isEmpty() else overlap.width() * overlap.height()
        if area > best_area:
            best, best_area = screen, area
    return best


def _reduced_motion() -> bool:
    try:
        return not animations_enabled()
    except OSError:
        log.exception("Could not read the Windows animation setting; keeping animations on")
        return False


class Overlay(QObject):
    """The floating pill shown while dictating."""

    stop_clicked = Signal()
    cancel_clicked = Signal()
    idle_clicked = Signal()

    def __init__(self, settings: SettingsStore) -> None:
        super().__init__()
        self._settings = settings
        self._target: TargetWindow | None = None
        self._screen: QScreen | None = None
        self._shut_down = False

        self._scene = OverlayScene()
        self._scene.reduced_motion = _reduced_motion()
        self._window = OverlayWindow(self._scene)
        self._window.clicked.connect(self._on_clicked)

        settings.changed.connect(self._on_settings_changed)
        app = QGuiApplication.instance()
        app.screenAdded.connect(self._on_screen_added)
        app.screenRemoved.connect(self._on_screens_changed)
        app.primaryScreenChanged.connect(self._on_screens_changed)
        for screen in QGuiApplication.screens():
            self._watch_screen(screen)
        self._apply_settings()

    # -- for tests and diagnostics ----------------------------------------------

    @property
    def scene(self) -> OverlayScene:
        return self._scene

    @property
    def window(self) -> OverlayWindow:
        return self._window

    @property
    def screen(self) -> QScreen | None:
        """The monitor the pill currently lives on."""
        return self._screen

    # -- public API ---------------------------------------------------------------

    def show_recording(self, kind: SessionKind, mode: RecordingMode) -> None:
        if self._shut_down:
            return
        # Read the Windows setting per session, so turning animations off applies without a restart.
        self._scene.reduced_motion = _reduced_motion()
        self._reposition()
        self._scene.show_recording(kind, mode)
        self._window.kick()

    def set_mode(self, mode: RecordingMode) -> None:
        if self._shut_down:
            return
        self._scene.set_mode(mode)
        self._window.kick()

    def set_level(self, level: float) -> None:
        """Microphone level, 0-1. Cheap: the frame loop picks it up on the next frame."""
        if not self._shut_down:
            self._scene.set_level(level)

    def set_preview(self, text: str) -> None:
        if self._shut_down:
            return
        self._scene.set_preview(text)
        self._window.kick()

    def set_note(self, text: str | None) -> None:
        if self._shut_down:
            return
        self._scene.set_note(text)
        self._window.kick()

    def show_processing(self) -> None:
        if self._shut_down:
            return
        self._scene.show_processing()
        self._window.kick()

    def show_outcome(self, outcome: Outcome, message: str | None = None) -> None:
        if self._shut_down:
            return
        self._scene.show_outcome(outcome, message)
        self._window.kick()

    def show_notice(self, text: str, kind: str = "info") -> None:
        if self._shut_down:
            return
        self._reposition()
        self._scene.show_notice(text, kind)
        self._window.kick()

    def set_target(self, target: TargetWindow | None) -> None:
        if self._shut_down:
            return
        self._target = target
        self._reposition()
        self._update_rest()
        self._window.kick()

    def set_hotkey_hint(self, keys: list[str]) -> None:
        if self._shut_down:
            return
        self._scene.set_hint_keys(keys)
        self._window.kick()

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        for signal, slot in self._connections():
            # RuntimeError/TypeError: already disconnected, or Qt is tearing the application down.
            with contextlib.suppress(RuntimeError, TypeError):
                signal.disconnect(slot)
        self._window.shutdown()
        self._window.deleteLater()

    # -- internals ------------------------------------------------------------------

    def _connections(self) -> list[tuple[object, object]]:
        app = QGuiApplication.instance()
        connections: list[tuple[object, object]] = [(self._settings.changed, self._on_settings_changed)]
        if app is not None:
            connections += [
                (app.screenAdded, self._on_screen_added),
                (app.screenRemoved, self._on_screens_changed),
                (app.primaryScreenChanged, self._on_screens_changed),
            ]
        return connections

    def _on_clicked(self, element: str) -> None:
        if element == "idle":
            self.idle_clicked.emit()
        elif element == "cancel":
            self.cancel_clicked.emit()
        elif element == "stop":
            self.stop_clicked.emit()

    def _on_settings_changed(self, sections: object) -> None:
        """``SettingsStore.changed`` carries the set of changed section names."""
        if self._shut_down:
            return
        try:
            relevant = "overlay" in sections  # type: ignore[operator]
        except TypeError:
            relevant = True
        if relevant:
            self._apply_settings()

    def _apply_settings(self) -> None:
        self._scene.placement = self._settings.current.overlay.position
        self._reposition()
        self._update_rest()
        self._window.kick()

    def _watch_screen(self, screen: QScreen) -> None:
        screen.geometryChanged.connect(self._on_screens_changed)
        screen.availableGeometryChanged.connect(self._on_screens_changed)

    def _on_screen_added(self, screen: QScreen) -> None:
        self._watch_screen(screen)
        self._on_screens_changed()

    def _on_screens_changed(self, *_args: object) -> None:
        if self._shut_down:
            return
        self._reposition()
        self._update_rest()
        self._window.kick()

    def _target_screen(self) -> QScreen | None:
        if self._target is not None and self._target.monitor_rect is not None:
            return screen_for_monitor(self._target.monitor_rect)
        return None

    def _pick_screen(self) -> QScreen | None:
        screens = QGuiApplication.screens()
        if not screens:
            return None
        primary = QGuiApplication.primaryScreen()
        if self._settings.current.overlay.show_on == "primary":
            return primary
        target = self._target_screen()
        if target is not None:
            return target
        if self._screen is not None and self._screen in screens:
            return self._screen  # no better information: stay where we are
        return QGuiApplication.screenAt(QCursor.pos()) or primary

    def _reposition(self) -> None:
        """Anchor the pill centred on the chosen monitor's work area, 24 px from the bottom (or top) edge."""
        screen = self._pick_screen()
        if screen is None:
            return
        self._screen = screen
        area = screen.availableGeometry()
        x = area.x() + area.width() // 2
        if self._scene.placement == "top":
            y = area.y() + style.SCREEN_MARGIN
        else:
            y = area.y() + area.height() - style.SCREEN_MARGIN
        self._window.set_anchor(QPointF(x, y))

    def _update_rest(self) -> None:
        cfg = self._settings.current.overlay
        target = self._target
        fullscreen_here = (
            target is not None
            and target.is_fullscreen
            and (target.monitor_rect is None or self._target_screen() is self._screen)
        )
        self._scene.set_rest(cfg.idle_pill and not (cfg.hide_in_fullscreen and fullscreen_here))
