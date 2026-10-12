"""The overlay's native window: frameless, translucent, always on top, and never focused.

The window is only ever as large as the pill plus room for its glow. While a
morph runs it covers the start and end sizes at once and shrinks back when the
pill settles, so it never resizes on every frame (resizing a layered window
each frame can shimmer) and never blocks clicks far from the pill. Outside
interactive states (idle pill, hands-free buttons) it lets every click through.

Frames are painted at ~60 fps only while something moves; otherwise the frame
timer stops and a single-shot timer wakes the window for the next scheduled
change (a message expiring, a hover delay).
"""

from __future__ import annotations

import logging
import math

from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QEnterEvent, QGuiApplication, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from whisp.platform.win32.window import set_click_through, style_overlay_window
from whisp.ui.overlay.scene import OverlayScene

log = logging.getLogger(__name__)

FRAME_INTERVAL_MS = 16


def _on_windows() -> bool:
    return QGuiApplication.platformName() == "windows"


class OverlayWindow(QWidget):
    """Hosts an ``OverlayScene`` on screen. ``clicked`` reports "idle", "cancel" or "stop"."""

    clicked = Signal(str)

    def __init__(self, scene: OverlayScene) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint
        )
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setWindowTitle("Whisp overlay")

        self._scene = scene
        self._anchor = QPointF()
        self._styled = False
        self._click_through: bool | None = None
        self._paint_failed = False

        self._frame_timer = QTimer(self)
        self._frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._frame_timer.setInterval(FRAME_INTERVAL_MS)
        self._frame_timer.timeout.connect(self._tick)
        self._wake_timer = QTimer(self)
        self._wake_timer.setSingleShot(True)
        self._wake_timer.timeout.connect(self._tick)

    # -- control ---------------------------------------------------------------

    @property
    def anchor(self) -> QPointF:
        """The pill's anchor in global (logical) screen coordinates."""
        return QPointF(self._anchor)

    @property
    def animating(self) -> bool:
        """True while the 60 fps frame timer runs."""
        return self._frame_timer.isActive()

    @property
    def waiting(self) -> bool:
        """True while a single-shot wake-up is scheduled."""
        return self._wake_timer.isActive()

    @property
    def click_through(self) -> bool:
        return bool(self._click_through)

    def set_anchor(self, anchor: QPointF) -> None:
        if anchor != self._anchor:
            self._anchor = QPointF(anchor)
            if self.isVisible():
                # Jump straight to the new place: the pill belongs to the new monitor now.
                self._sync_geometry(force=True)

    def kick(self) -> None:
        """Bring the window up to date after the scene changed state."""
        self._tick()

    def shutdown(self) -> None:
        self._frame_timer.stop()
        self._wake_timer.stop()
        self.hide()

    # -- frame loop --------------------------------------------------------------

    def _tick(self) -> None:
        animating = self._scene.advance()
        if not self._scene.visible:
            self._frame_timer.stop()
            if self.isVisible():
                self.hide()
            self._schedule_wake()
            return
        if not self.isVisible():
            self._show()
        self._sync_click_through()
        if not self._sync_geometry():
            self.update()
        if animating:
            if not self._frame_timer.isActive():
                self._frame_timer.start()
            self._wake_timer.stop()
        else:
            self._frame_timer.stop()
            self._schedule_wake()

    def _schedule_wake(self) -> None:
        deadline = self._scene.next_deadline()
        if deadline is None:
            self._wake_timer.stop()
            return
        delay_ms = max(1, math.ceil((deadline - self._scene.now()) * 1000.0))
        self._wake_timer.start(delay_ms)

    def _show(self) -> None:
        self._ensure_styled()
        self._sync_geometry(force=True)
        self.show()
        if _on_windows():
            # Re-assert topmost without activating (another topmost window may have been raised since).
            self._style_native()

    def _ensure_styled(self) -> None:
        if self._styled:
            return
        self._styled = True
        self.winId()  # create the native window so it can be styled before it is first shown
        self._style_native()

    def _style_native(self) -> None:
        # Only real Win32 windows can be styled (tests may run on Qt's offscreen platform).
        if not _on_windows():
            return
        try:
            style_overlay_window(self)
        except OSError:
            log.exception("Could not apply overlay window styles")

    def _sync_click_through(self) -> None:
        wanted = not self._scene.interactive
        if wanted == self._click_through:
            return
        self._click_through = wanted
        if wanted:
            self.setCursor(Qt.CursorShape.ArrowCursor)  # no stale hand cursor when the pill is clickable again
        if _on_windows():
            try:
                set_click_through(self, wanted)
            except OSError:
                log.exception("Could not change overlay click-through")

    # -- geometry ------------------------------------------------------------------

    def _sync_geometry(self, *, force: bool = False) -> bool:
        """Cover the pill now and where it is heading; grow at once, shrink only when the morph has settled.

        Returns True when the window moved or resized (and was repainted on the spot).
        """
        needed = _outer_rect(self._scene.required_rect(self._anchor))
        current = self.geometry()
        if needed == current:
            return False
        if not force and self.isVisible() and current.contains(needed) and not self._scene.geometry_settled:
            return False
        self.setGeometry(needed)
        if self.isVisible():
            # Paint and flush now: a layered window shows its new size and new pixels in the same update,
            # instead of briefly showing the previous frame at the new position.
            self.repaint()
        return True

    # -- painting --------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.fillRect(event.rect(), Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            origin = QPointF(self.geometry().topLeft())
            self._scene.paint(painter, self._anchor - origin)
        except Exception:
            # Never let a painting bug take the app down, and never flood the log at 60 fps.
            if not self._paint_failed:
                self._paint_failed = True
                log.exception("Overlay painting failed")
        finally:
            painter.end()

    # -- pointer ---------------------------------------------------------------------

    def _local_anchor(self) -> QPointF:
        return self._anchor - QPointF(self.geometry().topLeft())

    def _update_hover(self, pos: QPointF | None) -> None:
        self._scene.pointer_moved(pos, self._local_anchor())
        hovering = self._scene.hover is not None
        self.setCursor(Qt.CursorShape.PointingHandCursor if hovering else Qt.CursorShape.ArrowCursor)
        self.kick()

    def enterEvent(self, event: QEnterEvent) -> None:
        self._update_hover(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._update_hover(event.position())

    def leaveEvent(self, event: QEvent) -> None:
        self._update_hover(None)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._scene.press(event.position(), self._local_anchor())
            self.kick()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            event.accept()
            return
        element = self._scene.release(event.position(), self._local_anchor())
        self.kick()
        event.accept()
        if element is not None:
            self.clicked.emit(element)


def _outer_rect(rect: QRectF) -> QRect:
    """The smallest whole-pixel rectangle containing ``rect``."""
    left, top = math.floor(rect.left()), math.floor(rect.top())
    right, bottom = math.ceil(rect.right()), math.ceil(rect.bottom())
    return QRect(left, top, max(1, right - left), max(1, bottom - top))
