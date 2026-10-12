import math
import numpy as np
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QPainterPath


class WaveformWidget(QWidget):
    # Colors
    COLOR_BG = QColor(20, 20, 20)
    COLOR_BORDER = QColor(60, 60, 60)
    COLOR_GREY = QColor(100, 100, 100)
    COLOR_GREEN = QColor(50, 205, 50)

    WIDGET_HEIGHT = 70
    CORNER_RADIUS = 16
    BORDER_WIDTH = 1.5

    BAR_WIDTH = 8
    BAR_GAP = 6
    H_PADDING = 16
    BAR_MIN_HEIGHT = 8
    BAR_MAX_HEIGHT = 40
    BAR_RADIUS = 4  # rounded endcap radius

    def __init__(self, num_bars=7):
        super().__init__()
        self.num_bars = num_bars
        self._widget_width = 2 * self.H_PADDING + num_bars * self.BAR_WIDTH + (num_bars - 1) * self.BAR_GAP
        self.bar_color = self.COLOR_GREY
        self._target_levels = np.zeros(num_bars)
        self._current_levels = np.zeros(num_bars)
        self._bell_envelope = self._make_bell_envelope(num_bars)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(self._widget_width, self.WIDGET_HEIGHT)

        # Animation timer ~60fps
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.start()

    def _make_bell_envelope(self, n):
        """Gaussian bell curve centered on the middle bar."""
        center = (n - 1) / 2.0
        sigma = n / 4.0
        envelope = np.array([math.exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(n)])
        return envelope

    def position_on_screen(self):
        """Position at bottom-center of primary screen, 20% from bottom."""
        screen = QApplication.primaryScreen().geometry()
        x = screen.x() + (screen.width() - self._widget_width) // 2
        y = screen.y() + int(screen.height() * 0.80) - self.WIDGET_HEIGHT // 2
        self.move(x, y)

    def set_listening(self, listening: bool):
        self.bar_color = self.COLOR_GREEN if listening else self.COLOR_GREY

    def update_levels(self, fft_magnitudes: np.ndarray):
        """Accept FFT magnitudes (one per bar), normalized 0-1."""
        n = min(len(fft_magnitudes), self.num_bars)
        self._target_levels[:n] = np.clip(fft_magnitudes[:n], 0.0, 1.0)

    def _animate(self):
        # Smooth interpolation toward target
        self._current_levels += (self._target_levels - self._current_levels) * 0.25
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background rounded rect
        bg_rect = QRectF(0, 0, self.width(), self.height())
        bg_path = QPainterPath()
        bg_path.addRoundedRect(bg_rect, self.CORNER_RADIUS, self.CORNER_RADIUS)
        painter.fillPath(bg_path, self.COLOR_BG)

        # Border
        pen = QPen(self.COLOR_BORDER, self.BORDER_WIDTH)
        painter.setPen(pen)
        painter.drawPath(bg_path)

        # Draw bars
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.bar_color)

        center_y = self.height() / 2.0

        for i in range(self.num_bars):
            level = self._current_levels[i] * self._bell_envelope[i]
            bar_h = self.BAR_MIN_HEIGHT + level * (self.BAR_MAX_HEIGHT - self.BAR_MIN_HEIGHT)
            x = self.H_PADDING + i * (self.BAR_WIDTH + self.BAR_GAP)
            y = center_y - bar_h / 2.0
            bar_rect = QRectF(x, y, self.BAR_WIDTH, bar_h)
            bar_path = QPainterPath()
            bar_path.addRoundedRect(bar_rect, self.BAR_RADIUS, self.BAR_RADIUS)
            painter.drawPath(bar_path)

        painter.end()

    def show_widget(self):
        self.position_on_screen()
        self._target_levels = np.zeros(self.num_bars)
        self._current_levels = np.zeros(self.num_bars)
        self.show()

    def hide_widget(self):
        self._target_levels = np.zeros(self.num_bars)
        self._current_levels = np.zeros(self.num_bars)
        self.hide()
