"""Vector glyphs for the overlay, drawn as paths so they stay crisp at any DPI and never depend on a font."""

from __future__ import annotations

import itertools
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

from whisp.ui.overlay import style


def _pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def draw_cross(painter: QPainter, center: QPointF, size: float, color: QColor, width: float = 1.6) -> None:
    """The ✕ of the cancel button."""
    h = size / 2.0
    painter.setPen(_pen(color, width))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawLine(QPointF(center.x() - h, center.y() - h), QPointF(center.x() + h, center.y() + h))
    painter.drawLine(QPointF(center.x() - h, center.y() + h), QPointF(center.x() + h, center.y() - h))


def draw_stop(painter: QPainter, center: QPointF, size: float, color: QColor) -> None:
    """The ■ of the stop button: a softly rounded square."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    rect = QRectF(center.x() - size / 2.0, center.y() - size / 2.0, size, size)
    painter.drawRoundedRect(rect, size * 0.22, size * 0.22)


# Lucide "check" (24 px grid), centred on the glyph's visual middle.
_CHECK_POINTS = ((-8.0, 0.5), (-3.0, 5.5), (8.0, -5.5))


def draw_check(
    painter: QPainter, center: QPointF, scale: float, progress: float, color: QColor, width: float = 2.25
) -> None:
    """A check mark drawn up to ``progress`` (0-1) of its length, so it can draw itself."""
    if progress <= 0.0:
        return
    points = [QPointF(center.x() + x * scale, center.y() + y * scale) for x, y in _CHECK_POINTS]
    segments = list(itertools.pairwise(points))
    lengths = [math.dist((a.x(), a.y()), (b.x(), b.y())) for a, b in segments]
    remaining = sum(lengths) * min(1.0, progress)
    path = QPainterPath(points[0])
    for (a, b), length in zip(segments, lengths, strict=True):
        if remaining <= 0.0:
            break
        t = min(1.0, remaining / length)
        path.lineTo(QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t))
        remaining -= length
    painter.setPen(_pen(color, width))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)


def sparkle_path(center: QPointF, radius: float) -> QPainterPath:
    """A four-pointed sparkle (✦) with gently concave sides."""
    r, c = radius, radius * 0.16
    cx, cy = center.x(), center.y()
    path = QPainterPath(QPointF(cx, cy - r))
    path.quadTo(QPointF(cx + c, cy - c), QPointF(cx + r, cy))
    path.quadTo(QPointF(cx + c, cy + c), QPointF(cx, cy + r))
    path.quadTo(QPointF(cx - c, cy + c), QPointF(cx - r, cy))
    path.quadTo(QPointF(cx - c, cy - c), QPointF(cx, cy - r))
    path.closeSubpath()
    return path


def draw_copy(painter: QPainter, rect: QRectF, color: QColor, width: float = 1.4) -> None:
    """Lucide "copy": two overlapping rounded squares."""
    s = rect.width() / 24.0
    ox, oy = rect.x(), rect.y()

    def pt(x: float, y: float) -> QPointF:
        return QPointF(ox + x * s, oy + y * s)

    painter.setPen(_pen(color, width))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(ox + 8 * s, oy + 8 * s, 14 * s, 14 * s), 2.5 * s, 2.5 * s)
    back = QPainterPath(pt(4, 16))
    back.cubicTo(pt(2.9, 16), pt(2, 15.1), pt(2, 14))
    back.lineTo(pt(2, 4))
    back.cubicTo(pt(2, 2.9), pt(2.9, 2), pt(4, 2))
    back.lineTo(pt(14, 2))
    back.cubicTo(pt(15.1, 2), pt(16, 2.9), pt(16, 4))
    painter.drawPath(back)


def draw_badge(painter: QPainter, center: QPointF, kind: str, diameter: float = style.ICON_SIZE) -> None:
    """A small round status badge: "i" for info, "!" for warnings and errors."""
    color = style.KIND_COLORS.get(kind, style.ACCENT)
    r = diameter / 2.0
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(style.tinted(color, 0.2))
    painter.drawEllipse(center, r, r)
    painter.setPen(_pen(color, 1.6))
    cx, cy = center.x(), center.y()
    dot = QPointF(cx, cy - r * 0.42) if kind == "info" else QPointF(cx, cy + r * 0.45)
    if kind == "info":
        painter.drawLine(QPointF(cx, cy - r * 0.02), QPointF(cx, cy + r * 0.45))
    else:
        painter.drawLine(QPointF(cx, cy - r * 0.45), QPointF(cx, cy + r * 0.05))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawEllipse(dot, 1.0, 1.0)
