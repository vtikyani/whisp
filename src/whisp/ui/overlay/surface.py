"""The pill's glass surface: soft shadow, opal glow, translucent fill and a lit hairline border.

Shadows and glows are built from gradients laid out like a nine-slice around a
rounded rectangle (four radial corners, four linear edges), so they are exact
for any pill size and cost a handful of fills instead of a blur.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient

from whisp.ui import theme
from whisp.ui.overlay import style

Profile = tuple[tuple[float, float], ...]

# Strength of a halo against its distance from the shape's edge (0 = at the edge, 1 = at full spread).
GLOW_PROFILE: Profile = tuple((t, (1.0 - t) ** 2.4) for t in (0.0, 0.1, 0.22, 0.36, 0.52, 0.7, 0.86, 1.0))
SHADOW_PROFILE: Profile = tuple(
    (t, math.exp(-3.2 * t * t) * (1.0 - t)) for t in (0.0, 0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
)


def pill_path(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


def _with_alpha(color: QColor, alpha: float) -> QColor:
    c = QColor(color)
    c.setAlphaF(max(0.0, min(1.0, alpha)))
    return c


def paint_halo(
    painter: QPainter,
    rect: QRectF,
    radius: float,
    spread: float,
    color: QColor,
    profile: Profile,
    *,
    fill_interior: bool = False,
) -> None:
    """A soft halo around a rounded rectangle whose alpha follows ``profile`` outward from the edge."""
    if spread <= 0.0 or rect.width() <= 0.0 or rect.height() <= 0.0:
        return
    r = max(0.0, min(radius, rect.width() / 2.0, rect.height() / 2.0))
    x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
    base_alpha = color.alphaF()
    outer = r + spread
    painter.setPen(Qt.PenStyle.NoPen)

    def radial(cx: float, cy: float) -> QBrush:
        gradient = QRadialGradient(QPointF(cx, cy), outer)
        if r > 0.0:
            gradient.setColorAt(0.0, _with_alpha(color, base_alpha * profile[0][1]))
        for t, strength in profile:
            gradient.setColorAt((r + t * spread) / outer, _with_alpha(color, base_alpha * strength))
        return QBrush(gradient)

    def linear(ax: float, ay: float, bx: float, by: float) -> QBrush:
        # From the edge (a) outward to full spread (b).
        gradient = QLinearGradient(ax, ay, bx, by)
        for t, strength in profile:
            gradient.setColorAt(t, _with_alpha(color, base_alpha * strength))
        return QBrush(gradient)

    # Corners.
    painter.fillRect(QRectF(x0 - spread, y0 - spread, r + spread, r + spread), radial(x0 + r, y0 + r))
    painter.fillRect(QRectF(x1 - r, y0 - spread, r + spread, r + spread), radial(x1 - r, y0 + r))
    painter.fillRect(QRectF(x0 - spread, y1 - r, r + spread, r + spread), radial(x0 + r, y1 - r))
    painter.fillRect(QRectF(x1 - r, y1 - r, r + spread, r + spread), radial(x1 - r, y1 - r))
    # Edges.
    inner_w, inner_h = rect.width() - 2 * r, rect.height() - 2 * r
    if inner_w > 0.0:
        painter.fillRect(QRectF(x0 + r, y0 - spread, inner_w, spread), linear(0, y0, 0, y0 - spread))
        painter.fillRect(QRectF(x0 + r, y1, inner_w, spread), linear(0, y1, 0, y1 + spread))
    if inner_h > 0.0:
        painter.fillRect(QRectF(x0 - spread, y0 + r, spread, inner_h), linear(x0, 0, x0 - spread, 0))
        painter.fillRect(QRectF(x1, y0 + r, spread, inner_h), linear(x1, 0, x1 + spread, 0))
    if fill_interior:
        interior = _with_alpha(color, base_alpha * profile[0][1])
        painter.fillRect(QRectF(x0 + r, y0, max(0.0, inner_w), rect.height()), interior)
        if inner_h > 0.0:
            painter.fillRect(QRectF(x0, y0 + r, r, inner_h), interior)
            painter.fillRect(QRectF(x1 - r, y0 + r, r, inner_h), interior)


GLOW_SPREAD = 16.0
SHADOW_MARGIN = 26.0


def _draw_shadow(painter: QPainter, pill: QRectF, radius: float) -> None:
    """Soft drop shadow plus a tight contact shadow (DESIGN: 0 8px 24px, dark)."""
    soft = pill.adjusted(3.0, 7.0, -3.0, 5.0)
    paint_halo(painter, soft, radius, 18.0, style.SHADOW, SHADOW_PROFILE, fill_interior=True)
    contact = pill.adjusted(0.5, 1.5, -0.5, 1.0)
    paint_halo(painter, contact, radius, 3.0, style.SHADOW_CONTACT, SHADOW_PROFILE, fill_interior=True)


def _draw_glow(painter: QPainter, pill: QRectF, radius: float) -> None:
    """A white halo, coloured afterwards with the opal gradient running across the pill."""
    paint_halo(painter, pill, radius, GLOW_SPREAD, QColor(255, 255, 255), GLOW_PROFILE)


def _opal_tint(painter: QPainter, pill: QRectF, image_rect: QRectF) -> None:
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(image_rect, QBrush(theme.opal_gradient(pill.left(), pill.right())))


class HaloLayer:
    """A halo around the pill (shadow or glow), rendered once per pill size and reused while the size holds still.

    The image sits on whole device pixels and the pill is drawn into it at the
    matching sub-pixel offset, so it lines up exactly with the glass. The pill's
    interior is cleared: halos live outside the glass only, where they cannot
    muddy the translucent fill, and no clip path is needed per frame.
    """

    def __init__(
        self,
        margin: float,
        draw: Callable[[QPainter, QRectF, float], None],
        tint: Callable[[QPainter, QRectF, QRectF], None] | None = None,
    ) -> None:
        self._margin = margin
        self._draw = draw
        self._tint = tint
        self._key: tuple[float, ...] | None = None
        self._image: QImage | None = None

    def paint(self, painter: QPainter, rect: QRectF, radius: float, opacity: float) -> None:
        if opacity <= 0.003 or rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        dpr = painter.device().devicePixelRatioF()
        m = self._margin
        origin = QPointF(math.floor((rect.left() - m) * dpr) / dpr, math.floor((rect.top() - m) * dpr) / dpr)
        local = rect.translated(-origin)
        key = (local.left(), local.top(), rect.width(), rect.height(), radius, dpr)
        key = tuple(round(v, 3) for v in key)
        if key != self._key or self._image is None:
            self._image = self._render(local, radius, dpr)
            self._key = key
        painter.save()
        painter.setOpacity(painter.opacity() * min(1.0, opacity))
        painter.drawImage(origin, self._image)
        painter.restore()

    def _render(self, pill: QRectF, radius: float, dpr: float) -> QImage:
        image_rect = QRectF(0.0, 0.0, pill.right() + self._margin, pill.bottom() + self._margin)
        image = QImage(
            max(1, math.ceil(image_rect.width() * dpr)),
            max(1, math.ceil(image_rect.height() * dpr)),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.setDevicePixelRatio(dpr)
        image.fill(Qt.GlobalColor.transparent)
        p = QPainter(image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw(p, pill, radius)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.fillPath(pill_path(pill, radius), QColor(0, 0, 0))
        if self._tint is not None:
            self._tint(p, pill, image_rect)
        p.end()
        return image


def shadow_layer() -> HaloLayer:
    return HaloLayer(SHADOW_MARGIN, _draw_shadow)


def glow_layer() -> HaloLayer:
    """The opal outer glow; paint it with a per-frame opacity so it breathes with the voice."""
    return HaloLayer(GLOW_SPREAD + 1.0, _draw_glow, _opal_tint)


def paint_glass(
    painter: QPainter,
    rect: QRectF,
    radius: float,
    opacity: float,
    tint: QColor | None = None,
    tint_amount: float = 0.0,
    border_opacity: float | None = None,
) -> None:
    """The pill body: translucent ink fill and a 1 px border lit slightly from above.

    ``tint`` (e.g. danger red) blends into the border by ``tint_amount``.
    ``border_opacity`` defaults to ``opacity``; the idle hairline keeps its edge
    brighter than its fill so it stays findable on dark desktops.
    """
    path = pill_path(rect, radius)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(style.tinted(style.FILL, opacity))
    painter.drawPath(path)
    opacity = opacity if border_opacity is None else border_opacity
    top, bottom = style.BORDER_TOP, style.BORDER_BOTTOM
    if tint is not None and tint_amount > 0.0:
        lit = _with_alpha(tint, 0.62)
        top = style.mix(top, lit, tint_amount)
        bottom = style.mix(bottom, _with_alpha(tint, 0.42), tint_amount)
    gradient = QLinearGradient(0.0, rect.top(), 0.0, rect.bottom())
    gradient.setColorAt(0.0, style.tinted(top, opacity))
    gradient.setColorAt(0.5, style.tinted(style.mix(top, bottom, 0.65), opacity))
    gradient.setColorAt(1.0, style.tinted(bottom, opacity))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QBrush(gradient), 1.0))
    inset = rect.adjusted(0.5, 0.5, -0.5, -0.5)
    painter.drawRoundedRect(inset, max(0.0, radius - 0.5), max(0.0, radius - 0.5))
