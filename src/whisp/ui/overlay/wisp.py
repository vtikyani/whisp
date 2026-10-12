"""The wisp: three luminous opal ribbons that breathe when you're silent and swell when you speak.

Each ribbon's centre line is a sum of sines with its own phase speeds, shaped
by a bell envelope so both ends taper to nothing. Its thickness also varies
slowly along its length, so it reads as a ribbon turning in light rather than
a flat line. Back ribbons are dimmer with a soft glow; the front ribbon has a
crisp core. Ribbons are blended additively, so where they cross they glow.

``calm`` (0-1) settles the ribbons into a single hairline for the processing
state, where an opal shimmer sweeps across it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen, QPolygonF

from whisp.ui import theme
from whisp.ui.overlay import style
from whisp.ui.overlay.motion import LevelFollower, Spring, clamp

TAU = 2.0 * math.pi


@dataclass(frozen=True)
class _Ribbon:
    amplitude: float
    """Relative to the front ribbon."""
    freqs: tuple[float, float, float]
    """Cycles across the wisp's width."""
    weights: tuple[float, float, float]
    speeds: tuple[float, float, float]
    """Phase speeds in rad/s at rest; they quicken as the voice gets louder."""
    offsets: tuple[float, float, float]
    core: float
    """Core thickness in px at the widest point."""
    core_alpha: float
    glow: float
    """Glow stroke width in px."""
    glow_alpha: float
    reverse: bool
    """Run the opal gradient right-to-left, so crossing ribbons trade colours."""
    twist_freq: float
    twist_speed: float


# Back to front.
_RIBBONS = (
    _Ribbon(
        amplitude=0.72,
        freqs=(1.8, 2.25, 4.6),
        weights=(1.0, 0.6, 0.25),
        speeds=(1.3, -2.2, 3.4),
        offsets=(0.4, 2.1, 4.0),
        core=1.5,
        core_alpha=0.55,
        glow=6.0,
        glow_alpha=0.10,
        reverse=False,
        twist_freq=1.1,
        twist_speed=-0.9,
    ),
    _Ribbon(
        amplitude=0.88,
        freqs=(1.2, 2.9, 3.7),
        weights=(1.0, 0.45, 0.3),
        speeds=(-1.7, 2.6, -3.9),
        offsets=(2.6, 0.9, 5.1),
        core=1.7,
        core_alpha=0.65,
        glow=6.5,
        glow_alpha=0.11,
        reverse=True,
        twist_freq=0.9,
        twist_speed=1.2,
    ),
    _Ribbon(
        amplitude=1.0,
        freqs=(1.5, 2.6, 4.1),
        weights=(1.0, 0.55, 0.22),
        speeds=(2.1, -3.0, 4.3),
        offsets=(1.2, 3.7, 0.3),
        core=2.1,
        core_alpha=1.0,
        glow=5.0,
        glow_alpha=0.16,
        reverse=False,
        twist_freq=1.3,
        twist_speed=0.8,
    ),
)
_TWIST_SPEEDS = np.array([r.twist_speed for r in _RIBBONS], dtype=np.float64)

REST_AMPLITUDE = 1.3
"""Peak deflection in px while silent: a calm hairline that only just undulates."""
BREATH_SECONDS = 3.8
END_FADE = 0.14
"""Fraction of the width over which each end fades to nothing."""
ENVELOPE_POWER = 1.25
"""Shape of the bell that tapers the swell toward both ends (higher = more centred)."""
SOFT_LIMIT = 2.0
"""Soft limiter on the summed sines: typical sums still reach most of the amplitude, peaks never overshoot it."""


def _end_fade(u: float) -> float:
    a = clamp(u / END_FADE)
    b = clamp((1.0 - u) / END_FADE)
    return a * a * (3 - 2 * a) * b * b * (3 - 2 * b)


_GRADIENT_STOPS = (0.0, 0.06, 0.14, 0.24, 0.36, 0.5, 0.64, 0.76, 0.86, 0.94, 1.0)
_STOP_COLORS = tuple(theme.opal_at(u) for u in _GRADIENT_STOPS)
_STOP_FADES = tuple(_end_fade(u) for u in _GRADIENT_STOPS)


def _band(xs: np.ndarray, ys: np.ndarray, ox: np.ndarray, oy: np.ndarray) -> QPolygonF:
    """A closed band around the centre line (xs, ys), offset by (ox, oy) to each side."""
    px = np.concatenate((xs + ox, (xs - ox)[::-1])).tolist()
    py = np.concatenate((ys + oy, (ys - oy)[::-1])).tolist()
    return QPolygonF(list(map(QPointF, px, py)))


def opal_ribbon_gradient(x0: float, x1: float, alpha: float, reverse: bool = False) -> QLinearGradient:
    """The opal gradient across [x0, x1] with both ends fading to transparent."""
    gradient = QLinearGradient(x0, 0.0, x1, 0.0)
    n = len(_GRADIENT_STOPS)
    for i, offset in enumerate(_GRADIENT_STOPS):
        c = QColor(_STOP_COLORS[n - 1 - i] if reverse else _STOP_COLORS[i])
        c.setAlphaF(clamp(alpha * _STOP_FADES[i]))
        gradient.setColorAt(offset, c)
    return gradient


class Wisp:
    """Animated state and painting for the wisp. Advance with ``step(dt)``, then ``paint``."""

    def __init__(self) -> None:
        self.level = LevelFollower(attack=0.06, release=0.25)
        self.calm = Spring(0.0, omega=12.0, precision=0.002)
        """0 = alive and listening, 1 = settled into the processing hairline."""
        self.reduced_motion = False
        self._time = 0.0
        self._phases = np.array([r.offsets for r in _RIBBONS], dtype=np.float64)
        self._speeds = np.array([r.speeds for r in _RIBBONS], dtype=np.float64)
        self._twist = np.zeros(len(_RIBBONS))
        self._shimmer = 0.0
        self._grids: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._brushes: dict[tuple[float, float, bool], QBrush] = {}

    # -- state -------------------------------------------------------------

    def set_level(self, level: float) -> None:
        self.level.set(level)

    def reset(self) -> None:
        """Back to a silent, live wisp (a new recording)."""
        self.level.reset()

    def set_calm(self, calm: bool, *, instantly: bool = False) -> None:
        self.calm.target = 1.0 if calm else 0.0
        if instantly:
            self.calm.snap()
        if calm and self.calm.value < 0.05:
            self._shimmer = 0.0

    @property
    def energy(self) -> float:
        """Perceptual loudness after attack/release smoothing, eased so quiet speech still shows (0-1)."""
        return 1.0 - (1.0 - self.level.value) ** 2.2

    @property
    def breath(self) -> float:
        """The slow 0-1 breathing cycle that keeps a silent wisp alive."""
        return 0.5 + 0.5 * math.sin(TAU * self._time / BREATH_SECONDS)

    def step(self, dt: float) -> None:
        self._time += dt
        self.level.step(dt)
        self.calm.step(dt)
        pace = 0.55 + 1.3 * self.energy
        if self.reduced_motion:
            pace *= 0.45
        self._phases = (self._phases + self._speeds * (pace * dt)) % TAU
        self._twist = (self._twist + _TWIST_SPEEDS * dt) % TAU
        self._shimmer = (self._shimmer + dt / style.SHIMMER_PERIOD) % 1.0

    # -- painting ------------------------------------------------------------

    def paint(self, painter: QPainter, x0: float, x1: float, cy: float, max_amplitude: float, opacity: float) -> None:
        width = x1 - x0
        if opacity <= 0.0 or width < 8.0:
            return
        painter.save()
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        calm = clamp(self.calm.value)
        live = 1.0 - calm
        if live > 0.001:
            self._paint_ribbons(painter, x0, width, cy, max_amplitude, opacity, live)
        if calm > 0.001:
            self._paint_shimmer(painter, x0, x1, cy, opacity * calm)
        painter.restore()

    def _amplitude(self, max_amplitude: float) -> float:
        rest = REST_AMPLITUDE * (0.75 + 0.5 * self.breath)
        peak = max_amplitude * (0.55 if self.reduced_motion else 1.0)
        return rest + max(0.0, peak - rest) * self.energy

    def _paint_ribbons(
        self, painter: QPainter, x0: float, width: float, cy: float, max_amplitude: float, opacity: float, live: float
    ) -> None:
        n = int(clamp(width / 3.0, 28, 96))
        u, envelope, taper = self._grid(n)
        xs = x0 + u * width
        amplitude = self._amplitude(max_amplitude) * live
        swell = 1.0 + 0.35 * self.energy
        limit = math.tanh(SOFT_LIMIT)
        base_opacity = painter.opacity()
        glow_strength = 0.75 + 0.6 * self.energy
        painter.setPen(Qt.PenStyle.NoPen)
        for index, ribbon in enumerate(_RIBBONS):
            phases = self._phases[index]
            wave = np.zeros(n)
            for k in range(3):
                wave += ribbon.weights[k] * np.sin(TAU * ribbon.freqs[k] * u + phases[k])
            wave = np.tanh(SOFT_LIMIT * wave / sum(ribbon.weights)) / limit
            ys = cy + amplitude * ribbon.amplitude * envelope * wave
            dx = np.gradient(xs)
            dy = np.gradient(ys)
            norm = np.hypot(dx, dy)
            nx, ny = -dy / norm, dx / norm
            painter.setBrush(self._brush(x0, width, ribbon.reverse))

            # Glow: a wide, faint band around the ribbon that also tapers to nothing at the ends.
            half = 0.5 * ribbon.glow * swell * (0.3 + 0.7 * taper)
            painter.setOpacity(base_opacity * min(1.0, ribbon.glow_alpha * opacity * glow_strength))
            painter.drawPolygon(_band(xs, ys, nx * half, ny * half))

            # Core: a filled ribbon whose thickness tapers at the ends and turns slowly along its length.
            twist = 0.6 + 0.4 * np.abs(np.cos(TAU * ribbon.twist_freq * u + self._twist[index]))
            half = 0.5 * ribbon.core * swell * (0.25 + 0.75 * taper) * twist
            painter.setOpacity(base_opacity * min(1.0, ribbon.core_alpha * opacity))
            painter.drawPolygon(_band(xs, ys, nx * half, ny * half))
        painter.setOpacity(base_opacity)

    def _grid(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample positions along the wisp with their swell envelope and thickness taper (cached per count)."""
        grid = self._grids.get(n)
        if grid is None:
            u = np.linspace(0.0, 1.0, n)
            bell = np.sin(np.pi * u)
            grid = (u, bell**ENVELOPE_POWER, bell**0.7)
            self._grids[n] = grid
        return grid

    def _brush(self, x0: float, width: float, reverse: bool) -> QBrush:
        """The ribbon's opal brush at full strength (opacity is applied by the painter), cached per geometry."""
        key = (round(x0, 2), round(width, 2), reverse)
        brush = self._brushes.get(key)
        if brush is None:
            if len(self._brushes) > 24:
                self._brushes.clear()  # the wisp is morphing; keep the cache from growing
            brush = QBrush(opal_ribbon_gradient(x0, x0 + width, 1.0, reverse))
            self._brushes[key] = brush
        return brush

    def _paint_shimmer(self, painter: QPainter, x0: float, x1: float, cy: float, opacity: float) -> None:
        width = x1 - x0
        # The settled hairline.
        base = QPen(QBrush(opal_ribbon_gradient(x0, x1, 0.72 * opacity)), 1.5)
        base.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(base)
        painter.drawLine(QPointF(x0, cy), QPointF(x1, cy))
        if self.reduced_motion:
            # No movement: a slow brightness pulse instead of a sweep.
            pulse = 0.5 - 0.5 * math.cos(TAU * self._time / (style.SHIMMER_PERIOD * 1.6))
            pen = QPen(QBrush(opal_ribbon_gradient(x0, x1, 0.55 * opacity * pulse)), 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(x0, cy), QPointF(x1, cy))
            return
        band = max(18.0, width * 0.32)
        # Sweep for 90% of each period with a short rest; the band's centre stays near the line, so it never looks idle.
        progress = 0.5 - 0.5 * math.cos(math.pi * clamp(self._shimmer / 0.9))
        centre = x0 - band * 0.5 + (width + band) * progress
        for pen_width, alpha in ((8.0, 0.22), (3.5, 0.45), (1.8, 1.0)):
            gradient = QLinearGradient(centre - band, 0.0, centre + band, 0.0)
            for offset, strength in ((0.0, 0.0), (0.3, 0.35), (0.5, 1.0), (0.7, 0.35), (1.0, 0.0)):
                x = centre - band + 2 * band * offset
                c = QColor(theme.opal_at((x - x0) / width))
                # Lift the highlight toward white at its centre so it reads as light, not paint.
                c = QColor.fromRgbF(
                    c.redF() + (1 - c.redF()) * 0.45 * strength,
                    c.greenF() + (1 - c.greenF()) * 0.45 * strength,
                    c.blueF() + (1 - c.blueF()) * 0.45 * strength,
                )
                c.setAlphaF(clamp(alpha * strength * opacity * _end_fade(clamp((x - x0) / width))))
                gradient.setColorAt(offset, c)
            pen = QPen(QBrush(gradient), pen_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(x0, cy), QPointF(x1, cy))
