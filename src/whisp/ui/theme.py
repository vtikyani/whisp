"""Design tokens for Qt-painted UI. Mirrors docs/DESIGN.md — keep them in sync."""

from __future__ import annotations

import itertools
import logging

from PySide6.QtGui import QColor, QFont, QFontDatabase, QLinearGradient

from whisp.storage.paths import resource_path

log = logging.getLogger(__name__)

OPAL_STOPS: tuple[tuple[float, str], ...] = (
    (0.0, "#67E8F9"),
    (0.35, "#818CF8"),
    (0.65, "#C084FC"),
    (1.0, "#F9A8D4"),
)

ACCENT_DARK = "#8B9BFF"
ACCENT_LIGHT = "#5B63E6"

DARK = {
    "bg": "#0B0B0E",
    "surface_1": "#121216",
    "surface_2": "#18181E",
    "surface_3": "#202028",
    "border": QColor(255, 255, 255, 18),
    "border_strong": QColor(255, 255, 255, 31),
    "text_1": "#F4F4F6",
    "text_2": "#B3B3BD",
    "text_3": "#7B7B86",
    "success": "#4ADE80",
    "warning": "#FBBF24",
    "danger": "#F87171",
}

OVERLAY_FILL = QColor(14, 14, 18, 219)
OVERLAY_BORDER = QColor(255, 255, 255, 26)

DURATION_FAST_MS = 120
DURATION_MEDIUM_MS = 200
DURATION_SLOW_MS = 320

FONT_FAMILY = "Inter"
_font_loaded: bool | None = None


def color(token: str) -> QColor:
    value = DARK[token]
    return value if isinstance(value, QColor) else QColor(value)


def opal_gradient(x0: float, x1: float, alpha: float = 1.0) -> QLinearGradient:
    """Horizontal opal gradient across [x0, x1] in the painter's coordinates."""
    gradient = QLinearGradient(x0, 0.0, x1, 0.0)
    for offset, hex_color in OPAL_STOPS:
        c = QColor(hex_color)
        c.setAlphaF(max(0.0, min(1.0, alpha)))
        gradient.setColorAt(offset, c)
    return gradient


def opal_at(t: float) -> QColor:
    """Colour of the opal gradient at position ``t`` in [0, 1]."""
    t = max(0.0, min(1.0, t))
    for (o0, c0), (o1, c1) in itertools.pairwise(OPAL_STOPS):
        if t <= o1:
            k = 0.0 if o1 == o0 else (t - o0) / (o1 - o0)
            a, b = QColor(c0), QColor(c1)
            return QColor(
                round(a.red() + (b.red() - a.red()) * k),
                round(a.green() + (b.green() - a.green()) * k),
                round(a.blue() + (b.blue() - a.blue()) * k),
            )
    return QColor(OPAL_STOPS[-1][1])


def ensure_font_loaded() -> bool:
    """Register the bundled Inter font with Qt once. Safe to call repeatedly."""
    global _font_loaded
    if _font_loaded is None:
        font_id = QFontDatabase.addApplicationFont(str(resource_path("assets", "fonts", "InterVariable.ttf")))
        _font_loaded = font_id != -1
        if not _font_loaded:
            log.warning("Could not load the bundled Inter font; falling back to the system UI font")
    return _font_loaded


def ui_font(pixel_size: int, weight: QFont.Weight = QFont.Weight.Normal, *, tabular: bool = False) -> QFont:
    """Inter at a pixel size. ``tabular=True`` gives fixed-width digits (timers, counters)."""
    ensure_font_loaded()
    font = QFont(FONT_FAMILY if _font_loaded else "Segoe UI Variable")
    font.setPixelSize(pixel_size)
    font.setWeight(weight)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    if tabular:
        font.setFeature(QFont.Tag("tnum"), 1)
    return font
