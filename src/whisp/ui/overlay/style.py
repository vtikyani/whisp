"""Overlay design tokens.

Shared tokens (colours, fonts, durations) come from ``whisp.ui.theme`` and
docs/DESIGN.md. The values defined here are specific to the pill; the ones
marked "overlay-only" are refinements that DESIGN.md does not spell out.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

from whisp.ui import theme


def _rgba(hex_color: str, alpha: float) -> QColor:
    c = QColor(hex_color)
    c.setAlphaF(alpha)
    return c


# -- Geometry (logical px) ---------------------------------------------------

SCREEN_MARGIN = 24
"""Gap between the pill and the edge of the monitor's work area."""

ROW_HEIGHT = 44.0
"""Height of the main row: listening, hands-free, processing and done."""

IDLE_WIDTH, IDLE_HEIGHT = 40.0, 6.0
IDLE_HOVER_HEIGHT = 30.0
IDLE_OPACITY = 0.35
IDLE_HIT_WIDTH, IDLE_HIT_HEIGHT = 76.0, 24.0
"""Overlay-only: invisible hover target around the 40 x 6 hairline, so it is easy to find with the mouse."""

HOLD_WIDTH = 184.0
HANDS_FREE_WIDTH = 264.0
PROCESSING_WIDTH = 120.0
DONE_SIZE = 44.0
PREVIEW_HEIGHT = 28.0
"""Extra height for the live preview / note line beneath the wisp."""

MESSAGE_HEIGHT = 36.0
"""Overlay-only: height of message pills (copied, no speech, errors, notices)."""
MESSAGE_MAX_WIDTH = 460.0
MESSAGE_PADDING = 16.0
ICON_SIZE = 16.0
ICON_GAP = 8.0

WISP_INSET = 18.0
"""Horizontal breathing room between the pill edge and the ends of the wisp."""
BUTTON_SIZE = 28.0
BUTTON_INSET = 8.0
TIMER_GAP = 10.0
CHIP_HEIGHT = 22.0
CHIP_GAP = 10.0

GLOW_MARGIN = 30
"""Room around the pill inside the window for the shadow, the opal glow and the shake."""

SHAKE_AMPLITUDE = 6.0
SHAKE_SECONDS = 0.42

# -- Colour ------------------------------------------------------------------

FILL = QColor(theme.OVERLAY_FILL)
BORDER = QColor(theme.OVERLAY_BORDER)
BORDER_TOP = QColor(255, 255, 255, 46)
"""Overlay-only: the border is a touch brighter along the top edge, like light catching glass."""
BORDER_BOTTOM = QColor(255, 255, 255, 18)
SHADOW = QColor(0, 0, 0, 92)
SHADOW_CONTACT = QColor(0, 0, 0, 60)

TEXT_1 = theme.color("text_1")
TEXT_2 = theme.color("text_2")
TEXT_3 = theme.color("text_3")
SUCCESS = theme.color("success")
WARNING = theme.color("warning")
DANGER = theme.color("danger")
ACCENT = QColor(theme.ACCENT_DARK)
ORCHID = QColor(theme.OPAL_STOPS[2][1])
ORCHID_TEXT = QColor("#D8B4FE")
"""Overlay-only: a lighter orchid for the "Edit" chip label so it stays readable on dark glass."""

BUTTON_REST = QColor(255, 255, 255, 12)
BUTTON_HOVER = QColor(255, 255, 255, 26)
BUTTON_PRESSED = QColor(255, 255, 255, 40)
STOP_FILL = QColor(255, 255, 255, 30)
STOP_FILL_HOVER = QColor(255, 255, 255, 48)
STOP_FILL_PRESSED = QColor(255, 255, 255, 64)
KEYCAP_FILL = QColor(theme.DARK["surface_3"])
KEYCAP_BORDER = QColor(255, 255, 255, 34)
KEYCAP_SHADOW = QColor(0, 0, 0, 120)

KIND_COLORS: dict[str, QColor] = {"info": ACCENT, "warning": WARNING, "error": DANGER}


def tinted(color: QColor, alpha: float) -> QColor:
    """``color`` with its alpha multiplied by ``alpha``."""
    c = QColor(color)
    c.setAlphaF(max(0.0, min(1.0, c.alphaF() * alpha)))
    return c


def mix(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
        a.alphaF() + (b.alphaF() - a.alphaF()) * t,
    )


ORCHID_CHIP_FILL = _rgba(theme.OPAL_STOPS[2][1], 0.16)
ORCHID_CHIP_BORDER = _rgba(theme.OPAL_STOPS[2][1], 0.38)

# -- Type --------------------------------------------------------------------

FONT_MESSAGE = 13
FONT_PREVIEW = 13
FONT_NOTE = 12
FONT_TIMER = 12
FONT_CHIP = 12
FONT_HINT = 12
FONT_KEYCAP = 11

# -- Motion (seconds) --------------------------------------------------------

SPRING_OMEGA = 21.0
"""Pill morph: critically damped, ~99% settled after ~320 ms (DESIGN: 320 ms for the overlay morph)."""
FADE_IN = theme.DURATION_MEDIUM_MS / 1000
FADE_OUT = theme.DURATION_FAST_MS / 1000
CONTROLS_DELAY = 0.07
"""Overlay-only: buttons and the Edit chip wait this long before fading in, so the pill has room for them."""
MESSAGE_DELAY = 0.09
"""Overlay-only: message text waits this long, until the fading wisp has cleared the pill's middle."""

IDLE_HOVER_DELAY = 0.12
"""Overlay-only: hover must rest this long before the idle pill expands (avoids flicker on fly-overs)."""
IDLE_LEAVE_DELAY = 0.35

DONE_CHECK_DELAY = 0.1
DONE_CHECK_SECONDS = 0.26
DONE_HOLD_SECONDS = 0.62
"""Check drawn and held; then the pill retracts (≈600 ms in total, per DESIGN)."""
COPIED_SECONDS = 2.5
NO_SPEECH_SECONDS = 1.6
CANCEL_SECONDS = 0.0
ERROR_MIN_SECONDS = 3.0
NOTICE_MIN_SECONDS = 2.5
MESSAGE_MAX_SECONDS = 7.0
READING_BASE_SECONDS = 0.6
READING_SECONDS_PER_CHAR = 0.05
"""Long messages stay up long enough to read (0.6 s + 20 characters a second), within the min/max above."""

SHIMMER_PERIOD = 1.35
