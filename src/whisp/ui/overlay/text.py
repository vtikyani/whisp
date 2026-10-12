"""Text for the overlay: short messages with inline key caps.

Key combinations written the Windows way ("Ctrl + V", "Ctrl+Win") are drawn
as raised key caps, matching the main window's hotkey style. Lines wrap
word-by-word (a key combination never splits) and very long text is elided.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPen

from whisp.ui import theme
from whisp.ui.overlay import style

_KEY = (
    r"(?:Ctrl|Shift|Alt|Win|Esc|Tab|Enter|Space|Backspace|Delete|Insert|Home|End|PgUp|PgDn"
    r"|F(?:2[0-4]|1[0-9]|[1-9])|[A-Z0-9])"
)
_COMBO = re.compile(rf"(?<![\w+])({_KEY}(?:\s*\+\s*{_KEY})+)(?![\w+])")

KEYCAP_HEIGHT = 20.0
KEYCAP_PAD_X = 6.0
KEYCAP_MIN_WIDTH = 20.0
KEYCAP_GAP = 4.0
KEYCAP_MARGIN = 2.0
ELLIPSIS = "…"


@dataclass(frozen=True)
class Run:
    """Either plain text or a key combination."""

    text: str = ""
    keys: tuple[str, ...] = ()


def parse_runs(message: str) -> tuple[Run, ...]:
    """Split a message into text and key-combination runs."""
    runs: list[Run] = []
    pos = 0
    for match in _COMBO.finditer(message):
        if match.start() > pos:
            runs.append(Run(text=message[pos : match.start()]))
        keys = tuple(k.strip() for k in match.group(1).split("+"))
        runs.append(Run(keys=keys))
        pos = match.end()
    if pos < len(message):
        runs.append(Run(text=message[pos:]))
    return tuple(runs)


def hint_runs(keys: list[str] | tuple[str, ...]) -> tuple[Run, ...]:
    """'Hold [Ctrl] [Win] to dictate', or a click hint when no hotkey is known."""
    keys = tuple(k for k in keys if k)
    if not keys:
        return (Run(text="Click to start dictating"),)
    return (Run(text="Hold "), Run(keys=keys), Run(text=" to dictate"))


@lru_cache(maxsize=32)
def _font(pixel_size: int, weight: int, tabular: bool = False) -> QFont:
    return theme.ui_font(pixel_size, QFont.Weight(weight), tabular=tabular)


def font(pixel_size: int, weight: QFont.Weight = QFont.Weight.Normal, *, tabular: bool = False) -> QFont:
    """Cached Inter font (fonts are created on every frame otherwise)."""
    return _font(pixel_size, int(weight.value), tabular)


@lru_cache(maxsize=32)
def _metrics(pixel_size: int, weight: int, tabular: bool = False) -> QFontMetricsF:
    return QFontMetricsF(_font(pixel_size, weight, tabular))


def metrics(pixel_size: int, weight: QFont.Weight = QFont.Weight.Normal, *, tabular: bool = False) -> QFontMetricsF:
    return _metrics(pixel_size, int(weight.value), tabular)


def snap(value: float, dpr: float) -> float:
    """Round a coordinate to the nearest device pixel so text and hairlines stay crisp."""
    return round(value * dpr) / dpr


def keycap_width(label: str) -> float:
    fm = metrics(style.FONT_KEYCAP, QFont.Weight.Medium)
    return max(KEYCAP_MIN_WIDTH, fm.horizontalAdvance(label) + 2 * KEYCAP_PAD_X)


def keys_width(keys: tuple[str, ...]) -> float:
    return sum(keycap_width(k) for k in keys) + KEYCAP_GAP * (len(keys) - 1) + 2 * KEYCAP_MARGIN


def draw_keycaps(painter: QPainter, left: float, center_y: float, keys: tuple[str, ...], opacity: float) -> None:
    """Raised key caps: surface fill, 1 px strong border, a 1 px darker bottom edge."""
    dpr = painter.device().devicePixelRatioF()
    fm = metrics(style.FONT_KEYCAP, QFont.Weight.Medium)
    top = snap(center_y - KEYCAP_HEIGHT / 2.0, dpr)
    x = left + KEYCAP_MARGIN
    painter.setFont(font(style.FONT_KEYCAP, QFont.Weight.Medium))
    for label in keys:
        width = round(keycap_width(label))
        x = snap(x, dpr)
        rect = QRectF(x, top, width, KEYCAP_HEIGHT)
        # Bottom edge first, one pixel lower, so it shows as a thin ledge under the cap.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(style.tinted(style.KEYCAP_SHADOW, opacity))
        painter.drawRoundedRect(rect.translated(0, 1.0), 5.0, 5.0)
        painter.setBrush(style.tinted(style.KEYCAP_FILL, opacity))
        painter.setPen(QPen(style.tinted(style.KEYCAP_BORDER, opacity), 1.0))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4.5, 4.5)
        painter.setPen(style.tinted(style.TEXT_1, opacity))
        text_x = x + (width - fm.horizontalAdvance(label)) / 2.0
        baseline = snap(top + KEYCAP_HEIGHT / 2.0 + fm.capHeight() / 2.0, dpr)
        painter.drawText(QPointF(text_x, baseline), label)
        x += width + KEYCAP_GAP


@dataclass(frozen=True)
class _Token:
    text: str
    keys: tuple[str, ...]
    width: float
    space_before: bool


@dataclass(frozen=True)
class Segment:
    """A piece of one laid-out line: text or key caps, at an x offset from the line start."""

    x: float
    text: str
    keys: tuple[str, ...]
    width: float


class RichText:
    """Runs laid out into at most ``max_lines`` lines no wider than ``max_width``."""

    LINE_HEIGHT = 18.0

    def __init__(
        self,
        runs: tuple[Run, ...],
        pixel_size: int = style.FONT_MESSAGE,
        weight: QFont.Weight = QFont.Weight.Medium,
        max_width: float = 10_000.0,
        max_lines: int = 3,
    ) -> None:
        self.runs = runs
        self.pixel_size = pixel_size
        self.weight = weight
        self._fm = metrics(pixel_size, weight)
        self.lines: list[list[Segment]] = self._layout(self._tokens(runs), max_width, max(1, max_lines))
        self.width = max((self._line_width(line) for line in self.lines), default=0.0)

    @property
    def height(self) -> float:
        return self.LINE_HEIGHT * len(self.lines)

    def _tokens(self, runs: tuple[Run, ...]) -> list[_Token]:
        tokens: list[_Token] = []
        pending_space = False
        for run in runs:
            if run.keys:
                tokens.append(_Token("", run.keys, keys_width(run.keys), pending_space))
                pending_space = False
                continue
            text = run.text
            parts = text.split(" ")
            for i, word in enumerate(parts):
                if i > 0:
                    pending_space = True
                if word:
                    tokens.append(_Token(word, (), self._fm.horizontalAdvance(word), pending_space))
                    pending_space = False
        return tokens

    def _layout(self, tokens: list[_Token], max_width: float, max_lines: int) -> list[list[Segment]]:
        space = self._fm.horizontalAdvance(" ")
        lines: list[list[_Token]] = [[]]
        widths = [0.0]
        for token in tokens:
            gap = space if token.space_before and lines[-1] else 0.0
            if lines[-1] and widths[-1] + gap + token.width > max_width:
                lines.append([])
                widths.append(0.0)
                gap = 0.0
            lines[-1].append(token)
            widths[-1] += gap + token.width
        if len(lines) > max_lines:
            kept = lines[: max_lines - 1]
            rest = [t for line in lines[max_lines - 1 :] for t in line]
            kept.append(self._elide(rest, max_width, space))
            lines = kept
        return [self._segments(line, space) for line in lines if line]

    def _elide(self, tokens: list[_Token], max_width: float, space: float) -> list[_Token]:
        text = " ".join(t.text if t.text else " + ".join(t.keys) for t in tokens)
        elided = self._fm.elidedText(text, Qt.TextElideMode.ElideRight, max_width)
        return [_Token(elided, (), self._fm.horizontalAdvance(elided), False)]

    def _segments(self, tokens: list[_Token], space: float) -> list[Segment]:
        segments: list[Segment] = []
        x = 0.0
        for i, token in enumerate(tokens):
            gap = space if token.space_before and i > 0 else 0.0
            last = segments[-1] if segments else None
            if token.text and last is not None and last.text:
                # Merge neighbouring words into one draw call (keeps kerning and is cheaper).
                merged = last.text + (" " if gap else "") + token.text
                segments[-1] = Segment(last.x, merged, (), self._fm.horizontalAdvance(merged))
                x = last.x + segments[-1].width
                continue
            x += gap
            segments.append(Segment(x, token.text, token.keys, token.width))
            x += token.width
        return segments

    @staticmethod
    def _line_width(line: list[Segment]) -> float:
        return (line[-1].x + line[-1].width) if line else 0.0

    def paint(
        self, painter: QPainter, left: float, center_y: float, color: QColor, opacity: float, *, centred: bool = True
    ) -> None:
        """Paint the block with its vertical centre at ``center_y``; lines are centred unless ``centred`` is False."""
        if opacity <= 0.0 or not self.lines:
            return
        dpr = painter.device().devicePixelRatioF()
        cap = self._fm.capHeight()
        top = center_y - self.height / 2.0
        for index, line in enumerate(self.lines):
            line_center = top + self.LINE_HEIGHT * (index + 0.5)
            line_left = left + (self.width - self._line_width(line)) / 2.0 if centred else left
            baseline = snap(line_center + cap / 2.0, dpr)
            for segment in line:
                x = snap(line_left + segment.x, dpr)
                if segment.keys:
                    draw_keycaps(painter, x, line_center, segment.keys, opacity)
                else:
                    painter.setFont(font(self.pixel_size, self.weight))
                    painter.setPen(style.tinted(color, opacity))
                    painter.drawText(QPointF(x, baseline), segment.text)


def faded_text_pen(left: float, right: float, color: QColor, fade_left: float, fade_right: float) -> QPen:
    """A pen whose brush fades text out over ``fade_left`` px at the left and ``fade_right`` px at the right."""
    width = max(1.0, right - left)
    gradient = QLinearGradient(left, 0.0, right, 0.0)
    clear = QColor(color)
    clear.setAlpha(0)
    gradient.setColorAt(0.0, clear)
    gradient.setColorAt(min(0.49, fade_left / width), color)
    gradient.setColorAt(max(0.51, 1.0 - fade_right / width), color)
    gradient.setColorAt(1.0, clear if fade_right > 0 else color)
    return QPen(QBrush(gradient), 1.0)
