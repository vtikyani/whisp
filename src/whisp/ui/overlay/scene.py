"""The overlay's visual state machine: what the pill shows, how it morphs between states, and how it paints.

The scene knows nothing about windows or screens. It is driven by a clock,
advanced once per frame, and painted around an *anchor*: the pill's horizontal
centre on its fixed edge (the bottom edge when the pill sits at the bottom of
the screen, the top edge when it sits at the top). ``OverlayWindow`` hosts it
on screen; the render script paints it straight into images.

Every visual property is a spring or a fade, so any state can follow any other
mid-animation without a jump.
"""

from __future__ import annotations

import enum
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QRadialGradient

from whisp.core.types import Outcome, RecordingMode, SessionKind
from whisp.ui.overlay import glyphs, style, surface
from whisp.ui.overlay.motion import Fade, Slot, Spring, clamp, ease_out_cubic
from whisp.ui.overlay.text import RichText, faded_text_pen, font, hint_runs, metrics, parse_runs, snap
from whisp.ui.overlay.wisp import Wisp

COPIED_TEXT = "Copied — press Ctrl + V to paste"
NO_SPEECH_TEXT = "Didn't catch that"
FAILED_TEXT = "Something went wrong. Please try again."
PREVIEW_TAIL_CHARS = 240
"""Only the end of a long dictation can be seen; laying out more is wasted work."""
LINE_PADDING = 16.0
"""Horizontal padding of the preview / note line."""
PREVIEW_FADE = 28.0
"""Width of the fade where older preview words leave on the left."""


class Display(enum.Enum):
    HIDDEN = "hidden"
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    DONE = "done"
    MESSAGE = "message"


@dataclass(frozen=True)
class Message:
    """A message pill: text (key combinations become key caps), an optional icon and an optional tint."""

    text: str
    icon: str | None = None
    """"copy", "info", "warning", "error" or None."""
    tint: str | None = None
    """"warning" or "error" tints the border; None keeps it neutral."""


@dataclass(frozen=True)
class Line:
    """What the secondary line beneath the wisp shows."""

    kind: str
    """"preview" (live words), "note" (e.g. loading) or "message" (a short-lived outcome or notice)."""
    text: str = ""
    tone: str = "info"


@dataclass
class _Transient:
    display: Display
    message: Message | None
    started: float
    expires: float


def reading_seconds(text: str, minimum: float) -> float:
    """How long a message stays up: long enough to read, never forever."""
    reading = style.READING_BASE_SECONDS + len(text) * style.READING_SECONDS_PER_CHAR
    return min(style.MESSAGE_MAX_SECONDS, max(minimum, reading))


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class OverlayScene:
    """See the module docstring. All methods run on the GUI thread."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last = clock()
        self._was_animating = False
        self.placement = "bottom"
        self.reduced_motion = False

        # What to show.
        self._rest_idle = False
        self._hint_keys: tuple[str, ...] = ()
        self._base = Display.HIDDEN
        self._kind = SessionKind.DICTATION
        self._mode = RecordingMode.HOLD_PENDING
        self._started = 0.0
        self._preview = ""
        self._note: str | None = None
        self._inline: Line | None = None
        self._inline_expires = 0.0
        self._transient: _Transient | None = None
        self._display = Display.HIDDEN

        # Pointer.
        self._pointer_inside = False
        self._pointer_changed = 0.0
        self._idle_expanded = False
        self.hover: str | None = None
        self.pressed: str | None = None

        # Animated values.
        omega = style.SPRING_OMEGA
        self.width = Spring(style.IDLE_WIDTH * 0.6, omega, 0.05)
        self.height = Spring(style.IDLE_HEIGHT, omega, 0.05)
        self.alpha = Spring(0.0, 18.0, 0.002)
        # The wisp's ends move a little faster than the pill, so they clear the way before buttons fade in.
        self.wisp_left = Spring(style.WISP_INSET, omega * 1.45, 0.05)
        self.wisp_right = Spring(style.WISP_INSET, omega * 1.45, 0.05)
        self.chip_left = Spring(12.0, omega, 0.05)
        self.glow = Spring(0.0, 9.0, 0.002)
        self.tint = Spring(0.0, 14.0, 0.002)
        self._tint_kind: str | None = None
        self.preview_offset = Spring(0.0, 13.0, 0.05)

        self.wisp = Wisp()
        self.wisp_fade = Fade(False, 0.22, 0.1)
        self.controls_fade = Fade(False, style.FADE_IN, style.FADE_OUT, delay=style.CONTROLS_DELAY)
        self.chip_fade = Fade(False, style.FADE_IN, style.FADE_OUT, delay=style.CONTROLS_DELAY)
        self.hint_fade = Fade(False, 0.2, 0.1)
        self.check_fade = Fade(False, 0.12, 0.12)
        self.line: Slot[Line] = Slot(0.2, 0.12)
        # Messages wait for the wisp to clear the middle row, so the hairline never strikes through the text.
        self.message: Slot[Message] = Slot(0.2, 0.1, delay=style.MESSAGE_DELAY)
        self._glow_layer = surface.glow_layer()
        self._shadow_layer = surface.shadow_layer()
        self._rich_cache: dict[tuple[str, int], RichText] = {}
        self._shake_started: float | None = None
        self._check_started: float | None = None

    # -- inputs ----------------------------------------------------------------

    def set_rest(self, idle: bool) -> None:
        """Whether the pill rests as the idle hairline (True) or disappears (False) when nothing is happening."""
        if idle != self._rest_idle:
            self._rest_idle = idle
            if not idle:
                self._idle_expanded = False
            self._apply()

    def set_hint_keys(self, keys: list[str] | tuple[str, ...]) -> None:
        keys = tuple(str(k) for k in keys if k)
        if keys != self._hint_keys:
            self._hint_keys = keys
            self._apply()

    def show_recording(self, kind: SessionKind, mode: RecordingMode) -> None:
        now = self._clock()
        self._base = Display.LISTENING
        self._kind = kind
        self._mode = mode
        self._started = now
        self._preview = ""
        self._inline = None
        self._transient = None
        self._shake_started = None
        self.preview_offset.snap(0.0)
        if self.wisp_fade.value < 0.05:
            self.wisp.reset()
        self.wisp.set_calm(False)
        self._apply()

    def set_mode(self, mode: RecordingMode) -> None:
        if mode != self._mode:
            self._mode = mode
            if self._base is Display.LISTENING:
                self._apply()

    def set_level(self, level: float) -> None:
        self.wisp.set_level(level)

    def set_preview(self, text: str) -> None:
        text = " ".join(text.split())
        if len(text) > PREVIEW_TAIL_CHARS:
            tail = text[-PREVIEW_TAIL_CHARS:]
            cut = tail.find(" ")
            text = tail[cut + 1 :] if 0 <= cut < 40 else tail
        if text == self._preview:
            return
        if self._preview and text and not self.reduced_motion:
            # Keep the words already on screen where they are, then glide the line left to make room.
            fm = metrics(style.FONT_PREVIEW)
            area = max(0.0, self.width.target - 2 * LINE_PADDING)
            old_x = self._preview_x(fm.horizontalAdvance(self._preview), area)
            new_x = self._preview_x(fm.horizontalAdvance(text), area)
            self.preview_offset.value = clamp(self.preview_offset.value + old_x - new_x, -400.0, 400.0)
            self.preview_offset.velocity = 0.0
        self._preview = text
        self._apply()

    @staticmethod
    def _preview_x(text_width: float, area: float) -> float:
        """Where the preview starts within its line: left-aligned while it fits, then the newest words stay in view."""
        return min(0.0, area - text_width)

    def set_note(self, text: str | None) -> None:
        text = text.strip() if text else None
        if text != self._note:
            self._note = text or None
            self._apply()

    def show_processing(self) -> None:
        self._base = Display.PROCESSING
        self._preview = ""
        self._transient = None
        self._shake_started = None
        self.wisp.set_calm(True, instantly=self.reduced_motion)
        self._apply()

    def show_outcome(self, outcome: Outcome, message: str | None = None) -> None:
        """Show how a dictation ended, then settle back to rest.

        While a recording is showing, INSERTED, COPIED and NO_SPEECH can only
        belong to an earlier session (a recording is delivered after it stops),
        so they never cover the live recording: COPIED appears on the note line
        and the others are skipped. CANCELLED and FAILED end the recording view.
        """
        now = self._clock()
        if self._base is Display.LISTENING and outcome in (Outcome.INSERTED, Outcome.COPIED, Outcome.NO_SPEECH):
            if outcome is Outcome.COPIED:
                self._show_inline(message or COPIED_TEXT, "info", now)
            return
        self._base = Display.HIDDEN
        self._note = None
        self._inline = None
        self._shake_started = None
        self._transient = None
        if outcome is Outcome.INSERTED:
            self._transient = _Transient(Display.DONE, None, now, now + style.DONE_HOLD_SECONDS)
            self._check_started = now
        elif outcome is Outcome.COPIED:
            self._show_message(Message(message or COPIED_TEXT, icon="copy"), style.COPIED_SECONDS, now)
        elif outcome is Outcome.NO_SPEECH:
            self._show_message(Message(message or NO_SPEECH_TEXT), style.NO_SPEECH_SECONDS, now)
            self._shake_started = None if self.reduced_motion else now
        elif outcome is Outcome.FAILED:
            msg = Message(message or FAILED_TEXT, icon="error", tint="error")
            self._show_message(msg, style.ERROR_MIN_SECONDS, now)
        # CANCELLED: no transient, so the pill retracts straight to rest.
        self._apply()

    def _show_message(self, message: Message, minimum_seconds: float, now: float) -> None:
        expires = now + reading_seconds(message.text, minimum_seconds)
        self._transient = _Transient(Display.MESSAGE, message, now, expires)

    def show_notice(self, text: str, kind: str = "info") -> None:
        now = self._clock()
        kind = kind if kind in style.KIND_COLORS else "info"
        text = text.strip()
        if not text:
            return
        if self._base in (Display.LISTENING, Display.PROCESSING):
            self._show_inline(text, kind, now)
            return
        minimum = style.ERROR_MIN_SECONDS if kind == "error" else style.NOTICE_MIN_SECONDS
        self._show_message(Message(text, icon=kind, tint=kind if kind != "info" else None), minimum, now)
        self._shake_started = None
        self._apply()

    def _show_inline(self, text: str, tone: str, now: float) -> None:
        self._inline = Line("message", text, tone)
        self._inline_expires = now + reading_seconds(text, style.NOTICE_MIN_SECONDS)
        self._apply()

    # -- derived state -----------------------------------------------------------

    def now(self) -> float:
        """The scene's clock (monotonic seconds)."""
        return self._clock()

    @property
    def display(self) -> Display:
        return self._display

    @property
    def geometry_settled(self) -> bool:
        """True once the pill has reached its size and is not shaking."""
        return self.width.settled and self.height.settled and self._shake_started is None

    @property
    def hands_free(self) -> bool:
        return self._display is Display.LISTENING and self._mode is RecordingMode.HANDS_FREE

    @property
    def interactive(self) -> bool:
        """True while the pill has something to click (idle pill, hands-free buttons)."""
        return self._display is Display.IDLE or self.hands_free

    @property
    def visible(self) -> bool:
        return self.alpha.value > 0.001 or self.alpha.target > 0.0

    @property
    def settled(self) -> bool:
        springs = (self.width, self.height, self.alpha, self.wisp_left, self.wisp_right, self.chip_left, self.tint)
        fades = (self.wisp_fade, self.controls_fade, self.chip_fade, self.hint_fade, self.check_fade)
        return (
            all(s.settled for s in springs)
            and all(f.settled for f in fades)
            and self.line.settled
            and self.message.settled
            and self.preview_offset.settled
        )

    @property
    def animating(self) -> bool:
        """True while another frame would look different from this one."""
        if not self.visible:
            return not self.settled
        if self.wisp_fade.value > 0.0 or self.wisp_fade.visible:
            return True  # the wisp and the shimmer are always moving
        if self._shake_started is not None or self._check_drawing():
            return True
        return not self.settled or not self.glow.settled

    def next_deadline(self) -> float | None:
        """When something will change on its own (a message expiring, a hover delay), if anything."""
        deadlines: list[float] = []
        if self._transient is not None:
            deadlines.append(self._transient.expires)
        if self._inline is not None:
            deadlines.append(self._inline_expires)
        if self._display is Display.IDLE and self._pointer_inside != self._idle_expanded:
            delay = style.IDLE_HOVER_DELAY if self._pointer_inside else style.IDLE_LEAVE_DELAY
            deadlines.append(self._pointer_changed + delay)
        return min(deadlines) if deadlines else None

    def _check_drawing(self) -> bool:
        if self._check_started is None or self._display is not Display.DONE or self.reduced_motion:
            return False
        return self._clock() - self._check_started < style.DONE_CHECK_DELAY + style.DONE_CHECK_SECONDS

    def _resolve_display(self) -> Display:
        if self._transient is not None:
            return self._transient.display
        if self._base in (Display.LISTENING, Display.PROCESSING):
            return self._base
        return Display.IDLE if self._rest_idle else Display.HIDDEN

    def _current_line(self) -> Line | None:
        if self._display is Display.LISTENING:
            if self._inline is not None:
                return self._inline
            if self._note:
                return Line("note", self._note)
            if self._preview:
                return Line("preview")
        elif self._display is Display.PROCESSING:
            if self._inline is not None:
                return self._inline
            if self._note:
                return Line("note", self._note)
        return None

    def rich(
        self, text: str, pixel_size: int, weight: QFont.Weight = QFont.Weight.Medium, max_lines: int = 3
    ) -> RichText:
        """Laid-out text, cached: layout happens once per message, not once per frame."""
        key = (text, pixel_size * 1000 + int(weight.value) * 10 + max_lines)
        cached = self._rich_cache.get(key)
        if cached is None:
            if len(self._rich_cache) > 48:
                self._rich_cache.clear()
            max_width = style.MESSAGE_MAX_WIDTH - 2 * style.MESSAGE_PADDING - style.ICON_SIZE - style.ICON_GAP
            cached = RichText(parse_runs(text), pixel_size, weight, max_width=max_width, max_lines=max_lines)
            self._rich_cache[key] = cached
        return cached

    def _hint(self) -> RichText:
        key = ("\x00hint\x00" + "+".join(self._hint_keys), style.FONT_HINT)
        cached = self._rich_cache.get(key)
        if cached is None:
            cached = RichText(hint_runs(self._hint_keys), style.FONT_HINT, QFont.Weight.Medium)
            self._rich_cache[key] = cached
        return cached

    def chip_width(self) -> float:
        fm = metrics(style.FONT_CHIP, QFont.Weight.Medium)
        return 9.0 + 9.0 + 5.0 + fm.horizontalAdvance("Edit") + 10.0

    def timer_width(self) -> float:
        return metrics(style.FONT_TIMER, QFont.Weight.Medium, tabular=True).horizontalAdvance("00:00")

    def message_size(self, message: Message) -> tuple[float, float]:
        rich = self.rich(message.text, style.FONT_MESSAGE)
        icon = style.ICON_SIZE + style.ICON_GAP if message.icon else 0.0
        width = 2 * style.MESSAGE_PADDING + icon + rich.width
        return math.ceil(width), max(style.MESSAGE_HEIGHT, math.ceil(rich.height + 18.0))

    def _line_width(self, line: Line | None) -> float:
        if line is None or line.kind == "preview":
            return 0.0
        return self.rich(line.text, style.FONT_NOTE, max_lines=1).width + 2 * LINE_PADDING

    # -- targets -----------------------------------------------------------------

    def _retarget(self, spring: Spring, value: float) -> None:
        spring.target = value
        if self.reduced_motion:
            spring.snap()

    def _apply(self) -> None:
        """Point every spring and fade at the current state's targets."""
        display = self._resolve_display()
        self._display = display
        if display is not Display.IDLE:
            # Hover belongs to the idle pill; coming back to rest later starts collapsed until the pointer returns.
            self._idle_expanded = False
            self._pointer_inside = False
        line = self._current_line()
        self.line.set(line, instantly=False)
        listening = display is Display.LISTENING
        processing = display is Display.PROCESSING
        command = listening and self._kind is SessionKind.COMMAND
        hands_free = listening and self._mode is RecordingMode.HANDS_FREE
        chip = self.chip_width()

        self.wisp_fade.show(listening or processing)
        self.controls_fade.show(hands_free)
        self.chip_fade.show(command)
        self.hint_fade.show(display is Display.IDLE and self._idle_expanded)
        self.check_fade.show(display is Display.DONE, instantly=False)
        self.message.set(self._transient.message if display is Display.MESSAGE and self._transient else None)
        if processing:
            self.wisp.set_calm(True, instantly=self.reduced_motion)
        elif listening:
            self.wisp.set_calm(False)

        line_height = style.PREVIEW_HEIGHT if line is not None else 0.0
        alpha = 1.0
        if listening:
            button_room = style.BUTTON_INSET + style.BUTTON_SIZE
            width = style.HANDS_FREE_WIDTH if hands_free else style.HOLD_WIDTH
            left = button_room + 10.0 if hands_free else style.WISP_INSET
            right = button_room + style.TIMER_GAP + self.timer_width() + 10.0 if hands_free else style.WISP_INSET
            chip_left = button_room + 6.0 if hands_free else 12.0
            if command:
                width += chip + style.CHIP_GAP - 6.0
                left = chip_left + chip + style.CHIP_GAP
            self._retarget(self.chip_left, chip_left)
            self._retarget(self.wisp_left, left)
            self._retarget(self.wisp_right, right)
            width = max(width, self._line_width(line))
            height = style.ROW_HEIGHT + line_height
        elif processing:
            width = max(style.PROCESSING_WIDTH, self._line_width(line))
            height = style.ROW_HEIGHT + line_height
            self._retarget(self.wisp_left, style.WISP_INSET)
            self._retarget(self.wisp_right, style.WISP_INSET)
        elif display is Display.DONE:
            width = height = style.DONE_SIZE
        elif display is Display.MESSAGE and self._transient is not None and self._transient.message is not None:
            width, height = self.message_size(self._transient.message)
        elif display is Display.IDLE:
            if self._idle_expanded:
                width, height = math.ceil(self._hint().width + 2 * 13.0), style.IDLE_HOVER_HEIGHT
            else:
                width, height, alpha = style.IDLE_WIDTH, style.IDLE_HEIGHT, style.IDLE_OPACITY
        else:  # HIDDEN: shrink toward a small hairline while fading out.
            width, height, alpha = style.IDLE_WIDTH * 0.6, style.IDLE_HEIGHT, 0.0

        if self.alpha.value <= 0.001 and alpha > 0.0:
            # Appearing from nothing: grow out of a small hairline rather than from wherever we last were.
            self.width.snap(style.IDLE_WIDTH * 0.6)
            self.height.snap(style.IDLE_HEIGHT)
        self._retarget(self.width, width)
        self._retarget(self.height, height)
        self.alpha.target = alpha

        message = self._transient.message if display is Display.MESSAGE and self._transient else None
        tint = message.tint if message is not None else None
        if tint is not None:
            self._tint_kind = tint
        self.tint.target = 1.0 if tint is not None else 0.0
        if display is not Display.DONE:
            self._check_started = None
        if not self.interactive:
            self.hover = None
            self.pressed = None

    # -- frame -------------------------------------------------------------------

    def advance(self) -> bool:
        """Move every animation forward to the clock's current time. Returns ``animating``."""
        now = self._clock()
        # After a rest (no frames were running), start the new motion one frame in rather than jumping ahead by
        # however long we rested; while running, follow real time so a slow frame never slows the animation down.
        max_step = 0.05 if self._was_animating else 1.0 / 60.0
        dt = min(max(0.0, now - self._last), max_step)
        self._last = now
        self._process_deadlines(now)
        for spring in (
            self.width,
            self.height,
            self.alpha,
            self.wisp_left,
            self.wisp_right,
            self.chip_left,
            self.tint,
            self.preview_offset,
        ):
            spring.step(dt)
        for fade in (self.wisp_fade, self.controls_fade, self.chip_fade, self.hint_fade, self.check_fade):
            fade.step(dt)
        self.line.step(dt)
        self.message.step(dt)
        if self.wisp_fade.value > 0.0 or self.wisp_fade.visible:
            self.wisp.reduced_motion = self.reduced_motion
            self.wisp.step(dt)
        live = self.wisp_fade.value * (1.0 - clamp(self.wisp.calm.value))
        calm = self.wisp_fade.value * clamp(self.wisp.calm.value)
        # The outer glow breathes with the wisp: faint at rest, warmer as the voice swells.
        self.glow.target = live * (0.14 + 0.06 * self.wisp.breath + 0.42 * self.wisp.energy) + calm * 0.16
        self.glow.step(dt)
        if self._shake_started is not None and now - self._shake_started > style.SHAKE_SECONDS:
            self._shake_started = None
        self._was_animating = self.animating
        return self._was_animating

    def _process_deadlines(self, now: float) -> None:
        changed = False
        if self._transient is not None and now >= self._transient.expires:
            self._transient = None
            self._check_started = None
            changed = True
        if self._inline is not None and now >= self._inline_expires:
            self._inline = None
            changed = True
        if self._display is Display.IDLE and self._pointer_inside != self._idle_expanded:
            delay = style.IDLE_HOVER_DELAY if self._pointer_inside else style.IDLE_LEAVE_DELAY
            if now - self._pointer_changed >= delay:
                self._idle_expanded = self._pointer_inside
                changed = True
        if changed:
            self._apply()

    # -- geometry ----------------------------------------------------------------

    def _shake_offset(self) -> float:
        if self._shake_started is None:
            return 0.0
        t = clamp((self._clock() - self._shake_started) / style.SHAKE_SECONDS)
        return style.SHAKE_AMPLITUDE * math.sin(2.0 * math.tau * t) * (1.0 - t) ** 1.5

    def _rect(self, anchor: QPointF, width: float, height: float, shake: float = 0.0) -> QRectF:
        width, height = max(0.0, width), max(0.0, height)
        top = anchor.y() - height if self.placement == "bottom" else anchor.y()
        return QRectF(anchor.x() - width / 2.0 + shake, top, width, height)

    def pill_rect(self, anchor: QPointF) -> QRectF:
        return self._rect(anchor, self.width.value, self.height.value, self._shake_offset())

    def target_rect(self, anchor: QPointF) -> QRectF:
        return self._rect(anchor, self.width.target, self.height.target)

    def required_rect(self, anchor: QPointF) -> QRectF:
        """Everything that may be painted this frame or at the end of the current morph, with glow room."""
        m = float(style.GLOW_MARGIN)
        return self.pill_rect(anchor).united(self.target_rect(anchor)).adjusted(-m, -m, m, m)

    def settled_rect(self, anchor: QPointF) -> QRectF:
        m = float(style.GLOW_MARGIN)
        return self.target_rect(anchor).adjusted(-m, -m, m, m)

    @staticmethod
    def radius(height: float) -> float:
        return min(height / 2.0, style.ROW_HEIGHT / 2.0)

    def _row_center(self, pill: QRectF) -> float:
        return pill.top() + min(pill.height(), style.ROW_HEIGHT) / 2.0

    def _buttons(self, pill: QRectF) -> dict[str, QPointF]:
        cy = self._row_center(pill)
        half = style.BUTTON_SIZE / 2.0
        return {
            "cancel": QPointF(pill.left() + style.BUTTON_INSET + half, cy),
            "stop": QPointF(pill.right() - style.BUTTON_INSET - half, cy),
        }

    # -- pointer -------------------------------------------------------------------

    def hit_test(self, pos: QPointF, anchor: QPointF) -> str | None:
        pill = self.pill_rect(anchor)
        if self._display is Display.IDLE:
            if self._idle_expanded:
                area = pill.adjusted(-6.0, -6.0, 6.0, 6.0)
            else:
                c = pill.center()
                w, h = style.IDLE_HIT_WIDTH, style.IDLE_HIT_HEIGHT
                area = QRectF(c.x() - w / 2.0, c.y() - h / 2.0, w, h)
            return "idle" if area.contains(pos) else None
        if self.hands_free and self.controls_fade.value > 0.5:
            reach = style.BUTTON_SIZE / 2.0 + 4.0
            for name, centre in self._buttons(pill).items():
                if math.hypot(pos.x() - centre.x(), pos.y() - centre.y()) <= reach:
                    return name
        return None

    def pointer_moved(self, pos: QPointF | None, anchor: QPointF) -> None:
        """Track hover. ``pos`` is None when the pointer left the window."""
        target = self.hit_test(pos, anchor) if pos is not None else None
        inside = target == "idle"
        if inside != self._pointer_inside:
            self._pointer_inside = inside
            self._pointer_changed = self._clock()
        self.hover = target

    def press(self, pos: QPointF, anchor: QPointF) -> None:
        self.pressed = self.hit_test(pos, anchor)

    def release(self, pos: QPointF, anchor: QPointF) -> str | None:
        """Finish a click. Returns the element clicked ("idle", "cancel", "stop") or None."""
        pressed, self.pressed = self.pressed, None
        hit = self.hit_test(pos, anchor)
        return hit if hit is not None and hit == pressed else None

    # -- painting ------------------------------------------------------------------

    def paint(self, painter: QPainter, anchor: QPointF) -> None:
        """Paint the pill around ``anchor`` (in the painter's coordinates)."""
        alpha = clamp(self.alpha.value)
        pill = self.pill_rect(anchor)
        if alpha <= 0.001 or pill.width() < 0.5 or pill.height() < 0.5:
            return
        radius = self.radius(pill.height())
        path = surface.pill_path(pill, radius)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._display is Display.IDLE and not self._idle_expanded:
            # Invisible (alpha 1/255) but solid to Windows' hit testing: a forgiving hover target.
            c = pill.center()
            w, h = style.IDLE_HIT_WIDTH, style.IDLE_HIT_HEIGHT
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 1))
            painter.drawRoundedRect(QRectF(c.x() - w / 2.0, c.y() - h / 2.0, w, h), h / 2.0, h / 2.0)

        # Shadow and glow are cached per pill size and live outside the glass only.
        self._shadow_layer.paint(painter, pill, radius, alpha)
        self._glow_layer.paint(painter, pill, radius, self.glow.value * alpha)

        tint = style.KIND_COLORS.get(self._tint_kind) if self._tint_kind else None
        # sqrt keeps the edge brighter than the fill while faint (idle: 35% fill, ~60% edge) and equal at full.
        surface.paint_glass(painter, pill, radius, alpha, tint, self.tint.value, border_opacity=math.sqrt(alpha))

        painter.setClipPath(path, Qt.ClipOperation.IntersectClip)
        self._paint_content(painter, pill, alpha)
        painter.restore()

    def _paint_content(self, painter: QPainter, pill: QRectF, alpha: float) -> None:
        cy = self._row_center(pill)
        wisp = self.wisp_fade.value * alpha
        if wisp > 0.0:
            room = min(pill.height(), style.ROW_HEIGHT) / 2.0
            x0 = pill.left() + self.wisp_left.value
            x1 = pill.right() - self.wisp_right.value
            self.wisp.paint(painter, x0, x1, cy, max(0.0, min(15.0, room - 6.0)), wisp)
        controls = self.controls_fade.value * alpha
        if controls > 0.0:
            self._paint_controls(painter, pill, cy, controls)
        chip = self.chip_fade.value * alpha
        if chip > 0.0:
            self._paint_chip(painter, pill.left() + self.chip_left.value, cy, chip)
        for line, opacity in self.line:
            self._paint_line(painter, pill, line, opacity * alpha)
        check = self.check_fade.value * alpha
        if check > 0.0:
            self._paint_check(painter, pill, check)
        for message, opacity in self.message:
            self._paint_message(painter, pill, message, opacity * alpha)
        hint = self.hint_fade.value * alpha
        if hint > 0.0:
            rich = self._hint()
            rich.paint(painter, pill.center().x() - rich.width / 2.0, pill.center().y(), style.TEXT_2, hint)

    def _paint_controls(self, painter: QPainter, pill: QRectF, cy: float, opacity: float) -> None:
        dpr = painter.device().devicePixelRatioF()
        buttons = self._buttons(pill)
        r = style.BUTTON_SIZE / 2.0
        painter.setPen(Qt.PenStyle.NoPen)

        cancel = buttons["cancel"]
        if self.pressed == "cancel":
            fill = style.BUTTON_PRESSED
        elif self.hover == "cancel":
            fill = style.BUTTON_HOVER
        else:
            fill = style.BUTTON_REST
        painter.setBrush(style.tinted(fill, opacity))
        painter.drawEllipse(cancel, r, r)
        icon = style.TEXT_1 if self.hover == "cancel" else style.TEXT_2
        glyphs.draw_cross(painter, cancel, 8.0, style.tinted(icon, opacity), 1.6)

        stop = buttons["stop"]
        if self.pressed == "stop":
            fill = style.STOP_FILL_PRESSED
        elif self.hover == "stop":
            fill = style.STOP_FILL_HOVER
        else:
            fill = style.STOP_FILL
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(style.tinted(fill, opacity))
        painter.drawEllipse(stop, r, r)
        glyphs.draw_stop(painter, stop, 9.0, style.tinted(style.TEXT_1, opacity))

        elapsed = format_elapsed(self._clock() - self._started)
        fm = metrics(style.FONT_TIMER, QFont.Weight.Medium, tabular=True)
        right = stop.x() - r - style.TIMER_GAP
        painter.setFont(font(style.FONT_TIMER, QFont.Weight.Medium, tabular=True))
        painter.setPen(style.tinted(style.TEXT_2, opacity))
        x = snap(right - fm.horizontalAdvance(elapsed), dpr)
        painter.drawText(QPointF(x, snap(cy + fm.capHeight() / 2.0, dpr)), elapsed)

    def _paint_chip(self, painter: QPainter, left: float, cy: float, opacity: float) -> None:
        """The orchid "✦ Edit" chip that marks command mode."""
        dpr = painter.device().devicePixelRatioF()
        h = style.CHIP_HEIGHT
        rect = QRectF(snap(left, dpr), snap(cy - h / 2.0, dpr), round(self.chip_width()), h)
        painter.setPen(QPen(style.tinted(style.ORCHID_CHIP_BORDER, opacity), 1.0))
        painter.setBrush(style.tinted(style.ORCHID_CHIP_FILL, opacity))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), h / 2.0 - 0.5, h / 2.0 - 0.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(style.tinted(style.ORCHID_TEXT, opacity))
        painter.drawPath(glyphs.sparkle_path(QPointF(rect.left() + 9.0 + 4.5, rect.center().y()), 4.8))
        fm = metrics(style.FONT_CHIP, QFont.Weight.Medium)
        painter.setFont(font(style.FONT_CHIP, QFont.Weight.Medium))
        painter.setPen(style.tinted(style.ORCHID_TEXT, opacity))
        baseline = snap(rect.center().y() + fm.capHeight() / 2.0, dpr)
        painter.drawText(QPointF(snap(rect.left() + 23.0, dpr), baseline), "Edit")

    def _line_center(self, pill: QRectF) -> float:
        return pill.top() + style.ROW_HEIGHT + style.PREVIEW_HEIGHT / 2.0 - 6.0

    def _paint_line(self, painter: QPainter, pill: QRectF, line: Line, opacity: float) -> None:
        if opacity <= 0.0:
            return
        cy = self._line_center(pill)
        if line.kind == "preview":
            self._paint_preview(painter, pill, cy, opacity)
            return
        if line.kind == "note":
            color = style.TEXT_3
        else:
            color = {"error": style.DANGER, "warning": style.WARNING}.get(line.tone, style.TEXT_2)
        rich = self.rich(line.text, style.FONT_NOTE, max_lines=1)
        rich.paint(painter, pill.center().x() - rich.width / 2.0, cy, color, opacity)

    def _paint_preview(self, painter: QPainter, pill: QRectF, cy: float, opacity: float) -> None:
        if not self._preview:
            return
        left, right = pill.left() + LINE_PADDING, pill.right() - LINE_PADDING
        area = right - left
        if area < 24.0:
            return
        dpr = painter.device().devicePixelRatioF()
        fm = metrics(style.FONT_PREVIEW)
        width = fm.horizontalAdvance(self._preview)
        offset = self.preview_offset.value
        x = left + self._preview_x(width, area) + offset
        # Fade the left edge only once words actually run off it; feather the right edge while words glide in.
        fade_left = clamp(width - area + offset + 4.0, 0.0, PREVIEW_FADE)
        fade_right = clamp(offset, 0.0, 10.0)
        painter.save()
        painter.setClipRect(QRectF(left, cy - 14.0, area, 28.0), Qt.ClipOperation.IntersectClip)
        painter.setFont(font(style.FONT_PREVIEW))
        painter.setPen(faded_text_pen(left, right, style.tinted(style.TEXT_2, opacity), fade_left, fade_right))
        painter.drawText(QPointF(x, snap(cy + fm.capHeight() / 2.0, dpr)), self._preview)
        painter.restore()

    def _paint_check(self, painter: QPainter, pill: QRectF, opacity: float) -> None:
        progress = 1.0
        if self._check_started is not None and not self.reduced_motion:
            elapsed = self._clock() - self._check_started - style.DONE_CHECK_DELAY
            progress = ease_out_cubic(elapsed / style.DONE_CHECK_SECONDS)
        centre = pill.center()
        size = min(pill.width(), pill.height())
        # A soft success-tinted light behind the check, so the circle reads as "lit", not just outlined.
        halo = QRadialGradient(centre, size * 0.5)
        halo.setColorAt(0.0, style.tinted(style.SUCCESS, 0.2 * opacity * max(progress, 0.0)))
        halo.setColorAt(1.0, style.tinted(style.SUCCESS, 0.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(halo))
        painter.drawEllipse(centre, size * 0.5, size * 0.5)
        scale = 0.92 * min(1.0, size / style.DONE_SIZE)
        glyphs.draw_check(painter, centre, scale, progress, style.tinted(style.SUCCESS, opacity), 2.4)

    def _paint_message(self, painter: QPainter, pill: QRectF, message: Message, opacity: float) -> None:
        rich = self.rich(message.text, style.FONT_MESSAGE)
        icon_room = style.ICON_SIZE + style.ICON_GAP if message.icon else 0.0
        left = pill.center().x() - (icon_room + rich.width) / 2.0
        cy = pill.center().y()
        if message.icon:
            painter.save()
            painter.setOpacity(painter.opacity() * opacity)
            icon_cy = cy - rich.height / 2.0 + rich.LINE_HEIGHT / 2.0
            if message.icon == "copy":
                s = style.ICON_SIZE
                glyphs.draw_copy(painter, QRectF(left, icon_cy - s / 2.0, s, s), style.TEXT_2)
            else:
                glyphs.draw_badge(painter, QPointF(left + style.ICON_SIZE / 2.0, icon_cy), message.icon)
            painter.restore()
        # Wrapped lines beside an icon read best flush left; a single line is the same either way.
        rich.paint(painter, left + icon_room, cy, style.TEXT_1, opacity, centred=not message.icon)
