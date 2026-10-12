"""Render every overlay state to PNG files for visual review.

    python scripts/render_overlay.py OUT_DIR [--scale 2]

Each state is painted by the real ``OverlayScene`` (the same code the overlay
window paints with), driven by a simulated clock at 60 fps so mid-animation
frames are reproducible. The pill is drawn into a transparent layer first and
then composited over a dark and a light desktop-like background, exactly as
Windows composites the translucent overlay window, so glass, shadow and glow
can be judged on both. Nothing is shown on screen.

Writes one PNG per state (dark on the left, light on the right) plus
``contact-sheet.png`` with every state.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter, QRadialGradient  # noqa: E402

from whisp.core.types import Outcome, RecordingMode, SessionKind  # noqa: E402
from whisp.ui.overlay.scene import OverlayScene  # noqa: E402
from whisp.ui.theme import ui_font  # noqa: E402

PANEL_W, PANEL_H = 560, 190
TASKBAR_H = 48
FPS = 60


class SimClock:
    """A clock the script moves by hand, so every frame is reproducible."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def speech(t: float) -> float:
    """A speech-like microphone level: syllable bursts under a slower phrase contour."""
    syllables = abs(math.sin(math.tau * 2.7 * t)) ** 1.5
    phrase = 0.55 + 0.45 * math.sin(math.tau * 0.45 * t + 0.6)
    flutter = 0.08 * math.sin(math.tau * 11.0 * t)
    return max(0.0, min(1.0, 0.08 + 0.75 * syllables * phrase + flutter))


class Rig:
    """A scene plus its clock, with helpers to drive it like the app does."""

    def __init__(self, *, rest_idle: bool = True, reduced_motion: bool = False, placement: str = "bottom") -> None:
        self.clock = SimClock()
        self.scene = OverlayScene(clock=self.clock)
        self.scene.reduced_motion = reduced_motion
        self.scene.placement = placement
        self.scene.set_hint_keys(["Ctrl", "Win"])
        self.scene.set_rest(rest_idle)
        self.level: Callable[[float], float] | None = None
        self.run(1.0)

    def run(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds * FPS))):
            self.clock.now += 1.0 / FPS
            if self.level is not None:
                self.scene.set_level(self.level(self.clock.now))
            self.scene.advance()

    def anchor(self) -> QPointF:
        if self.scene.placement == "top":
            return QPointF(PANEL_W / 2.0, 24.0)
        return QPointF(PANEL_W / 2.0, PANEL_H - TASKBAR_H - 24.0)


def desktop(dark: bool, scale: float) -> QImage:
    image = QImage(round(PANEL_W * scale), round(PANEL_H * scale), QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(scale)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    base = QLinearGradient(0, 0, PANEL_W, PANEL_H)
    if dark:
        base.setColorAt(0.0, QColor("#141a33"))
        base.setColorAt(0.55, QColor("#231a3a"))
        base.setColorAt(1.0, QColor("#0d1424"))
        blobs = (
            (0.18, 0.2, 0.55, "#3b4aa8", 0.55),
            (0.82, 0.35, 0.5, "#6b2f7a", 0.45),
            (0.5, 1.0, 0.6, "#1d6b85", 0.35),
        )
    else:
        base.setColorAt(0.0, QColor("#e9eef8"))
        base.setColorAt(0.55, QColor("#f4eef6"))
        base.setColorAt(1.0, QColor("#eef3f1"))
        blobs = ((0.15, 0.25, 0.55, "#c9d6ff", 0.7), (0.85, 0.3, 0.5, "#f3d3ea", 0.7), (0.5, 1.0, 0.6, "#cfeee8", 0.6))
    p.fillRect(QRectF(0, 0, PANEL_W, PANEL_H), base)
    for x, y, r, color, alpha in blobs:
        g = QRadialGradient(QPointF(x * PANEL_W, y * PANEL_H), r * PANEL_W)
        c = QColor(color)
        c.setAlphaF(alpha)
        g.setColorAt(0.0, c)
        c.setAlphaF(0.0)
        g.setColorAt(1.0, c)
        p.fillRect(QRectF(0, 0, PANEL_W, PANEL_H), g)
    # A window edge and some text, so translucency has something to show through.
    card = QRectF(28, 18, 250, 70)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(32, 32, 40, 235) if dark else QColor(255, 255, 255, 240))
    p.drawRoundedRect(card, 8, 8)
    p.setFont(ui_font(13, QFont.Weight.Medium))
    p.setPen(QColor("#d8d8e0") if dark else QColor("#2a2a33"))
    p.drawText(QPointF(card.left() + 14, card.top() + 28), "Thanks for the update — I'll take")
    p.drawText(QPointF(card.left() + 14, card.top() + 48), "a look this afternoon and reply.")
    taskbar = QRectF(0, PANEL_H - TASKBAR_H, PANEL_W, TASKBAR_H)
    p.fillRect(taskbar, QColor(28, 28, 30, 240) if dark else QColor(243, 243, 243, 245))
    p.fillRect(QRectF(0, taskbar.top(), PANEL_W, 1), QColor(255, 255, 255, 20) if dark else QColor(0, 0, 0, 22))
    p.end()
    return image


def frame(rig: Rig, scale: float) -> QImage:
    """The pill as Windows would show it: painted into a transparent layer, composited over each desktop."""
    layer = QImage(round(PANEL_W * scale), round(PANEL_H * scale), QImage.Format.Format_ARGB32_Premultiplied)
    layer.setDevicePixelRatio(scale)
    layer.fill(Qt.GlobalColor.transparent)
    p = QPainter(layer)
    rig.scene.paint(p, rig.anchor())
    p.end()
    out = QImage(round(PANEL_W * 2 * scale), round(PANEL_H * scale), QImage.Format.Format_ARGB32_Premultiplied)
    out.setDevicePixelRatio(scale)
    p = QPainter(out)
    for i, dark in enumerate((True, False)):
        p.drawImage(QPointF(i * PANEL_W, 0), desktop(dark, scale))
        p.drawImage(QPointF(i * PANEL_W, 0), layer)
    p.end()
    return out


def strip(frames: list[QImage]) -> QImage:
    """Stack frames vertically (for animations)."""
    width = max(f.width() for f in frames)
    height = sum(f.height() for f in frames)
    out = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    out.fill(QColor("#000000"))
    p = QPainter(out)
    y = 0
    for f in frames:
        # Target rect in device pixels: frames carry a device pixel ratio that would otherwise shrink them.
        p.drawImage(QRectF(0, y, f.width(), f.height()), f)
        y += f.height()
    p.end()
    return out


def recording(rig: Rig, kind: SessionKind = SessionKind.DICTATION, mode: RecordingMode = RecordingMode.HOLD) -> None:
    rig.scene.show_recording(kind, mode)


def build_states(scale: float) -> dict[str, QImage]:
    states: dict[str, QImage] = {}

    rig = Rig()
    states["01-idle"] = frame(rig, scale)
    rig.scene.pointer_moved(rig.scene.pill_rect(rig.anchor()).center(), rig.anchor())
    rig.run(0.8)
    states["02-idle-hover"] = frame(rig, scale)

    for name, level in (("03-listening-silent", 0.0), ("04-listening-soft", 0.3), ("05-listening-loud", 0.85)):
        rig = Rig()
        recording(rig)
        rig.level = lambda _t, v=level: v
        rig.run(1.4)
        states[name] = frame(rig, scale)

    rig = Rig()
    recording(rig)
    rig.level = speech
    frames = []
    for _ in range(4):
        rig.run(0.23)
        frames.append(frame(rig, scale))
    states["06-listening-speech"] = strip(frames)

    rig = Rig()
    recording(rig, mode=RecordingMode.HANDS_FREE)
    rig.level = speech
    rig.run(12.4)
    states["07-hands-free"] = frame(rig, scale)
    rig.scene.pointer_moved(rig.scene._buttons(rig.scene.pill_rect(rig.anchor()))["stop"], rig.anchor())
    rig.run(0.1)
    states["08-hands-free-stop-hover"] = frame(rig, scale)

    rig = Rig()
    recording(rig, kind=SessionKind.COMMAND)
    rig.level = speech
    rig.run(1.3)
    states["09-command-hold"] = frame(rig, scale)
    rig.scene.set_mode(RecordingMode.HANDS_FREE)
    rig.run(0.6)
    states["10-command-hands-free"] = frame(rig, scale)

    rig = Rig()
    recording(rig)
    rig.level = speech
    rig.run(0.8)
    words = "so I was thinking we could move the launch to Thursday and give the team a little more time"
    for i in range(3, len(words.split()) + 1, 3):
        rig.scene.set_preview(" ".join(words.split()[:i]))
        rig.run(0.35)
    rig.run(0.6)
    states["11-preview"] = frame(rig, scale)

    rig = Rig()
    recording(rig)
    rig.scene.set_note("Loading model…")
    rig.level = speech
    rig.run(1.2)
    states["12-note-loading"] = frame(rig, scale)

    rig = Rig()
    recording(rig)
    rig.level = speech
    rig.run(1.0)
    rig.level = None
    rig.scene.set_level(0.0)
    rig.scene.show_processing()
    frames = []
    for wait in (0.12, 0.5, 0.35, 0.35):
        rig.run(wait)
        frames.append(frame(rig, scale))
    states["13-processing"] = strip(frames)

    frames = []
    rig.scene.show_outcome(Outcome.INSERTED)
    for wait in (0.06, 0.1, 0.12, 0.2, 0.25, 0.2):
        rig.run(wait)
        frames.append(frame(rig, scale))
    states["14-inserted"] = strip(frames)

    rig = Rig()
    recording(rig)
    rig.run(0.5)
    rig.scene.show_processing()
    rig.run(0.4)
    rig.scene.show_outcome(Outcome.COPIED)
    rig.run(0.6)
    states["15-copied"] = frame(rig, scale)

    rig = Rig()
    recording(rig)
    rig.run(0.5)
    rig.scene.show_processing()
    rig.run(0.4)
    rig.scene.show_outcome(Outcome.NO_SPEECH)
    frames = []
    for wait in (0.1, 0.1, 0.5):
        rig.run(wait)
        frames.append(frame(rig, scale))
    states["16-no-speech"] = strip(frames)

    rig = Rig()
    rig.scene.show_outcome(Outcome.FAILED, "Microphone not found — check Settings → Microphone")
    rig.run(0.6)
    states["17-error"] = frame(rig, scale)

    rig = Rig()
    rig.scene.show_notice("AI polish is off, so commands can't run", "warning")
    rig.run(0.6)
    states["18-notice-warning"] = frame(rig, scale)

    rig = Rig()
    rig.scene.show_notice("Model ready. Hold Ctrl + Win to dictate")
    rig.run(0.6)
    states["19-notice-info"] = frame(rig, scale)

    rig = Rig()
    recording(rig)
    rig.level = speech
    rig.run(0.8)
    rig.scene.set_mode(RecordingMode.HANDS_FREE)
    frames = []
    for wait in (0.05, 0.08, 0.1, 0.4):
        rig.run(wait)
        frames.append(frame(rig, scale))
    states["20-morph-hold-to-hands-free"] = strip(frames)

    rig = Rig(reduced_motion=True)
    recording(rig, mode=RecordingMode.HANDS_FREE)
    rig.level = speech
    rig.run(1.0)
    states["21-reduced-motion"] = frame(rig, scale)

    rig = Rig(placement="top")
    recording(rig)
    rig.level = speech
    rig.run(0.6)
    rig.scene.set_preview("the pill can also sit at the top of the screen")
    rig.run(0.8)
    states["22-top-placement"] = frame(rig, scale)

    rig = Rig()
    rig.scene.show_outcome(
        Outcome.FAILED,
        "Couldn't reach your AI server, so the text was inserted without polish. Check Settings → AI polish.",
    )
    rig.run(0.6)
    states["23-long-error"] = frame(rig, scale)

    rig = Rig()
    recording(rig, mode=RecordingMode.HANDS_FREE)
    rig.level = speech
    rig.run(0.8)
    rig.scene.show_outcome(Outcome.COPIED)  # an earlier session finished while this one records
    rig.run(0.6)
    states["24-earlier-session-copied"] = frame(rig, scale)

    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="folder for the PNG files")
    parser.add_argument("--scale", type=float, default=2.0, help="device pixel ratio to render at (default 2)")
    args = parser.parse_args()
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841 - keeps Qt alive
    args.out.mkdir(parents=True, exist_ok=True)
    states = build_states(args.scale)
    for name, image in states.items():
        path = args.out / f"{name}.png"
        image.save(str(path))
        print(f"wrote {path}")
    sheet = strip([img for name, img in states.items() if img.height() <= round(PANEL_H * args.scale)])
    sheet.save(str(args.out / "contact-sheet.png"))
    print(f"wrote {args.out / 'contact-sheet.png'}")


if __name__ == "__main__":
    main()
