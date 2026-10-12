"""Generate Whisp's logo and tray-icon SVGs.

The mark is two interlaced "wisps": sine ribbons whose thickness tapers to
nothing at both ends, filled with the opal gradient from docs/DESIGN.md.

    python scripts/make_brand_assets.py            # writes src/whisp/assets/brand/*.svg
    python scripts/make_brand_assets.py --preview sheet.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

OPAL = [(0.0, "#67E8F9"), (0.35, "#818CF8"), (0.65, "#C084FC"), (1.0, "#F9A8D4")]
BRAND_DIR = Path(__file__).resolve().parent.parent / "src" / "whisp" / "assets" / "brand"


def ribbon(x0: float, x1: float, cy: float, amp: float, thick: float, phase: float = 0.0, n: int = 48) -> str:
    """Closed SVG path of a one-period sine ribbon that tapers to zero thickness at both ends."""

    def centre(u: float) -> tuple[float, float]:
        envelope = math.sin(math.pi * u) ** 0.9
        return x0 + (x1 - x0) * u, cy - amp * envelope * math.sin(2 * math.pi * u + phase)

    top, bottom = [], []
    for i in range(n + 1):
        u = i / n
        x, y = centre(u)
        ax, ay = centre(max(0.0, u - 1e-3))
        bx, by = centre(min(1.0, u + 1e-3))
        length = math.hypot(bx - ax, by - ay) or 1.0
        nx, ny = -(by - ay) / length, (bx - ax) / length
        half = thick * math.sin(math.pi * u) ** 1.2 / 2
        top.append((x + nx * half, y + ny * half))
        bottom.append((x - nx * half, y - ny * half))
    return _smooth_closed_path(top + bottom[::-1])


def _smooth_closed_path(points: list[tuple[float, float]]) -> str:
    """Catmull-Rom spline through the points, as cubic Bezier segments."""
    n = len(points)
    parts = [f"M{points[0][0]:.2f} {points[0][1]:.2f}"]
    for i in range(n):
        p0, p1, p2, p3 = points[i - 1], points[i], points[(i + 1) % n], points[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        parts.append(f"C{c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p2[0]:.2f} {p2[1]:.2f}")
    parts.append("Z")
    return "".join(parts)


def _gradient(gid: str, x0: float, x1: float) -> str:
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in OPAL)
    return (
        f'<linearGradient id="{gid}" x1="{x0}" y1="0" x2="{x1}" y2="0" gradientUnits="userSpaceOnUse">'
        f"{stops}</linearGradient>"
    )


def logo(tile: bool) -> str:
    back = ribbon(13, 51, 32, 9.5, 2.6, phase=1.6)
    front = ribbon(9, 55, 32, 10.5, 8.0)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">', f"<defs>{_gradient('o', 9, 55)}</defs>"]
    if tile:
        parts.append('<rect x="2" y="2" width="60" height="60" rx="16" fill="#0E0E13"/>')
        parts.append(
            '<rect x="2.5" y="2.5" width="59" height="59" rx="15.5" fill="none" '
            'stroke="#FFFFFF" stroke-opacity="0.08"/>'
        )
    parts.append(f'<path d="{back}" fill="url(#o)" opacity="0.7"/>')
    parts.append(f'<path d="{front}" fill="url(#o)"/>')
    parts.append("</svg>")
    return "".join(parts)


def tray_mark(fill: str | None) -> str:
    """Single bold wisp sized for 16-32 px. ``fill=None`` uses the opal gradient."""
    path = ribbon(6, 58, 32, 11.0, 10.0)
    if fill is None:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs>{_gradient("o", 6, 58)}</defs>'
            f'<path d="{path}" fill="url(#o)"/></svg>'
        )
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="{path}" fill="{fill}"/></svg>'


ASSETS = {
    "logo.svg": lambda: logo(tile=True),
    "logo-glyph.svg": lambda: logo(tile=False),
    "tray-idle-light.svg": lambda: tray_mark("#FFFFFF"),  # for dark taskbars
    "tray-idle-dark.svg": lambda: tray_mark("#1A1A20"),  # for light taskbars
    "tray-active.svg": lambda: tray_mark(None),
}


def write_assets(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in ASSETS.items():
        path = out_dir / name
        path.write_text(build() + "\n", encoding="utf-8")
        written.append(path)
    return written


def render_preview(out_dir: Path, target: Path) -> None:
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 - keeps Qt alive
    sheet = QImage(900, 300, QImage.Format.Format_ARGB32)
    sheet.fill(QColor("#2A2A30"))
    painter = QPainter(sheet)
    x = 20
    for size in (256, 128, 64, 32, 16):
        QSvgRenderer(str(out_dir / "logo.svg")).render(painter, QRectF(x, 20, size, size))
        x += size + 20
    x = 20
    tray_previews = (
        ("tray-idle-light.svg", "#202024"),
        ("tray-active.svg", "#202024"),
        ("tray-idle-dark.svg", "#E8E8EC"),
    )
    for name, bg in tray_previews:
        for size in (32, 24, 16):
            painter.fillRect(QRectF(x - 4, 236, size + 8, size + 8), QColor(bg))
            QSvgRenderer(str(out_dir / name)).render(painter, QRectF(x, 240, size, size))
            x += size + 16
        x += 20
    painter.end()
    sheet.save(str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=BRAND_DIR)
    parser.add_argument("--preview", type=Path, help="also render a PNG contact sheet here")
    args = parser.parse_args()
    for path in write_assets(args.out):
        print(f"wrote {path}")
    if args.preview:
        render_preview(args.out, args.preview)
        print(f"preview {args.preview}")


if __name__ == "__main__":
    main()
