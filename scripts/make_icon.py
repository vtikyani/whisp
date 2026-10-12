"""Render Whisp's Windows icon and installer images from the logo SVG.

    python scripts/make_icon.py                        # writes the files below into packaging/
    python scripts/make_icon.py --preview sheet.png    # also writes a contact sheet to check every icon size

Writes:

``whisp.ico``
    The app icon: 16, 20, 24, 32, 40, 48, 64, 128 and 256 px.
``wizard-small.png``
    The logo in the top corner of the installer's pages.
``wizard.png``
    The tall panel on the installer's first and last pages.

Every image is painted by Qt's SVG renderer (QSvgRenderer) from
src/whisp/assets/brand/logo.svg (logo-glyph.svg for the tall panel). Two
small-size touches keep the icon crisp:

* At each size the logo's rounded tile is snapped to whole pixels, so its
  edges stay sharp instead of fading into a half-transparent fringe.
* At 32 px and below, the logo's translucent back ribbon is left out. At
  those sizes it is only a pixel wide and reads as a smudge under the main
  wisp.

The ICO file is written by hand (no Pillow): an ICONDIR header, one
ICONDIRENTRY per image, then the images. Images below 256 px are 32-bit BMP
(DIB) images with an alpha channel, which every part of Windows reads; the
256 px image is PNG-compressed, as Windows expects at that size. For a given
Qt version the output is the same, byte for byte, on every run.
"""

from __future__ import annotations

import argparse
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QRadialGradient
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "src" / "whisp" / "assets" / "brand"
LOGO = BRAND / "logo.svg"
GLYPH = BRAND / "logo-glyph.svg"
OUT_DIR = ROOT / "packaging"

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
"""Every size Windows picks from: small icons at 100-250% scaling, Explorer views, Start and the taskbar."""
PNG_FROM = 256
"""Images this big or bigger are stored PNG-compressed; smaller ones as BMP."""
FOREGROUND_ONLY_UP_TO = 32
"""At this size and below, translucent shapes (the back ribbon) are left out."""

WIZARD_SMALL_SIZE = 159
"""Inno Setup shows the small image in a square of 58 px at 100% scaling and 159 px at 250%; it shrinks larger ones."""
WIZARD_SIZE = (534, 1022)
"""Inno Setup's tall image area at 250% scaling (164:314, its fixed aspect ratio); it shrinks it for lower scaling."""
WIZARD_GLOW = QColor(0x81, 0x8C, 0xF8)
"""Periwinkle, the middle of the opal gradient (docs/DESIGN.md)."""

_SVG_URI = "http://www.w3.org/2000/svg"
_SVG_NS = "{" + _SVG_URI + "}"
_ICONDIR = struct.Struct("<HHH")  # reserved, type (1 = icon), image count
_ICONDIRENTRY = struct.Struct("<BBBBHHII")  # width, height, colours, reserved, planes, bits, bytes, offset
_BITMAPINFOHEADER = struct.Struct("<IiiHHIIiiII")


def tile_geometry(svg: Path) -> tuple[QRectF, QRectF, QColor]:
    """The logo's viewBox, its background tile (the first ``<rect>``) in SVG units, and the tile's colour."""
    root = ET.parse(svg).getroot()
    x, y, w, h = (float(v) for v in root.attrib["viewBox"].replace(",", " ").split())
    rect = root.find(f"{_SVG_NS}rect")
    if rect is None:
        raise SystemExit(f"{svg} has no <rect> tile to fit to the icon")
    tile = QRectF(*(float(rect.attrib[name]) for name in ("x", "y", "width", "height")))
    return QRectF(x, y, w, h), tile, QColor(rect.attrib["fill"])


def foreground_only(svg: Path) -> QByteArray:
    """The logo without its translucent shapes (any ``<path>`` with an ``opacity`` below 1)."""
    ET.register_namespace("", _SVG_URI)
    root = ET.parse(svg).getroot()
    for path in root.findall(f"{_SVG_NS}path"):
        if float(path.get("opacity", "1")) < 1:
            root.remove(path)
    return QByteArray(ET.tostring(root, encoding="utf-8"))


def tile_inset(size: int, view_box: QRectF, tile: QRectF) -> int:
    """Whole-pixel margin around the tile at ``size``: the logo's own margin, rounded half up."""
    return int(size * (tile.left() - view_box.left()) / view_box.width() + 0.5)


def _canvas(width: int, height: int) -> tuple[QImage, QPainter]:
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    return image, painter


def render(renderer: QSvgRenderer, size: int, view_box: QRectF, tile: QRectF) -> QImage:
    """The logo at ``size`` x ``size`` px, with the tile's edges on whole pixels. Straight (not premultiplied) ARGB."""
    inset = tile_inset(size, view_box, tile)
    scale = (size - 2 * inset) / tile.width()
    # Place the whole viewBox so that the tile lands exactly on [inset, size - inset].
    target = QRectF(
        inset - (tile.left() - view_box.left()) * scale,
        inset - (tile.top() - view_box.top()) * scale,
        view_box.width() * scale,
        view_box.height() * scale,
    )
    image, painter = _canvas(size, size)
    renderer.render(painter, target)
    painter.end()
    return image.convertToFormat(QImage.Format.Format_ARGB32)


def render_wizard_panel(glyph: QSvgRenderer, ink: QColor) -> QImage:
    """The installer's tall panel: the logo's ink with the wisp glyph, softly lit, in the upper middle."""
    width, height = WIZARD_SIZE
    image, painter = _canvas(width, height)
    painter.fillRect(0, 0, width, height, ink)
    centre = QPointF(width / 2, height * 0.42)
    glow = QRadialGradient(centre, width * 0.62)
    glow.setColorAt(0.0, QColor(WIZARD_GLOW.red(), WIZARD_GLOW.green(), WIZARD_GLOW.blue(), 46))
    glow.setColorAt(1.0, QColor(WIZARD_GLOW.red(), WIZARD_GLOW.green(), WIZARD_GLOW.blue(), 0))
    painter.fillRect(0, 0, width, height, glow)
    side = width * 1.0  # the glyph's 64-unit square; its wisp spans about 72% of it
    glyph.render(painter, QRectF(centre.x() - side / 2, centre.y() - side / 2, side, side))
    painter.end()
    return image.convertToFormat(QImage.Format.Format_RGB32)


def _rows(image: QImage) -> list[bytes]:
    """Top-down rows of BGRA bytes (``Format_ARGB32`` is stored B, G, R, A on little-endian Windows)."""
    data = bytes(image.constBits())
    stride = image.bytesPerLine()
    width = image.width() * 4
    return [data[y * stride : y * stride + width] for y in range(image.height())]


def dib_bytes(image: QImage) -> bytes:
    """A 32-bit BMP image as stored in an ICO: BITMAPINFOHEADER, bottom-up BGRA pixels, then the AND mask.

    Windows draws 32-bit icons with their alpha channel. The 1-bit AND mask is
    still part of the format; it marks the fully transparent pixels for programs
    that ignore alpha.
    """
    width, height = image.width(), image.height()
    rows = _rows(image)[::-1]  # BMP rows run bottom-up
    pixels = b"".join(rows)
    mask_stride = (width + 31) // 32 * 4
    mask = bytearray()
    for row in rows:
        line = bytearray(mask_stride)
        for x in range(width):
            if row[x * 4 + 3] == 0:
                line[x // 8] |= 0x80 >> (x % 8)
        mask += line
    header = _BITMAPINFOHEADER.pack(
        _BITMAPINFOHEADER.size,
        width,
        height * 2,  # an icon's DIB counts the colour image and the mask together
        1,  # planes
        32,  # bits per pixel
        0,  # BI_RGB: uncompressed
        len(pixels) + len(mask),
        0,
        0,
        0,
        0,
    )
    return header + pixels + bytes(mask)


def png_bytes(image: QImage) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG", 0):  # quality 0 = strongest (lossless) compression
        raise SystemExit("Qt couldn't encode a PNG image")
    buffer.close()
    return bytes(data.data())


def ico_bytes(images: list[QImage]) -> bytes:
    """An ICO file holding ``images`` (square, 1-256 px each), in the order given."""
    header = _ICONDIR.pack(0, 1, len(images))
    offset = _ICONDIR.size + _ICONDIRENTRY.size * len(images)
    entries, blobs = [], []
    for image in images:
        size = image.width()
        if size != image.height() or not 1 <= size <= 256:
            raise ValueError(f"icon images must be square and 1-256 px, not {size}x{image.height()}")
        blob = png_bytes(image) if size >= PNG_FROM else dib_bytes(image)
        side = size % 256  # 0 means 256
        entries.append(_ICONDIRENTRY.pack(side, side, 0, 0, 1, 32, len(blob), offset))
        blobs.append(blob)
        offset += len(blob)
    return header + b"".join(entries) + b"".join(blobs)


def write_preview(images: list[QImage], target: Path) -> None:
    """A contact sheet: every size at 1:1 on a dark and a light background, and the small ones enlarged 8x."""
    zoom = 8
    small = [image for image in images if image.width() <= 48]
    width = max(sum(i.width() + 16 for i in images) + 16, sum(i.width() * zoom + 16 for i in small) + 16)
    height = 2 * (256 + 32) + 48 * zoom + 32
    sheet = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    sheet.fill(QColor("#FFFFFF"))
    painter = QPainter(sheet)
    for band, background in enumerate(("#202024", "#F3F3F3")):
        top = 16 + band * (256 + 32)
        painter.fillRect(0, top - 16, width, 256 + 32, QColor(background))
        x = 16
        for image in images:
            painter.drawImage(x, top + 256 - image.height(), image)
            x += image.width() + 16
    top = 2 * (256 + 32) + 16
    painter.fillRect(0, top - 16, width, 48 * zoom + 32, QColor("#202024"))
    x = 16
    for image in small:
        enlarged = image.scaled(
            image.width() * zoom,
            image.height() * zoom,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter.drawImage(x, top, enlarged)
        x += enlarged.width() + 16
    painter.end()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not sheet.save(str(target)):
        raise SystemExit(f"Couldn't write {target}")


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    print(f"wrote {path} ({len(data):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="folder to write into (default: %(default)s)")
    parser.add_argument("--preview", type=Path, help="also write a PNG contact sheet of every icon size here")
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window or display needed
    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 - Qt's painting needs an application
    full = QSvgRenderer(str(LOGO))
    small = QSvgRenderer(foreground_only(LOGO))
    glyph = QSvgRenderer(str(GLYPH))
    if not (full.isValid() and small.isValid() and glyph.isValid()):
        raise SystemExit(f"Qt couldn't read the logo SVGs in {BRAND}")
    view_box, tile, ink = tile_geometry(LOGO)
    icons = [render(small if size <= FOREGROUND_ONLY_UP_TO else full, size, view_box, tile) for size in SIZES]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / "whisp.ico", ico_bytes(icons))
    _write(args.out_dir / "wizard-small.png", png_bytes(render(full, WIZARD_SMALL_SIZE, view_box, tile)))
    _write(args.out_dir / "wizard.png", png_bytes(render_wizard_panel(glyph, ink)))
    if args.preview:
        write_preview(icons, args.preview)
        print(f"preview {args.preview}")


if __name__ == "__main__":
    main()
