"""Writing systems, as far as spacing between pieces of text depends on them.

Chinese and Japanese are written without spaces between words or sentences, so
pieces of recognised speech in those scripts join with nothing between them:
"你好。" and "我是小明。" make "你好。我是小明。". Thai, Lao, Khmer and Burmese put
spaces between phrases rather than words, and pieces always meet at a pause, so
they join with a space like every other language.
"""

from __future__ import annotations

from collections.abc import Iterable

_SPACELESS_RANGES = (
    (0x2E80, 0x2FDF),  # CJK and Kangxi radicals
    (0x3000, 0x303F),  # CJK symbols and punctuation: 、。「」 and the ideographic space
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x31C0, 0x31FF),  # CJK strokes, Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK unified ideographs, extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFE30, 0xFE4F),  # CJK compatibility forms
    (0xFF00, 0xFFEF),  # full-width punctuation and letters (，！？), half-width Katakana
    (0x20000, 0x3134F),  # CJK unified ideographs, extensions B to G
)


def is_spaceless(char: str) -> bool:
    """True if ``char`` belongs to Chinese or Japanese writing, which puts no spaces between words."""
    code = ord(char)
    return any(low <= code <= high for low, high in _SPACELESS_RANGES)


def join_spoken(parts: Iterable[str]) -> str:
    """Join pieces of recognised speech with a space, except where Chinese or Japanese is on either side."""
    text = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if text and not (is_spaceless(text[-1]) or is_spaceless(part[0])):
            text += " "
        text += part
    return text
