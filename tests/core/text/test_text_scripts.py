"""Joining pieces of recognised speech, with or without spaces depending on the writing system."""

from __future__ import annotations

import pytest

from whisp.core.text.scripts import is_spaceless, join_spoken


@pytest.mark.parametrize("char", ["你", "。", "，", "！", "「", "あ", "カ", "ｶ", "　", "𠀀"])
def test_chinese_and_japanese_are_written_without_spaces(char: str) -> None:
    assert is_spaceless(char)


@pytest.mark.parametrize("char", ["a", "é", ".", "한", "ก", "မ", "ا", "1", " "])
def test_other_scripts_use_spaces(char: str) -> None:
    # Korean spaces its words; Thai and Burmese space their phrases, and pieces meet at pauses.
    assert not is_spaceless(char)


@pytest.mark.parametrize(
    ("parts", "joined"),
    [
        (["Hello there.", "How are you?"], "Hello there. How are you?"),
        (["你好。", "我是小明。"], "你好。我是小明。"),
        (["今日は", "いい天気ですね。"], "今日はいい天気ですね。"),
        (["我用", "iPhone"], "我用iPhone"),
        (["안녕하세요.", "반갑습니다."], "안녕하세요. 반갑습니다."),
        ([" padded ", "", "   ", "text "], "padded text"),
        ([], ""),
    ],
)
def test_join_spoken(parts: list[str], joined: str) -> None:
    assert join_spoken(parts) == joined
