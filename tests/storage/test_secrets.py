"""DPAPI encryption of API keys."""

from __future__ import annotations

import base64

import pytest

from whisp.storage.secrets import protect, unprotect


@pytest.mark.parametrize(
    "plain",
    ["sk-test-1234567890", "ünïcødé key ✓ 鍵", "x", "k" * 4096, "line1\nline2\ttab"],
)
def test_round_trip(plain: str) -> None:
    token = protect(plain)
    assert token and token.isascii()
    if len(plain) >= 8:
        assert plain not in token  # (a one-letter key can appear in base64 by chance)
    assert unprotect(token) == plain


def test_tokens_are_salted() -> None:
    assert protect("same key") != protect("same key")  # DPAPI adds a random salt each time


def test_empty_stays_empty() -> None:
    assert protect("") == ""
    assert unprotect("") == ""


@pytest.mark.parametrize("token", ["not base64!!", "AAAA", base64.b64encode(b"garbage that is not dpapi").decode()])
def test_bad_tokens_give_an_empty_string(token: str) -> None:
    assert unprotect(token) == ""


def test_tampered_token_gives_an_empty_string() -> None:
    raw = bytearray(base64.b64decode(protect("sk-secret")))
    raw[-5] ^= 0xFF
    assert unprotect(base64.b64encode(bytes(raw)).decode()) == ""


def test_surrounding_whitespace_is_ignored() -> None:
    assert unprotect("  " + protect("key") + "\n") == "key"


def test_non_string_input_is_rejected_quietly() -> None:
    assert unprotect(None) == ""  # type: ignore[arg-type]
