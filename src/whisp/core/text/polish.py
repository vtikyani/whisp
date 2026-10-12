"""AI polish (cleaning up a dictation) and command mode (rewriting selected text) with a language model.

Three providers, chosen in ``settings.polish.provider``:

- ``"builtin"``: Whisp's own llama.cpp server (``LlmRuntime``), on 127.0.0.1
  with a per-launch API key.
- ``"ollama"``: Ollama's native ``/api/chat`` with thinking off, the model kept
  loaded for 30 minutes and an 8K context.
- ``"openai"``: any OpenAI-compatible server (LM Studio, vLLM, a remote
  service). The API key is stored encrypted (``storage.secrets``).

A server address typed without ``http://`` or ``https://`` gets http:// on this
PC and https:// anywhere else; ``test()`` warns about plain http to another
machine. A user name or password in an address never appears in messages or logs.

Sampling is greedy (temperature 0) with no repetition or presence penalties,
because those push the model towards synonyms, which is paraphrasing. Output
length is bounded, stop sequences catch an echoed tag, and thinking is
switched off (any ``<think>`` block that slips through is removed).

**Nothing the model says is trusted.** Every answer goes through the checks
below; if one fails, the caller gets its own input back (``used=False``) with
a short reason in ``error``:

1. tidy: strip thinking, zero-width characters, echoed tags, wrapping quotes
   or code fences and a "Here is the cleaned text:" line;
2. the model must have finished (``finish_reason`` "stop");
3. it must not return nothing for real input;
4. it must answer within ``polish.timeout_seconds``;
5. the length must stay within 0.5–1.4× of the input (inputs of 6+ words);
6. at most ~15% new content words (number words count as their digits), which
   catches answers, additions and leaked examples;
7. a question must stay a question, not get answered;
8. no doubled text or repetition loops;
9. no change of script or language.

``polish`` and ``rewrite`` never raise. Never log transcript text, selections,
prompts, model output or API keys: lengths, timings and reason codes only.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
import ssl
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from PySide6.QtCore import Qt

from whisp import __version__
from whisp.core.net import tls_context, tls_error
from whisp.core.text import rules
from whisp.core.text.context import TextContext
from whisp.core.text.prompts import (
    CLEANUP_STOP,
    COMMAND_STOP,
    cleanup_messages,
    command_messages,
)
from whisp.core.types import AppCategory
from whisp.storage import secrets

if TYPE_CHECKING:
    from whisp.storage.settings import PolishSettings, SettingsStore

log = logging.getLogger(__name__)

CONTEXT_TOKENS = 8192
"""Context of the built-in server and of Ollama requests (``num_ctx``)."""
PROMPT_RESERVE_TOKENS = 1400
"""Room kept for the fixed system prompt and examples."""
MAX_INPUT_TOKENS = 2200
"""Longer texts (roughly 1,600+ English words) are not sent: they wouldn't fit the context with their answer."""
OLLAMA_KEEP_ALIVE = "30m"
SERVER_DOWN_RETRY_S = 30.0
"""After a server refuses a connection, dictations skip it this long instead of each waiting to find out."""
SPEED_SAMPLES = 5
"""Recent answers per server and model that its speed is judged on."""
_WARMUP_TIMEOUT_S = 120.0
_CONNECT_TIMEOUT_S = 3.0

MSG_EMPTY = "The AI returned nothing, so its answer wasn't used."
MSG_CUT_OFF = "The AI's answer was cut off, so it wasn't used."
MSG_LENGTH = "The AI changed the length too much, so its answer wasn't used."
MSG_ADDED = "The AI added words that weren't said, so its answer wasn't used."
MSG_ANSWERED = "The AI answered instead of writing down the question, so its answer wasn't used."
MSG_REPEATED = "The AI repeated the text, so its answer wasn't used."
MSG_LANGUAGE = "The AI changed the language, so its answer wasn't used."
MSG_TOO_LONG = "This is too long for AI polish, so it was inserted as dictated."
MSG_TOO_SLOW = "This is too long to polish within the time limit, so it was inserted as dictated."
MSG_BUILTIN_NOT_READY = "The built-in AI isn't ready yet, so the text was inserted without polish."
MSG_NO_SELECTION = "Select the text you want to change, then try again."
MSG_NO_INSTRUCTION = "No instruction was heard. Hold the command key and say what to change."
MSG_COMMAND_NEEDS_AI = "Command mode needs AI polish. Turn it on in Styles → AI polish."
MSG_QUESTION = (
    "Command mode changes the selected text. Ask it to rewrite, shorten, fix or translate it, "
    "rather than asking a question about it."
)
MSG_SELECTION_TOO_LONG = "The selection is too long for command mode. Select less text and try again."
MSG_NOT_CONFIGURED = "AI polish isn't set up yet. Choose an AI server and model in Styles → AI polish."
MSG_UNREADABLE = "The AI server sent a reply Whisp couldn't read."
MSG_UNEXPECTED = "AI polish failed unexpectedly, so the text was inserted without it."
MSG_NOT_ENCRYPTED = (
    "The connection isn't encrypted (http), so others on the network could read your text. "
    "Use https:// if the server supports it."
)


@dataclass(frozen=True)
class PolishResult:
    text: str = field(repr=False)
    used: bool
    """False when the input came back unchanged: a deliberate skip (``error`` None) or a failure (``error`` set)."""
    error: str | None
    """Short user-facing reason when the AI's answer couldn't be used."""
    elapsed_ms: int


class PolishError(Exception):
    """A provider problem; the message is user-facing."""


class Runtime(Protocol):
    """What the polisher needs from ``LlmRuntime``."""

    def base_url(self) -> str | None: ...

    def api_key(self) -> str | None: ...

    def installed_models(self) -> list[str]: ...


# -- output clean-up (check 1) --------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_ONLY_RE = re.compile(r"^.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_ZERO_WIDTH_RE = re.compile(
    "[\N{ZERO WIDTH SPACE}\N{ZERO WIDTH NON-JOINER}\N{ZERO WIDTH JOINER}\N{WORD JOINER}\N{ZERO WIDTH NO-BREAK SPACE}]"
)
_ECHO_TAG_RE = re.compile(
    r"</?(?:transcript|selected_text|instruction|style|preferences|output|cleaned|result|text)>", re.IGNORECASE
)
_FENCE_RE = re.compile(r"^```[\w-]*\n(?P<body>.*?)\n?```$", re.DOTALL)
_PREAMBLE_RE = re.compile(
    r"^(?:(?:sure|okay|ok|certainly|of course|here you go)[!,.]?\s*)?"
    r"(?:(?:here(?:'s|’s| is| are)|below is)\b[^\n]{0,80}|"
    r"(?:the\s+)?(?:cleaned|corrected|polished|edited|revised|rewritten|final|updated)\b[^\n]{0,40}|"
    r"(?:output|result|answer))\s*:\s*$",
    re.IGNORECASE,
)
_TRAILING_NOTE_RE = re.compile(r"\n\s*\n\s*\(?(?:note|notes|explanation|changes)\b[^\n]*\)?\s*$", re.IGNORECASE)
_QUOTE_PAIRS = {'"': '"', "“": "”", "'": "'", "‘": "’", "«": "»", "„": "“"}


def _edge_whitespace(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped:
        return "", ""
    start = text.index(stripped[0])
    end = len(text) - len(text.rstrip())
    return text[:start], text[len(text) - end :] if end else ""


def tidy_output(raw: str, source: str) -> str:
    """Check 1: remove the wrapping a model sometimes adds around its answer.

    Strips thinking blocks, echoed tags, a preamble line ("Here is the cleaned
    text:"), a trailing note, wrapping quotes or code fences (unless the source
    had them too) and zero-width characters, then puts back the line breaks
    or spaces the source had at its start and end.
    """
    text = raw or ""
    text = _THINK_BLOCK_RE.sub("", text)
    if re.search(r"</think>", text, re.IGNORECASE):
        text = _THINK_CLOSE_ONLY_RE.sub("", text)  # the opening tag was in the prompt template
    text = _THINK_OPEN_RE.sub("", text)  # unfinished thinking: nothing usable after it
    text = text.replace("\r\n", "\n").strip()
    if not _ECHO_TAG_RE.search(source):
        text = _ECHO_TAG_RE.sub("", text).strip()
    text = _ZERO_WIDTH_RE.sub("", text)
    source_core = source.strip()
    if "```" not in source_core:
        fence = _FENCE_RE.match(text)
        if fence:
            text = fence.group("body").strip()
    lines = text.split("\n", 1)
    if len(lines) == 2 and _PREAMBLE_RE.match(lines[0].strip()) and not _PREAMBLE_RE.match(source_core.split("\n")[0]):
        text = lines[1].strip()
    if "note" not in source_core.lower() and "explanation" not in source_core.lower():
        text = _TRAILING_NOTE_RE.sub("", text).strip()
    quoted = len(text) >= 2 and text[0] in _QUOTE_PAIRS and text[-1] == _QUOTE_PAIRS[text[0]]
    if quoted and not (source_core[:1] == text[0] and source_core[-1:] == text[-1]):
        text = text[1:-1].strip()
    if not text:
        return ""
    lead, trail = _edge_whitespace(source)
    return lead + text + trail


# -- words, numbers and languages -------------------------------------------------------------

_WORD_RE = re.compile(r"\w+(?:['’]\w+)*")
# Scripts written without spaces between words (Thai, Lao, Myanmar, Khmer, Japanese kana, Chinese):
# each character counts as one unit.
_SPLIT_SCRIPT_RE = re.compile(
    "[\U00000e00-\U00000eff\U00001000-\U0000109f\U00001780-\U000017ff\U00003040-\U000030ff"
    "\U00003400-\U00004dbf\U00004e00-\U00009fff\U0000f900-\U0000faff]"
)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


def units(text: str) -> list[str]:
    """Lower-case words; characters of scripts without spaces count one each."""
    out: list[str] = []
    for match in _WORD_RE.finditer(text.lower().replace("’", "'")):
        word = match.group()
        if _SPLIT_SCRIPT_RE.search(word):
            out.extend(ch for ch in word if ch.isalnum())
        else:
            out.append(word)
    return out


_SMALL_NUMBERS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
    "fortieth": 40, "fiftieth": 50, "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
    "hundredth": 100, "thousandth": 1000,
}  # fmt: skip
_ORDINAL_SUFFIX_RE = re.compile(r"(\d+)(?:st|nd|rd|th)")
_NUMBER_WORDS = set(_SMALL_NUMBERS) | set(_TENS) | set(_SCALES)

_STOPWORDS = rules.word_set(
    "a an the and or but nor so to of in on at for with from by as into onto about over under than then "
    "is are was were be been being am do does did done have has had having i you he she it we they me him her us "
    "them my your his its our their mine yours ours theirs this that these those there here what which who whom "
    "whose when where why how not no yes if too very just also only all any some can could will would shall "
    "should may might must let s t re ll ve m d okay ok um uh oh well like really"
)
_CONTRACTION_RE = re.compile(r"(?:'s|'re|'ll|'ve|'m|'d|n't)$")


def _number_values(words: list[str]) -> set[str]:
    """Digit forms of every number spoken in ``words`` ("twenty twenty six" -> {"20", "26", "2026", ...})."""
    values: set[str] = set()
    i = 0
    while i < len(words):
        if words[i] not in _NUMBER_WORDS and words[i] not in _ORDINALS:
            i += 1
            continue
        run: list[str] = []
        while i < len(words) and (words[i] in _NUMBER_WORDS or words[i] in _ORDINALS or (words[i] == "and" and run)):
            if words[i] != "and":
                run.append(words[i])
            i += 1
        groups: list[int] = []
        total = current = 0
        previous = ""
        for word in run:
            if word in _ORDINALS:
                value = _ORDINALS[word]
                if previous in _TENS and value < 10:
                    current += value
                else:
                    if previous:
                        groups.append(total + current)
                        total = 0
                    current = value
            elif word in _SMALL_NUMBERS:
                if previous in _SMALL_NUMBERS or (previous in _TENS and current % 10) or previous in _ORDINALS:
                    groups.append(total + current)
                    total = current = 0
                current += _SMALL_NUMBERS[word]
            elif word in _TENS:
                if previous in _TENS or previous in _SMALL_NUMBERS or previous in _ORDINALS:
                    groups.append(total + current)
                    total = current = 0
                current += _TENS[word]
            elif word == "hundred":
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * _SCALES[word]
                current = 0
            previous = word
        groups.append(total + current)
        digits = [str(g) for g in groups[:_MAX_NUMBER_GROUPS]]
        # Every run of consecutive groups: "five five five one two" may be written "555-12" or "55512".
        for start in range(len(digits)):
            for end in range(start + 1, len(digits) + 1):
                values.add("".join(digits[start:end]))
        for group in groups:
            values.update(f"{group:,}".split(","))  # "1,000" is read as "1" and "000"
    return values


_MAX_NUMBER_GROUPS = 24


def _normalise(word: str) -> str:
    word = word.lower().replace("’", "'")
    if word in _SMALL_NUMBERS:
        return str(_SMALL_NUMBERS[word])
    if word in _TENS:
        return str(_TENS[word])
    if word in _ORDINALS:
        return str(_ORDINALS[word])
    ordinal = _ORDINAL_SUFFIX_RE.fullmatch(word)
    if ordinal:
        return ordinal.group(1)
    if word == "okay":
        return "ok"
    if word == "won't":
        return "will"
    if word == "can't":
        return "can"
    return _CONTRACTION_RE.sub("", word)


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


@dataclass(frozen=True)
class _Vocabulary:
    words: frozenset[str]
    stems: frozenset[str]
    prefixes: frozenset[str]

    def knows(self, word: str) -> bool:
        return word in self.words or _stem(word) in self.stems or (len(word) >= 7 and word[:6] in self.prefixes)


def _vocabulary(text: str, extra: Iterable[str] = ()) -> _Vocabulary:
    raw = units(text)
    normalised = [_normalise(w) for w in raw]
    words = set(normalised) | _number_values(raw)
    # Compounds written together or apart: "e mail" ~ "email", "svelte kit" ~ "sveltekit".
    words.update(a + b for a, b in itertools.pairwise(normalised))
    for term in extra:
        words.update(_normalise(w) for w in units(term))
        words.add("".join(units(term)))
    stems = {_stem(w) for w in words}
    prefixes = {w[:6] for w in words if len(w) >= 7}
    return _Vocabulary(frozenset(words), frozenset(stems), frozenset(prefixes))


def _content_words(text: str) -> list[str]:
    text = _LIST_MARKER_RE.sub("", text)
    return [n for n in (_normalise(w) for w in units(text)) if n and n not in _STOPWORDS]


_SCRIPTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("latin", re.compile("[A-Za-z\U000000c0-\U0000024f\U00001e00-\U00001eff]")),
    ("cyrillic", re.compile("[\U00000400-\U0000052f]")),
    ("greek", re.compile("[\U00000370-\U000003ff]")),
    ("arabic", re.compile("[\U00000600-\U000006ff\U00000750-\U0000077f]")),
    ("hebrew", re.compile("[\U00000590-\U000005ff]")),
    ("devanagari", re.compile("[\U00000900-\U0000097f]")),
    ("thai", re.compile("[\U00000e00-\U00000e7f]")),
    ("hangul", re.compile("[\U0000ac00-\U0000d7af\U00001100-\U000011ff]")),
    ("cjk", re.compile("[\U00003040-\U000030ff\U00003400-\U00004dbf\U00004e00-\U00009fff]")),
)


def dominant_script(text: str) -> str | None:
    """The writing system most letters belong to, or None for fewer than 8 letters."""
    counts = {name: len(pattern.findall(text)) for name, pattern in _SCRIPTS}
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] >= 8 else None


# -- checks 2-9 ---------------------------------------------------------------------------

_OK_FINISH = frozenset({"stop", "eos", "end_turn", "stop_sequence"})
_QUESTION_WORDS = rules.word_set(
    "what why who whom whose when where which how is are was were do does did can could would should will "
    "shall may has have am isn't aren't don't doesn't didn't can't won't"
)
_TRANSLATE_RE = re.compile(
    r"translat|\b(?:english|spanish|french|german|italian|portuguese|dutch|russian|ukrainian|polish|czech|"
    r"swedish|norwegian|danish|finnish|greek|turkish|arabic|hebrew|hindi|chinese|mandarin|cantonese|japanese|"
    r"korean|vietnamese|thai|indonesian|malay|romanian|hungarian|bulgarian|croatian|serbian|slovak|catalan|"
    r"latin|filipino|tagalog|persian|farsi|urdu|bengali)\b",
    re.IGNORECASE,
)


def check_finish_reason(finish_reason: str | None) -> str | None:
    """Check 2: the model stopped on its own (not cut off by the token limit)."""
    if finish_reason is None or finish_reason.lower() in _OK_FINISH:
        return None
    return MSG_CUT_OFF


def check_not_empty(source: str, output: str) -> str | None:
    """Check 3: real input must not come back empty."""
    if source.strip() and not output.strip():
        return MSG_EMPTY
    return None


def check_length_ratio(source: str, output: str) -> str | None:
    """Check 5: output words ÷ input words within 0.5–1.4 (inputs of six or more words)."""
    n_in = len(units(source))
    if n_in < 6:
        return None
    ratio = len(units(output)) / n_in
    if ratio < 0.5 or ratio > 1.4:
        return MSG_LENGTH
    return None


def check_new_words(source: str, output: str, allowed: Iterable[str] = ()) -> str | None:
    """Check 6: the output may only use (almost) the input's content words.

    Number words match their digits, compounds match written together or
    apart, and the user's dictionary terms are always allowed. Fails when two
    or more content words are new and they make up over 15% of the output,
    or when more than half of them are new.
    """
    words = _content_words(output)
    if not words:
        return None
    vocab = _vocabulary(source, allowed)
    new = [w for w in words if not vocab.knows(w)]
    ratio = len(new) / len(words)
    if (len(new) >= 2 and ratio > 0.15) or ratio > 0.5:
        return MSG_ADDED
    return None


def looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    words = units(stripped)
    return bool(words) and _normalise(words[0]) in _QUESTION_WORDS and stripped[-1] not in ".!"


def check_question_kept(source: str, output: str, allowed: Iterable[str] = ()) -> str | None:
    """Check 7: a dictated question must come back as a question, not as its answer."""
    if not looks_like_question(source):
        return None
    src = Counter(_normalise(w) for w in units(source))
    out = Counter(_normalise(w) for w in units(output))
    kept = sum(min(count, out[w]) for w, count in src.items())
    if kept < 0.6 * sum(src.values()):
        return MSG_ANSWERED
    if "?" not in output:
        vocab = _vocabulary(source, allowed)
        if any(not vocab.knows(w) for w in _content_words(output)):
            return MSG_ANSWERED
    return None


def _sentences(text: str) -> list[str]:
    return [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def check_duplication(source: str, output: str) -> str | None:
    """Check 8: the text twice, or the same sentence looping."""
    out, src = units(output), units(source)
    half = len(out) // 2
    output_doubled = len(out) >= 4 and len(out) % 2 == 0 and out[:half] == out[half:]
    source_doubled = len(src) % 2 == 0 and src[: len(src) // 2] == src[len(src) // 2 :]
    if output_doubled and not source_doubled:
        return MSG_REPEATED
    if len(src) >= 3:
        joined, needle = " ".join(out), " ".join(src)
        if joined.count(needle) >= 2:
            return MSG_REPEATED
    sentences = _sentences(output)
    source_sentences = Counter(_sentences(source))
    run = 1
    for previous, current in itertools.pairwise(sentences):
        run = run + 1 if current == previous and len(current.split()) >= 3 else 1
        if run >= 3 and source_sentences[current] < run:
            return MSG_REPEATED
    return None


def check_language(source: str, output: str) -> str | None:
    """Check 9: same writing system, and for Latin text the same language when both are clear."""
    source_script, output_script = dominant_script(source), dominant_script(output)
    if source_script and output_script and source_script != output_script:
        return MSG_LANGUAGE
    if source_script == output_script == "latin":
        source_language = rules.guess_latin_language(source)
        output_language = rules.guess_latin_language(output)
        if source_language and output_language and source_language != output_language:
            return MSG_LANGUAGE
    return None


def validate_cleanup(source: str, output: str, finish_reason: str | None, allowed: Iterable[str] = ()) -> str | None:
    """Checks 2, 3 and 5-9 for a cleaned dictation. Returns the first failure's message, or None."""
    allowed = tuple(allowed)
    return (
        check_finish_reason(finish_reason)
        or check_not_empty(source, output)
        or check_length_ratio(source, output)
        or check_new_words(source, output, allowed)
        or check_question_kept(source, output, allowed)
        or check_duplication(source, output)
        or check_language(source, output)
    )


def wants_translation(instruction: str) -> bool:
    return _TRANSLATE_RE.search(instruction) is not None


def validate_rewrite(selection: str, output: str, finish_reason: str | None, instruction: str) -> str | None:
    """Checks for command mode, where the wording is meant to change: finished, not empty, not doubled,
    and the same language unless the instruction asks for another."""
    return (
        check_finish_reason(finish_reason)
        or check_not_empty(selection, output)
        or check_duplication(selection, output)
        or (None if wants_translation(instruction) else check_language(selection, output))
    )


_CLEAR_EDIT_VERBS = rules.word_set(
    "make rewrite rephrase reword fix translate shorten lengthen expand extend convert summarize summarise simplify "
    "condense add remove delete replace capitalize capitalise write improve edit tidy proofread punctuate sort "
    "reorder merge combine swap insert append prepend uppercase lowercase say"
)
"""Words that only ever ask for an edit."""
_EDIT_VERBS = _CLEAR_EDIT_VERBS | rules.word_set(
    "change correct turn format put use keep polish clean bold bullet list number split drop cut trim spell"
)
"""Also words that can describe the text instead ("is this correct?", "a clean draft")."""
_INFO_VERBS = rules.word_set("explain describe define answer tell analyze analyse evaluate critique rate")
_AUX_QUESTION_STARTS = rules.word_set("is are was were am do does did has have isn't aren't doesn't don't didn't")


def instruction_is_question(instruction: str) -> bool:
    """True for "what does this mean?", "is this correct?" or "explain this": asking about the text, not editing it.

    "Can you make this shorter?" and "how would you say this in French?" are edits.
    """
    words = [_normalise(w) if w.endswith("n't") else w for w in re.findall(r"[a-z']+", instruction.lower())]
    if not words:
        return False
    if words[0] in _AUX_QUESTION_STARTS:
        # "is this correct?" asks about the text; "is it possible to shorten this?" still asks for an edit.
        return not any(w in _CLEAR_EDIT_VERBS for w in words)
    if any(w in _EDIT_VERBS for w in words):
        return False
    return words[0] in _QUESTION_WORDS or words[0] in _INFO_VERBS or instruction.strip().endswith("?")


# -- token budget ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count: ~3.5 characters per token for ASCII, ~1 per character otherwise (on the high side)."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return math.ceil(ascii_chars / 3.5 + (len(text) - ascii_chars) * 1.1)


def _cleanup_max_tokens(text: str) -> int:
    return min(2 * estimate_tokens(text) + 64, CONTEXT_TOKENS - PROMPT_RESERVE_TOKENS - estimate_tokens(text))


def _command_max_tokens(selection: str, instruction: str) -> int:
    budget = CONTEXT_TOKENS - PROMPT_RESERVE_TOKENS - estimate_tokens(selection) - estimate_tokens(instruction)
    return max(64, min(3 * estimate_tokens(selection) + 256, budget))


# -- providers --------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reply:
    text: str = field(repr=False)
    finish_reason: str | None
    completion_tokens: int | None
    generation_s: float | None = None
    """How long the server says it spent writing the answer (llama-server ``timings``, Ollama ``eval_duration``):
    no model loading or prompt reading in it. None when it doesn't say."""


@dataclass
class _Speeds:
    """Recent writing speeds (tokens/s) of one server and model, for guessing whether a text will finish in time.

    A server's own timings leave out loading the model, so one of them is enough. Wall-clock time doesn't: one
    slow answer may have been a cold start, so a guess needs a second sample, and it is the median (the faster
    middle one of an even number), which no single slow answer can drag down.
    """

    samples: deque[float] = field(default_factory=lambda: deque(maxlen=SPEED_SAMPLES))
    timed: bool = False
    """At least one sample came from the server's own timings."""

    def add(self, tokens_per_s: float, *, timed: bool) -> None:
        self.samples.append(tokens_per_s)
        self.timed = self.timed or timed

    def estimate(self) -> float | None:
        if not self.samples or (not self.timed and len(self.samples) < 2):
            return None
        ordered = sorted(self.samples)
        return ordered[len(ordered) // 2]


@dataclass(frozen=True)
class _Target:
    provider: str
    url: str
    """Full request URL."""
    model: str
    api_key: str | None
    local: bool


def _is_local(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".localhost")


def _with_scheme(url: str) -> str:
    """``url`` as typed if it has a scheme; otherwise http:// for this PC and https:// for any other server."""
    if "://" in url:
        return url
    return ("http://" if _is_local("http://" + url) else "https://") + url


def _unencrypted(url: str) -> bool:
    """True for plain http to another machine: what is sent can be read on the way."""
    return urlsplit(url).scheme.lower() == "http" and not _is_local(url)


def normalise_openai_base(url: str) -> str:
    """Base URL for an OpenAI-compatible server: no trailing slash, no endpoint path, "/v1" when no path is given.

    Without a scheme, a server on this PC gets http:// and any other https://.
    """
    url = url.strip().rstrip("/")
    if not url:
        return ""
    url = _with_scheme(url)
    for suffix in ("/chat/completions", "/completions", "/models"):
        if url.lower().endswith(suffix):
            url = url[: -len(suffix)]
    parts = urlsplit(url)
    if parts.path in ("", "/"):
        parts = parts._replace(path="/v1")
    return urlunsplit(parts).rstrip("/")


def normalise_ollama_base(url: str) -> str:
    """Base URL for Ollama: no trailing slash, and a scheme (as for ``normalise_openai_base``)."""
    url = url.strip().rstrip("/")
    return _with_scheme(url) if url else ""


def _host_of(url: str) -> str:
    """The server's host and port, for messages and logs: never a user name or password written into the URL."""
    parts = urlsplit(url if "://" in url else "http://" + url)
    return parts.netloc.rpartition("@")[2] or "the configured address"


def _server_of(url: str) -> str:
    parts = urlsplit(url if "://" in url else "http://" + url)
    return f"{parts.scheme}://{parts.netloc.rpartition('@')[2]}".lower()


def _unreachable(url: str) -> str:
    return f"Couldn't reach the AI server at {_host_of(url)}. Is it running?"


def _connection_problem(url: str, exc: BaseException) -> str:
    """What to tell the user when connecting to the AI server at ``url`` failed with ``exc``."""
    error = tls_error(exc)
    if isinstance(error, ssl.SSLCertVerificationError):
        host = _host_of(url)
        return f"The security certificate of the AI server at {host} couldn't be verified, so nothing was sent."
    if error is not None and urlsplit(url).scheme.lower() == "https":
        # Typically a server that only speaks plain http, given an address without http:// in front.
        return (
            f"Couldn't make a secure (https) connection to the AI server at {_host_of(url)}. "
            "If it only supports http, type http:// in front of its address."
        )
    return _unreachable(url)


def _server_error(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        error = error.get("message")
    return str(error)[:200] if error else ""


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class Polisher:
    """Runs AI polish and command mode against the configured provider. Thread-safe."""

    def __init__(self, settings: SettingsStore, runtime: Runtime) -> None:
        self._settings = settings
        self._runtime = runtime
        self._local = threading.local()
        self._lock = threading.Lock()
        self._speed: dict[tuple[str, str, str], _Speeds] = {}
        """Measured writing speed per (provider, url, model)."""
        self._no_extras: set[str] = set()
        """OpenAI-compatible base URLs that rejected Whisp's extra request fields."""
        self._resolved_models: dict[str, str] = {}
        self._warmed: str | None = None
        self._down: dict[str, tuple[float, str]] = {}
        """Servers that couldn't be connected to: until when to stop trying them, and what went wrong."""
        status_changed = getattr(runtime, "status_changed", None)
        if status_changed is not None:
            status_changed.connect(self._on_runtime_status, Qt.ConnectionType.DirectConnection)

    # -- public API ----------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True when polish is on and the provider looks usable (no network calls). Any thread.

        A server that refused a connection in the last ``SERVER_DOWN_RETRY_S`` seconds counts as not ready.
        """
        polish = self._settings.current.polish
        if not polish.enabled:
            return False
        if polish.provider == "builtin":
            return self._runtime.base_url() is not None
        if polish.provider == "ollama":
            base = normalise_ollama_base(polish.ollama_url)
            return bool(base and polish.ollama_model.strip()) and self._down_reason(base) is None
        base = normalise_openai_base(polish.openai_url)
        return bool(base) and self._down_reason(base) is None

    def polish(self, text: str, ctx: TextContext) -> PolishResult:
        """Clean up ``text`` (already through the rules). Blocking; never raises.

        Skips (``used=False``, no error) when polish is off, the text is
        empty, or the style is verbatim.
        """
        started = time.monotonic()
        try:
            polish = self._settings.current.polish
            if not polish.enabled or not text.strip() or ctx.style == "verbatim":
                return PolishResult(text, False, None, 0)
            return self._cleanup(text, ctx, polish, started)
        except Exception:
            log.exception("AI polish failed")
            return PolishResult(text, False, MSG_UNEXPECTED, _ms_since(started))

    def rewrite(self, selection: str, instruction: str, ctx: TextContext) -> PolishResult:
        """Apply a spoken ``instruction`` to ``selection`` (command mode). Blocking; never raises.

        On any failure the selection comes back unchanged with ``error`` set.
        """
        started = time.monotonic()
        try:
            return self._rewrite(selection, instruction, ctx, started)
        except Exception:
            log.exception("Command mode failed")
            return PolishResult(selection, False, MSG_UNEXPECTED, _ms_since(started))

    def test(self) -> tuple[bool, str]:
        """Send a short sample through the configured provider with all checks. Blocking.

        Works whether or not polish is switched on, so settings can be tried first.
        """
        sample = "um so this is a quick test of the AI polish and it should come back clean"
        ctx = TextContext(AppCategory.OTHER, "clean", "", "en", (), (), True, True, "")
        started = time.monotonic()
        polish = self._settings.current.polish
        try:
            result = self._cleanup(
                rules.apply_rules(sample, ctx), ctx, polish, started, record_speed=False, retry_down=True
            )
        except Exception:
            log.exception("AI polish test failed")
            return False, MSG_UNEXPECTED
        if result.used:
            model = self._describe_model(polish)
            message = f"Connected to {model}. It answered in {result.elapsed_ms} ms."
            if polish.provider != "builtin" and _unencrypted(self._server_base(polish)):
                message += " " + MSG_NOT_ENCRYPTED
            return True, message
        if result.error in (MSG_EMPTY, MSG_CUT_OFF, MSG_LENGTH, MSG_ADDED, MSG_ANSWERED, MSG_REPEATED, MSG_LANGUAGE):
            return False, f"Connected, but the answer didn't pass Whisp's checks: {result.error}"
        return False, result.error or MSG_UNEXPECTED

    def list_models(self, provider: str) -> list[str]:
        """Models the provider offers. Blocking; returns [] when the server can't be reached (and logs why)."""
        polish = self._settings.current.polish
        try:
            if provider == "builtin":
                return list(self._runtime.installed_models())
            if provider == "ollama":
                base = normalise_ollama_base(polish.ollama_url)
                if not base:
                    return []
                response = self._client(_is_local(base)).get(f"{base}/api/tags", timeout=self._list_timeout())
                response.raise_for_status()
                return sorted(m["name"] for m in response.json().get("models", []) if m.get("name"))
            if provider == "openai":
                base = normalise_openai_base(polish.openai_url)
                if not base:
                    return []
                return self._openai_models(base, self._openai_key(polish))
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("Could not list %s models: %s", provider, type(exc).__name__)
        return []

    # -- cleanup ---------------------------------------------------------------------------

    def _cleanup(
        self,
        text: str,
        ctx: TextContext,
        polish: PolishSettings,
        started: float,
        *,
        record_speed: bool = True,
        retry_down: bool = False,
    ) -> PolishResult:
        tokens = estimate_tokens(text)
        if tokens > MAX_INPUT_TOKENS:
            return PolishResult(text, False, MSG_TOO_LONG, _ms_since(started))
        try:
            target = self._target(polish)
        except PolishError as exc:
            return PolishResult(text, False, str(exc), _ms_since(started))
        if not retry_down and (down := self._down_reason(target.url)) is not None:
            return PolishResult(text, False, down, _ms_since(started))
        with self._lock:
            speeds = self._speed.get((target.provider, target.url, target.model))
            speed = speeds.estimate() if speeds else None
        if speed and tokens / speed > polish.timeout_seconds:
            log.info("Skipping AI polish: ~%d tokens at %.0f tokens/s won't finish in time", tokens, speed)
            return PolishResult(text, False, MSG_TOO_SLOW, _ms_since(started))
        messages = cleanup_messages(text, ctx)
        try:
            reply = self._chat(
                target,
                messages,
                max_tokens=_cleanup_max_tokens(text),
                stop=CLEANUP_STOP,
                deadline=started + polish.timeout_seconds,
            )
        except PolishError as exc:
            log.info("AI polish skipped after %d ms: %s", _ms_since(started), exc)
            return PolishResult(text, False, str(exc), _ms_since(started))
        elapsed = time.monotonic() - started
        if record_speed:
            self._record_speed(target, reply, elapsed)
        output = tidy_output(reply.text, text)
        problem = validate_cleanup(text, output, reply.finish_reason, (e.term for e in ctx.dictionary))
        log.info(
            "AI polish via %s: %d -> %d chars in %d ms (finish %s)%s",
            target.provider,
            len(text),
            len(output),
            int(elapsed * 1000),
            reply.finish_reason,
            f"; rejected: {_reason_code(problem)}" if problem else "",
        )
        if problem:
            return PolishResult(text, False, problem, _ms_since(started))
        return PolishResult(output, True, None, _ms_since(started))

    def _rewrite(self, selection: str, instruction: str, ctx: TextContext, started: float) -> PolishResult:
        if not selection.strip():
            return PolishResult(selection, False, MSG_NO_SELECTION, 0)
        if not instruction.strip():
            return PolishResult(selection, False, MSG_NO_INSTRUCTION, 0)
        polish = self._settings.current.polish
        if not polish.enabled:
            return PolishResult(selection, False, MSG_COMMAND_NEEDS_AI, 0)
        if instruction_is_question(instruction):
            return PolishResult(selection, False, MSG_QUESTION, 0)
        if estimate_tokens(selection) + estimate_tokens(instruction) > MAX_INPUT_TOKENS:
            return PolishResult(selection, False, MSG_SELECTION_TOO_LONG, 0)
        try:
            target = self._target(polish)
            if (down := self._down_reason(target.url)) is not None:
                return PolishResult(selection, False, down, _ms_since(started))
            reply = self._chat(
                target,
                command_messages(selection, instruction, ctx),
                max_tokens=_command_max_tokens(selection, instruction),
                stop=COMMAND_STOP,
                deadline=started + polish.timeout_seconds,
            )
        except PolishError as exc:
            log.info("Command mode failed after %d ms: %s", _ms_since(started), exc)
            return PolishResult(selection, False, str(exc), _ms_since(started))
        output = tidy_output(reply.text, selection)
        problem = validate_rewrite(selection, output, reply.finish_reason, instruction)
        log.info(
            "Command mode via %s: %d -> %d chars in %d ms (finish %s)%s",
            target.provider,
            len(selection),
            len(output),
            _ms_since(started),
            reply.finish_reason,
            f"; rejected: {_reason_code(problem)}" if problem else "",
        )
        if problem:
            return PolishResult(selection, False, problem, _ms_since(started))
        return PolishResult(output, True, None, _ms_since(started))

    def _down_reason(self, url: str) -> str | None:
        """Why the server at ``url`` is being skipped (it couldn't be connected to just now), or None."""
        with self._lock:
            down = self._down.get(_server_of(url))
        if down is None or time.monotonic() >= down[0]:
            return None
        return down[1]

    def _mark_down(self, url: str, reason: str) -> None:
        log.info("AI server %s isn't reachable; skipping it for %.0f s", _host_of(url), SERVER_DOWN_RETRY_S)
        with self._lock:
            self._down[_server_of(url)] = (time.monotonic() + SERVER_DOWN_RETRY_S, reason)

    def _mark_up(self, url: str) -> None:
        with self._lock:
            self._down.pop(_server_of(url), None)

    def _record_speed(self, target: _Target, reply: _Reply, elapsed: float) -> None:
        """Remember how fast the server wrote: by its own timings when it reports them, else by the clock."""
        tokens = reply.completion_tokens or estimate_tokens(reply.text)
        seconds = reply.generation_s or elapsed
        if tokens < 32 or seconds <= 0:
            return
        key = (target.provider, target.url, target.model)
        with self._lock:
            self._speed.setdefault(key, _Speeds()).add(tokens / seconds, timed=reply.generation_s is not None)

    @staticmethod
    def _server_base(polish: PolishSettings) -> str:
        """Base URL of the configured Ollama or OpenAI-compatible server ("" when unset)."""
        if polish.provider == "ollama":
            return normalise_ollama_base(polish.ollama_url)
        return normalise_openai_base(polish.openai_url)

    def _describe_model(self, polish: PolishSettings) -> str:
        if polish.provider == "builtin":
            return "the built-in AI"
        if polish.provider == "ollama":
            return f"{polish.ollama_model} on Ollama"
        base = normalise_openai_base(polish.openai_url)
        model = polish.openai_model or self._resolved_models.get(base, "")
        return f"{model} at {_host_of(base)}" if model else _host_of(base)

    # -- transport -------------------------------------------------------------------------

    def _client(self, local: bool) -> httpx.Client:
        """One client per thread and kind: local servers bypass any system proxy."""
        attr = "local_client" if local else "remote_client"
        client = getattr(self._local, attr, None)
        if client is None:
            client = httpx.Client(
                verify=tls_context(),
                trust_env=not local,
                headers={"User-Agent": f"Whisp/{__version__}"},
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
            setattr(self._local, attr, client)
        return client

    def _list_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(10.0, connect=_CONNECT_TIMEOUT_S)

    def _openai_key(self, polish: PolishSettings) -> str | None:
        if not polish.openai_api_key_protected:
            return None
        key = secrets.unprotect(polish.openai_api_key_protected)
        return key or None

    def _openai_models(self, base: str, api_key: str | None) -> list[str]:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = self._client(_is_local(base)).get(f"{base}/models", headers=headers, timeout=self._list_timeout())
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", []) if isinstance(m, dict) and m.get("id")]

    def _target(self, polish: PolishSettings) -> _Target:
        if polish.provider == "builtin":
            base, key = self._runtime.base_url(), self._runtime.api_key()
            if not base:
                raise PolishError(MSG_BUILTIN_NOT_READY)
            running = getattr(getattr(self._runtime, "status", None), "model_id", None)
            model = running or polish.builtin_model or "builtin"
            return _Target("builtin", f"{base}/v1/chat/completions", model, key, True)
        if polish.provider == "ollama":
            base = normalise_ollama_base(polish.ollama_url)
            if not base or not polish.ollama_model.strip():
                raise PolishError(MSG_NOT_CONFIGURED)
            return _Target("ollama", f"{base}/api/chat", polish.ollama_model.strip(), None, _is_local(base))
        base = normalise_openai_base(polish.openai_url)
        if not base:
            raise PolishError(MSG_NOT_CONFIGURED)
        key = self._openai_key(polish)
        model = polish.openai_model.strip() or self._resolve_model(base, key)
        return _Target("openai", f"{base}/chat/completions", model, key, _is_local(base))

    def _resolve_model(self, base: str, api_key: str | None) -> str:
        """No model chosen: use the first one the server lists (LM Studio's loaded model, for example)."""
        with self._lock:
            cached = self._resolved_models.get(base)
        if cached:
            return cached
        try:
            models = self._openai_models(base, api_key)
        except httpx.HTTPError as exc:
            raise PolishError(_connection_problem(base, exc)) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise PolishError(MSG_UNREADABLE) from exc
        if not models:
            host = _host_of(base)
            raise PolishError(f"The AI server at {host} has no model loaded. Choose one in Styles → AI polish.")
        with self._lock:
            self._resolved_models[base] = models[0]
        return models[0]

    def _chat(
        self,
        target: _Target,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        stop: tuple[str, ...],
        deadline: float,
    ) -> _Reply:
        if target.provider == "ollama":
            body: dict[str, Any] = {
                "model": target.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "repeat_penalty": 1.0,
                    "num_ctx": CONTEXT_TOKENS,
                    "num_predict": max_tokens,
                    "stop": list(stop),
                },
            }
            try:
                return self._send(target, body, deadline)
            except _Rejected as exc:
                if "think" not in exc.detail.lower():
                    raise PolishError(exc.message) from None
                body.pop("think")  # older Ollama, or a model without a thinking switch
                return self._send(target, body, deadline)
        body = {
            "model": target.model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "max_tokens": max_tokens,
            "stop": list(stop),
        }
        if target.provider == "builtin":
            body.update(
                {
                    "top_k": 1,
                    "min_p": 0,
                    "repeat_penalty": 1.0,
                    "cache_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
            try:
                return self._send(target, body, deadline)
            except _Rejected as exc:
                raise PolishError(exc.message) from None
        # Any other OpenAI-compatible server: ask for thinking off, unless it already said it doesn't
        # accept the extra field (the real OpenAI API rejects unknown fields).
        base = target.url.rsplit("/chat/completions", 1)[0]
        with self._lock:
            extras = base not in self._no_extras
        if extras:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return self._send(target, body, deadline)
        except _Rejected as exc:
            if not extras:
                raise PolishError(exc.message) from None
            with self._lock:
                self._no_extras.add(base)
            body.pop("chat_template_kwargs", None)
            try:
                return self._send(target, body, deadline)
            except _Rejected as again:
                raise PolishError(again.message) from None

    def _send(self, target: _Target, body: dict[str, Any], deadline: float) -> _Reply:
        remaining = deadline - time.monotonic()
        timeout_s = self._settings.current.polish.timeout_seconds
        if remaining <= 0:
            raise PolishError(_timeout_message(timeout_s))
        headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
        timeout = httpx.Timeout(remaining, connect=min(_CONNECT_TIMEOUT_S, remaining))
        try:
            response = self._client(target.local).post(target.url, json=body, headers=headers, timeout=timeout)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Nothing is listening (Windows takes a couple of seconds to say so), or a secure connection
            # couldn't be made. Either way the next try would fail the same way, so stop trying for a while.
            message = _connection_problem(target.url, exc)
            self._mark_down(target.url, message)
            raise PolishError(message) from exc
        except httpx.TimeoutException as exc:
            raise PolishError(_timeout_message(timeout_s)) from exc
        except httpx.HTTPError as exc:
            raise PolishError(_connection_problem(target.url, exc)) from exc
        self._mark_up(target.url)
        if time.monotonic() > deadline:
            raise PolishError(_timeout_message(timeout_s))
        if response.status_code in (401, 403):
            raise PolishError("The AI server rejected the API key. Check it in Styles → AI polish.")
        if response.status_code == 404:
            if target.provider == "ollama" or "model" in _server_error(response).lower():
                raise PolishError(f"The AI server doesn't have the model “{target.model}”. Choose another in Styles.")
            raise PolishError(f"The AI server at {_host_of(target.url)} doesn't offer chat. Check its address.")
        if response.status_code in (400, 422):
            # The server's own error text can quote the request, so it is never shown or logged.
            message = f"The AI server rejected the request (HTTP {response.status_code}). Check the model in Styles."
            raise _Rejected(message, _server_error(response))
        if response.status_code >= 400:
            raise PolishError(f"The AI server had a problem (HTTP {response.status_code}).")
        try:
            data = response.json()
        except ValueError as exc:
            raise PolishError(MSG_UNREADABLE) from exc
        return _parse_reply(target.provider, data)

    # -- warm-up -----------------------------------------------------------------------------

    def _on_runtime_status(self, status: object) -> None:
        """When the built-in server becomes ready, load the fixed prompt prefixes into its cache (in the background)."""
        base_url = getattr(status, "base_url", None)
        state = getattr(getattr(status, "state", None), "value", None)
        if state != "ready" or not base_url:
            return
        with self._lock:
            if self._warmed == base_url:
                return
            self._warmed = base_url
        threading.Thread(target=self._warm_up, name="whisp-llm-warmup", daemon=True).start()

    def _warm_up(self) -> None:
        base, key = self._runtime.base_url(), self._runtime.api_key()
        if not base:
            return
        ctx = TextContext(AppCategory.OTHER, "clean", "", None, (), (), True, True, "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        started = time.monotonic()
        try:
            with httpx.Client(trust_env=False, timeout=_WARMUP_TIMEOUT_S) as client:
                for messages, stop in (
                    (cleanup_messages("Hello.", ctx), CLEANUP_STOP),
                    (command_messages("Hello.", "make it formal", ctx), COMMAND_STOP),
                ):
                    body = {
                        "messages": messages,
                        "max_tokens": 1,
                        "temperature": 0,
                        "stop": list(stop),
                        "cache_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    client.post(f"{base}/v1/chat/completions", json=body, headers=headers)
            log.info("Built-in AI prompt cache warmed in %d ms", _ms_since(started))
        except httpx.HTTPError as exc:
            log.info("Built-in AI warm-up skipped: %s", type(exc).__name__)


class _Rejected(Exception):
    """HTTP 400/422 from the server; ``detail`` is the server's own error text."""

    def __init__(self, message: str, detail: str) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


def _parse_reply(provider: str, data: Any) -> _Reply:
    try:
        if provider == "ollama":
            message = data.get("message") or {}
            finish = data.get("done_reason") or ("stop" if data.get("done") else None)
            generation_s = _duration(data.get("eval_duration"), 1e9)  # nanoseconds; load_duration is separate
            return _Reply(_message_text(message.get("content")), finish, data.get("eval_count"), generation_s)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}  # llama-server's own, in milliseconds
        return _Reply(
            _message_text(message.get("content")),
            choice.get("finish_reason"),
            usage.get("completion_tokens") or timings.get("predicted_n"),
            _duration(timings.get("predicted_ms"), 1e3),
        )
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise PolishError(MSG_UNREADABLE) from exc


def _duration(value: Any, per_second: float) -> float | None:
    """A server-reported duration in seconds, or None when it's missing or makes no sense."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        return None
    return value / per_second


def _timeout_message(seconds: float) -> str:
    return f"The AI took longer than {seconds:g} seconds, so the text was inserted without polish."


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


_REASON_CODES = {
    MSG_EMPTY: "empty",
    MSG_CUT_OFF: "cut_off",
    MSG_LENGTH: "length",
    MSG_ADDED: "new_words",
    MSG_ANSWERED: "answered",
    MSG_REPEATED: "repeated",
    MSG_LANGUAGE: "language",
}


def _reason_code(message: str) -> str:
    return _REASON_CODES.get(message, "other")
