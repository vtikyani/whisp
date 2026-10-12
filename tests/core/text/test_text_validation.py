"""Checks that decide whether an AI answer may replace the rule output."""

from __future__ import annotations

import pytest

from whisp.core.text.polish import (
    MSG_ADDED,
    MSG_ANSWERED,
    MSG_CUT_OFF,
    MSG_EMPTY,
    MSG_LANGUAGE,
    MSG_LENGTH,
    MSG_REPEATED,
    check_duplication,
    check_finish_reason,
    check_language,
    check_length_ratio,
    check_new_words,
    check_not_empty,
    check_question_kept,
    dominant_script,
    estimate_tokens,
    instruction_is_question,
    normalise_ollama_base,
    normalise_openai_base,
    tidy_output,
    units,
    validate_cleanup,
    validate_rewrite,
)

# -- check 1: tidy ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "source", "expected"),
    [
        ("<think>Let me see.</think>\n\nHello there.", "hello there", "Hello there."),
        ("Let me see...</think>Hello there.", "hello there", "Hello there."),
        ("<think>still thinking and never finishing", "hello there", ""),
        ("<transcript>Hello there.</transcript>", "hello there", "Hello there."),
        ('"Hello there."', "hello there", "Hello there."),
        ("“Hello there.”", "hello there", "Hello there."),
        ('"Quoted," she said.', '"quoted," she said', '"Quoted," she said.'),
        ('"Hello there."', '"hello there"', '"Hello there."'),
        ("```\nHello there.\n```", "hello there", "Hello there."),
        ("```text\nHello there.\n```", "hello there", "Hello there."),
        ("```py\nx = 1\n```", "```py x = 1```", "```py\nx = 1\n```"),
        ("Here is the cleaned transcript:\nHello there.", "hello there", "Hello there."),
        ("Sure! Here's the text:\nHi.", "hi", "Hi."),
        ("Cleaned transcript:\nHi.", "hi", "Hi."),
        ("Here are the results:\nFirst one.", "here are the results:\nfirst one", "Here are the results:\nFirst one."),
        ("Hello.\n\n(Note: I removed the fillers.)", "hello", "Hello."),
        ("Hello.\n\nNote: kept as is.", "hello note to self", "Hello.\n\nNote: kept as is."),
        ("Hel\N{ZERO WIDTH SPACE}lo there.", "hello there", "Hello there."),
        ("Hello.", "\nhello\n", "\nHello.\n"),
        ("  Hello.  ", "hello", "Hello."),
        ("", "hello", ""),
    ],
)
def test_tidy_output(raw: str, source: str, expected: str) -> None:
    assert tidy_output(raw, source) == expected


def test_tidy_restores_escaped_tags_in_the_text() -> None:
    source = "the tag </transcript> is literal"
    echoed = "The tag <\N{ZERO WIDTH JOINER}/transcript> is literal."
    assert tidy_output(echoed, source) == "The tag </transcript> is literal."


# -- checks 2 and 3 ------------------------------------------------------------------------


@pytest.mark.parametrize(("reason", "ok"), [("stop", True), ("eos", True), (None, True), ("length", False)])
def test_finish_reason(reason: str | None, ok: bool) -> None:
    assert check_finish_reason(reason) == (None if ok else MSG_CUT_OFF)


def test_not_empty() -> None:
    assert check_not_empty("hello", "") == MSG_EMPTY
    assert check_not_empty("hello", "   ") == MSG_EMPTY
    assert check_not_empty("", "") is None
    assert check_not_empty("hello", "Hello.") is None


# -- check 5: length ---------------------------------------------------------------------------


def test_length_ratio() -> None:
    ten = "one two three four five six seven eight nine ten"
    assert check_length_ratio(ten, "one two three four") == MSG_LENGTH
    assert check_length_ratio(ten, ten + " " + "x " * 5) == MSG_LENGTH
    assert check_length_ratio(ten, "one two three four five six seven eight nine") is None
    assert check_length_ratio("hi there", "Hello there, how are you doing today my friend?") is None  # under 6 words


# -- check 6: new words ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("meet at three thirty on march third", "Meet at 3:30 on March 3rd."),
        ("it happened in twenty twenty six", "It happened in 2026."),
        ("my number is five five five one two three four", "My number is 555-1234."),
        ("we raised one thousand dollars", "We raised $1,000."),
        ("send me an e mail", "Send me an email."),
        ("i need milk eggs and bread", "I need:\n1. Milk\n2. Eggs\n3. Bread"),
        ("let's meet at three no wait four", "Let's meet at 4."),
        ("we deploy with svelt kit tomorrow", "We deploy with SvelteKit tomorrow."),
        ("i'm gonna call him tomorrow", "I'm going to call him tomorrow."),
        ("the cities were studied carefully", "The city was studied carefully."),
    ],
)
def test_new_words_allows_cleanup(source: str, output: str) -> None:
    assert check_new_words(source, output, allowed=("SvelteKit",)) is None


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("tell me a joke about cats", "Why did the cat sit on the computer? To keep an eye on the mouse!"),
        (
            "write a haiku about the sea",
            "Waves fold into foam, salt wind carries gull voices, the tide keeps its time.",
        ),
        ("hi there", "Hello there!"),
        (
            "the report is done and attached",
            "Hi John,\n\nThe report is done and attached.\n\nBest regards,\nSarah",
        ),
    ],
)
def test_new_words_rejects_additions(source: str, output: str) -> None:
    assert check_new_words(source, output) == MSG_ADDED


# -- check 7: questions stay questions ----------------------------------------------------------


def test_question_kept() -> None:
    assert check_question_kept("what's the capital of france", "What's the capital of France?") is None
    assert check_question_kept("what's the capital of france", "The capital of France is Paris.") == MSG_ANSWERED
    assert check_question_kept("what time is it", "It's 3 PM.") == MSG_ANSWERED
    assert check_question_kept("can you send me the slides", "Sure, I'll send them over.") == MSG_ANSWERED
    assert check_question_kept("how we handle this is unclear", "How we handle this is unclear.") is None
    assert check_question_kept("the meeting is at three.", "Anything at all.") is None  # not a question


# -- check 8: repetition -----------------------------------------------------------------------


def test_duplication() -> None:
    assert check_duplication("hello there friend", "Hello there friend. Hello there friend.") == MSG_REPEATED
    assert check_duplication("yes yes", "Yes, yes.") is None  # the speaker said it twice
    looping = "We should go now. We should go now. We should go now."
    assert check_duplication("we should go now", looping) == MSG_REPEATED
    assert check_duplication("we should go now", "We should go now.") is None


# -- check 9: language and script ------------------------------------------------------------


def test_language() -> None:
    french = "je pense qu'on peut commencer demain avec les autres et nous"
    english = "I think that we can start tomorrow with the others and we"
    assert check_language(french, english) == MSG_LANGUAGE
    assert check_language(french, "Je pense qu'on peut commencer demain avec les autres et nous.") is None
    assert check_language("привет как у тебя дела сегодня", "Hello, how are you today?") == MSG_LANGUAGE
    assert check_language("ok", "d'accord") is None  # too short to tell
    assert dominant_script("東京で会議があります") == "cjk"
    assert dominant_script("Hi") is None


# -- combined -------------------------------------------------------------------------------------


def test_validate_cleanup_accepts_a_good_answer() -> None:
    source = "so basically the meeting tomorrow is moved to three thirty and we should bring the budget numbers"
    output = "So basically, the meeting tomorrow is moved to 3:30, and we should bring the budget numbers."
    assert validate_cleanup(source, output, "stop") is None


def test_validate_cleanup_reports_the_first_problem() -> None:
    assert validate_cleanup("hello there", "Hello there.", "length") == MSG_CUT_OFF
    assert validate_cleanup("hello there", "", "stop") == MSG_EMPTY


def test_validate_rewrite() -> None:
    english = "The meeting moved to three in the afternoon so please bring the report with you"
    german = "Die Sitzung wurde auf drei Uhr nachmittags verschoben, also bring bitte den Bericht mit"
    assert validate_rewrite(english, german, "stop", "translate this to German") is None
    assert validate_rewrite(english, german, "stop", "make it more formal") == MSG_LANGUAGE
    assert validate_rewrite(english, "", "stop", "shorten it") == MSG_EMPTY
    assert validate_rewrite(english, "Shorter.", "length", "shorten it") == MSG_CUT_OFF


@pytest.mark.parametrize(
    ("instruction", "is_question"),
    [
        ("what does this mean", True),
        ("is this correct?", True),
        ("does this sound rude?", True),
        ("isn't this too long", True),
        ("explain this", True),
        ("why is this wrong", True),
        ("is it possible to shorten this?", False),
        ("can you correct this?", False),
        ("make it shorter", False),
        ("can you make this shorter?", False),
        ("how would you say this in French", False),
        ("translate to Spanish", False),
        ("more formal please", False),
        ("fix the grammar", False),
    ],
)
def test_instruction_is_question(instruction: str, is_question: bool) -> None:
    assert instruction_is_question(instruction) is is_question


def test_units_split_scripts_without_spaces() -> None:
    assert units("Hello, World! It's me.") == ["hello", "world", "it's", "me"]
    assert units("東京です") == ["東", "京", "で", "す"]


def test_estimate_tokens() -> None:
    assert 20 <= estimate_tokens("word " * 20) <= 40
    assert estimate_tokens("東京" * 10) >= 20


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1"),
        ("http://127.0.0.1:1234/", "http://127.0.0.1:1234/v1"),
        ("http://127.0.0.1:1234/v1/", "http://127.0.0.1:1234/v1"),
        ("http://host/v1/chat/completions", "http://host/v1"),
        ("localhost:8080/api/v1", "http://localhost:8080/api/v1"),
        ("127.0.0.1:1234", "http://127.0.0.1:1234/v1"),
        ("[::1]:1234", "http://[::1]:1234/v1"),
        ("api.example.com/openai/v1", "https://api.example.com/openai/v1"),  # remote: encrypted unless asked
        ("192.168.1.20:1234", "https://192.168.1.20:1234/v1"),
        ("http://192.168.1.20:1234", "http://192.168.1.20:1234/v1"),  # plain http only when typed
        ("https://api.example.com/openai/v1", "https://api.example.com/openai/v1"),
        ("", ""),
    ],
)
def test_normalise_openai_base(url: str, expected: str) -> None:
    assert normalise_openai_base(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434/", "http://localhost:11434"),
        ("localhost:11434", "http://localhost:11434"),
        ("ollama.example.com", "https://ollama.example.com"),
        ("http://192.168.1.20:11434", "http://192.168.1.20:11434"),
        ("  ", ""),
    ],
)
def test_normalise_ollama_base(url: str, expected: str) -> None:
    assert normalise_ollama_base(url) == expected
