"""The deterministic text rules: voice commands, fillers, stutters, dictionary, snippets, styles, whitespace."""

from __future__ import annotations

import random
import re
import statistics
import time
from dataclasses import replace
from datetime import datetime

import pytest

from whisp.core.text.context import TextContext
from whisp.core.text.rules import (
    apply_rules,
    expand_variables,
    guess_latin_language,
    is_english,
    normalise_whitespace,
    word_set,
)
from whisp.core.types import AppCategory
from whisp.storage.settings import DictionaryEntry, Snippet

SVELTE = DictionaryEntry(term="SvelteKit", sounds_like=("svelte kit",))
KUBECTL = DictionaryEntry(term="kubectl", sounds_like=("cube control", "cube cuddle"))
DOTNET = DictionaryEntry(term=".NET", sounds_like=("dot net",))
DON = DictionaryEntry(term="Don")
CAL = Snippet(trigger="my calendar link", text="https://cal.com/sam-rivera")
ADDRESS = Snippet(trigger="my address", text="1 Main St\nSpringfield")
LINK = Snippet(trigger="my link", text="https://example.com/a")


def make_ctx(
    style: str = "clean",
    *,
    language: str | None = None,
    dictionary: tuple[DictionaryEntry, ...] = (),
    snippets: tuple[Snippet, ...] = (),
    fillers: bool = True,
    commands: bool = True,
) -> TextContext:
    return TextContext(
        category=AppCategory.OTHER,
        style=style,
        app_name="",
        language=language,
        dictionary=dictionary,
        snippets=snippets,
        remove_fillers=fillers,
        voice_commands=commands,
        custom_instructions="",
    )


CASES: list[tuple[str, str, str, dict]] = [
    # -- voice commands ------------------------------------------------------------------
    ("new line between sentences", "Hello there. New line. How are you?", "Hello there.\nHow are you?", {}),
    ("new line between commas", "Hello there, new line, how are you", "Hello there,\nHow are you", {}),
    ("new-line spelled with a hyphen", "Hello. New-line. Bye.", "Hello.\nBye.", {}),
    ("newline as one word at the start", "Newline. Done.", "\nDone.", {}),
    ("ordinary 'a new line of'", "Start a new line of products.", "Start a new line of products.", {}),
    ("ordinary 'new line items'", "New line items are listed below.", "New line items are listed below.", {}),
    ("ordinary 'a new paragraph'", "We need a new paragraph here.", "We need a new paragraph here.", {}),
    (
        "letter with paragraphs",
        "Dear John, new paragraph, I hope you're well. New paragraph. Thanks, new line, Mike.",
        "Dear John,\n\nI hope you're well.\n\nThanks,\nMike.",
        {},
    ),
    ("capitalised word after the command", "Hi. New line Thanks for coming.", "Hi.\nThanks for coming.", {}),
    ("proper noun New Line Cinema", "I loved New Line Cinema films.", "I loved New Line Cinema films.", {}),
    ("two commands in a row", "Hello. New line new line. Bye", "Hello.\n\nBye", {}),
    ("question mark moves before the break", "How are you, new line?", "How are you?\n", {}),
    ("list with line breaks", "Buy milk, new line, eggs, new line, bread.", "Buy milk,\nEggs,\nBread.", {}),
    (
        "scratch that after a sentence",
        "I'll be there at five. Scratch that. I'll be there at six.",
        "I'll be there at six.",
        {},
    ),
    ("scratch that mid-sentence", "I'll be there at five, scratch that, six.", "Six.", {}),
    (
        "scratch that keeps earlier sentences",
        "First point. Second point, scratch that, third point.",
        "First point. Third point.",
        {},
    ),
    ("scratch that twice", "One. Two. Scratch that. Scratch that. Three.", "Three.", {}),
    ("scratch that after a line", "Line one. New line. Line two. Scratch that.", "Line one.\n", {}),
    ("scratch that with nothing before", "Scratch that.", "", {}),
    ("ordinary 'scratch that itch'", "Please scratch that itch.", "Please scratch that itch.", {}),
    ("commands off", "Hello. New line. Bye.", "Hello. New line. Bye.", {"commands": False}),
    ("commands still work in verbatim", "hello world. new line. done", "hello world.\ndone", {"style": "verbatim"}),
    ("next line", "Item one. Next line. Item two.", "Item one.\nItem two.", {}),
    ("skip a line", "Hello. Skip a line. Bye.", "Hello.\n\nBye.", {}),
    ("ordinary 'the next line'", "Read the next line.", None, {}),
    ("ordinary 'skip a line between'", "Please skip a line between items.", None, {}),
    # -- spoken punctuation ------------------------------------------------------------------
    ("spoken comma, recogniser comma after", "hello comma, how are you period.", "Hello, how are you.", {}),
    (
        "recogniser comma before the spoken one",
        "Hello, comma, how are you question mark?",
        "Hello, how are you?",
        {},
    ),
    ("question mark at the end", "Is it ready question mark", "Is it ready?", {}),
    ("exclamation point", "Wow exclamation point!", "Wow!", {}),
    ("exclamation mark", "That's great exclamation mark", "That's great!", {}),
    ("full stop", "That is it full stop.", "That is it.", {}),
    ("colon continues the sentence", "Note colon. The budget is ready.", "Note: the budget is ready.", {}),
    ("semicolon", "I came, I saw semicolon, I conquered.", "I came, I saw; I conquered.", {}),
    ("dash between clauses", "He was there dash, and then he left.", "He was there — and then he left.", {}),
    ("dash at the end", "He was there dash.", "He was there —", {}),
    ("sentence mark before a capital", "See you then period Bye.", "See you then. Bye.", {}),
    ("two marks in a row", "Really question mark exclamation point", "Really?!", {}),
    ("mark then a line command", "I am done period new paragraph thanks", "I am done.\n\nThanks", {}),
    ("comma then a line command", "Dear Sam comma new line thanks for coming", "Dear Sam,\nThanks for coming", {}),
    ("no pause after the word", "Buy milk comma eggs and bread.", None, {}),
    ("ordinary 'trial period ended'", "The trial period ended.", None, {}),
    ("ordinary 'within the trial period'", "It is within the trial period.", None, {}),
    ("ordinary 'comma-separated'", "Send a comma-separated list.", None, {}),
    ("ordinary 'use a colon here'", "Use a colon here.", None, {}),
    ("emphatic period", "I'm not going, period.", None, {}),
    ("emphatic Period sentence", "I'm not going. Period.", None, {}),
    ("Period alone", "Period.", None, {}),
    ("ordinary 'a short period'", "It ended after a short period.", None, {}),
    ("ordinary 'a 30-day period'", "You have a 30-day period.", None, {}),
    ("ordinary 'a thirty day period'", "You have a thirty day period.", None, {}),
    ("ordinary 'a historical period'", "It was a historical period.", None, {}),
    ("ordinary 'a difficult period'", "We went through a difficult period.", None, {}),
    ("ordinary 'an intense period'", "It was an intense period.", None, {}),
    ("ordinary 'the prior period'", "Sales grew over the prior period.", None, {}),
    ("ordinary 'a final dash'", "He made a final dash.", None, {}),
    ("spoken period after a noun", "Send it to the team period.", "Send it to the team.", {}),
    ("ordinary 'a full stop'", "The car came to a full stop.", None, {}),
    ("ordinary 'big question mark'", "There's a big question mark over it.", None, {}),
    ("ordinary 'my colon'", "The doctor checked my colon.", None, {}),
    ("ordinary 'a dash of salt'", "Add a dash of salt.", None, {}),
    ("name Dash", "I met Dash yesterday. Thanks Dash.", None, {}),
    ("Oxford comma", "I love the Oxford comma.", None, {}),
    ("the word comma", "Say the word comma.", None, {}),
    ("verbatim keeps punctuation words", "hello comma, how are you period.", None, {"style": "verbatim"}),
    ("French 'colon' is a word", "C'est un colon.", None, {"language": "fr"}),
    ("French 'colon' when the model can't tell", "C'est un colon.", None, {}),
    ("Italian 'comma' is a word", "Si applica il secondo comma.", None, {"language": "it"}),
    ("spoken punctuation in English", "Hello comma, how are you period.", "Hello, how are you.", {"language": "en"}),
    ("editing commands work in every language", "Bonjour. New line. Merci.", "Bonjour.\nMerci.", {"language": "fr"}),
    ("commands off keeps punctuation words", "hello comma, how", "Hello comma, how", {"commands": False}),
    # -- fillers --------------------------------------------------------------------------
    ("filler at the start", "Um, so I think we should go.", "So I think we should go.", {}),
    (
        "fillers between commas",
        "I was, uh, thinking that, um, we could, uh, go.",
        "I was thinking that we could go.",
        {},
    ),
    ("list commas survive", "We need eggs, um, milk, and bread.", "We need eggs, milk, and bread.", {}),
    ("comma after yes survives", "Yes, um, we can.", "Yes, we can.", {}),
    ("filler before a full stop", "We should go, um.", "We should go.", {}),
    ("filler as its own sentence", "Okay. Um. Let's go.", "Okay. Let's go.", {}),
    ("repeated fillers", "Uh, uh, I think so.", "I think so.", {}),
    ("hesitation ellipsis", "I was uh... thinking about it.", "I was thinking about it.", {}),
    ("clause comma after a filler", "I like it um, but not much.", "I like it, but not much.", {}),
    ("ah and hmm and erm", "Hmm, let me think. Erm, yes. Ah, I see.", "Let me think. Yes. I see.", {}),
    ("quoted filler", '"Um, hello," she said.', '"Hello," she said.', {}),
    ("after an ellipsis", "Okay… um, so we start.", "Okay… so we start.", {}),
    ("only a filler", "Um.", "", {}),
    ("long hum", "Hmmmmmm.", "", {}),
    ("mm as a unit", "The screw is 5 mm long.", "The screw is 5 mm long.", {}),
    ("mm as a filler", "Mm, that's good.", "That's good.", {}),
    ("uh-huh as an answer", "Uh-huh. That works.", "Uh-huh. That works.", {}),
    ("uh-huh inside a sentence", "And then, uh-huh, we left.", "And then, we left.", {}),
    ("ER the abbreviation", "We went to the ER yesterday.", "We went to the ER yesterday.", {}),
    ("UM the abbreviation", "UM is a university.", "UM is a university.", {}),
    ("er as a filler", "Er, I'm not sure.", "I'm not sure.", {}),
    ("German um and er are words", "Um 5 Uhr ist er hier.", "Um 5 Uhr ist er hier.", {}),
    ("German with language set", "Um 5 Uhr ist er hier.", "Um 5 Uhr ist er hier.", {"language": "de"}),
    ("German filler", "Äh, ich weiß nicht.", "Ich weiß nicht.", {"language": "de"}),
    ("French filler", "Euh, je pense que oui.", "Je pense que oui.", {"language": "fr"}),
    ("Portuguese um is a word", "Eu comprei um carro novo.", "Eu comprei um carro novo.", {}),
    ("fillers off", "Um, so I think.", "Um, so I think.", {"fillers": False}),
    (
        "verbatim keeps fillers",
        "um so this is verbatim, uh, text",
        "um so this is verbatim, uh, text",
        {"style": "verbatim"},
    ),
    # -- stutters --------------------------------------------------------------------------
    ("stutters", "I I think the the meeting is fine.", "I think the meeting is fine.", {}),
    ("sentence-initial stutter", "The the plan works.", "The plan works.", {}),
    ("triple stutter", "I I I think so.", "I think so.", {}),
    ("pronoun stutter", "We we we should go.", "We should go.", {}),
    ("article stutter", "It was a a great day.", "It was a great day.", {}),
    ("had had", "He had had enough.", "He had had enough.", {}),
    ("that that", "I know that that is true.", "I know that that is true.", {}),
    ("is is", "What it is is a problem.", "What it is is a problem.", {}),
    ("her her and you you", "I gave her her keys and told you you were right.", None, {}),
    ("phone number words", "My number is five five five one two three.", None, {}),
    ("digits", "Room 4 4 is free.", None, {}),
    ("place names", "We visited Bora Bora last year.", None, {}),
    ("emphasis", "It's a very very big deal, bye bye.", None, {}),
    ("spelled letters", "The B B C reported it.", None, {}),
    ("French reflexive", "Nous nous sommes levés tôt.", None, {"language": "fr"}),
    # -- dictionary ------------------------------------------------------------------------
    ("variant", "I love svelte kit and its docs.", "I love SvelteKit and its docs.", {"dictionary": (SVELTE,)}),
    ("possessive", "sveltekit's router is nice.", "SvelteKit's router is nice.", {"dictionary": (SVELTE,)}),
    ("hyphenated variant", "Try svelte-kit today.", "Try SvelteKit today.", {"dictionary": (SVELTE,)}),
    ("not inside a contraction", "I don't like Don.", "I don't like Don.", {"dictionary": (DON,)}),
    ("multi-word variant", "We use cube control daily.", "We use kubectl daily.", {"dictionary": (KUBECTL,)}),
    ("lower-case term at sentence start", "Kubectl is great.", "kubectl is great.", {"dictionary": (KUBECTL,)}),
    ("term with punctuation", "I like dot net and C++.", "I like .NET and C++.", {"dictionary": (DOTNET,)}),
    (
        "longest match first",
        "Read the svelte kit docs now.",
        "Read the SvelteKit Docs now.",
        {"dictionary": (SVELTE, DictionaryEntry(term="SvelteKit Docs", sounds_like=("svelte kit docs",)))},
    ),
    ("no match inside words", "The rapid apis help.", None, {"dictionary": (DictionaryEntry(term="API"),)}),
    # -- snippets --------------------------------------------------------------------------
    ("whole utterance", "My calendar link.", "https://cal.com/sam-rivera", {"snippets": (CAL,)}),
    ("whole utterance without punctuation", "my calendar link", "https://cal.com/sam-rivera", {"snippets": (CAL,)}),
    (
        "inline trigger",
        "Here is my calendar link, thanks.",
        "Here is https://cal.com/sam-rivera, thanks.",
        {"snippets": (CAL,)},
    ),
    (
        "multi-line expansion untouched by formal style",
        "here is my address",
        "Here is 1 Main St\nSpringfield",
        {"style": "formal", "snippets": (ADDRESS,)},
    ),
    ("expansion never capitalised", "Sure. my link", "Sure. https://example.com/a", {"snippets": (LINK,)}),
    # -- styles -----------------------------------------------------------------------------
    ("clean capitals and i", "i think i'm ready. ok then.", "I think I'm ready. Ok then.", {}),
    ("abbreviation doesn't end a sentence", "Bring snacks, e.g. chips and dips.", None, {}),
    ("casual drops a single period", "sounds good.", "Sounds good", {"style": "casual"}),
    ("casual keeps two sentences", "Sounds good. See you soon.", None, {"style": "casual"}),
    ("casual keeps a question mark", "Are you coming?", None, {"style": "casual"}),
    ("casual keeps an exclamation mark", "Great!", None, {"style": "casual"}),
    ("casual keeps an abbreviation", "See you at 5 p.m.", None, {"style": "casual"}),
    ("formal adds a period", "thanks so much for the help", "Thanks so much for the help.", {"style": "formal"}),
    (
        "formal email lines",
        "Hi Sarah,\n\nI need the budget\n\nThanks,\nMike",
        "Hi Sarah,\n\nI need the budget.\n\nThanks,\nMike",
        {"style": "formal"},
    ),
    ("formal leaves short lines", "Call me", None, {"style": "formal"}),
    ("verbatim changes nothing", "hello world", "hello world", {"style": "verbatim"}),
    # -- whitespace -------------------------------------------------------------------------
    ("spaces before punctuation", "Hello , world . How are  you ?", "Hello, world. How are you?", {}),
    ("French keeps high punctuation spacing", "Bonjour ! Ça va ?", None, {"language": "fr"}),
    ("line breaks are kept", "First line\nsecond line", "First line\nSecond line", {}),
    ("empty", "   ", "", {}),
]


@pytest.mark.parametrize(("name", "raw", "expected", "options"), CASES, ids=[c[0] for c in CASES])
def test_rules(name: str, raw: str, expected: str | None, options: dict) -> None:
    ctx = make_ctx(**options)
    assert apply_rules(raw, ctx) == (raw if expected is None else expected)


def test_carriage_returns_and_invisible_characters_are_cleaned() -> None:
    raw = "Hello\r\nworld\N{ZERO WIDTH SPACE}.\N{NO-BREAK SPACE}Bye."
    assert apply_rules(raw, make_ctx()) == "Hello\nWorld. Bye."


def test_whole_snippet_is_inserted_exactly() -> None:
    snippet = Snippet(trigger="sign off", text="Best,\n  Sam")
    assert apply_rules("Sign off.", make_ctx(snippets=(snippet,))) == "Best,\n  Sam"


# -- snippet triggers and variables ---------------------------------------------------------

MONDAY = datetime(2026, 10, 12, 15, 45)


def test_overlapping_triggers_longest_wins() -> None:
    short = Snippet(trigger="my link", text="SHORT")
    long = Snippet(trigger="my link two", text="LONG")
    ctx = make_ctx(snippets=(short, long))
    assert apply_rules("Here is my link two now.", ctx) == "Here is LONG now."
    assert apply_rules("Here is my link now.", ctx) == "Here is SHORT now."
    reversed_order = make_ctx(snippets=(long, short))
    assert apply_rules("Here is my link two now.", reversed_order) == "Here is LONG now."


@pytest.mark.parametrize(
    "text",
    ["My signature is long.", "Use sig2 here.", "Use 2sig here.", "Use sig_x here.", "Mysig."],
)
def test_trigger_glued_to_letters_or_digits_does_not_fire(text: str) -> None:
    ctx = make_ctx(snippets=(Snippet(trigger="sig", text="Best, Sam"), Snippet(trigger="my sig", text="X")))
    assert apply_rules(text, ctx) == text


def test_text_after_a_snippet_is_capitalised_only_when_the_snippet_ends_a_sentence() -> None:
    sign_off = Snippet(trigger="sign off", text="Talk soon.")
    ctx = make_ctx(snippets=(LINK, sign_off))
    assert apply_rules("my link is below", ctx) == "https://example.com/a is below"
    assert apply_rules("thanks sign off see you", ctx) == "Thanks Talk soon. See you"


def test_trigger_next_to_punctuation_fires() -> None:
    ctx = make_ctx(snippets=(Snippet(trigger="sig", text="Best, Sam"),))
    assert apply_rules("Add (sig), then send.", ctx) == "Add (Best, Sam), then send."


@pytest.mark.parametrize(
    ("template", "now", "expected"),
    [
        ("{date}", MONDAY, "October 12, 2026"),
        ("{time}", MONDAY, "3:45 PM"),
        ("{time}", datetime(2026, 1, 5, 0, 5), "12:05 AM"),
        ("{time}", datetime(2026, 1, 5, 12, 0), "12:00 PM"),
        ("{time}", datetime(2026, 1, 5, 9, 7), "9:07 AM"),
        ("{weekday}", MONDAY, "Monday"),
        ("{year}", MONDAY, "2026"),
        ("{DATE} and {Weekday}", MONDAY, "October 12, 2026 and Monday"),
        ("Sent {weekday}, {date} at {time}", MONDAY, "Sent Monday, October 12, 2026 at 3:45 PM"),
        ("{name} and {date", MONDAY, "{name} and {date"),  # unknown or unfinished: literal
        ("{ date }", MONDAY, "{ date }"),
        ("{{date}}", MONDAY, "{date}"),  # escaped braces
        ("{{{date}}}", MONDAY, "{October 12, 2026}"),
        ("{{ and }}", MONDAY, "{ and }"),
        ("no variables", MONDAY, "no variables"),
    ],
)
def test_expand_variables(template: str, now: datetime, expected: str) -> None:
    assert expand_variables(template, now) == expected


def test_snippet_variables_are_filled_in() -> None:
    sign_off = Snippet(trigger="sign off", text="Thanks,\nSam ({date})")
    today = Snippet(trigger="today's date", text="{weekday}, {date}")
    ctx = replace(make_ctx(snippets=(sign_off, today)), now=MONDAY)
    assert apply_rules("Sign off.", ctx) == "Thanks,\nSam (October 12, 2026)"
    assert apply_rules("Today is today's date.", ctx) == "Today is Monday, October 12, 2026."


def test_snippet_variables_without_a_time_use_the_clock() -> None:
    ctx = make_ctx(snippets=(Snippet(trigger="this year", text="{year}"),))
    assert apply_rules("This year.", ctx) == str(datetime.now().year)


def test_language_detection() -> None:
    assert is_english("en", "Um 5 Uhr")
    assert not is_english("de", "I think so")
    assert is_english(None, "Um, yes.")  # no evidence of another language
    assert not is_english(None, "Um 5 Uhr ist er hier und wir sind da.")
    assert not is_english(None, "Привет, как дела?")
    assert guess_latin_language("Je pense que nous pouvons commencer demain avec les autres") == "fr"
    assert guess_latin_language("I think that we can start tomorrow with the others") == "en"
    assert guess_latin_language("Hello") is None


def test_normalise_whitespace_keeps_leading_dot_terms() -> None:
    assert normalise_whitespace("I like .NET and  Node.js .") == "I like .NET and Node.js."


# -- never invent words ----------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")


def _words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD_RE.findall(text)}


_VOCABULARY = [
    *sorted(
        word_set(
            "um uh erm er ah hmm mm uh-huh new next line paragraph skip scratch that the a I we had is was it you "
            "know svelte kit cube control meeting tomorrow at five six send me slides please thanks comma period "
            "full stop question mark exclamation point colon semicolon dash trial"
        )
    ),
    ",",
    ".",
    "?",
    "!",
    "...",
    "-",
]


@pytest.mark.parametrize("style", ["verbatim", "clean", "casual", "formal"])
def test_never_invents_words(style: str) -> None:
    rng = random.Random(1234 + len(style))
    ctx = make_ctx(style, dictionary=(SVELTE, KUBECTL), snippets=(CAL,))
    allowed = _words(" ".join([SVELTE.term, KUBECTL.term, CAL.text]))
    for _ in range(400):
        tokens = [rng.choice(_VOCABULARY) for _ in range(rng.randint(1, 25))]
        raw = " ".join(tokens)
        out = apply_rules(raw, ctx)
        invented = _words(out) - _words(raw) - allowed
        assert not invented, f"{raw!r} -> {out!r} invented {invented}"


def test_never_invents_words_in_the_table() -> None:
    for _name, raw, _expected, options in CASES:
        ctx = make_ctx(**options)
        allowed = set()
        for entry in ctx.dictionary:
            allowed |= _words(entry.term)
        for snippet in ctx.snippets:
            allowed |= _words(snippet.text)
        out = apply_rules(raw, ctx)
        assert not (_words(out) - _words(raw) - allowed), (raw, out)


# -- speed ---------------------------------------------------------------------------------------

_TYPICAL = (
    "Um, so I think, uh, we should go to the the store tomorrow and buy eggs, milk, and bread. New line. "
    "Then we can meet John at five, scratch that, at six, and talk about the svelte kit project and the "
    "cube control setup. I I think it's going to be a a long day, but, um, that's fine. New paragraph. "
    "Let me know if that works for you and whether the budget numbers are ready, because the meeting "
    "with the finance team was moved to Thursday afternoon and we still need to finish the slides."
)


def test_typical_dictation_takes_a_couple_of_milliseconds() -> None:
    dictionary = tuple(DictionaryEntry(term=f"Term{i}", sounds_like=(f"term {i}",)) for i in range(50))
    ctx = make_ctx(dictionary=(SVELTE, KUBECTL, *dictionary), snippets=(CAL, ADDRESS))
    assert 90 <= len(_TYPICAL.split()) <= 160
    apply_rules(_TYPICAL, ctx)  # compile and cache the dictionary pattern
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        apply_rules(_TYPICAL, ctx)
        timings.append(time.perf_counter() - started)
    # Typically about 1 ms; the bound leaves room for a busy test machine.
    assert statistics.median(timings) < 0.010
