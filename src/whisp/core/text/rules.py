"""Deterministic text rules: the fast, predictable clean-up applied to every dictation.

``apply_rules(raw, ctx)`` runs these steps in order:

1. **Voice commands** (when ``ctx.voice_commands``).

   *Editing commands* work in every style: "new line"/"next line" → line
   break, "new paragraph"/"skip a line" → blank line, "scratch that" → delete
   the sentence just dictated. They only fire when they stand on their own as
   a clause: punctuation, a line break or the start/end of the text on both
   sides ("Thanks. New paragraph. Bye." fires; "start a new line of products"
   does not). After a sentence end, one followed by a capitalised word also
   fires ("Hi. New line Thanks.").

   *Spoken punctuation* ("comma", "period"/"full stop", "question mark",
   "exclamation point"/"exclamation mark", "colon", "semicolon", "dash") works
   in English text only (French "colon" and Italian "comma" are real words),
   in every style except verbatim, where these words stay text (verbatim is
   the default for code, where "dash" or "colon" is often meant literally).
   The language is the one the speech model reports: the language setting, an
   English-only model, or Whisper's own detection. When it can't tell (Parakeet
   v3 on auto), the text's common words decide, and text with no sign of
   another language counts as English, since the command words are English.
   The recogniser usually adds its own punctuation where the speaker paused,
   so the spoken word arrives as "hello comma, how are you period." A
   punctuation word fires only when all of these hold:

   - something comes before it: a word, or the recogniser's own mark after a
     word ("Hello, comma, how" works too; the duplicate mark is dropped);
   - it is followed by a pause (the recogniser's punctuation), the end of the
     text, a line break or another command. A sentence mark ("period",
     "question mark", "exclamation point") may instead be followed by a
     capitalised word that starts the next sentence;
   - the word before it doesn't make it a noun: not an article, possessive,
     quantifier or preposition ("a comma", "my colon", "of"), not a verb of
     saying or writing ("the word comma", "type period"), and not a typical
     modifier ("Oxford comma", "trial period", "em dash", "a 30-day period",
     "the probationary period"); "period" and "dash" also stay words after
     "a"/"an" and one more word ("a difficult period", "a final dash");
   - it isn't capitalised in mid-sentence (a name: "Dash", "Colon");
   - "period" also doesn't follow the recogniser's own punctuation, which keeps
     the emphatic "I'm not going, period." as written.

   So "the trial period ended", "a comma-separated list" and "use a colon
   here" stay intact. Punctuation the recogniser put right next to a spoken
   mark is replaced by it. "dash" becomes a spaced em dash (" — "). After a
   spoken comma, colon, semicolon or dash, a common word the recogniser
   capitalised ("Note colon. The budget") goes back to lower case.
2. **Fillers** (when ``ctx.remove_fillers`` and the style isn't verbatim): um,
   uh, erm, er, ah, hmm, mm… are removed with their commas, and the next word
   is capitalised when the filler started the sentence ("Um, so I think" →
   "So I think"). "mm" after a number is a unit, and "uh-huh"/"mm-hmm" at the
   start of a sentence mean "yes", so those stay. English-only fillers (um, er,
   ah) are skipped for other languages, where they can be real words (German
   "um 5 Uhr", "er ist", Portuguese "um carro").
3. **Stutters** (same switch, English only): "I I think" → "I think". Doubles
   that are often correct are never collapsed: "had had", "that that", "is is"
   ("what it is is…"), "was was", "do do", "her her", "you you" ("I told you
   you were right"), emphasis ("very very", "bye bye", "no no"), numbers and
   spelled-out letters ("five five five", "B B C"), and capitalised repeats
   ("Bora Bora").
4. **Dictionary**: every ``sounds_like`` variant and the term itself, matched
   case-insensitively on word boundaries (longest first), becomes the
   canonical spelling. Adjacent punctuation is kept. It runs again after the
   style step, so a capital at the start of a sentence never overrides a
   term's spelling ("kubectl", "iPhone").
5. **Snippets**: an utterance that is only a trigger (ignoring case and
   punctuation) becomes the snippet text exactly. A trigger inside a longer
   utterance is replaced in place (overlapping triggers: the longest wins; a
   trigger glued to letters or digits doesn't fire), and later steps never
   touch the inserted text. Snippet text may use ``{date}``, ``{time}``,
   ``{weekday}`` and ``{year}``, filled in with the dictation's time
   (``ctx.now``); see ``expand_variables``.
6. **Style**: verbatim changes nothing; clean capitalises sentence starts and a
   lone "i"; casual also drops a single trailing period from a one-sentence
   message (keeping "?" and "!"); formal also ends sentences with terminal
   punctuation.
7. **Whitespace**: no double spaces, no space before , . ! ? : ; (French keeps
   its space before ! ? : ;), no spaces around line breaks. Line breaks stay.

The rules only ever remove words, change letter case, add punctuation, or put
in the user's own dictionary terms and snippets. They never invent words.
Pure and fast: no I/O, no locks, a couple of milliseconds for long texts.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from datetime import datetime

from whisp.core.text.context import TextContext
from whisp.storage.settings import DictionaryEntry, Snippet

# -- tokens -------------------------------------------------------------------

_NL, _SPACE, _WORD, _PUNCT = 0, 1, 2, 3

_TOKEN_RE = re.compile(
    r"(?P<nl>\n)"
    r"|(?P<space>[^\S\n]+)"
    r"|(?P<word>\w(?:[\w'’]|[-.](?=\w))*)"
    r"|(?P<punct>\.{2,}|[^\w\s])"
)
_KIND = {"nl": _NL, "space": _SPACE, "word": _WORD, "punct": _PUNCT}


def _tokenize(text: str) -> list[tuple[int, str]]:
    return [(_KIND[m.lastgroup], m.group()) for m in _TOKEN_RE.finditer(text)]  # type: ignore[index]


_TERMINATORS = frozenset({".", "!", "?", "?!", "!?"})
_ELLIPSES = frozenset({"…"})
_CLAUSE_PUNCT = frozenset({",", ";", ":", "-", "—", "–"})


def _is_terminator(tok: str) -> bool:
    return tok in _TERMINATORS


def _is_ellipsis(tok: str) -> bool:
    return tok in _ELLIPSES or (len(tok) >= 2 and set(tok) == {"."})


def word_set(text: str) -> frozenset[str]:
    """The space-separated words of ``text`` as a set (keeps long word lists readable)."""
    return frozenset(text.split())


# -- language -------------------------------------------------------------------

# Function words used to tell English from the other common Latin-script
# languages when the speech model didn't report a language. Words shared with
# English ("a", "in", "is", "was", "so", "on", "me") and the fillers that are
# real words elsewhere ("um", "er") are left out on purpose.
_EN_WORDS = word_set(
    "the and to of you we they he she it that this for with have has not what but my your our can just if at "
    "there about would should could i it's i'm don't yes yeah okay please thanks think know going want need "
    "we're they're you're that's let's are were been from which when how why who all some more very really "
    "because maybe"
)
_OTHER_WORDS: dict[str, frozenset[str]] = {
    "de": word_set(
        "der die das und ist nicht ich du wir sie ein eine einen mit auf für zu von den dem des es sich auch noch "
        "aber wenn dass wie hier kann haben sein bin sind war uhr oder nur schon jetzt mal heute morgen ja nein "
        "bitte danke"
    ),
    "fr": word_set(
        "le la les et est je tu nous vous ils elle il un une des du pas que qui dans pour avec sur ce cette mais "
        "ou bien très oui non merci c'est j'ai qu'on demain aujourd'hui avoir être fait peut suis sont"
    ),
    "es": word_set(
        "el la los las y es que de en un una por para con no sí pero como más muy yo tú nosotros está están "
        "esto eso hola gracias mañana hoy porque cuando también hay tengo tiene del al"
    ),
    "it": word_set(
        "il lo la gli le e è che di un una per con non sono ma come più molto io tu noi questo quello ciao "
        "grazie domani oggi perché quando anche ho ha del della nel"
    ),
    "pt": word_set(
        "o os as e é que de uma para com não sim mas como mais muito eu você nós isso esse essa obrigado "
        "obrigada amanhã hoje porque quando também tem tenho está estou do da dos das no na"
    ),
    "nl": word_set(
        "de het een en is niet dat ik je jij we wij zij van voor met op te maar ook nog wel dan als hoe wat "
        "hier morgen vandaag dank bedankt alstublieft graag ja nee zijn heb heeft"
    ),
}
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏ]")
_LETTER_RE = re.compile(r"[^\W\d_]")
_GUESS_CHARS = 3000
"""The language guess looks at this much of the text; more adds time, not accuracy."""


def _function_word_hits(text: str) -> dict[str, int]:
    words = [w.lower().replace("’", "'") for kind, w in _tokenize(text[:_GUESS_CHARS]) if kind == _WORD]
    hits = {"en": sum(1 for w in words if w in _EN_WORDS)}
    for code, vocab in _OTHER_WORDS.items():
        hits[code] = sum(1 for w in words if w in vocab)
    return hits


def is_mostly_latin(text: str) -> bool:
    """True when at least half the letters in ``text`` are Latin (or there are no letters)."""
    text = text[:_GUESS_CHARS]
    letters = _LETTER_RE.findall(text)
    return not letters or len(_LATIN_RE.findall(text)) >= 0.5 * len(letters)


def guess_latin_language(text: str, *, min_hits: int = 3) -> str | None:
    """Best guess among en/de/fr/es/it/pt/nl from common function words, or None when unsure.

    "Sure" means at least ``min_hits`` function words and at least twice as many
    as the runner-up.
    """
    if not is_mostly_latin(text):
        return None
    ranked = sorted(_function_word_hits(text).items(), key=lambda kv: kv[1], reverse=True)
    (best, best_hits), (_, second_hits) = ranked[0], ranked[1]
    if best_hits >= min_hits and best_hits >= 2 * second_hits:
        return best
    return None


def is_english(language: str | None, text: str) -> bool:
    """True if ``text`` should get the English-only rules.

    Uses the language the speech model reported; without one, guesses from
    common function words (mostly-Latin text with no evidence of another
    language counts as English).
    """
    if language:
        return language.lower().split("-")[0] == "en"
    if not is_mostly_latin(text):
        return False
    hits = _function_word_hits(text)
    en_hits = hits.pop("en")
    return en_hits >= max(hits.values())


# -- 1. voice commands --------------------------------------------------------------

_S = r"[^\S\n]+"  # spaces inside a spoken phrase
_COMMAND_RE = re.compile(
    r"(?<![\w'’-])(?:"
    rf"(?P<newline>(?:new|next)(?:[^\S\n]*-[^\S\n]*|[^\S\n]*)(?P<line>line))"
    rf"|(?P<paragraph>new{_S}(?P<para>paragraph)|skip{_S}a{_S}(?P<skip>line))"
    rf"|(?P<scratch>scratch{_S}(?P<that>that))"
    rf"|(?P<mark>comma|period|full{_S}stop|question{_S}mark|exclamation{_S}(?:point|mark)|semicolon|colon|dash)"
    r")(?![\w'’-])",
    re.IGNORECASE,
)
_BETWEEN_COMMANDS_RE = re.compile(r"[\s,.;:!?…—–]*")
_LEFT_BOUNDARY = frozenset(",.;:!?…—–\n")
_STRONG_LEFT = frozenset(".!?…\n")
_RECOGNISER_PUNCT = ",.;:!?…—–"
_AFTER_COMMAND_RE = re.compile(r"[^\S\n]*(?P<punct>[.,;:!?…—–]+)?")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?:(?<=[.!?…])[\"'”’)\]]*[^\S\n]+|\n)")
_LAST_WORD_RE = re.compile(r"(?<![\w'’-])([\w'’]+(?:-[\w'’]+)*)$")
_NEXT_WORD_RE = re.compile(r"[^\S\n]*([A-Za-z]+(?:'[a-z]+)?)")

# Spoken punctuation. "dash" always becomes a spaced em dash (" — "): words joined
# by a hyphen arrive from the recogniser already joined, so a spoken "dash" is
# nearly always the pause-like dash between clauses.
_MARK_KINDS = {
    "comma": "comma",
    "period": "period",
    "full stop": "period",
    "question mark": "question",
    "exclamation point": "exclamation",
    "exclamation mark": "exclamation",
    "colon": "colon",
    "semicolon": "semicolon",
    "dash": "dash",
}
_MARK_TEXT = {
    "comma": ",",
    "period": ".",
    "question": "?",
    "exclamation": "!",
    "colon": ":",
    "semicolon": ";",
    "dash": " — ",
}
_SENTENCE_MARKS = frozenset({"period", "question", "exclamation"})
_CLAUSE_MARKS = frozenset({"comma", "colon", "semicolon", "dash"})

# Words after which a punctuation word is a noun or is being talked about, not
# spoken punctuation: "a comma", "the period", "my colon", "of", "the word comma".
_NOUN_CONTEXT = word_set(
    "a an the this that these those each every any some no another one which what whose my your his her its our "
    "their per of in on at to for with from by about during within over after before until till since through "
    "throughout into onto between among like than word words called named say said says saying type typed typing "
    "spell spelled spelt literal literally insert inserting add adding remove removing delete use using missing "
    "extra"
)
_MARK_MODIFIERS: dict[str, frozenset[str]] = {
    "comma": word_set(
        "oxford serial harvard inverted stray misplaced final trailing leading decimal double single upside-down"
    ),
    "period": word_set(
        "trial grace waiting cooling billing notice probation reporting holiday rest transition recovery payment "
        "warranty test testing school class free initial extended menstrual refractory interwar postwar prewar "
        "early late same short long brief given certain specific limited comment review exclusivity lockup "
        "lock-up vesting evaluation assessment accounting fiscal financial tax election campaign quiet blackout "
        "settling honeymoon adjustment incubation gestation sample sampling study observation follow-up lunch prep "
        "planning homeroom first second third fourth fifth sixth seventh eighth last next previous entire whole "
        "open closing opening remaining following preceding intervening qualifying rolling trailing measurement "
        "operating booking cooldown cool-down ramp-up peak off-peak busy slow dry wet rainy warm cold hot contract "
        "lease rental subscription retention holding stability warmup warm-up burn-in break-in adaptation safe "
        "safety lock run running sentencing return refund cancellation enrollment enrolment registration "
        "application submission voting polling nesting breeding mating growing flowering dormant latency latent "
        "ice stone bronze iron renaissance baroque romantic modern edwardian georgian tudor cretaceous prior current "
        "comparable relevant specified interim recent corresponding equivalent allotted designated applicable"
    ),
    "question": word_set("big huge giant little small real serious large massive"),
    "exclamation": word_set("big huge little small double triple single"),
    "colon": word_set("sigmoid ascending descending transverse spastic irritable inflamed healthy double"),
    "dash": word_set("em en hundred yard metre meter dot quick mad short long double single"),
}
_ARTICLE_NOUN_KINDS = frozenset({"period", "dash"})
"""After "a"/"an" and one more word these are nouns: "a difficult period", "an intense period", "a final dash"."""
_PERIOD_UNITS = word_set("second minute hour day week fortnight month quarter semester term year decade century time")
_ADJECTIVE_ENDINGS = ("al", "ary", "ory", "ic", "ian", "ive")
"""Adjectives before "period" ("a historical period", "the probationary period") make it a noun."""
_LOWER_AFTER_CLAUSE_MARK = word_set(
    "the a an and but or so then how what why when where who which we you they he she it this that there here my "
    "your our their his her its if because as to in on at for with from by of not no yes please thanks thank let's "
    "let can could would should will do does did is are was were have has had"
)
"""Common words the recogniser capitalised after a pause; after a spoken comma or colon they continue the sentence."""


def _last_word(text: str) -> str:
    match = _LAST_WORD_RE.search(text)
    return match.group(1) if match else ""


def _is_noun_context(word: str, kind: str) -> bool:
    """True when ``word`` before a punctuation word makes that word a noun ("the period", "trial period")."""
    lw = word.lower().replace("’", "'")
    if lw in _NOUN_CONTEXT or lw in _MARK_MODIFIERS.get(kind, frozenset()):
        return True
    if kind in ("period", "dash") and (any(c.isdigit() for c in lw) or lw in _NUMBER_WORDS):
        return True  # "a 30-day period", "the 100-meter dash"
    # "a two-week period", "the probationary period"
    return kind == "period" and (lw in _PERIOD_UNITS or (len(lw) > 4 and lw.endswith(_ADJECTIVE_ENDINGS)))


def _mark_fires(
    match: re.Match[str], kind: str, left_text: str, *, after_command: bool, pause_after: bool, rest: str
) -> bool:
    """Whether a spoken punctuation word is a command here (see ``apply_rules``)."""
    phrase = match.group("mark")
    left_char = left_text[-1:]
    if not after_command:
        if not left_text or left_char == "\n":
            return False  # nothing to punctuate ("Period." alone is a word)
        if phrase[0].isupper() and left_char not in ".!?…":
            return False  # a capitalised word mid-sentence is a name ("Dash", "Colon")
        if left_char in _RECOGNISER_PUNCT:
            if kind == "period":
                return False  # "I'm not going, period." is the emphatic idiom
            left_text = left_text.rstrip(_RECOGNISER_PUNCT + " \t")
        word = _last_word(left_text)
        if not word or _is_noun_context(word, kind):
            return False
        if kind in _ARTICLE_NOUN_KINDS and _last_word(left_text[: -len(word)].rstrip(" \t")).lower() in ("a", "an"):
            return False  # "a difficult period", "a final dash": the article makes it a noun
    if pause_after:
        return True
    # No pause after it: only a sentence mark followed by a new, capitalised sentence counts.
    return kind in _SENTENCE_MARKS and rest.lstrip(" \t")[:1].isupper()


def _sentence_start(text: str) -> int:
    """Index where the last sentence (or line) in ``text`` starts."""
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        if match.end() < len(text):
            start = match.end()
    return start


def _scratch(out: str) -> str:
    """Delete the sentence just dictated from ``out`` (see "scratch that")."""
    body = out.rstrip()
    trimmed = body.rstrip(",;:—– \t")
    if not trimmed.strip():
        return ""
    if trimmed[-1] in ".!?…" or trimmed.endswith(('"', "”", "'", "’")):
        # The command follows a finished sentence: remove that sentence.
        cut = _sentence_start(trimmed[:-1])
    else:
        # The command interrupts a sentence: remove what was said of it so far.
        cut = _sentence_start(trimmed)
    return trimmed[:cut]


def _apply_voice_commands(text: str, *, punctuation: bool) -> str:
    """Line commands, "scratch that" and (when ``punctuation``) spoken punctuation. See ``apply_rules``."""
    matches = [m for m in _COMMAND_RE.finditer(text) if punctuation or not m.group("mark")]
    if not matches:
        return text
    out = ""
    pos = 0
    fired_end = -1  # where the last command that fired ended, punctuation included
    for index, match in enumerate(matches):
        if match.start() < pos:
            continue
        before = (out + text[pos : match.start()]).rstrip(" \t")
        after_command = fired_end >= 0 and _BETWEEN_COMMANDS_RE.fullmatch(text[fired_end : match.start()]) is not None
        after = _AFTER_COMMAND_RE.match(text, match.end())
        assert after is not None  # the pattern can match the empty string
        rest = text[after.end() :]
        following = matches[index + 1] if index + 1 < len(matches) else None
        before_command = following is not None and bool(
            _BETWEEN_COMMANDS_RE.fullmatch(text[match.end() : following.start()])
        )
        pause_after = after.group("punct") is not None or not rest.strip(" \t") or rest.startswith("\n")
        if match.group("mark"):
            kind = _MARK_KINDS[" ".join(match.group("mark").lower().split())]
            if not _mark_fires(
                match, kind, before, after_command=after_command, pause_after=pause_after or before_command, rest=rest
            ):
                continue
            if not after_command and before[-1:] in _RECOGNISER_PUNCT:
                # "Hello, comma, how": the recogniser's own mark is replaced by the spoken one.
                before = before.rstrip(_RECOGNISER_PUNCT + " \t")
            out = before + _MARK_TEXT[kind]
            pos = after.end()  # the recogniser's mark after the spoken one is dropped too
            while pos < len(text) and text[pos] in " \t":
                pos += 1
            if pos < len(text) and text[pos] != "\n" and not out.endswith(" "):
                out += " "  # "Hello, how", not "Hello,how"
            if kind in _CLAUSE_MARKS:
                # "Note colon. The budget" -> "Note: the budget": the sentence goes on.
                word = _NEXT_WORD_RE.match(text, pos)
                if word and word.group(1).lower() in _LOWER_AFTER_CLAUSE_MARK and word.group(1).istitle():
                    start = word.start(1)
                    text = text[:start] + text[start].lower() + text[start + 1 :]
            fired_end = after.end()
            continue
        left = before[-1:] if before else ""
        left_ok = not before or left in _LEFT_BOUNDARY or after_command
        # Right after another command ("done period new paragraph thanks") the speaker is clearly dictating commands.
        right_ok = pause_after or before_command or after_command
        if not right_ok and (not before or left in _STRONG_LEFT):
            # "Hi. New line Thanks." - accept a capitalised next word, but not "New Line Cinema".
            tail_word = match.group("line") or match.group("para") or match.group("skip") or match.group("that")
            right_ok = tail_word.islower() and rest.lstrip(" \t")[:1].isupper()
        if not (left_ok and right_ok):
            continue
        out = before
        trailing = after.group("punct") or ""
        question = next((c for c in trailing if c in "?!"), "")
        if question and not match.group("scratch"):
            # "How are you, new line?" - the question mark belongs to the sentence before.
            stripped = out.rstrip(",;: \t")
            if stripped and stripped[-1] not in ".!?…\n":
                out = stripped + question
        if match.group("newline"):
            out += "\n"
        elif match.group("paragraph"):
            out += "\n\n"
        else:
            out = _scratch(out)
        pos = after.end()
        # The text after a command starts a new line or sentence: drop the space before it.
        while pos < len(text) and text[pos] in " \t":
            pos += 1
        fired_end = after.end()
    return out + text[pos:]


# -- 2. fillers ---------------------------------------------------------------------

_UNIVERSAL_FILLER_RE = re.compile(r"(?:h+m+|m{3,}|u+h+|u+h+m+|e+h+m+|ä+h+m*|ö+h+m*|e+u+h+|h+e+u+)", re.IGNORECASE)
"""Fillers that aren't words in any language Whisp is likely to hear."""
_EN_FILLER_RE = re.compile(r"(?:u+m+|e+r+m+|er|a+h+)", re.IGNORECASE)
"""English fillers that are real words elsewhere ("um" in German and Portuguese, "er" in German and Dutch)."""
_EN_MM_RE = re.compile(r"m{2}", re.IGNORECASE)
_EN_MID_ONLY_RE = re.compile(r"(?:uh-huh|mm-hmm|mm-hm|mhm|uh-hum)", re.IGNORECASE)
"""Fillers inside a sentence, but "yes" when they start one ("Uh-huh, that works")."""

_NUMBER_WORDS = word_set(
    "zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
    "billion trillion double triple"
)

# Words after which a comma can't end a phrase, so a comma before a removed
# filler there was only the pause ("I was, uh, thinking" -> "I was thinking").
_CONTINUING = word_set(
    "a an the to of in on at for with from by about into onto over under through between than "
    "am is are was were be been being have has had do does did will would can could shall should may might must "
    "and or but nor that because if i we you they he she it my your our their his her its "
    "this these those some any each every very really just not"
)
_CONTINUING_SUFFIXES = ("'s", "'re", "'m", "'ve", "'ll", "'d", "n't", "’s", "’re", "’m", "’ve", "’ll", "’d", "n’t")

_FILLER_ALWAYS, _FILLER_MM, _FILLER_MID = 1, 2, 3


_FILLER_INITIALS = frozenset("uehmaäö")
_MAX_FILLER_LEN = 12


def _filler_kind(word: str, english: bool) -> int:
    if len(word) > _MAX_FILLER_LEN or word[0].lower() not in _FILLER_INITIALS:
        return 0  # cheap pre-check: most words can't be fillers
    if len(word) >= 2 and word.isupper():
        return 0  # "UM", "ER", "AH" are abbreviations, not hesitations
    if _UNIVERSAL_FILLER_RE.fullmatch(word):
        return _FILLER_ALWAYS
    if not english:
        return 0
    if _EN_FILLER_RE.fullmatch(word):
        return _FILLER_ALWAYS
    if _EN_MM_RE.fullmatch(word):
        return _FILLER_MM
    if _EN_MID_ONLY_RE.fullmatch(word):
        return _FILLER_MID
    return 0


def _is_number(word: str) -> bool:
    return any(c.isdigit() for c in word) or word.lower() in _NUMBER_WORDS


def _continues_phrase(word: str) -> bool:
    lw = word.lower()
    return lw in _CONTINUING or lw.endswith(_CONTINUING_SUFFIXES)


def _prev_significant(tokens: list[tuple[int, str]], keep: list[bool], i: int) -> int:
    j = i - 1
    while j >= 0 and (not keep[j] or tokens[j][0] == _SPACE):
        j -= 1
    return j


def _remove_fillers(text: str, english: bool) -> str:
    tokens = _tokenize(text)
    keep = [True] * len(tokens)
    capitalise_from: list[int] = []
    n = len(tokens)
    for i, (kind, word) in enumerate(tokens):
        if kind != _WORD:
            continue
        filler = _filler_kind(word, english)
        if not filler:
            continue
        p = _prev_significant(tokens, keep, i)
        prev_kind, prev_tok = tokens[p] if p >= 0 else (_NL, "")
        # Start of a sentence, line or quotation: after ". ! ?", an ellipsis, a line break or an opening quote.
        at_start = p < 0 or prev_kind == _NL or (prev_kind == _PUNCT and prev_tok not in _CLAUSE_PUNCT)
        if filler == _FILLER_MM and prev_kind == _WORD and _is_number(prev_tok):
            continue  # "5 mm"
        if filler == _FILLER_MID and at_start:
            continue  # "Uh-huh." means yes
        # Punctuation attached after the filler ("um," "um." "um...").
        after: list[int] = []
        k = i + 1
        while (
            k < n
            and tokens[k][0] == _PUNCT
            and (tokens[k][1] in _TERMINATORS or tokens[k][1] in _CLAUSE_PUNCT or _is_ellipsis(tokens[k][1]))
        ):
            after.append(k)
            k += 1
        after_toks = [tokens[a][1] for a in after]
        ends_sentence = any(_is_terminator(t) for t in after_toks)
        rest_blank = all(t[0] == _SPACE for t in tokens[k:])
        keep[i] = False
        if at_start:
            for a in after:
                keep[a] = False
            if k < n and tokens[k][0] == _SPACE:
                keep[k] = False  # '"Um, hello"' -> '"Hello"', not '" Hello"'
            if word[:1].isupper():
                capitalise_from.append(i)
            continue
        if prev_kind == _PUNCT and prev_tok in _CLAUSE_PUNCT:
            before_comma = _prev_significant(tokens, keep, p)
            if ends_sentence or rest_blank:
                keep[p] = False  # "go, um." -> "go."
            elif (
                prev_tok == ","
                and before_comma >= 0
                and tokens[before_comma][0] == _WORD
                and _continues_phrase(tokens[before_comma][1])
            ):
                keep[p] = False  # "I was, uh, thinking" -> "I was thinking"
            for a in after:
                if not _is_terminator(tokens[a][1]):
                    keep[a] = False
            continue
        # The filler follows a word: keep a sentence end or a clause comma, drop a hesitation ellipsis.
        for a in after:
            if _is_ellipsis(tokens[a][1]) or tokens[a][1] in {"-", "—", "–"}:
                keep[a] = False
    for i in capitalise_from:
        for j in range(i + 1, n):
            if keep[j] and tokens[j][0] == _WORD:
                tokens[j] = (_WORD, _capitalise(tokens[j][1]))
                break
            if keep[j] and tokens[j][0] in (_NL,):
                break
    return "".join(tok for (kind, tok), kept in zip(tokens, keep, strict=True) if kept)


# -- 3. stutters --------------------------------------------------------------------

_LEGIT_DOUBLES = word_set(
    # grammar: "had had", "that that", "what it is is", "it was was", "I do do that", "gave her her keys",
    # "told you you were"
    "had that is was do does did her you "
    # emphasis, reduplication and sounds
    "very really so no yes yeah yep bye ha haha hey now there well knock tut chop night far long many much more "
    "too oh ok okay go come wait please pretty blah bla la na ho hip ta tick ding bang boom beep choo cha dum tsk "
    "shh hush boo aye can will"
)


def _collapse_stutters(text: str) -> str:
    tokens = _tokenize(text)
    keep = [True] * len(tokens)
    last_word = -1  # index of the previous kept word, if only spaces separate it from the current one
    for i, (kind, word) in enumerate(tokens):
        if kind == _SPACE:
            continue
        if kind != _WORD:
            last_word = -1
            continue
        if last_word >= 0 and _is_stutter(tokens[last_word][1], word):
            keep[i] = False
            # Drop the space before the removed word.
            if i > 0 and tokens[i - 1][0] == _SPACE:
                keep[i - 1] = False
            continue
        last_word = i
    return "".join(tok for (_, tok), kept in zip(tokens, keep, strict=True) if kept)


def _is_stutter(first: str, second: str) -> bool:
    if first.lower() != second.lower():
        return False
    lower = second.lower()
    if lower in _LEGIT_DOUBLES or _is_number(second):
        return False
    if len(second) == 1:
        # "I I think", "a a book" - but not spelled-out letters ("B B C", "A A").
        return lower == "i" or second == "a"
    # A capitalised repeat mid-sentence is a name ("Bora Bora", "Walla Walla").
    return second.islower()


# -- 4. dictionary -------------------------------------------------------------------

_SEPARATOR = r"(?:[^\S\n]*-[^\S\n]*|[^\S\n]+)"


def _phrase_key(phrase: str) -> str:
    return " ".join(re.split(r"[\s\-]+", phrase.strip().lower())).strip()


# Word edges for dictionary terms and snippet triggers. A match must not start
# inside a contraction ("don't" never yields "t") but may follow a French
# elision ("l'API"); it may be followed by a possessive ("SvelteKit's") but
# not by the rest of a contraction ("don" never matches in "don't").
_PHRASE_START = r"(?<!\w)(?<!\w\w['’])"
_PHRASE_END = r"(?!\w)(?!['’](?!s(?!\w))\w)"


_TRIE_END = ""
_TRIE_SEP = " "


def _trie_pattern(node: dict[str, dict], last: str) -> str:
    """Regex for one prefix-tree node. Longer continuations come before the end of a phrase."""
    branches = []
    for unit in sorted(k for k in node if k != _TRIE_END):
        unit_re = _SEPARATOR if unit == _TRIE_SEP else re.escape(unit)
        branches.append(unit_re + _trie_pattern(node[unit], unit))
    if _TRIE_END in node:
        branches.append(_PHRASE_END if re.match(r"\w", last) else "")
    if len(branches) == 1:
        return branches[0]
    return "(?:" + "|".join(branches) + ")"


def _phrases_pattern(keys: Iterable[str]) -> str:
    """One regex for many phrases, built as a prefix tree.

    A plain alternation makes the regex engine try every phrase at every
    position, which is slow for big dictionaries. Sharing prefixes lets it rule
    out most positions after one character.
    """
    trie: dict[str, dict] = {}
    for key in keys:
        node = trie
        for char in key:
            node = node.setdefault(char, {})
        node[_TRIE_END] = {}
    word_start = {k: v for k, v in trie.items() if re.match(r"\w", k)}
    other_start = {k: v for k, v in trie.items() if not re.match(r"\w", k)}
    parts = []
    if word_start:
        parts.append(_PHRASE_START + _trie_pattern(word_start, ""))
    if other_start:
        parts.append(_trie_pattern(other_start, ""))
    return "|".join(parts)


@functools.lru_cache(maxsize=32)
def _phrase_matcher(pairs: tuple[tuple[str, str], ...]) -> tuple[re.Pattern[str], dict[str, str]] | None:
    """Regex matching any phrase in ``pairs`` (phrase, replacement), longest first, plus the key → replacement map."""
    mapping: dict[str, str] = {}
    for phrase, replacement in pairs:
        key = _phrase_key(phrase)
        if key and key not in mapping:
            mapping[key] = replacement
    if not mapping:
        return None
    return re.compile(_phrases_pattern(mapping), re.IGNORECASE), mapping


def _dictionary_pairs(entries: Iterable[DictionaryEntry]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        pairs.append((entry.term, entry.term))
        pairs.extend((variant, entry.term) for variant in entry.sounds_like)
    return tuple(pairs)


def _apply_dictionary(text: str, entries: tuple[DictionaryEntry, ...]) -> str:
    if not entries:
        return text
    matcher = _phrase_matcher(_dictionary_pairs(entries))
    if matcher is None:
        return text
    pattern, mapping = matcher
    return pattern.sub(lambda m: mapping.get(_phrase_key(m.group()), m.group()), text)


# -- 5. snippets ---------------------------------------------------------------------

_PLACEHOLDER_BASE = 0xF0000  # Supplementary Private Use Area-A: never produced by speech recognition
_PLACEHOLDER_RE = re.compile("[\U000f0000-\U000ffffd]")
_SNIPPET_KEY_RE = re.compile(r"[^\w\s]+")


def _snippet_key(text: str) -> str:
    return " ".join(_SNIPPET_KEY_RE.sub(" ", text.lower()).split())


def _whole_snippet(text: str, snippets: tuple[Snippet, ...]) -> str | None:
    key = _snippet_key(text)
    if not key:
        return None
    for snippet in snippets:
        if _snippet_key(snippet.trigger) == key:
            return snippet.text
    return None


def _insert_snippets(text: str, snippets: tuple[Snippet, ...]) -> tuple[str, list[str]]:
    """Replace triggers with placeholders; returns the text and the expansions in placeholder order."""
    if not snippets:
        return text, []
    matcher = _phrase_matcher(tuple((s.trigger, s.text) for s in snippets))
    if matcher is None:
        return text, []
    pattern, mapping = matcher
    expansions: list[str] = []

    def replace(match: re.Match[str]) -> str:
        expansion = mapping.get(_phrase_key(match.group()))
        if expansion is None:
            return match.group()
        expansions.append(expansion)
        return chr(_PLACEHOLDER_BASE + len(expansions) - 1)

    return pattern.sub(replace, text), expansions


def _restore_snippets(text: str, expansions: list[str]) -> str:
    if not expansions:
        return text
    return _PLACEHOLDER_RE.sub(lambda m: expansions[ord(m.group()) - _PLACEHOLDER_BASE], text)


_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_VARIABLE_RE = re.compile(r"\{\{|\}\}|\{([A-Za-z]+)\}")


def expand_variables(text: str, now: datetime) -> str:
    """Fill in the variables a snippet's text may contain. Pure: the caller passes the time.

    ``{date}`` → "October 12, 2026", ``{time}`` → "3:45 PM", ``{weekday}`` →
    "Sunday", ``{year}`` → "2026" (names are case-insensitive). ``{{`` and ``}}``
    are literal braces, so "{{date}}" gives "{date}". Any other ``{…}`` stays as
    written. Names and formats are English and fixed, so a snippet reads the same
    on every PC.
    """

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        name = match.group(1).lower()
        if name == "date":
            return f"{_MONTHS[now.month - 1]} {now.day}, {now.year}"
        if name == "time":
            return f"{now.hour % 12 or 12}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"
        if name == "weekday":
            return _WEEKDAYS[now.weekday()]
        if name == "year":
            return str(now.year)
        return token

    return _VARIABLE_RE.sub(replace, text)


# -- 6. style ----------------------------------------------------------------------

_ABBREVIATIONS = word_set("etc vs approx mr mrs ms dr prof st jr sr inc ltd corp dept fig cf eg ie al vol pp ca")
_PRONOUN_I_RE = re.compile(r"(?<![\w'’.-])i(?=(?:['’](?:m|d|ll|ve))?(?![\w'’-]|\.\w))")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s")


def _capitalise(word: str) -> str:
    """Upper-case the first letter of an all-lower-case word ("so" -> "So"; "iPhone" and "x.com" unchanged)."""
    letters = word.replace("'", "").replace("’", "")
    if not letters.isalpha() or not word.islower():
        return word
    return word[0].upper() + word[1:]


def _is_abbreviation(word: str) -> bool:
    return "." in word or word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


def _capitalise_sentences(text: str, expansions: list[str]) -> str:
    tokens = _tokenize(text)
    at_start = True
    prev_word = ""
    for i, (kind, tok) in enumerate(tokens):
        if kind == _SPACE:
            continue
        if kind == _NL:
            at_start = True
            continue
        if kind == _WORD:
            if at_start:
                tokens[i] = (_WORD, _capitalise(tok))
            at_start = False
            prev_word = tok
            continue
        if _PLACEHOLDER_RE.fullmatch(tok):
            # A snippet (inserted as written) takes a word's place; the next sentence starts after it only if
            # its text ends one: "my link is below" keeps "is" as it was.
            index = ord(tok) - _PLACEHOLDER_BASE
            ending = expansions[index].rstrip(" \t") if index < len(expansions) else ""
            at_start = ending.endswith(("\n", "!", "?")) or (ending.endswith(".") and not ending.endswith(".."))
            prev_word = ""
            continue
        if _is_terminator(tok):
            # A period after "e.g" or "Dr" doesn't end the sentence. An ellipsis ("I was… thinking")
            # usually doesn't either, so it never triggers a capital.
            at_start = not (tok == "." and prev_word and _is_abbreviation(prev_word))
        prev_word = ""
    return "".join(tok for _, tok in tokens)


def _ends_with_placeholder(line: str) -> bool:
    stripped = line.rstrip()
    return bool(stripped) and _PLACEHOLDER_RE.fullmatch(stripped[-1]) is not None


def _drop_single_trailing_period(text: str) -> str:
    """Casual style: "Sounds good." -> "Sounds good" for one-sentence messages."""
    stripped = text.rstrip()
    if "\n" in stripped or not stripped.endswith(".") or stripped.endswith(".."):
        return text
    tokens = _tokenize(stripped)
    words_before = [tok for kind, tok in tokens[:-1] if kind == _WORD]
    if not words_before or _is_abbreviation(words_before[-1]):
        return text
    for idx, (kind, tok) in enumerate(tokens[:-1]):
        if kind == _PUNCT and (_is_terminator(tok) or tok == "…"):
            prev = next((t for k, t in reversed(tokens[:idx]) if k == _WORD), "")
            if not (tok == "." and prev and _is_abbreviation(prev)):
                return text  # more than one sentence: keep the punctuation as it is
    return stripped[:-1] + text[len(stripped) :]


def _ensure_terminal_punctuation(text: str) -> str:
    """Formal style: end each sentence-like line (three words or more) with a period."""
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        body = line.rstrip()
        if not body or _LIST_MARKER_RE.match(body) or _ends_with_placeholder(body):
            continue
        last = body[-1]
        if not (last.isalnum() or last in "\"'”’)]"):
            continue
        if sum(1 for kind, _ in _tokenize(body) if kind == _WORD) < 3:
            continue  # greetings, sign-offs and names ("Thanks,", "Mike")
        lines[idx] = body + "." + line[len(body) :]
    return "\n".join(lines)


def _apply_style(text: str, style: str, english: bool, expansions: list[str]) -> str:
    if style == "verbatim":
        return text
    text = _capitalise_sentences(text, expansions)
    if english:
        text = _PRONOUN_I_RE.sub("I", text)
    if style == "casual":
        text = _drop_single_trailing_period(text)
    elif style == "formal":
        text = _ensure_terminal_punctuation(text)
    return text


# -- 7. whitespace -------------------------------------------------------------------

_HSPACE_RE = re.compile(r"[^\S\n]+")
_AROUND_NL_RE = re.compile(r" *\n *")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +(?=[,.!?:;](?:\s|$|[,.!?:;)\]\"'”’…]))")
_SPACE_BEFORE_LOW_PUNCT_RE = re.compile(r" +(?=[,.](?:\s|$|[,.!?:;)\]\"'”’…]))")
_INVISIBLE_RE = re.compile(
    "[\N{ZERO WIDTH SPACE}\N{ZERO WIDTH NON-JOINER}\N{ZERO WIDTH JOINER}\N{WORD JOINER}\N{ZERO WIDTH NO-BREAK SPACE}]"
)


def normalise_whitespace(text: str, language: str | None = None) -> str:
    """Collapse runs of spaces, remove spaces before punctuation and around line breaks. Keeps line breaks."""
    text = _HSPACE_RE.sub(" ", text)
    text = _AROUND_NL_RE.sub("\n", text)
    # French typography puts a space before ! ? : ; - keep it.
    french = bool(language) and language.lower().startswith("fr")
    text = (_SPACE_BEFORE_LOW_PUNCT_RE if french else _SPACE_BEFORE_PUNCT_RE).sub("", text)
    return text.strip(" ")


def _prepare(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE_RE.sub("", text)
    text = _PLACEHOLDER_RE.sub("", text)
    return text.replace("\N{NO-BREAK SPACE}", " ")


# -- public -----------------------------------------------------------------------------


def apply_rules(raw: str, ctx: TextContext) -> str:
    """Run the deterministic rules (see the module docstring) and return the finished text."""
    text = _prepare(raw)
    if not text.strip():
        return ""
    english = is_english(ctx.language, text)
    if ctx.voice_commands:
        # Spoken punctuation is English words that are real words elsewhere (French "colon", Italian "comma"), so
        # it works only in English text, and stays text in verbatim style (the default for code, where "dash" or
        # "colon" is often meant literally). Line breaks and "scratch that" are phrases no other language uses by
        # accident: they work in every style and language.
        text = _apply_voice_commands(text, punctuation=english and ctx.style != "verbatim")
    if ctx.remove_fillers and ctx.style != "verbatim":
        text = _remove_fillers(text, english)
        if english:
            text = _collapse_stutters(text)
    text = normalise_whitespace(text, ctx.language)
    text = _apply_dictionary(text, ctx.dictionary)
    now = ctx.now or datetime.now()
    whole = _whole_snippet(text, ctx.snippets)
    if whole is not None:
        return expand_variables(whole, now)
    text, expansions = _insert_snippets(text, ctx.snippets)
    text = _apply_style(text, ctx.style, english, expansions)
    # The dictionary has the last word on spelling: "kubectl" stays lower-case even at the start of a sentence.
    text = _apply_dictionary(text, ctx.dictionary)
    text = normalise_whitespace(text, ctx.language)
    if not text.strip():
        return ""
    return _restore_snippets(text, [expand_variables(e, now) for e in expansions])
