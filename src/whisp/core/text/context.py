"""Where the text is going: app categories and the per-dictation text context.

``categorize`` decides what kind of app a window belongs to (messaging, email,
code, document or other). Native apps are recognised by their executable
name. Browsers and a few multi-purpose hosts (LibreOffice) are recognised by
keywords in the window title, because the same process can show Gmail in one
tab and GitHub in the next. The user's own overrides always win.

``build_context`` bundles everything the text rules and AI polish need for one
dictation into an immutable ``TextContext``.

Window titles are used only here, in memory. Never log them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from whisp.core.types import AppCategory, TargetWindow
from whisp.storage.settings import DictionaryEntry, Settings, Snippet

STYLES = ("verbatim", "clean", "casual", "formal")
DEFAULT_STYLE = "clean"


@dataclass(frozen=True)
class TextContext:
    """Everything the text pipeline needs to finish one dictation."""

    category: AppCategory
    style: str
    """``"verbatim"`` | ``"clean"`` | ``"casual"`` | ``"formal"``."""
    app_name: str
    language: str | None
    """ISO-639-1 code of the spoken language (``"en"``), or None when unknown."""
    dictionary: tuple[DictionaryEntry, ...]
    snippets: tuple[Snippet, ...]
    remove_fillers: bool
    voice_commands: bool
    custom_instructions: str
    now: datetime | None = None
    """Local time of the dictation, for snippet variables such as {date}. None means "when the rules run"."""


# -- built-in categorisation ------------------------------------------------------

_MESSAGING = {
    "slack.exe",
    "ms-teams.exe",  # new Teams
    "teams.exe",  # classic Teams
    "discord.exe",
    "discordptb.exe",
    "discordcanary.exe",
    "whatsapp.exe",
    "whatsapp.root.exe",
    "telegram.exe",
    "signal.exe",
    "messenger.exe",
    "zoom.exe",
    "skype.exe",
    "element.exe",
    "mattermost.exe",
    "rocket.chat.exe",
    "wechat.exe",
    "weixin.exe",
    "line.exe",
    "viber.exe",
    "beeper.exe",
    "ferdium.exe",
    "rambox.exe",
    "franz.exe",
    "wire.exe",
    "threema.exe",
    "chatwork.exe",
    "lark.exe",
    "feishu.exe",
    "dingtalk.exe",
    "kakaotalk.exe",
    "googlechat.exe",
}

_EMAIL = {
    "outlook.exe",  # classic Outlook
    "olk.exe",  # "new Outlook"
    "hxoutlook.exe",  # Windows Mail
    "thunderbird.exe",
    "betterbird.exe",
    "mailspring.exe",
    "mailclient.exe",  # eM Client
    "postbox.exe",
    "superhuman.exe",
    "spark.exe",
    "spark desktop.exe",
    "mailbird.exe",
    "thebat.exe",
    "notion mail.exe",
}

_JETBRAINS = (
    "idea",
    "pycharm",
    "webstorm",
    "phpstorm",
    "rubymine",
    "clion",
    "rider",
    "goland",
    "datagrip",
    "rustrover",
    "dataspell",
    "aqua",
    "writerside",
    "studio",  # Android Studio
)

_CODE = {
    "code.exe",
    "code - insiders.exe",
    "vscodium.exe",
    "cursor.exe",
    "windsurf.exe",
    "zed.exe",
    "antigravity.exe",
    "trae.exe",
    "kiro.exe",
    "positron.exe",
    "devenv.exe",
    "sublime_text.exe",
    "notepad++.exe",
    "windowsterminal.exe",
    "wt.exe",
    "openconsole.exe",
    "powershell.exe",
    "pwsh.exe",
    "powershell_ise.exe",
    "cmd.exe",
    "conhost.exe",
    "wezterm.exe",
    "wezterm-gui.exe",
    "alacritty.exe",
    "mintty.exe",
    "hyper.exe",
    "tabby.exe",
    "warp.exe",
    "fleet.exe",
    "gvim.exe",
    "nvim-qt.exe",
    "neovide.exe",
    "emacs.exe",
    "runemacs.exe",
    "putty.exe",
    "kitty.exe",
    "github desktop.exe",
    "githubdesktop.exe",
    "sourcetree.exe",
    "fork.exe",
    "gitkraken.exe",
    "postman.exe",
    "insomnia.exe",
    "ssms.exe",
    "azuredatastudio.exe",
    "dbeaver.exe",
    "rstudio.exe",
    "spyder.exe",
    "matlab.exe",
    "unity.exe",
    "godot.exe",
    *(f"{name}{suffix}.exe" for name in _JETBRAINS for suffix in ("", "64")),
}

_DOCUMENT = {
    "winword.exe",
    "powerpnt.exe",
    "onenote.exe",
    "onenoteim.exe",
    "notion.exe",
    "obsidian.exe",
    "evernote.exe",
    "wordpad.exe",
    "notepad.exe",
    "swriter.exe",
    "typora.exe",
    "logseq.exe",
    "joplin.exe",
    "scrivener.exe",
    "ia writer.exe",
    "anytype.exe",
    "affine.exe",
    "capacities.exe",
    "craft.exe",
    "bear.exe",
    "simplenote.exe",
    "standard notes.exe",
    "zettlr.exe",
    "marktext.exe",
    "ghostwriter.exe",
    "abiword.exe",
    "wps.exe",
    "acrobat.exe",
    "notion calendar.exe",
}

_PROCESS_CATEGORIES: dict[str, AppCategory] = {
    **dict.fromkeys(_DOCUMENT, AppCategory.DOCUMENT),
    **dict.fromkeys(_CODE, AppCategory.CODE),
    **dict.fromkeys(_EMAIL, AppCategory.EMAIL),
    **dict.fromkeys(_MESSAGING, AppCategory.MESSAGING),
}

BROWSERS = frozenset(
    {
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
        "brave.exe",
        "arc.exe",
        "opera.exe",
        "opera_gx.exe",
        "vivaldi.exe",
        "chromium.exe",
        "thorium.exe",
        "librewolf.exe",
        "waterfox.exe",
        "floorp.exe",
        "zen.exe",
        "comet.exe",
        "dia.exe",
        "yandex.exe",
        "browser.exe",  # Yandex Browser
        "iexplore.exe",
    }
)
"""Processes whose category depends on the site shown, read from the window title."""

# Site names as they appear in browser window titles. When several match, the
# one nearest the end of the title wins: sites put their own name last
# ("Inbox - me@x.com - Gmail - Google Chrome"), while the page content comes first.
#
# Distinctive brand names match anywhere (as whole words). Names that are also
# ordinary words ("Outlook", "Word", "Teams") only count when they make up a
# whole title segment, as in "Mail - Jo Smith - Outlook", so a page titled
# "Weather outlook for Boston" is not mistaken for email.
_ANYWHERE = False
_SEGMENT = True
_TITLE_KEYWORDS: tuple[tuple[str, AppCategory, bool], ...] = (
    ("Gmail", AppCategory.EMAIL, _ANYWHERE),
    ("Outlook", AppCategory.EMAIL, _SEGMENT),
    ("Yahoo Mail", AppCategory.EMAIL, _ANYWHERE),
    ("Proton Mail", AppCategory.EMAIL, _ANYWHERE),
    ("Fastmail", AppCategory.EMAIL, _ANYWHERE),
    ("Zoho Mail", AppCategory.EMAIL, _ANYWHERE),
    ("iCloud Mail", AppCategory.EMAIL, _ANYWHERE),
    ("Superhuman", AppCategory.EMAIL, _SEGMENT),
    ("Slack", AppCategory.MESSAGING, _ANYWHERE),
    ("Discord", AppCategory.MESSAGING, _ANYWHERE),
    ("WhatsApp", AppCategory.MESSAGING, _ANYWHERE),
    ("Messenger", AppCategory.MESSAGING, _SEGMENT),
    ("Microsoft Teams", AppCategory.MESSAGING, _ANYWHERE),
    ("Teams", AppCategory.MESSAGING, _SEGMENT),
    ("Telegram", AppCategory.MESSAGING, _ANYWHERE),
    ("Telegram Web", AppCategory.MESSAGING, _ANYWHERE),
    ("Google Chat", AppCategory.MESSAGING, _ANYWHERE),
    ("Element", AppCategory.MESSAGING, _SEGMENT),
    ("Mattermost", AppCategory.MESSAGING, _ANYWHERE),
    ("Rocket.Chat", AppCategory.MESSAGING, _ANYWHERE),
    ("Google Docs", AppCategory.DOCUMENT, _ANYWHERE),
    ("Notion", AppCategory.DOCUMENT, _SEGMENT),
    ("Confluence", AppCategory.DOCUMENT, _ANYWHERE),
    ("Coda", AppCategory.DOCUMENT, _SEGMENT),
    ("Dropbox Paper", AppCategory.DOCUMENT, _ANYWHERE),
    ("Quip", AppCategory.DOCUMENT, _SEGMENT),
    ("OneNote", AppCategory.DOCUMENT, _ANYWHERE),
    ("Evernote", AppCategory.DOCUMENT, _ANYWHERE),
    ("Google Keep", AppCategory.DOCUMENT, _ANYWHERE),
    ("Microsoft Word", AppCategory.DOCUMENT, _ANYWHERE),
    ("Word", AppCategory.DOCUMENT, _SEGMENT),
    ("LibreOffice Writer", AppCategory.DOCUMENT, _ANYWHERE),
    ("GitHub", AppCategory.CODE, _ANYWHERE),
    ("GitLab", AppCategory.CODE, _ANYWHERE),
    ("Bitbucket", AppCategory.CODE, _ANYWHERE),
    ("Stack Overflow", AppCategory.CODE, _ANYWHERE),
    ("CodeSandbox", AppCategory.CODE, _ANYWHERE),
    ("StackBlitz", AppCategory.CODE, _ANYWHERE),
    ("CodePen", AppCategory.CODE, _ANYWHERE),
    ("JSFiddle", AppCategory.CODE, _ANYWHERE),
    ("Replit", AppCategory.CODE, _ANYWHERE),
    ("Colab", AppCategory.CODE, _SEGMENT),
    ("JupyterLab", AppCategory.CODE, _ANYWHERE),
    ("Jupyter Notebook", AppCategory.CODE, _ANYWHERE),
    ("Azure DevOps", AppCategory.CODE, _ANYWHERE),
    ("Visual Studio Code", AppCategory.CODE, _ANYWHERE),
)

# A title segment boundary: the start/end of the title or a " - ", " | ", " · ", " — " separator.
_SEP = r"[-|·—–•]"
_SEGMENT_START = rf"(?:^|\s{_SEP}\s+)(?:\(\d+\+?\)\s*)?"  # also skips unread counters like "(3) "
_SEGMENT_END = rf"(?=\s+{_SEP}(?:\s|$)|\s*$)"


def _title_pattern(keyword: str, whole_segment: bool) -> re.Pattern[str]:
    body = re.escape(keyword)
    if whole_segment:
        return re.compile(rf"{_SEGMENT_START}(?P<kw>{body}){_SEGMENT_END}", re.IGNORECASE)
    return re.compile(rf"(?<![\w])(?P<kw>{body})(?![\w])", re.IGNORECASE)


_TITLE_PATTERNS: tuple[tuple[re.Pattern[str], AppCategory], ...] = tuple(
    (_title_pattern(keyword, segment), category) for keyword, category, segment in _TITLE_KEYWORDS
)

_TITLE_HOSTS = frozenset({"soffice.bin", "soffice.exe"})
"""Native processes that host several kinds of app (LibreOffice), told apart by title."""

_CATEGORY_BY_ID = {c.value: c for c in AppCategory}


def _normalise_process(name: str) -> str:
    return name.strip().lower()


def _override_for(process: str, overrides: Mapping[str, str]) -> AppCategory | None:
    if not overrides:
        return None
    candidates = [process]
    if not process.endswith(".exe"):
        candidates.append(process + ".exe")
    else:
        candidates.append(process.removesuffix(".exe"))
    for key in candidates:
        value = overrides.get(key)
        if value is not None:
            category = _CATEGORY_BY_ID.get(str(value).strip().lower())
            if category is not None:
                return category
    return None


def categorize_title(title: str) -> AppCategory:
    """Category of a site from a browser window title; OTHER if nothing matches."""
    if not title:
        return AppCategory.OTHER
    best: tuple[int, int, AppCategory] | None = None
    for pattern, category in _TITLE_PATTERNS:
        for match in pattern.finditer(title):
            # Prefer the match that ends last, then the longer keyword ("Microsoft Teams" over "Teams").
            start, end = match.span("kw")
            key = (end, end - start, category)
            if best is None or key[:2] > best[:2]:
                best = key
    return best[2] if best else AppCategory.OTHER


def categorize(process_name: str, title: str, overrides: Mapping[str, str]) -> AppCategory:
    """Decide which kind of app a window belongs to.

    ``overrides`` maps lower-case process names (``"notion.exe"``) to category
    ids (``"document"``) and always wins. Unknown override values are ignored.
    """
    process = _normalise_process(process_name)
    override = _override_for(process, overrides)
    if override is not None:
        return override
    if process in BROWSERS or process in _TITLE_HOSTS:
        return categorize_title(title)
    return _PROCESS_CATEGORIES.get(process, AppCategory.OTHER)


def style_for(settings: Settings, category: AppCategory) -> str:
    style = settings.text.styles.get(category.value, DEFAULT_STYLE)
    return style if style in STYLES else DEFAULT_STYLE


def _normalise_language(language: str | None) -> str | None:
    if not language:
        return None
    code = language.strip().lower().replace("_", "-").split("-")[0]
    if not code or code == "auto":
        return None
    return code


def build_context(settings: Settings, target: TargetWindow | None, language: str | None) -> TextContext:
    """The text context for one dictation into ``target`` (None when the focused window is unknown)."""
    if target is not None:
        category = categorize(target.process_name, target.title, settings.text.app_categories)
        app_name = target.app_name or target.process_name
    else:
        category = AppCategory.OTHER
        app_name = ""
    return TextContext(
        category=category,
        style=style_for(settings, category),
        app_name=app_name,
        language=_normalise_language(language),
        dictionary=tuple(settings.text.dictionary),
        snippets=tuple(settings.text.snippets),
        remove_fillers=settings.text.remove_fillers,
        voice_commands=settings.text.voice_commands,
        custom_instructions=settings.polish.custom_instructions.strip(),
        now=datetime.now(),
    )
