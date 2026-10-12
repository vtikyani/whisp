"""App categorisation and the per-dictation text context."""

from __future__ import annotations

from datetime import datetime

import pytest

from whisp.core.text.context import BROWSERS, build_context, categorize, categorize_title
from whisp.core.types import AppCategory, TargetWindow
from whisp.storage.settings import DictionaryEntry, Settings, Snippet

M, E, C, D = AppCategory.MESSAGING, AppCategory.EMAIL, AppCategory.CODE, AppCategory.DOCUMENT
OTHER = AppCategory.OTHER


@pytest.mark.parametrize(
    ("process", "expected"),
    [
        ("slack.exe", M),
        ("Slack.exe", M),
        ("ms-teams.exe", M),
        ("teams.exe", M),
        ("discord.exe", M),
        ("WhatsApp.exe", M),
        ("telegram.exe", M),
        ("signal.exe", M),
        ("zoom.exe", M),
        ("outlook.exe", E),
        ("olk.exe", E),
        ("thunderbird.exe", E),
        ("mailspring.exe", E),
        ("code.exe", C),
        ("cursor.exe", C),
        ("windsurf.exe", C),
        ("zed.exe", C),
        ("idea64.exe", C),
        ("pycharm64.exe", C),
        ("rider64.exe", C),
        ("devenv.exe", C),
        ("sublime_text.exe", C),
        ("notepad++.exe", C),
        ("WindowsTerminal.exe", C),
        ("wt.exe", C),
        ("powershell.exe", C),
        ("pwsh.exe", C),
        ("cmd.exe", C),
        ("conhost.exe", C),
        ("wezterm-gui.exe", C),
        ("alacritty.exe", C),
        ("winword.exe", D),
        ("notion.exe", D),
        ("obsidian.exe", D),
        ("onenote.exe", D),
        ("evernote.exe", D),
        ("wordpad.exe", D),
        ("excel.exe", OTHER),
        ("explorer.exe", OTHER),
        ("", OTHER),
    ],
)
def test_native_apps(process: str, expected: AppCategory) -> None:
    assert categorize(process, "anything", {}) is expected


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Inbox (3) - me@gmail.com - Gmail - Google Chrome", E),
        ("Mail - Jo Smith - Outlook - Microsoft Edge", E),
        ("general (Channel) - Acme - Slack - Google Chrome", M),
        ("#general | Friends - Discord - Brave", M),
        ("(3) WhatsApp - Google Chrome", M),
        ("Chat | Microsoft Teams - Microsoft Edge", M),
        ("Messenger - Mozilla Firefox", M),
        ("Telegram Web - Google Chrome", M),
        ("Quarterly plan - Google Docs - Google Chrome", D),
        ("Roadmap - Engineering - Confluence - Vivaldi", D),
        ("Document1 - Word - Mozilla Firefox", D),
        ("Notion - Opera", D),
        ("Pull requests · whisp/whisp · GitHub - Google Chrome", C),
        ("How do I exit vim? - Stack Overflow - Google Chrome", C),
        ("Merge requests · GitLab - Firefox", C),
        ("my-app - CodeSandbox - Chrome", C),
        # The site nearest the end of the title wins.
        ("Gmail clone · GitHub - Google Chrome", C),
        # Ordinary words are not site names.
        ("Weather outlook for Boston - Google Chrome", OTHER),
        ("Word of the Day - Merriam-Webster - Google Chrome", OTHER),
        ("Building great teams at work - Medium - Firefox", OTHER),
        ("New Tab - Google Chrome", OTHER),
        ("", OTHER),
    ],
)
def test_browser_titles(title: str, expected: AppCategory) -> None:
    assert categorize("chrome.exe", title, {}) is expected
    assert categorize_title(title) is expected


@pytest.mark.parametrize("browser", sorted(BROWSERS))
def test_every_browser_uses_the_title(browser: str) -> None:
    assert categorize(browser, "Inbox - Gmail", {}) is E
    assert categorize(browser, "Some page", {}) is OTHER


def test_libreoffice_is_told_apart_by_title() -> None:
    assert categorize("soffice.bin", "Untitled 1 - LibreOffice Writer", {}) is D
    assert categorize("soffice.bin", "Untitled 1 - LibreOffice Calc", {}) is OTHER


def test_user_overrides_win() -> None:
    assert categorize("notion.exe", "", {"notion.exe": "messaging"}) is M
    assert categorize("Notion.exe", "", {"notion": "code"}) is C  # key without ".exe"
    assert categorize("chrome.exe", "Inbox - Gmail", {"chrome.exe": "document"}) is D
    assert categorize("notion.exe", "", {"notion.exe": "nonsense"}) is D  # unknown values are ignored
    assert categorize("custom.exe", "", {"custom.exe": "email"}) is E


def _target(process: str, title: str = "", app: str = "App") -> TargetWindow:
    return TargetWindow(hwnd=1, pid=2, process_name=process, app_name=app, title=title)


def test_build_context_uses_settings() -> None:
    settings = Settings.model_validate(
        {
            "text": {
                "remove_fillers": False,
                "voice_commands": True,
                "dictionary": [{"term": "SvelteKit", "sounds_like": ["svelte kit"]}],
                "snippets": [{"trigger": "my link", "text": "https://x.y"}],
                "styles": {"messaging": "formal"},
                "app_categories": {"notepad.exe": "code"},
            },
            "polish": {"custom_instructions": "  Use British spelling.  "},
        }
    )
    ctx = build_context(settings, _target("slack.exe", app="Slack"), "en-GB")
    assert ctx.category is M
    assert ctx.style == "formal"
    assert ctx.app_name == "Slack"
    assert ctx.language == "en"
    assert ctx.dictionary == (DictionaryEntry(term="SvelteKit", sounds_like=("svelte kit",)),)
    assert ctx.snippets == (Snippet(trigger="my link", text="https://x.y"),)
    assert ctx.remove_fillers is False
    assert ctx.voice_commands is True
    assert ctx.custom_instructions == "Use British spelling."

    code = build_context(settings, _target("notepad.exe"), None)
    assert code.category is C
    assert code.style == "verbatim"  # default style for code


def test_build_context_without_a_target() -> None:
    before = datetime.now()
    ctx = build_context(Settings(), None, "auto")
    assert ctx.category is OTHER
    assert ctx.style == "clean"
    assert ctx.app_name == ""
    assert ctx.language is None
    assert ctx.now is not None and before <= ctx.now <= datetime.now()  # snippet variables use delivery time


def test_default_styles_per_category() -> None:
    settings = Settings()
    assert build_context(settings, _target("slack.exe"), None).style == "casual"
    assert build_context(settings, _target("outlook.exe"), None).style == "formal"
    assert build_context(settings, _target("code.exe"), None).style == "verbatim"
    assert build_context(settings, _target("winword.exe"), None).style == "clean"
    assert build_context(settings, _target("excel.exe"), None).style == "clean"
