"""SettingsStore and the settings schema: defaults, merging, validation, salvage and recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from whisp.storage.paths import app_paths
from whisp.storage.settings import (
    SCHEMA_VERSION,
    DictionaryEntry,
    Settings,
    SettingsStore,
    Snippet,
    TextSettings,
    parse_settings,
)


def settings_path() -> Path:
    return app_paths().settings_file


def write_raw(content: str) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def read_saved() -> dict:
    return json.loads(settings_path().read_text(encoding="utf-8"))


# -- defaults ----------------------------------------------------------------------------


def test_fresh_store_uses_defaults_and_writes_them(settings_store: SettingsStore) -> None:
    assert settings_store.current == Settings()
    assert settings_store.path == settings_path()
    saved = read_saved()
    assert saved["schema_version"] == SCHEMA_VERSION
    assert saved["hotkeys"]["dictate"] == "ctrl+win"
    assert Settings.model_validate(saved) == Settings()


def test_defaults_are_sensible() -> None:
    s = Settings()
    assert s.general.onboarding_completed is False
    assert s.hotkeys.tap_to_lock_ms == 350
    assert s.audio.input_device is None
    assert s.polish.enabled is False
    assert s.insertion.on_focus_change == "clipboard"
    assert s.privacy.history_enabled is True
    assert s.text.styles == {
        "messaging": "casual",
        "email": "formal",
        "code": "verbatim",
        "document": "clean",
        "other": "clean",
    }


def test_settings_are_immutable(settings_store: SettingsStore) -> None:
    with pytest.raises(ValidationError):
        settings_store.current.audio.sounds_enabled = False  # type: ignore[misc]


# -- update ------------------------------------------------------------------------------


def test_update_merges_into_the_section_and_persists(settings_store: SettingsStore) -> None:
    after = settings_store.update({"audio": {"sounds_enabled": False}})
    assert after.audio.sounds_enabled is False
    assert after.audio.sounds_volume == Settings().audio.sounds_volume  # untouched field kept
    assert settings_store.current is after
    assert read_saved()["audio"]["sounds_enabled"] is False
    reloaded = SettingsStore(settings_path())
    assert reloaded.current == after


def test_update_emits_changed_with_the_sections_that_changed(settings_store: SettingsStore, qtbot) -> None:
    with qtbot.waitSignal(settings_store.changed, timeout=1000) as blocker:
        settings_store.update({"audio": {"sounds_volume": 0.2}, "overlay": {"position": "top"}, "general": {}})
    assert blocker.args == [{"audio", "overlay"}]


def test_update_without_changes_is_a_no_op(settings_store: SettingsStore, qtbot) -> None:
    before = settings_store.current
    with qtbot.assertNotEmitted(settings_store.changed):
        result = settings_store.update({"audio": {"sounds_enabled": before.audio.sounds_enabled}})
    assert result is before


def test_mappings_marked_replace_whole_are_replaced_not_merged(settings_store: SettingsStore) -> None:
    settings_store.update({"text": {"app_categories": {"Notion.exe": "document", "slack.exe": "messaging"}}})
    assert settings_store.current.text.app_categories == {"notion.exe": "document", "slack.exe": "messaging"}
    settings_store.update({"text": {"app_categories": {"slack.exe": "messaging"}}})
    assert settings_store.current.text.app_categories == {"slack.exe": "messaging"}
    settings_store.update({"insertion": {"app_methods": {"WindowsTerminal.exe": "type"}}})
    settings_store.update({"insertion": {"app_methods": {}}})
    assert settings_store.current.insertion.app_methods == {}


def test_other_mappings_are_merged(settings_store: SettingsStore) -> None:
    settings_store.update({"text": {"styles": {"email": "casual"}}})
    styles = settings_store.current.text.styles
    assert styles["email"] == "casual"
    assert styles["code"] == "verbatim"


def test_invalid_update_is_rejected_and_changes_nothing(settings_store: SettingsStore, qtbot) -> None:
    before = settings_store.current
    saved_before = settings_path().read_text(encoding="utf-8")
    with qtbot.assertNotEmitted(settings_store.changed), pytest.raises(ValueError, match=r"audio\.sounds_volume"):
        settings_store.update({"audio": {"sounds_volume": 2.0}})
    with pytest.raises(ValueError, match="tap_to_lock_ms"):
        settings_store.update({"hotkeys": {"tap_to_lock_ms": 5}})
    with pytest.raises(ValueError):
        settings_store.update({"text": {"styles": {"email": "shouty"}}})
    assert settings_store.current is before
    assert settings_path().read_text(encoding="utf-8") == saved_before


def test_a_change_that_couldnt_be_saved_isnt_applied(settings_store: SettingsStore, qtbot) -> None:
    # For example antivirus or a sync tool holding settings.json open while Whisp replaces it.
    before = settings_store.current
    saved_before = settings_path().read_text(encoding="utf-8")

    def locked(_settings: Settings) -> None:
        raise PermissionError(13, "The file is in use by another program")

    settings_store._save = locked  # shadows the method on this one store
    try:
        with qtbot.assertNotEmitted(settings_store.changed), pytest.raises(OSError):
            settings_store.update({"hotkeys": {"dictate": "f8"}, "audio": {"sounds_volume": 0.1}})
        with qtbot.assertNotEmitted(settings_store.changed), pytest.raises(OSError):
            settings_store.reset()
    finally:
        del settings_store._save
    assert settings_store.current is before
    assert settings_path().read_text(encoding="utf-8") == saved_before

    # Trying again once the file is free applies the change, and everyone hears about it.
    with qtbot.waitSignal(settings_store.changed, timeout=1000) as blocker:
        settings_store.update({"hotkeys": {"dictate": "f8"}})
    assert blocker.args == [{"hotkeys"}]
    assert settings_store.current.hotkeys.dictate == "f8"
    assert read_saved()["hotkeys"]["dictate"] == "f8"


def test_whisp_still_starts_when_settings_cant_be_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from whisp.storage import settings as settings_module

    def read_only(_self: SettingsStore, _settings: Settings) -> None:
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(SettingsStore, "_save", read_only)
    store = SettingsStore(tmp_path / "settings.json")  # a first start: there's no file yet
    assert store.current == Settings()
    assert store.save_problem == settings_module.MSG_CANT_SAVE
    with pytest.raises(OSError):  # later changes still report the problem to whoever made them
        store.update({"audio": {"sounds_enabled": False}})


def test_no_save_problem_normally(settings_store: SettingsStore) -> None:
    assert settings_store.save_problem is None


def test_unknown_section_is_rejected(settings_store: SettingsStore) -> None:
    with pytest.raises(ValueError, match="unknown settings section"):
        settings_store.update({"bogus": {"x": 1}})


# -- reset -------------------------------------------------------------------------------


def test_reset_one_section(settings_store: SettingsStore, qtbot) -> None:
    settings_store.update({"audio": {"sounds_volume": 0.1}, "overlay": {"position": "top"}})
    with qtbot.waitSignal(settings_store.changed, timeout=1000) as blocker:
        after = settings_store.reset("audio")
    assert blocker.args == [{"audio"}]
    assert after.audio == Settings().audio
    assert after.overlay.position == "top"
    with pytest.raises(ValueError):
        settings_store.reset("bogus")


def test_reset_everything_keeps_onboarding_state(settings_store: SettingsStore) -> None:
    settings_store.update(
        {"general": {"onboarding_completed": True, "theme": "light"}, "audio": {"sounds_volume": 0.3}}
    )
    after = settings_store.reset()
    assert after.general.onboarding_completed is True
    assert after.general.theme == "system"
    assert after.audio == Settings().audio


# -- loading, salvage and recovery -------------------------------------------------------


def test_invalid_section_is_reset_on_its_own(isolated_whisp_home: Path) -> None:
    write_raw(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "hotkeys": {"tap_to_lock_ms": 5, "dictate": "f8"},
                "audio": {"sounds_volume": 0.2},
                "general": {"onboarding_completed": True},
            }
        )
    )
    store = SettingsStore(settings_path())
    assert store.current.hotkeys == Settings().hotkeys
    assert store.current.audio.sounds_volume == 0.2
    assert store.current.general.onboarding_completed is True
    # The repaired settings were written back.
    assert read_saved()["hotkeys"]["tap_to_lock_ms"] == Settings().hotkeys.tap_to_lock_ms


def test_parse_settings_reports_what_it_reset() -> None:
    settings, problems = parse_settings({"audio": {"sounds_volume": "loud"}, "privacy": {"history_retention_days": -1}})
    assert settings.audio == Settings().audio
    assert settings.privacy == Settings().privacy
    assert len(problems) == 2
    assert any("audio" in p and "sounds_volume" in p for p in problems)
    assert any("privacy" in p for p in problems)


def test_parse_settings_accepts_a_valid_file_without_problems() -> None:
    raw = Settings(audio={"sounds_volume": 0.4}).model_dump(mode="json")
    settings, problems = parse_settings(raw)
    assert problems == []
    assert settings.audio.sounds_volume == 0.4


def test_unknown_keys_are_ignored(isolated_whisp_home: Path) -> None:
    write_raw(json.dumps({"schema_version": SCHEMA_VERSION, "audio": {"sounds_volume": 0.5, "legacy_key": True}}))
    store = SettingsStore(settings_path())
    assert store.current.audio.sounds_volume == 0.5


def test_corrupt_file_is_kept_aside_and_defaults_used(isolated_whisp_home: Path) -> None:
    path = write_raw("{this is not json")
    store = SettingsStore(path)
    assert store.current == Settings()
    backups = list(path.parent.glob("settings.invalid-*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{this is not json"
    assert read_saved()["schema_version"] == SCHEMA_VERSION


def test_non_object_file_gives_defaults(isolated_whisp_home: Path) -> None:
    write_raw("[1, 2, 3]")
    store = SettingsStore(settings_path())
    assert store.current == Settings()
    assert isinstance(read_saved(), dict)


def test_old_schema_version_is_upgraded_and_saved(isolated_whisp_home: Path) -> None:
    write_raw(json.dumps({"schema_version": 0, "audio": {"sounds_volume": 0.7}}))
    store = SettingsStore(settings_path())
    assert store.current.schema_version == SCHEMA_VERSION
    assert store.current.audio.sounds_volume == 0.7
    assert read_saved()["schema_version"] == SCHEMA_VERSION


def test_save_leaves_no_temporary_file(settings_store: SettingsStore) -> None:
    settings_store.update({"audio": {"sounds_volume": 0.9}})
    assert not settings_path().with_suffix(".json.tmp").exists()


# -- field validators --------------------------------------------------------------------


def test_styles_are_filled_with_defaults() -> None:
    text = TextSettings(styles={"email": "casual"})
    assert text.styles == {
        "messaging": "casual",
        "email": "casual",
        "code": "verbatim",
        "document": "clean",
        "other": "clean",
    }


def test_override_keys_are_normalised() -> None:
    text = TextSettings(app_categories={"  Notion.EXE ": "document", "  ": "code"})
    assert text.app_categories == {"notion.exe": "document"}


def test_dictionary_entries_are_cleaned_and_validated() -> None:
    entry = DictionaryEntry(term="  SvelteKit ", sounds_like=("svelte kit", "Svelte Kit", " ", "sveltex"))
    assert entry.term == "SvelteKit"
    assert entry.sounds_like == ("svelte kit", "sveltex")
    with pytest.raises(ValidationError):
        DictionaryEntry(term="   ")


def test_snippet_trigger_is_stripped_and_required() -> None:
    assert Snippet(trigger="  my link ", text="https://example.com").trigger == "my link"
    with pytest.raises(ValidationError):
        Snippet(trigger=" ", text="x")
    with pytest.raises(ValidationError):
        Snippet(trigger="ok", text="")
