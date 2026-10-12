"""User settings: schema, validation, and a thread-safe store.

Settings live in ``settings.json`` (see ``paths.AppPaths.settings_file``).
The file is validated on load. A section that fails validation is reset to
its defaults on its own, so one bad value never wipes the rest. A file that
isn't valid JSON is kept as ``settings.invalid-<time>.json`` and replaced
with defaults.

Every settings object is immutable. ``SettingsStore.current`` always returns
a complete, valid snapshot, so worker threads can read it without locks.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from PySide6.QtCore import QObject, Signal

from whisp.storage.paths import display_path

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

MSG_CANT_SAVE = (
    "Whisp can't save its settings, so changes won't be kept after you quit. "
    "Check that your user folder isn't full or read-only."
)

StyleId = Literal["verbatim", "clean", "casual", "formal"]
CategoryId = Literal["messaging", "email", "code", "document", "other"]
InsertMethod = Literal["paste", "paste_shift_insert", "type"]


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class DictionaryEntry(_Section):
    term: str = Field(min_length=1, max_length=100)
    """How the word should be written, e.g. ``"SvelteKit"``."""
    sounds_like: tuple[str, ...] = ()
    """What the recogniser tends to produce instead, e.g. ``("svelte kit", "sveltex")``."""

    @field_validator("term")
    @classmethod
    def _strip_term(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("term must not be blank")
        return v

    @field_validator("sounds_like")
    @classmethod
    def _clean_variants(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        out = []
        for item in v:
            item = item.strip()
            if item and len(item) <= 100 and item.lower() not in {o.lower() for o in out}:
                out.append(item)
        return tuple(out)


class Snippet(_Section):
    trigger: str = Field(min_length=1, max_length=100)
    """Phrase the user says, e.g. ``"my calendar link"``."""
    text: str = Field(min_length=1, max_length=5000)
    """What gets inserted instead."""

    @field_validator("trigger")
    @classmethod
    def _strip_trigger(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("trigger must not be blank")
        return v


class GeneralSettings(_Section):
    launch_at_startup: bool = True
    theme: Literal["system", "dark", "light"] = "system"
    onboarding_completed: bool = False
    show_notifications: bool = True


class HotkeySettings(_Section):
    dictate: str = "ctrl+win"
    """Hold to talk; a quick tap locks hands-free mode."""
    command: str = "ctrl+win+alt"
    """Hold and speak an instruction to rewrite the selected text (needs AI polish)."""
    paste_last: str = "alt+shift+z"
    """Insert the most recent transcript again."""
    copy_last: str = "alt+shift+x"
    """Copy the most recent transcript to the clipboard."""
    hands_free: str = "ctrl+win+space"
    """Dedicated toggle for hands-free mode. Empty = not bound (a quick tap of the dictate key still works)."""
    tap_to_lock_ms: int = Field(default=350, ge=150, le=1000)
    """Presses shorter than this lock hands-free mode instead of push-to-talk."""


class AudioSettings(_Section):
    input_device: str | None = None
    """Device name, or None for the Windows default microphone."""
    sounds_enabled: bool = True
    sounds_volume: float = Field(default=0.6, ge=0.0, le=1.0)
    max_recording_minutes: int = Field(default=10, ge=1, le=60)
    mute_other_apps: bool = True
    """Silence music and videos while dictating. Call apps and anything using a microphone are left alone."""


class TranscriptionSettings(_Section):
    model_id: str = "parakeet-tdt-0.6b-v2"
    language: str = "auto"
    """``"auto"`` or an ISO-639-1 code. Only used by multilingual models."""
    device: Literal["auto", "gpu", "cpu"] = "auto"
    unload_after_minutes: int = Field(default=0, ge=0, le=1440)
    """Free the model from memory after this much idle time. 0 = keep it loaded."""
    live_preview: bool = True


class TextSettings(_Section):
    remove_fillers: bool = True
    voice_commands: bool = True
    """Understand "new line", "new paragraph" and "scratch that"."""
    smart_spacing: Literal["auto", "always", "never"] = "auto"
    dictionary: tuple[DictionaryEntry, ...] = ()
    snippets: tuple[Snippet, ...] = ()
    styles: dict[CategoryId, StyleId] = Field(
        default_factory=lambda: {
            "messaging": "casual",
            "email": "formal",
            "code": "verbatim",
            "document": "clean",
            "other": "clean",
        }
    )
    app_categories: dict[str, CategoryId] = Field(default_factory=dict)
    """User overrides: lower-case process name (e.g. ``"notion.exe"``) → category."""

    @field_validator("styles")
    @classmethod
    def _fill_styles(cls, v: dict[str, str]) -> dict[str, str]:
        defaults = {"messaging": "casual", "email": "formal", "code": "verbatim", "document": "clean", "other": "clean"}
        return {**defaults, **v}

    @field_validator("app_categories")
    @classmethod
    def _lower_keys(cls, v: dict[str, str]) -> dict[str, str]:
        return {k.strip().lower(): c for k, c in v.items() if k.strip()}


class PolishSettings(_Section):
    enabled: bool = False
    provider: Literal["builtin", "ollama", "openai"] = "builtin"
    builtin_model: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    openai_url: str = ""
    """Any OpenAI-compatible server, e.g. LM Studio at http://127.0.0.1:1234/v1."""
    openai_model: str = ""
    openai_api_key_protected: str = ""
    """API key encrypted with Windows DPAPI (see ``storage.secrets``). Never stored in plain text."""
    timeout_seconds: float = Field(default=6.0, ge=1.0, le=60.0)
    custom_instructions: str = Field(default="", max_length=2000)


class InsertionSettings(_Section):
    method: Literal["paste", "type"] = "paste"
    restore_clipboard: bool = True
    on_focus_change: Literal["insert", "clipboard"] = "clipboard"
    """If the focused window changed while processing: insert anyway, or leave text on the clipboard."""
    app_methods: dict[str, InsertMethod] = Field(default_factory=dict)
    """User overrides: lower-case process name → insertion method."""

    @field_validator("app_methods")
    @classmethod
    def _lower_keys(cls, v: dict[str, str]) -> dict[str, str]:
        return {k.strip().lower(): m for k, m in v.items() if k.strip()}


class OverlaySettings(_Section):
    idle_pill: bool = False
    """Show a small resting pill at the screen edge when not dictating. Off by default: it can sit over
    buttons near the bottom of the screen, such as a chat app's Send button."""
    position: Literal["bottom", "top"] = "bottom"
    show_on: Literal["active", "primary"] = "active"
    """Monitor of the focused window, or always the primary monitor."""
    hide_in_fullscreen: bool = True


class PrivacySettings(_Section):
    history_enabled: bool = True
    """Keep transcript text in local history. When off, only word counts and timings are kept for stats."""
    history_retention_days: int = Field(default=30, ge=0, le=3650)
    """Delete history older than this. 0 = keep forever."""


class Settings(_Section):
    schema_version: int = SCHEMA_VERSION
    general: GeneralSettings = GeneralSettings()
    hotkeys: HotkeySettings = HotkeySettings()
    audio: AudioSettings = AudioSettings()
    transcription: TranscriptionSettings = TranscriptionSettings()
    text: TextSettings = TextSettings()
    polish: PolishSettings = PolishSettings()
    insertion: InsertionSettings = InsertionSettings()
    overlay: OverlaySettings = OverlaySettings()
    privacy: PrivacySettings = PrivacySettings()


SECTIONS: dict[str, type[_Section]] = {
    name: field.annotation  # type: ignore[misc]
    for name, field in Settings.model_fields.items()
    if name != "schema_version"
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in _REPLACE_WHOLE:
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# Mapping-valued settings that a patch replaces outright instead of merging,
# so the UI can remove entries.
_REPLACE_WHOLE = {"app_categories", "app_methods"}


def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade older settings files. Version 1 is the first schema."""
    raw = dict(raw)
    raw["schema_version"] = SCHEMA_VERSION
    return raw


def parse_settings(raw: Any) -> tuple[Settings, list[str]]:
    """Validate ``raw`` into Settings, resetting only the sections that are invalid.

    Returns the settings and a list of human-readable problems found.
    """
    problems: list[str] = []
    if not isinstance(raw, dict):
        return Settings(), ["settings file did not contain an object; using defaults"]
    raw = _migrate(raw)
    try:
        return Settings.model_validate(raw), problems
    except ValidationError:
        pass
    sections: dict[str, Any] = {}
    for name, model in SECTIONS.items():
        value = raw.get(name, {})
        try:
            sections[name] = model.model_validate(value)
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in err["loc"]) for err in exc.errors()})
            problems.append(f"reset '{name}' to defaults (invalid: {', '.join(fields)})")
            sections[name] = model()
    return Settings(**sections), problems


class SettingsStore(QObject):
    """Loads, validates, saves and broadcasts settings.

    ``changed`` is emitted on the thread that called ``update``/``reset``
    (normally the GUI thread) with the set of section names that changed.
    """

    changed = Signal(object)

    def __init__(self, path: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._path = path
        self._lock = threading.RLock()
        self._save_problem: str | None = None
        self._settings = self._load()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def save_problem(self) -> str | None:
        """A user-facing message if the settings couldn't be written at start-up, else None.

        Whisp then runs on the settings in memory; later changes raise ``OSError`` as usual.
        """
        return self._save_problem

    @property
    def current(self) -> Settings:
        return self._settings

    def as_dict(self) -> dict[str, Any]:
        return self._settings.model_dump(mode="json")

    def update(self, patch: dict[str, Any]) -> Settings:
        """Apply a partial update such as ``{"audio": {"sounds_enabled": False}}``.

        Raises ``ValueError`` with a readable message if the result is invalid,
        and ``OSError`` if it couldn't be saved; in both cases nothing is changed.
        """
        unknown = set(patch) - set(SECTIONS)
        if unknown:
            raise ValueError(f"unknown settings section(s): {', '.join(sorted(unknown))}")
        with self._lock:
            before = self._settings
            merged = _deep_merge(before.model_dump(mode="json"), patch)
            try:
                after = Settings.model_validate(merged)
            except ValidationError as exc:
                raise ValueError(_format_validation_error(exc)) from exc
            changed = {name for name in SECTIONS if getattr(before, name) != getattr(after, name)}
            if not changed:
                return before
            # Saved first: a change that only lived in memory would take effect without ``changed`` ever
            # being sent, and trying again would then look like no change at all.
            self._save(after)
            self._settings = after
        self.changed.emit(changed)
        return after

    def reset(self, section: str | None = None) -> Settings:
        """Restore one section (or everything, except onboarding state) to defaults.

        Raises ``OSError`` if the result couldn't be saved; nothing is changed then.
        """
        with self._lock:
            before = self._settings
            if section is None:
                after = Settings(general=GeneralSettings(onboarding_completed=before.general.onboarding_completed))
            else:
                if section not in SECTIONS:
                    raise ValueError(f"unknown settings section: {section}")
                after = before.model_copy(update={section: SECTIONS[section]()})
            changed = {name for name in SECTIONS if getattr(before, name) != getattr(after, name)}
            self._save(after)
            self._settings = after
        if changed:
            self.changed.emit(changed)
        return after

    # -- persistence -------------------------------------------------------

    def _save_on_load(self, settings: Settings) -> None:
        """Write settings during start-up. A failure (full or read-only folder) mustn't stop Whisp starting."""
        try:
            self._save(settings)
        except OSError as exc:
            self._save_problem = MSG_CANT_SAVE
            log.warning(
                "Couldn't save settings to %s (%s); running with them in memory",
                display_path(self._path),
                display_path(exc),
            )

    def _load(self) -> Settings:
        if not self._path.exists():
            settings = Settings()
            self._save_on_load(settings)
            return settings
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            backup = self._path.with_name(f"settings.invalid-{time.strftime('%Y%m%d-%H%M%S')}.json")
            reason = display_path(exc)
            try:
                os.replace(self._path, backup)
                log.warning(
                    "Settings file was unreadable (%s); kept a copy at %s and reset to defaults",
                    reason,
                    display_path(backup),
                )
            except OSError:
                log.warning("Settings file was unreadable (%s); reset to defaults", reason)
            settings = Settings()
            self._save_on_load(settings)
            return settings
        settings, problems = parse_settings(raw)
        for problem in problems:
            log.warning("Settings: %s", problem)
        if problems or raw.get("schema_version") != SCHEMA_VERSION:
            self._save_on_load(settings)
        return settings

    def _save(self, settings: Settings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        data = json.dumps(settings.model_dump(mode="json"), indent=2, ensure_ascii=False)
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._path)


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"])
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)
