"""Where Whisp keeps its files.

Settings roam with the Windows profile (%APPDATA%\\Whisp). Everything large or
machine-specific — models, runtimes, history, logs — stays local
(%LOCALAPPDATA%\\Whisp). Tests and portable setups can redirect both by setting
the WHISP_HOME environment variable to a single folder.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from whisp import APP_NAME


@dataclass(frozen=True)
class AppPaths:
    config_dir: Path
    data_dir: Path

    @property
    def settings_file(self) -> Path:
        return self.config_dir / "settings.json"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def runtime_dir(self) -> Path:
        """Downloaded runtimes (GPU libraries, local LLM server)."""
        return self.data_dir / "runtime"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def history_db(self) -> Path:
        return self.data_dir / "history.db"

    def ensure(self) -> None:
        for d in (self.config_dir, self.data_dir, self.models_dir, self.runtime_dir, self.logs_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


def _default_paths() -> AppPaths:
    override = os.environ.get("WHISP_HOME")
    if override:
        root = Path(override).expanduser().resolve()
        return AppPaths(config_dir=root / "config", data_dir=root / "data")
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return AppPaths(config_dir=Path(appdata) / APP_NAME, data_dir=Path(local) / APP_NAME)


_paths: AppPaths | None = None


def app_paths() -> AppPaths:
    """Process-wide paths, created on first use."""
    global _paths
    if _paths is None:
        _paths = _default_paths()
        _paths.ensure()
    return _paths


def display_path(value: object) -> str:
    """``value`` for the log, with the user's own folder written as ``~``, so logs don't carry the account name.

    Works on any text that contains a path, such as an error message (where paths appear with doubled
    backslashes), and with either kind of slash. Windows paths ignore case.
    """
    text = str(value)
    home = Path.home()
    if home.parent == home:  # no real home folder (a drive root): nothing to hide
        return text
    plain = str(home).rstrip("\\/")
    for folder in dict.fromkeys((plain, plain.replace("\\", "\\\\"), plain.replace("\\", "/"))):
        text = _replace_folder(text, folder)
    return text


def _replace_folder(text: str, folder: str) -> str:
    lowered, needle = text.lower(), folder.lower()
    out: list[str] = []
    start = 0
    while (found := lowered.find(needle, start)) >= 0:
        end = found + len(needle)
        # Only the whole folder name: C:\Users\sam must not swallow the start of C:\Users\samuel.
        if end == len(text) or text[end] in "\\/'\"":
            out.append(text[start:found] + "~")
        else:
            out.append(text[start:end])
        start = end
    out.append(text[start:])
    return "".join(out)


def resource_path(*parts: str) -> Path:
    """Path to a file shipped inside the package (works from source and frozen builds)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    if getattr(sys, "_MEIPASS", None):
        base = base / "whisp"
    return base.joinpath(*parts)
