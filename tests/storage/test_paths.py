"""storage.paths: how file paths appear in the log."""

from __future__ import annotations

from pathlib import Path

import pytest

from whisp.storage.paths import display_path


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Path:
    fake = Path(r"C:\Users\sam")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake))
    return fake


def test_the_users_folder_is_written_as_a_tilde(home: Path) -> None:
    log_file = home / "AppData" / "Local" / "Whisp" / "logs" / "whisp.log"
    assert display_path(log_file) == r"~\AppData\Local\Whisp\logs\whisp.log"
    assert display_path(home) == "~"
    assert display_path(r"c:\users\SAM\AppData") == r"~\AppData"  # Windows ignores case
    assert display_path("C:/Users/sam/AppData") == "~/AppData"


def test_other_folders_are_left_alone(home: Path) -> None:
    assert display_path(r"C:\Users\samuel\notes.txt") == r"C:\Users\samuel\notes.txt"
    assert display_path(r"D:\Whisp\data") == r"D:\Whisp\data"
    assert display_path("") == ""


def test_paths_inside_error_messages_are_covered_too(home: Path) -> None:
    error = FileNotFoundError(2, "No such file or directory", str(home / "AppData" / "Roaming" / "settings.json"))
    shown = display_path(error)
    assert "sam" not in shown.lower()
    assert r"~\\AppData\\Roaming" in shown  # error messages double their backslashes
