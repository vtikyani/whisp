"""Shared pytest setup.

Every test runs with WHISP_HOME pointing at a fresh temporary folder, so no
test can read or modify the real user's settings, history or models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolated_whisp_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path_factory.mktemp("whisp-home")
    monkeypatch.setenv("WHISP_HOME", str(home))
    import whisp.storage.paths as paths

    monkeypatch.setattr(paths, "_paths", None)
    yield home
    monkeypatch.setattr(paths, "_paths", None)


@pytest.fixture
def settings_store(isolated_whisp_home: Path):
    from whisp.storage.paths import app_paths
    from whisp.storage.settings import SettingsStore

    return SettingsStore(app_paths().settings_file)


@pytest.fixture(scope="session")
def shared_models_dir() -> Path:
    """Where tests marked ``models`` cache downloaded speech models between runs.

    Kept outside WHISP_HOME so large downloads happen once. Override with
    WHISP_TEST_MODELS.
    """
    root = Path(os.environ.get("WHISP_TEST_MODELS", Path(__file__).resolve().parent.parent / ".local" / "test-models"))
    root.mkdir(parents=True, exist_ok=True)
    return root
