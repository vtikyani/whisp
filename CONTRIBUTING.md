# Contributing to Whisp

Thanks for helping. Bug reports, ideas and pull requests are all welcome.

## Set up

You need Windows 10 or 11 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                    # .venv with Python 3.12, Whisp and the dev tools
uv run whisp               # start Whisp from source
```

- **Exact versions:** `uv sync` installs the versions pinned in `uv.lock`. If you change dependencies in `pyproject.toml`, run `uv lock` and commit both files (CI checks they match).
- **GPU speed from source:** `uv sync --extra cuda` installs NVIDIA's CUDA libraries into `.venv`. (The packaged app downloads them on demand instead.)
- **Keep your real data safe:** set `WHISP_HOME` to a scratch folder and Whisp keeps its settings, history, models and logs there.
- **Diagnostics:** `uv run python -m whisp --diagnostics` prints versions, GPU status and file locations.

## Tests

```powershell
uv run pytest
```

This runs the fast tests. They never touch your real settings or history: a fixture points `WHISP_HOME` at a temporary folder.

Some tests need more than a plain PC, so they are marked and skipped by default. Run them with `-m`:

| Marker | Needs | Run with |
|---|---|---|
| `gpu` | An NVIDIA GPU and the CUDA libraries (`uv sync --extra cuda`) | `uv run pytest -m gpu` |
| `models` | Speech models. The first run downloads them (several GB) into `.local\test-models`, or into `WHISP_TEST_MODELS` if you set it. | `uv run pytest -m models` |
| `desktop` | The real Windows desktop: keyboard, clipboard and focused window. Don't touch the keyboard or mouse while they run. | `uv run pytest -m desktop` |

Combine them as usual, for example `uv run pytest -m "gpu or models"`.

## Lint and format

```powershell
uv run ruff check .
uv run ruff format .
```

The rules live in `pyproject.toml`. Lines can be 120 characters long.

## Find your way around

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is the map: every module, its public interface, which thread it runs on, and how one dictation flows from the hotkey to the inserted text. If you change an interface described there, update it in the same pull request.
- [docs/DESIGN.md](docs/DESIGN.md) is the visual language: colours, type, motion, the overlay and the window.
- `src/whisp/` holds the app: `core/` (audio, speech recognition, text), `platform/win32/` (hotkeys, clipboard, typing), `storage/` (settings, history), `ui/` (overlay and window) and `app/` (start-up and wiring).
- `scripts/` has tools: `render_overlay.py` and `render_main_window.py` save screenshots of every UI state, `benchmark_asr.py` measures speech recognition speed, `make_brand_assets.py` and `make_icon.py` generate the logo, icon and installer images, and `build.ps1` builds the app and installer.

A few rules hold everywhere:

1. **Never lose words.** Audio captured before the user stops is always transcribed and delivered.
2. **Never block the GUI thread.** No waiting, sleeping, network, disk-heavy or model work on it.
3. **Private by default.** Transcript text is never logged, and nothing leaves the PC unless the user sets up an external AI server.

## Build the app

```powershell
.\scripts\build.ps1              # dist\Whisp\Whisp.exe
.\scripts\build.ps1 -Installer   # also dist\Whisp-Setup-<version>.exe (needs Inno Setup 6.7+)
```

The build runs PyInstaller from its own environment, `build\env`, which uv makes from `uv.lock` each time. So the app holds exactly the locked versions, never whatever else is in `.venv`, and nothing is installed globally. The recipe is `packaging\whisp.spec`; the installer script is `packaging\whisp.iss`.

## Pull request checklist

- [ ] `uv run ruff check .` passes, and your changes are formatted with `uv run ruff format`.
- [ ] `uv run pytest` passes. If you changed GPU, model or desktop code, run those markers too.
- [ ] **Never log transcript text,** selections, clipboard contents or window titles. Log lengths and counts instead.
- [ ] No placeholder code, stubs or mock data in the app. Tests may use fakes.
- [ ] Tests that use the clipboard save and restore it.
- [ ] New downloads are pinned to an exact version and checked by SHA-256.
- [ ] New third-party code, fonts, icons or models come with their licence file and an entry in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- [ ] If you added a dependency or data files, the packaged app still starts: run `.\scripts\build.ps1`, then `dist\Whisp\Whisp.exe`.
- [ ] User-facing text is short, plain and in sentence case (see docs/DESIGN.md).
- [ ] No personal data in the commit: no names, email addresses, file paths from your PC, logs, recordings or transcripts.

## Reporting a bug

Open an issue and include:

- what you did, what you expected and what happened;
- the output of `Whisp.exe --diagnostics` (run it from a terminal in the install folder, `%LOCALAPPDATA%\Programs\Whisp`);
- the log, from Settings → About → Logs (it opens `%LOCALAPPDATA%\Whisp\logs`).

Logs never contain what you said, but they do contain file paths, which include your Windows user name. Check before you share.

For security problems, please follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
