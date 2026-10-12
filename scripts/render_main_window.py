"""Render the main window's pages to PNG files for visual review.

Opens the real MainWindow and Bridge with a real SettingsStore in a temporary
WHISP_HOME. Real code is used wherever it is safe: HistoryStore (when it
exists), shortcut names (platform.win32.keys), the text rules behind the
Styles previews, and the LLM catalog. Fakes defined here stand in for the
speech-model modules that don't exist yet and for anything that would touch
hardware, the registry or the network (microphone, GPU, global hotkeys,
downloads, llama-server), so every render is repeatable. The sample
dictations, dictionary words and snippets below exist only for these
screenshots.

Usage (from the repo root):
    set PYTHONPATH=src
    .venv\\Scripts\\python.exe scripts\\render_main_window.py [--out DIR] [--only home,settings-model]
        [--themes dark,light]

PNGs go to DIR (default: whisp-window-renders in the system temp folder).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(tempfile.gettempdir()) / "whisp-window-renders"

# --------------------------------------------------------------------------------------------
# Shots: (file name, theme, page, section, (width, height), scenario)
# --------------------------------------------------------------------------------------------

# (name, page, section, scenario, JavaScript to run before the capture)
PAGES = [
    ("home", "home", None, "full", None),
    ("home-new-user", "home", None, "fresh", None),
    ("home-no-model", "home", None, "no-model", None),
    ("history", "history", None, "full", None),
    ("history-expanded", "history", None, "full", "document.querySelector('.h-body').click()"),
    ("history-dialog", "history", None, "full", "document.querySelector('.page-actions .btn').click()"),
    ("dictionary", "dictionary", None, "full", None),
    ("dictionary-editing", "dictionary", None, "full", "document.querySelector('.entry .icon-btn').click()"),
    ("dictionary-toast", "dictionary", None, "full", "document.querySelector('.entry .icon-btn.is-danger').click()"),
    ("snippets", "snippets", None, "full", None),
    ("styles", "styles", None, "full", None),
    ("styles-polish-builtin", "styles", None, "full", "document.querySelector('.polish-card').scrollIntoView()"),
    (
        "styles-polish-openai",
        "styles",
        None,
        "full",
        "[...document.querySelectorAll('.polish-card .segmented button')].find(b => b.textContent.includes('OpenAI'))"
        ".click(); setTimeout(() => document.querySelector('.polish-card').scrollIntoView(), 200)",
    ),
    ("settings-general", "settings", "general", "full", None),
    ("settings-shortcuts", "settings", "shortcuts", "full", None),
    (
        "settings-shortcuts-recording",
        "settings",
        "shortcuts",
        "full",
        "document.querySelector('.shortcut .btn').click()",
    ),
    ("settings-microphone", "settings", "microphone", "full", None),
    ("settings-model", "settings", "model", "downloading", None),
    ("settings-typing", "settings", "typing", "full", None),
    ("settings-overlay", "settings", "overlay", "full", None),
    ("settings-privacy", "settings", "privacy", "full", None),
    ("settings-about", "settings", "about", "full", None),
    ("onboarding-welcome", "onboarding", "welcome", "fresh", None),
    ("onboarding-microphone", "onboarding", "microphone", "fresh", None),
    ("onboarding-model", "onboarding", "model", "fresh", None),
    ("onboarding-shortcut", "onboarding", "shortcut", "fresh", None),
    ("onboarding-practice", "onboarding", "practice", "fresh", None),
    ("onboarding-done", "onboarding", "done", "fresh", None),
]

EXTRA_SIZES = [
    ("home-900x600", "home", None, "full", (900, 600)),
    ("settings-model-900x600", "settings", "model", "downloading", (900, 600)),
    ("styles-900x600", "styles", None, "full", (900, 600)),
    ("onboarding-model-900x600", "onboarding", "model", "fresh", (900, 600)),
    ("home-1600x1000", "home", None, "full", (1600, 1000)),
]

# Closes anything an action opened, so the next shot starts clean.
CLEANUP_JS = (
    "document.querySelector('dialog[open] .btn-secondary')?.click();"
    "window.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'}));"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", default="", help="comma-separated shot names (without theme suffix)")
    parser.add_argument("--themes", default="dark,light")
    parser.add_argument("--settle", type=int, default=900, help="milliseconds to wait after navigating")
    parser.add_argument("--no-extra", action="store_true", help="skip the extra window sizes")
    return parser.parse_args()


# --------------------------------------------------------------------------------------------
# Sample data (screenshots only)
# --------------------------------------------------------------------------------------------

SAMPLE_TEXTS = {
    "Slack": [
        "Sounds good, I'll pick this up after lunch and ping you when the PR is ready.",
        "Can we push the design review to Thursday? I want the latest numbers in the deck first.",
        "Nice work on the release notes, they read really well.",
        "I'm running about five minutes late to standup, start without me.",
    ],
    "Outlook": [
        "Hi Maya, thanks for sending the contract over. I've read it and everything looks good to me, so feel "
        "free to go ahead and countersign.",
        "Hi team, quick update on the launch. We're on track for the 14th, and the only open item is the "
        "pricing page copy.",
        "Thanks for the intro, Daniel. Priya, great to meet you. Would Tuesday afternoon work for a call?",
    ],
    "Visual Studio Code": [
        "Add a retry loop around the upload call and log the error if it still fails.",
        "Rename this function to parse settings file and make the path argument optional.",
    ],
    "Notion": [
        "The main finding is that response times dropped by about thirty percent after the cache change, "
        "with the biggest gains on cold starts.",
        "Next steps: confirm the rollout plan with infra, then schedule the migration for the second week of November.",
    ],
    "Microsoft Teams": [
        "Yes, I can join. I'll share my screen and walk through the dashboard.",
        "Let's keep the meeting to thirty minutes and send notes afterwards.",
    ],
    "Google Chrome": [
        "best hiking trails near Lake District with waterfalls",
        "how to set up a reverse proxy with caddy",
    ],
}
APP_PROCESSES = {
    "Slack": ("slack.exe", "messaging"),
    "Outlook": ("olk.exe", "email"),
    "Visual Studio Code": ("code.exe", "code"),
    "Notion": ("notion.exe", "document"),
    "Microsoft Teams": ("ms-teams.exe", "messaging"),
    "Google Chrome": ("chrome.exe", "other"),
}
APP_WEIGHTS = {"Slack": 9, "Outlook": 5, "Visual Studio Code": 4, "Notion": 4, "Microsoft Teams": 3, "Google Chrome": 2}

DICTIONARY = [
    {"term": "Kubernetes", "sounds_like": ["cooper netties", "kubernetties"]},
    {"term": "SvelteKit", "sounds_like": ["svelte kit"]},
    {"term": "Anaïs", "sounds_like": ["anise", "anna ease"]},
    {"term": "Postgres", "sounds_like": ["post grass", "postgress"]},
    {"term": "OKRs", "sounds_like": ["okay ours"]},
]
SNIPPETS = [
    {"trigger": "my calendar link", "text": "https://cal.com/sam-rivera/30min"},
    {"trigger": "my address", "text": "Flat 4, 12 Albion Road\nLondon N16 9PH"},
    {
        "trigger": "sign off",
        "text": "Thanks so much, and have a great rest of your week.\n\nSam Rivera\nProduct Lead",
    },
]


@dataclass(frozen=True)
class SampleEntry:
    """Mirrors HistoryEntry from docs/ARCHITECTURE.md."""

    id: int
    created_at: float
    kind: str
    app_name: str
    process_name: str
    category: str
    raw_text: str
    final_text: str
    word_count: int
    audio_ms: int
    speech_ms: int
    latency_ms: int
    model_id: str
    polished: bool
    outcome: str


def sample_entries(seed: int = 7) -> list[SampleEntry]:
    rng = random.Random(seed)
    now = datetime.now()
    entries: list[SampleEntry] = []
    next_id = 1
    for days_ago in range(29, -1, -1):
        if days_ago in (23, 22, 16, 9) or (days_ago > 3 and rng.random() < 0.12):
            continue
        count = rng.randint(2, 9) if days_ago else 5
        day = now - timedelta(days=days_ago)
        previous = ""
        for _ in range(count):
            app = rng.choices(list(APP_WEIGHTS), weights=list(APP_WEIGHTS.values()))[0]
            text = rng.choice([t for t in SAMPLE_TEXTS[app] if t != previous] or SAMPLE_TEXTS[app])
            previous = text
            words = len(text.split()) * rng.randint(1, 3)
            moment = day.replace(hour=rng.randint(8, 19), minute=rng.randint(0, 59), second=rng.randint(0, 59))
            if moment > now:
                moment = now - timedelta(minutes=rng.randint(5, 240))
            speech_ms = int(words / rng.uniform(125, 165) * 60000)
            process, category = APP_PROCESSES[app]
            polished = rng.random() < 0.3
            raw = "Um, " + text[0].lower() + text[1:] if polished else text
            entries.append(
                SampleEntry(
                    id=next_id,
                    created_at=moment.timestamp(),
                    kind="dictation",
                    app_name=app,
                    process_name=process,
                    category=category,
                    raw_text=raw,
                    final_text=text,
                    word_count=words,
                    audio_ms=speech_ms + rng.randint(400, 1500),
                    speech_ms=speech_ms,
                    latency_ms=rng.randint(180, 420),
                    model_id="parakeet-tdt-0.6b-v2",
                    polished=polished,
                    outcome="inserted",
                )
            )
            next_id += 1
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return entries


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------


def build_fakes() -> dict[str, Any]:
    from PySide6.QtCore import QObject, QTimer, Signal

    from whisp.core.types import AudioDevice, EngineState, EngineStatus
    from whisp.ui.main_window.bridge import BridgeAdapters

    @dataclass(frozen=True)
    class ModelFile:
        path: str
        size: int
        sha256: str = ""

    @dataclass(frozen=True)
    class ModelVariant:
        key: str
        quantization: str | None
        files: tuple[ModelFile, ...]

    @dataclass(frozen=True)
    class ModelInfo:
        id: str
        onnx_asr_name: str
        name: str
        description: str
        languages: tuple[str, ...]
        accuracy: int
        speed: int
        license: str
        license_url: str
        repo_id: str
        revision: str
        variants: tuple[ModelVariant, ...]
        recommended: bool

    european = ("bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt")
    european += ("pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk")
    catalog = (
        ModelInfo(
            "parakeet-tdt-0.6b-v2",
            "nemo-parakeet-tdt-0.6b-v2",
            "Parakeet TDT 0.6B v2",
            "NVIDIA's fast, very accurate English model. The best choice for most people.",
            ("en",),
            5,
            5,
            "CC-BY-4.0",
            "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
            "istupakov/parakeet-tdt-0.6b-v2-onnx",
            "main",
            (
                ModelVariant("gpu", None, (ModelFile("encoder.onnx", 2_440_000_000),)),
                ModelVariant("cpu", "int8", (ModelFile("encoder.int8.onnx", 652_000_000),)),
            ),
            True,
        ),
        ModelInfo(
            "parakeet-tdt-0.6b-v3",
            "nemo-parakeet-tdt-0.6b-v3",
            "Parakeet TDT 0.6B v3",
            "The same speed with 25 European languages, detected automatically.",
            european,
            4,
            5,
            "CC-BY-4.0",
            "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
            "istupakov/parakeet-tdt-0.6b-v3-onnx",
            "main",
            (
                ModelVariant("gpu", None, (ModelFile("encoder.onnx", 2_510_000_000),)),
                ModelVariant("cpu", "int8", (ModelFile("encoder.int8.onnx", 671_000_000),)),
            ),
            False,
        ),
        ModelInfo(
            "whisper-large-v3-turbo",
            "whisper-ort",
            "Whisper Large v3 Turbo",
            "OpenAI's Whisper, tuned for speed. Understands 99 languages.",
            ("en", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "ru", "sv", "tr", "uk", "zh"),
            4,
            3,
            "MIT",
            "https://huggingface.co/openai/whisper-large-v3-turbo",
            "onnx-community/whisper-large-v3-turbo",
            "main",
            (
                ModelVariant("gpu", None, (ModelFile("model.onnx", 1_620_000_000),)),
                ModelVariant("cpu", "int8", (ModelFile("model.int8.onnx", 870_000_000),)),
            ),
            False,
        ),
    )

    from whisp.core.asr.gpu import GpuInfo

    gpu = GpuInfo(
        True, "NVIDIA GeForce RTX 4070", 12282, True, "CUDA is ready. Speech models run on your graphics card."
    )

    class FakeStream:
        error: str | None = None

        def __init__(self, device: str | None, on_level: Any) -> None:
            self._on_level = on_level
            self._stop = threading.Event()
            self.device_name = device or "Microphone (Shure MV7)"
            threading.Thread(target=self._run, daemon=True).start()

        def _run(self) -> None:
            start = time.monotonic()
            while not self._stop.is_set():
                t = time.monotonic() - start
                syllables = max(0.0, math.sin(t * 9.0)) * (0.55 + 0.45 * math.sin(t * 1.3))
                level = 0.08 + 0.62 * syllables * (0.5 + 0.5 * math.sin(t * 0.7 + 1))
                self._on_level(max(0.0, min(1.0, level)))
                time.sleep(1 / 30)

        def stop(self) -> None:
            self._stop.set()

    class FakeAdapters(BridgeAdapters):
        def list_input_devices(self) -> list[AudioDevice]:
            return [
                AudioDevice("Microphone (Shure MV7)", "Microphone (Shure MV7)", "Windows WASAPI", True, 1, 48000.0),
                AudioDevice("Headset (WH-1000XM5)", "Headset (WH-1000XM5)", "Windows WASAPI", False, 1, 16000.0),
                AudioDevice(
                    "Microphone Array (Realtek)", "Microphone Array (Realtek)", "Windows WASAPI", False, 2, 48000.0
                ),
            ]

        def start_microphone(self, device: str | None, on_level: Any) -> FakeStream:
            return FakeStream(device, on_level)

        def detect_gpu(self) -> GpuInfo:
            return gpu

        def model_catalog(self) -> tuple[ModelInfo, ...]:
            return catalog

        def autostart_status(self) -> str:
            return "enabled"  # never touch the real Run key from a render

        def set_autostart(self, enabled: bool) -> None:
            return None

        def protect_secret(self, plain: str) -> str:
            return "render-only-token"

        def user_first_name(self) -> str | None:
            return "Sam"

    class FakeHistory(QObject):
        added = Signal(object)
        changed = Signal()

        persistent = True

        def __init__(self, entries: list[SampleEntry]) -> None:
            super().__init__()
            self.entries = list(entries)

        def set_entries(self, entries: list[SampleEntry]) -> None:
            self.entries = list(entries)
            self.changed.emit()

        def add(self, entry: SampleEntry) -> SampleEntry:
            entry = replace(entry, id=max((e.id for e in self.entries), default=0) + 1)
            self.entries.insert(0, entry)
            self.added.emit(entry)
            return entry

        def list(self, *, query: str = "", app: str = "", limit: int = 50, offset: int = 0) -> list[SampleEntry]:
            items = [
                e
                for e in self.entries
                if (not query or query.lower() in e.final_text.lower()) and (not app or e.app_name == app)
            ]
            return items[offset : offset + limit]

        def get(self, entry_id: int) -> SampleEntry | None:
            return next((e for e in self.entries if e.id == entry_id), None)

        def delete(self, entry_id: int) -> None:
            self.entries = [e for e in self.entries if e.id != entry_id]
            self.changed.emit()

        def clear(self) -> None:
            self.entries = []
            self.changed.emit()

        def prune(self) -> int:
            return 0

        def last_text(self) -> str | None:
            return self.entries[0].final_text if self.entries else None

        def apps(self) -> list[str]:
            return sorted({e.app_name for e in self.entries})

        def stats(self) -> dict[str, Any]:
            today = datetime.now().date()
            per_day = Counter()
            for e in self.entries:
                per_day[datetime.fromtimestamp(e.created_at).date()] += e.word_count
            daily = [
                {"date": (today - timedelta(days=d)).isoformat(), "words": per_day.get(today - timedelta(days=d), 0)}
                for d in range(29, -1, -1)
            ]
            words = sum(e.word_count for e in self.entries)
            speech_min = sum(e.speech_ms for e in self.entries) / 60000
            streak = 0
            for d in range(0, 60):
                if per_day.get(today - timedelta(days=d)):
                    streak += 1
                else:
                    break
            apps = Counter()
            for e in self.entries:
                apps[e.app_name] += e.word_count
            week_start = today - timedelta(days=today.weekday())
            return {
                "total_words": words,
                "total_dictations": len(self.entries),
                "words_today": per_day.get(today, 0),
                "words_this_week": sum(v for k, v in per_day.items() if k >= week_start),
                "avg_wpm": round(words / speech_min) if speech_min else 0,
                "time_saved_minutes": max(0, round(words / 40 - speech_min)),
                "streak_days": streak,
                "best_streak_days": max(streak, 9),
                "top_apps": [{"app": a, "words": w} for a, w in apps.most_common(5)],
                "daily": daily,
            }

    class FakeModels(QObject):
        progress = Signal(str, object, object)
        changed = Signal()
        download_failed = Signal(str, str)

        def __init__(self) -> None:
            super().__init__()
            self.installed: set[tuple[str, str]] = {("parakeet-tdt-0.6b-v2", "gpu")}
            self.downloading: dict[str, list[int]] = {}
            self._timer = QTimer(self)
            self._timer.setInterval(120)
            self._timer.timeout.connect(self._tick)

        def is_installed(self, model_id: str, variant: str) -> bool:
            return (model_id, variant) in self.installed

        def model_path(self, model_id: str, variant: str) -> Path:
            return Path(model_id) / variant

        def install_state(self) -> dict[str, dict]:
            return {mid: {"downloading": True, "progress": tuple(p)} for mid, p in self.downloading.items()}

        def start_fake_download(self, model_id: str, done_fraction: float) -> None:
            total = sum(f.size for f in catalog[2].variants[0].files)
            self.downloading[model_id] = [int(total * done_fraction), total]
            self._timer.start()

        def download(self, model_id: str, variant: str) -> None:
            info = next(m for m in catalog if m.id == model_id)
            total = sum(f.size for v in info.variants if v.key == variant for f in v.files)
            self.downloading[model_id] = [0, total]
            self._timer.start()

        def _tick(self) -> None:
            for model_id, state in list(self.downloading.items()):
                state[0] = min(state[1], state[0] + state[1] // 400)
                self.progress.emit(model_id, state[0], state[1])

        def cancel(self, model_id: str) -> None:
            self.downloading.pop(model_id, None)
            self.changed.emit()

        def delete(self, model_id: str) -> None:
            self.installed = {item for item in self.installed if item[0] != model_id}
            self.changed.emit()

        def set_installed(self, items: set[tuple[str, str]]) -> None:
            self.installed = set(items)
            self.changed.emit()

    class FakeAsr(QObject):
        status_changed = Signal(object)
        gpu_runtime_progress = Signal(object, object)
        gpu_runtime_finished = Signal(object)
        gpu_runtime_failed = Signal(str)

        def __init__(self) -> None:
            super().__init__()
            self.status = EngineStatus(EngineState.READY, "parakeet-tdt-0.6b-v2", "cuda", "NVIDIA GeForce RTX 4070")
            self.gpu_runtime_installing = False

        def set_status(self, status: EngineStatus) -> None:
            self.status = status
            self.status_changed.emit(status)

        def install_gpu_runtime(self) -> bool:
            self.gpu_runtime_installing = True
            return True

        def cancel_gpu_runtime(self) -> None:
            self.gpu_runtime_installing = False

    from whisp.core.llm.catalog import CATALOG as LLM_CATALOG
    from whisp.core.llm.catalog import recommended_model
    from whisp.core.llm.runtime import LlmState, LlmStatus

    class FakeLlm(QObject):
        """The real LLM catalog and status types, without starting llama-server or downloading."""

        status_changed = Signal(object)
        progress = Signal(str, object, object)
        install_finished = Signal(str)
        install_failed = Signal(str, str)

        def __init__(self) -> None:
            super().__init__()
            chosen = recommended_model().id
            self.status = LlmStatus(
                LlmState.READY, chosen, "http://127.0.0.1:8178", None, "NVIDIA graphics card (CUDA 13)"
            )
            self._installed = {chosen}

        def catalog(self) -> list[Any]:
            return list(LLM_CATALOG)

        def is_installed(self, model_id: str) -> bool:
            return model_id in self._installed

        def install(self, model_id: str) -> None:
            return None

        def cancel_install(self) -> None:
            return None

        def uninstall(self, model_id: str) -> None:
            self._installed.discard(model_id)

    class FakePolisher:
        def is_ready(self) -> bool:
            return True

        def test(self) -> tuple[bool, str]:
            time.sleep(0.4)
            return True, "Polish works. A test sentence came back in 0.6 s."

        def list_models(self, provider: str) -> list[str]:
            return ["llama3.2:3b", "qwen2.5:7b", "gemma3:4b"]

    class FakeHotkeys(QObject):
        pressed = Signal(str)
        released = Signal(str)
        interrupted = Signal(str)
        recorded = Signal(str)
        binding_error = Signal(str, str)

        def start_recording(self) -> None:
            return None

        def stop_recording(self) -> None:
            return None

    class FakeController(QObject):
        state_changed = Signal()
        notification = Signal(str, str, str)
        open_window_requested = Signal(str, str)

        def __init__(self) -> None:
            super().__init__()
            self.binding_errors: dict[str, str] = {}
            self.is_paused = False

        def problem(self) -> str | None:
            return None

        def toggle_paused(self) -> None:
            self.is_paused = not self.is_paused
            self.state_changed.emit()

    return {
        "controller": FakeController(),
        "adapters": FakeAdapters(),
        "history": FakeHistory,
        "models": FakeModels(),
        "asr": FakeAsr(),
        "llm": FakeLlm(),
        "polisher": FakePolisher(),
        "hotkeys": FakeHotkeys(),
        "EngineStatus": EngineStatus,
        "EngineState": EngineState,
    }


@dataclass
class Services:
    """The AppServices fields the main window uses."""

    paths: Any
    settings: Any
    history: Any
    models: Any
    asr: Any
    llm: Any
    polisher: Any
    hotkeys: Any
    controller: Any
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    home = Path(tempfile.mkdtemp(prefix="whisp-render-"))
    os.environ["WHISP_HOME"] = str(home)
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    sys.path.insert(0, str(REPO / "src"))

    from PySide6.QtCore import QTimer
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    from whisp.storage.paths import app_paths
    from whisp.storage.settings import SettingsStore
    from whisp.ui.main_window.window import MainWindow, register_web_scheme

    register_web_scheme()
    app = QApplication(sys.argv)
    paths = app_paths()
    settings = SettingsStore(paths.settings_file)
    fakes = build_fakes()
    settings.update(
        {
            "general": {"onboarding_completed": True, "theme": "dark"},
            "text": {"dictionary": DICTIONARY, "snippets": SNIPPETS, "app_categories": {"notion.exe": "document"}},
            "polish": {"enabled": True, "provider": "builtin", "builtin_model": fakes["llm"].status.model_id},
            "insertion": {"app_methods": {"windowsterminal.exe": "type"}},
        }
    )
    entries = sample_entries()
    try:
        from whisp.storage.history import HistoryEntry, HistoryStore  # the real store, once it exists
    except ImportError:
        history = fakes["history"](entries)
    else:
        from dataclasses import asdict

        history = HistoryStore(paths.history_db, settings)
        for entry in reversed(entries):
            history.add(HistoryEntry(**asdict(entry)))
        print("Using the real HistoryStore")

    services = Services(
        paths=paths,
        settings=settings,
        history=history,
        models=fakes["models"],
        asr=fakes["asr"],
        llm=fakes["llm"],
        polisher=fakes["polisher"],
        hotkeys=fakes["hotkeys"],
        controller=fakes["controller"],
    )
    window = MainWindow(services, adapters=fakes["adapters"])  # type: ignore[arg-type]
    window.move(40, 40)
    window.resize(1080, 720)
    window.show()
    view = window.findChild(QWebEngineView)

    themes = [t for t in args.themes.split(",") if t]
    only = {name for name in args.only.split(",") if name}
    Shot = tuple[str, str, str, str | None, tuple[int, int], str, str | None]
    shots: list[Shot] = []
    for theme in themes:
        for name, page, section, scenario, action in PAGES:
            if not only or name in only:
                shots.append((f"{name}-{theme}", theme, page, section, (1080, 720), scenario, action))
    if not args.no_extra and "dark" in themes:
        for name, page, section, scenario, size in EXTRA_SIZES:
            if not only or name in only:
                shots.append((f"{name}-dark", "dark", page, section, size, scenario, None))

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    EngineStatus, EngineState = fakes["EngineStatus"], fakes["EngineState"]

    def apply_scenario(scenario: str, page: str) -> None:
        fresh = scenario == "fresh"
        settings.update(
            {
                "general": {"onboarding_completed": page != "onboarding"},
                "text": {"dictionary": DICTIONARY, "snippets": SNIPPETS},
                "polish": {"provider": "builtin"},
            }
        )
        if scenario == "no-model":
            services.models.downloading.clear()
            services.models.set_installed(set())
            services.asr.set_status(EngineStatus(EngineState.NOT_INSTALLED, "parakeet-tdt-0.6b-v2"))
            return
        if fresh:
            if hasattr(history, "set_entries"):
                history.set_entries([])
            else:
                history.clear()
            if page == "onboarding":
                services.models.set_installed(set())
                services.models.downloading.clear()
                services.asr.set_status(EngineStatus(EngineState.NOT_INSTALLED, "parakeet-tdt-0.6b-v2"))
        else:
            if hasattr(history, "set_entries"):
                if not history.entries:
                    history.set_entries(entries)
            elif not history.list(limit=1):
                from dataclasses import asdict

                from whisp.storage.history import HistoryEntry

                for entry in reversed(entries):
                    history.add(HistoryEntry(**asdict(entry)))
            services.models.set_installed({("parakeet-tdt-0.6b-v2", "gpu"), ("parakeet-tdt-0.6b-v3", "cpu")})
            services.asr.set_status(
                EngineStatus(EngineState.READY, "parakeet-tdt-0.6b-v2", "cuda", "NVIDIA GeForce RTX 4070")
            )
            services.models.downloading.clear()
            if scenario == "downloading":
                services.models.start_fake_download("whisper-large-v3-turbo", 0.42)

    state = {"index": 0}

    def next_shot() -> None:
        if state["index"] >= len(shots):
            print(f"Saved {len(shots)} renders to {out_dir}")
            window.close()
            app.quit()
            return
        name, theme, page, section, size, scenario, action = shots[state["index"]]
        settings.update({"general": {"theme": theme}})
        apply_scenario(scenario, page)
        window.resize(*size)
        window.show_page(page, section)
        if action:
            QTimer.singleShot(args.settle, lambda: view.page().runJavaScript(action))
            QTimer.singleShot(args.settle + 600, lambda: capture(name))
        else:
            QTimer.singleShot(args.settle, lambda: capture(name))

    def capture(name: str) -> None:
        image = view.grab()
        path = out_dir / f"{name}.png"
        image.save(str(path))
        print(f"  {path.name}  {image.width()}x{image.height()}")
        state["index"] += 1
        view.page().runJavaScript(CLEANUP_JS)
        QTimer.singleShot(150, next_shot)

    def on_loaded(ok: bool) -> None:
        if not ok:
            print("The page failed to load", file=sys.stderr)
            app.exit(1)
            return
        view.loadFinished.disconnect(on_loaded)
        QTimer.singleShot(1200, next_shot)

    view.loadFinished.connect(on_loaded)
    code = app.exec()
    shutil.rmtree(home, ignore_errors=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
