"""AppServices: every long-lived service, built once in ``app.main`` and handed to the UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisp.app.controller import AppController
    from whisp.app.delivery import DeliveryService
    from whisp.app.sounds import SoundPlayer
    from whisp.core.asr.manager import ModelManager
    from whisp.core.asr.service import AsrService
    from whisp.core.llm.runtime import LlmRuntime
    from whisp.core.session import SessionPipeline
    from whisp.core.text.polish import Polisher
    from whisp.platform.win32.audio_mute import MuteService
    from whisp.platform.win32.foreground import ForegroundWatcher
    from whisp.platform.win32.hotkeys import HotkeyService
    from whisp.platform.win32.insert import TextInserter
    from whisp.storage.history import HistoryStore
    from whisp.storage.paths import AppPaths
    from whisp.storage.settings import SettingsStore
    from whisp.ui.overlay import Overlay


@dataclass
class AppServices:
    paths: AppPaths
    settings: SettingsStore
    history: HistoryStore
    models: ModelManager
    asr: AsrService
    pipeline: SessionPipeline
    llm: LlmRuntime
    polisher: Polisher
    hotkeys: HotkeyService
    inserter: TextInserter
    foreground: ForegroundWatcher
    overlay: Overlay
    sounds: SoundPlayer
    mute: MuteService
    delivery: DeliveryService
    controller: AppController
