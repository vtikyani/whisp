import os
import yaml

DEFAULTS = {
    "whisper_model": "large-v3",
    "hotkey": "ctrl+shift+space",
    "vad_silence_ms": 300,
    "vad_threshold": 0.5,
    "vad_max_phrase_ms": 3000,
    "audio_feedback": True,
    "auto_start": True,
    "waveform_bars": 7,
    "language": "en",
    "typing_delay_ms": 10,
    "hallucination_filter": True,
    "min_audio_rms": 0.005,
}

CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")


def load_config(path=None):
    path = path or CONFIG_PATH
    config = dict(DEFAULTS)

    if os.path.exists(path):
        with open(path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
    else:
        save_config(config, path)

    return config


def save_config(config, path=None):
    path = path or CONFIG_PATH
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
