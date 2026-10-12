"""Real speech audio for tests, made with Windows' built-in text-to-speech.

Everything returned here is mono float32 at 16 kHz (``whisp.core.types.SAMPLE_RATE``),
the format the whole app uses::

    from helpers.speech import concat, noise, silence, synthesize

    audio = concat(silence(0.5), synthesize("Turn left at the second light."), silence(1.0))

``synthesize`` drives ``System.Speech.Synthesis.SpeechSynthesizer`` (SAPI)
through Windows PowerShell 5.1, preferring an installed en-US voice. Results
are cached as WAV files in ``%TEMP%\\whisp-test-speech``, keyed by a hash of the
text and rate, so each phrase is synthesised once per machine.

Stable API: ``synthesize``, ``silence``, ``noise``, ``concat``, ``SAMPLE_RATE``.
"""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

from whisp.core.types import SAMPLE_RATE

__all__ = ["SAMPLE_RATE", "concat", "noise", "silence", "synthesize"]

_CACHE_DIR = Path(tempfile.gettempdir()) / "whisp-test-speech"
_CACHE_VERSION = 1
"""Bump when the synthesis script changes, so stale cached audio is not reused."""
_TIMEOUT_S = 120

# Windows PowerShell (not pwsh): System.Speech ships with the .NET Framework only.
_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $voice = $synth.GetInstalledVoices() |
        Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*' } |
        Sort-Object @{ Expression = { $_.VoiceInfo.Culture.Name -ne 'en-US' } },
                    @{ Expression = { $_.VoiceInfo.Name } } |
        Select-Object -First 1
    if ($voice) { $synth.SelectVoice($voice.VoiceInfo.Name) }
    $synth.Rate = [int]$env:WHISP_TTS_RATE
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
        16000,
        [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
        [System.Speech.AudioFormat.AudioChannel]::Mono)
    $synth.SetOutputToWaveFile($env:WHISP_TTS_OUT, $format)
    $synth.Speak($env:WHISP_TTS_TEXT)
    $synth.SetOutputToNull()
} finally {
    $synth.Dispose()
}
"""


def synthesize(text: str, voice_rate: int = 0) -> np.ndarray:
    """Speak ``text`` with Windows text-to-speech and return the audio.

    ``voice_rate`` is SAPI's speaking rate, from -10 (slowest) to 10 (fastest).
    Raises ``RuntimeError`` if speech synthesis is unavailable or fails.
    """
    if not text.strip():
        raise ValueError("text must not be blank")
    if not -10 <= voice_rate <= 10:
        raise ValueError("voice_rate must be between -10 and 10")
    key = hashlib.sha256(f"{_CACHE_VERSION}\n{voice_rate}\n{text}".encode()).hexdigest()
    path = _CACHE_DIR / f"{key}.wav"
    if not path.is_file():
        _render(text, voice_rate, path)
    return _read_wav(path)


def silence(seconds: float) -> np.ndarray:
    """Digital silence (exact zeros)."""
    if seconds < 0:
        raise ValueError("seconds must not be negative")
    return np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)


def noise(seconds: float, level_db: float = -60.0, seed: int = 0) -> np.ndarray:
    """White noise at ``level_db`` dBFS RMS: a stand-in for a quiet room heard through a real microphone."""
    if seconds < 0:
        raise ValueError("seconds must not be negative")
    rng = np.random.default_rng(seed)
    rms = 10.0 ** (level_db / 20.0)
    return (rng.standard_normal(round(seconds * SAMPLE_RATE)) * rms).astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    """Join audio clips end to end."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])


def _render(text: str, voice_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temporary name, then an atomic rename: parallel test runs never see half-written files.
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.wav")
    env = {**os.environ, "WHISP_TTS_TEXT": text, "WHISP_TTS_RATE": str(voice_rate), "WHISP_TTS_OUT": str(tmp)}
    encoded = base64.b64encode(_SCRIPT.encode("utf-16-le")).decode("ascii")
    command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
    try:
        proc = subprocess.run(
            [*command, "-EncodedCommand", encoded],
            env=env,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Windows text-to-speech could not run: {exc}") from exc
    if proc.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Windows text-to-speech failed (exit {proc.returncode}): {proc.stderr.strip()}")
    os.replace(tmp, path)


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = (np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        data = _resample(data, rate, SAMPLE_RATE)
    return np.ascontiguousarray(data, dtype=np.float32)


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited resampling in the frequency domain (fine for short test clips)."""
    if data.size == 0:
        return data.astype(np.float32)
    n_out = max(1, round(data.size * dst_rate / src_rate))
    spectrum = np.fft.rfft(data.astype(np.float64))
    bins = n_out // 2 + 1
    if bins <= spectrum.size:
        spectrum = spectrum[:bins]
    else:
        spectrum = np.concatenate([spectrum, np.zeros(bins - spectrum.size, dtype=spectrum.dtype)])
    return (np.fft.irfft(spectrum, n=n_out) * (n_out / data.size)).astype(np.float32)
