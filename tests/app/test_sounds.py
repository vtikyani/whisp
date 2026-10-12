"""SoundPlayer tests: synthesis quality, settings, mixing, device failures and clean shutdown.

No test opens a real audio device: sounddevice's stream and device query are replaced by fakes.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest
import sounddevice as sd

from whisp.app import sounds
from whisp.app.sounds import PEAK, SOUND_NAMES, SoundPlayer, synthesize


class FakeStream:
    instances: ClassVar[list[FakeStream]] = []
    fail_with: ClassVar[Exception | None] = None

    def __init__(self, *, samplerate: int, channels: int, dtype: str, latency: str, callback) -> None:
        if FakeStream.fail_with is not None:
            raise FakeStream.fail_with
        self.samplerate = samplerate
        self.channels = channels
        self.callback = callback
        self.active = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self) -> None:
        self.active = True

    def close(self, ignore_errors: bool = True) -> None:
        self.active = False
        self.closed = True

    def pull(self, frames: int = 4096) -> np.ndarray:
        """Run the audio callback once, as PortAudio would."""
        out = np.zeros((frames, self.channels), dtype=np.float32)
        self.callback(out, frames, None, sd.CallbackFlags())
        return out


class Holds:
    """Records ``hold_portaudio()`` usage."""

    def __init__(self) -> None:
        self.active = 0
        self.entered = 0

    @contextlib.contextmanager
    def __call__(self) -> Iterator[None]:
        self.active += 1
        self.entered += 1
        try:
            yield
        finally:
            self.active -= 1


@pytest.fixture
def fake_audio(monkeypatch) -> Holds:
    FakeStream.instances = []
    FakeStream.fail_with = None
    holds = Holds()
    monkeypatch.setattr(sounds.sd, "OutputStream", FakeStream)
    monkeypatch.setattr(sounds.sd, "query_devices", lambda kind=None: {"default_samplerate": 48000.0})
    monkeypatch.setattr(sounds, "hold_portaudio", holds)
    return holds


@pytest.fixture
def player(settings_store, fake_audio):
    p = SoundPlayer(settings_store)
    yield p
    p.shutdown()


def wait_for(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


# -- synthesis -------------------------------------------------------------------


@pytest.mark.parametrize("rate", [44_100, 48_000])
@pytest.mark.parametrize("name", SOUND_NAMES)
def test_sounds_are_short_smooth_and_at_the_right_level(name: str, rate: int) -> None:
    samples = synthesize(name, rate)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert np.all(np.isfinite(samples))
    duration_ms = len(samples) / rate * 1000
    assert 60 <= duration_ms <= 220
    peak = float(np.max(np.abs(samples)))
    assert 20 * np.log10(peak) == pytest.approx(-16.0, abs=0.05)  # -16 dBFS before volume, never clipping
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    assert 20 * np.log10(rms) > -40.0  # clearly audible, not silence
    # No clicks: it starts and ends at silence, ramps in, and never jumps between samples.
    assert samples[0] == 0.0 and samples[-1] == 0.0
    first_ms = samples[: rate // 1000]
    assert float(np.max(np.abs(first_ms))) < 0.5 * PEAK
    assert float(np.max(np.abs(np.diff(samples)))) < 0.4 * PEAK
    assert abs(float(np.mean(samples))) < 1e-3  # no DC offset


def test_unknown_sound_names_are_rejected_by_synthesis() -> None:
    with pytest.raises(KeyError):
        synthesize("boing")


# -- playback --------------------------------------------------------------------


def test_play_mixes_at_the_configured_volume(player: SoundPlayer, settings_store, fake_audio: Holds) -> None:
    settings_store.update({"audio": {"sounds_enabled": True, "sounds_volume": 0.5}})
    player.play("start")
    wait_for(lambda: FakeStream.instances and player._pending)
    stream = FakeStream.instances[0]
    assert stream.active
    out = stream.pull()
    expected_gain = 0.5**sounds.LOUDNESS_EXPONENT
    assert float(np.max(np.abs(out))) == pytest.approx(PEAK * expected_gain, rel=0.02)
    assert fake_audio.active == 1  # the open stream holds PortAudio against microphone re-scans


def test_overlapping_sounds_mix_without_clipping(player: SoundPlayer, settings_store) -> None:
    settings_store.update({"audio": {"sounds_enabled": True, "sounds_volume": 1.0}})
    for _ in range(3):
        player.play("error")
    wait_for(lambda: len(player._pending) == 3)
    out = FakeStream.instances[0].pull()
    assert float(np.max(np.abs(out))) <= 1.0
    assert float(np.max(np.abs(out))) > PEAK  # really mixed, not replaced


def test_disabled_or_silent_settings_play_nothing(player: SoundPlayer, settings_store) -> None:
    settings_store.update({"audio": {"sounds_enabled": False}})
    player.play("start")
    settings_store.update({"audio": {"sounds_enabled": True, "sounds_volume": 0.0}})
    player.play("stop")
    time.sleep(0.1)
    assert FakeStream.instances == []


def test_unknown_name_is_logged_not_raised(player: SoundPlayer, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="whisp.app.sounds"):
        player.play("boing")
    assert "boing" in caplog.text
    assert FakeStream.instances == []


def test_missing_device_is_logged_once_and_stays_silent(player: SoundPlayer, monkeypatch, caplog, fake_audio) -> None:
    FakeStream.fail_with = sd.PortAudioError("Device unavailable")
    monkeypatch.setattr(sounds, "RETRY_SECONDS", 0.0)
    with caplog.at_level(logging.WARNING, logger="whisp.app.sounds"):
        player.play("start")
        player.play("stop")
        time.sleep(0.15)
        player.play("lock")
        time.sleep(0.15)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert fake_audio.active == 0  # a failed open never keeps PortAudio held
    FakeStream.fail_with = None
    player.play("start")  # the device came back
    wait_for(lambda: FakeStream.instances)


def test_stream_closes_after_idle_and_on_shutdown(settings_store, fake_audio: Holds, monkeypatch) -> None:
    monkeypatch.setattr(sounds, "IDLE_CLOSE_SECONDS", 0.05)
    player = SoundPlayer(settings_store)
    try:
        player.play("lock")
        wait_for(lambda: FakeStream.instances)
        stream = FakeStream.instances[0]
        stream.pull(48_000)  # play it to the end
        wait_for(lambda: stream.closed, timeout=3.0)
        assert fake_audio.active == 0
        player.play("cancel")
        wait_for(lambda: len(FakeStream.instances) == 2)
    finally:
        player.shutdown()
    assert FakeStream.instances[1].closed
    assert fake_audio.active == 0
    assert not player._thread.is_alive()
    player.shutdown()  # idempotent
    player.play("start")  # ignored after shutdown
    assert len(FakeStream.instances) == 2


def test_play_returns_immediately(player: SoundPlayer, monkeypatch) -> None:
    def slow_query(kind=None):
        time.sleep(0.3)  # a sluggish audio driver
        return {"default_samplerate": 48000.0}

    monkeypatch.setattr(sounds.sd, "query_devices", slow_query)
    started = time.perf_counter()
    player.play("start")
    assert time.perf_counter() - started < 0.02
