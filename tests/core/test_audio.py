"""Microphone capture: device choice, frames, levels, stop, unplug and stall handling.

Most tests replace PortAudio with a scripted fake so failures can be staged on
demand. ``test_real_microphone_*`` (marked ``desktop``) use the real microphone.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sounddevice as sd

from whisp.core import audio
from whisp.core.audio import AudioError, MicrophoneStream, hold_portaudio, list_input_devices, perceptual_level
from whisp.core.types import FRAME_SAMPLES, SAMPLE_RATE

WASAPI = "Windows WASAPI"
MME = "MME"
HEADSET = "Headset Microphone (Contoso Wireless)"
HEADSET_MME = HEADSET[:31]
ARRAY = "Microphone Array (Realtek)"

TABLE = audio._DeviceTable(
    devices=(
        audio._Device(0, "Microsoft Sound Mapper - Input", MME, 2, 44100.0),
        audio._Device(1, HEADSET_MME, MME, 1, 44100.0),
        audio._Device(2, ARRAY, MME, 2, 44100.0),
        audio._Device(10, HEADSET, WASAPI, 1, 16000.0),
        audio._Device(11, ARRAY, WASAPI, 2, 48000.0),
    ),
    default_inputs={MME: 1, WASAPI: 10},
)


def frame(value: float) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


class FakeInputStream:
    """Stands in for ``sounddevice.RawInputStream``; the test plays the audio device."""

    def __init__(self, reported_latency: float, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.callback: Callable[..., None] = kwargs["callback"]
        self.finished_callback: Callable[[], None] = kwargs["finished_callback"]
        self.latency = reported_latency
        self.active = False
        self.closed = False
        self.aborted = False

    def start(self) -> None:
        self.active = True

    def stop(self, ignore_errors: bool = True) -> None:
        self._finish()

    def abort(self, ignore_errors: bool = True) -> None:
        self.aborted = True
        self._finish()

    def close(self, ignore_errors: bool = True) -> None:
        self.closed = True

    def feed(self, samples: np.ndarray) -> None:
        block = np.ascontiguousarray(samples, dtype=np.float32)
        self.callback(memoryview(block.tobytes()), block.size, None, sd.CallbackFlags())

    def unplug(self) -> None:
        """The device vanished: PortAudio ends the stream by itself."""
        self._finish()

    def _finish(self) -> None:
        if self.active:
            self.active = False
            self.finished_callback()


class FakePortAudio:
    def __init__(self) -> None:
        self.table = TABLE
        self.windows: audio._WindowsInputs | None = audio._WindowsInputs(count=2, default_name=HEADSET_MME)
        self.streams: list[FakeInputStream] = []
        self.failing: set[int] = set()
        self.reinitialized = 0
        self.latency = 0.0

    def scan(self) -> audio._DeviceTable:
        return self.table

    def windows_inputs(self) -> audio._WindowsInputs | None:
        return self.windows

    def reinitialize(self) -> None:
        self.reinitialized += 1

    def open_stream(self, **kwargs: Any) -> FakeInputStream:
        if kwargs["device"] in self.failing:
            raise sd.PortAudioError("Error opening InputStream: Invalid device")
        stream = FakeInputStream(self.latency, **kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture
def portaudio(monkeypatch: pytest.MonkeyPatch) -> FakePortAudio:
    fake = FakePortAudio()
    monkeypatch.setattr(audio, "_scan", fake.scan)
    monkeypatch.setattr(audio, "_windows_inputs", fake.windows_inputs)
    monkeypatch.setattr(audio, "_reinitialize_portaudio", fake.reinitialize)
    monkeypatch.setattr(audio.sd, "RawInputStream", fake.open_stream)
    monkeypatch.setattr(audio, "_portaudio_users", 0)
    return fake


def join_background_closes() -> None:
    for thread in threading.enumerate():
        if thread.name == "whisp-audio-close":
            thread.join(2.0)


def read_all(mic: MicrophoneStream) -> list[np.ndarray]:
    frames = []
    while (item := mic.read(0.0)) is not None:
        frames.append(item)
    return frames


# -- levels -----------------------------------------------------------------------


def test_perceptual_level_maps_decibels_to_zero_one() -> None:
    assert perceptual_level(0.0) == 0.0
    assert perceptual_level(float("nan")) == 0.0
    assert perceptual_level(10 ** (-60 / 10)) == pytest.approx(0.0, abs=1e-9)
    assert perceptual_level(10 ** (-36 / 10)) == pytest.approx(0.5)
    assert perceptual_level(10 ** (-12 / 10)) == pytest.approx(1.0)
    assert perceptual_level(1.0) == 1.0
    levels = [perceptual_level(10 ** (db / 10)) for db in range(-80, 1, 4)]
    assert levels == sorted(levels)


def test_level_callback_is_rate_limited(portaudio: FakePortAudio) -> None:
    levels: list[float] = []
    mic = MicrophoneStream(None, on_level=levels.append)
    mic.start()
    stream = portaudio.streams[-1]
    stream.feed(frame(10 ** (-12 / 20)))
    assert levels == [pytest.approx(1.0)]
    time.sleep(0.04)  # let the 30 ms reporting interval pass
    stream.feed(frame(10 ** (-36 / 20)))
    for _ in range(5):
        stream.feed(frame(0.5))
    assert levels == [pytest.approx(1.0), pytest.approx(0.5)]
    mic.stop()


# -- device choice ------------------------------------------------------------------


def test_plan_prefers_wasapi_and_matches_cut_mme_names() -> None:
    candidates, found = audio._plan(TABLE, HEADSET, None)
    assert found
    assert [(c.index, c.host_api, c.name) for c in candidates] == [(10, WASAPI, HEADSET), (1, MME, HEADSET)]


def test_plan_does_not_prefix_match_names_that_were_not_cut() -> None:
    candidates, found = audio._plan(TABLE, ARRAY + " 2", None)
    assert not found
    assert [c.index for c in candidates] == [10, 0]


def test_plan_follows_the_current_windows_default() -> None:
    candidates, found = audio._plan(TABLE, None, ARRAY)
    assert found
    assert [(c.index, c.host_api) for c in candidates] == [(11, WASAPI), (0, MME)]
    assert candidates[1].name == ARRAY  # the MME sound mapper, labelled with the real device


def test_plan_uses_portaudio_default_without_windows_information() -> None:
    candidates, _ = audio._plan(TABLE, None, None)
    assert [(c.index, c.host_api) for c in candidates] == [(10, WASAPI), (0, MME)]


def test_plan_without_wasapi_uses_the_mme_sound_mapper() -> None:
    table = audio._DeviceTable(tuple(d for d in TABLE.devices if d.host_api == MME), {MME: 1})
    candidates, _ = audio._plan(table, None, HEADSET_MME)
    assert [(c.index, c.host_api, c.name) for c in candidates] == [(0, MME, HEADSET_MME)]


def test_stale_device_list_is_detected() -> None:
    assert not audio._looks_stale(TABLE, None)
    assert not audio._looks_stale(TABLE, audio._WindowsInputs(2, HEADSET_MME))
    assert audio._looks_stale(TABLE, audio._WindowsInputs(3, HEADSET_MME))  # something was plugged in
    assert audio._looks_stale(TABLE, audio._WindowsInputs(2, "Brand New Microphone"))


def test_lists_wasapi_devices_with_full_names(portaudio: FakePortAudio) -> None:
    devices = list_input_devices()
    assert [d.id for d in devices] == [HEADSET, ARRAY]
    assert [d.is_default for d in devices] == [True, False]
    assert {d.host_api for d in devices} == {WASAPI}
    assert portaudio.reinitialized == 1


def test_lists_mme_devices_when_wasapi_is_missing(portaudio: FakePortAudio) -> None:
    portaudio.table = audio._DeviceTable(tuple(d for d in TABLE.devices if d.host_api == MME), {MME: 1})
    portaudio.windows = audio._WindowsInputs(2, ARRAY)
    devices = list_input_devices()
    assert [(d.id, d.is_default) for d in devices] == [(HEADSET_MME, False), (ARRAY, True)]


def test_scans_never_reinitialize_portaudio_under_an_open_stream(portaudio: FakePortAudio) -> None:
    mic = MicrophoneStream(None)
    mic.start()
    list_input_devices()
    assert portaudio.reinitialized == 0
    mic.stop()
    list_input_devices()
    assert portaudio.reinitialized == 1
    with hold_portaudio():
        list_input_devices()
    assert portaudio.reinitialized == 1


def test_opens_the_default_device_as_16khz_mono(portaudio: FakePortAudio) -> None:
    mic = MicrophoneStream(None)
    mic.start()
    kwargs = portaudio.streams[-1].kwargs
    assert kwargs["device"] == 10
    assert (kwargs["samplerate"], kwargs["channels"], kwargs["blocksize"], kwargs["dtype"]) == (
        SAMPLE_RATE,
        1,
        FRAME_SAMPLES,
        "float32",
    )
    assert isinstance(kwargs["extra_settings"], sd.WasapiSettings)
    assert mic.device_name == HEADSET
    assert portaudio.reinitialized == 0  # the device list was current, so no re-scan
    mic.stop()


def test_follows_a_changed_windows_default(portaudio: FakePortAudio) -> None:
    portaudio.windows = audio._WindowsInputs(2, ARRAY)
    mic = MicrophoneStream(None)
    mic.start()
    assert portaudio.streams[-1].kwargs["device"] == 11
    assert mic.device_name == ARRAY
    mic.stop()


def test_rescans_when_windows_reports_new_devices(portaudio: FakePortAudio) -> None:
    portaudio.windows = audio._WindowsInputs(3, HEADSET_MME)
    mic = MicrophoneStream(None)
    mic.start()
    assert portaudio.reinitialized == 1
    mic.stop()


def test_missing_device_falls_back_to_the_default(portaudio: FakePortAudio, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="whisp.core.audio"):
        mic = MicrophoneStream("USB Microphone (unplugged)")
        mic.start()
    assert portaudio.streams[-1].kwargs["device"] == 10
    assert portaudio.reinitialized == 1  # looked again before giving up on it
    assert "not available" in caplog.text
    mic.stop()


def test_falls_back_to_mme_when_wasapi_fails(portaudio: FakePortAudio) -> None:
    portaudio.failing.add(10)
    mic = MicrophoneStream(HEADSET)
    mic.start()
    kwargs = portaudio.streams[-1].kwargs
    assert kwargs["device"] == 1
    assert kwargs["extra_settings"] is None
    assert mic.device_name == HEADSET
    mic.stop()


def test_reports_a_user_facing_error_when_nothing_opens(portaudio: FakePortAudio) -> None:
    portaudio.failing.update({10, 0})
    mic = MicrophoneStream(None)
    with pytest.raises(AudioError) as caught:
        mic.start()
    assert str(caught.value) == audio.MSG_OPEN_FAILED
    assert audio._portaudio_users == 0
    mic.stop()  # harmless after a failed start
    assert mic.read(1.0) is None


def test_reports_when_there_is_no_microphone_at_all(portaudio: FakePortAudio) -> None:
    portaudio.table = audio._DeviceTable((), {MME: -1, WASAPI: -1})
    portaudio.windows = audio._WindowsInputs(0, None)
    with pytest.raises(AudioError) as caught:
        MicrophoneStream(None).start()
    assert str(caught.value) == audio.MSG_NO_DEVICE


def test_start_twice_is_a_programming_error(portaudio: FakePortAudio) -> None:
    mic = MicrophoneStream(None)
    mic.start()
    with pytest.raises(RuntimeError):
        mic.start()
    mic.stop()


# -- capture --------------------------------------------------------------------------


def test_frames_flow_from_the_callback_to_read(portaudio: FakePortAudio) -> None:
    mic = MicrophoneStream(None)
    mic.start()
    stream = portaudio.streams[-1]
    sent = [frame(0.1), frame(0.2), frame(0.3)]
    for item in sent:
        stream.feed(item)
    for item in sent:
        np.testing.assert_array_equal(mic.read(1.0), item)
    assert mic.read(0.01) is None
    assert mic.error is None
    mic.stop()


def test_a_stream_nobody_reads_keeps_only_the_newest_audio(
    portaudio: FakePortAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audio, "_MAX_BUFFERED_FRAMES", 3)
    levels: list[float] = []
    mic = MicrophoneStream(None, on_level=levels.append)  # like a settings-page level meter
    mic.start()
    stream = portaudio.streams[-1]
    for value in (0.1, 0.2, 0.3, 0.4, 0.5):
        stream.feed(frame(value))
    mic.stop()
    assert [f[0] for f in read_all(mic)] == [pytest.approx(0.3), pytest.approx(0.4), pytest.approx(0.5)]
    assert levels  # metering is unaffected


def test_stop_keeps_captured_frames_readable(portaudio: FakePortAudio) -> None:
    mic = MicrophoneStream(None)
    mic.start()
    stream = portaudio.streams[-1]
    stream.feed(frame(0.1))
    stream.feed(frame(0.2))
    mic.stop()
    assert stream.closed and not stream.active
    assert audio._portaudio_users == 0
    assert [f[0] for f in read_all(mic)] == [pytest.approx(0.1), pytest.approx(0.2)]
    started = time.monotonic()
    assert mic.read(5.0) is None  # ended: no waiting
    assert time.monotonic() - started < 1.0
    assert mic.error is None
    mic.stop()  # idempotent


def test_stop_collects_audio_still_on_its_way(portaudio: FakePortAudio) -> None:
    portaudio.latency = 0.064  # 1024 samples = two frames in flight
    mic = MicrophoneStream(None)
    mic.start()
    stream = portaudio.streams[-1]

    def device_delivers_late_audio() -> None:
        time.sleep(0.02)
        stream.feed(frame(0.4))
        stream.feed(frame(0.5))

    device = threading.Thread(target=device_delivers_late_audio)
    device.start()
    mic.stop()
    device.join()
    assert [f[0] for f in read_all(mic)] == [pytest.approx(0.4), pytest.approx(0.5)]


def test_unplugged_microphone_ends_the_stream_with_an_error(portaudio: FakePortAudio) -> None:
    mic = MicrophoneStream(None)
    mic.start()
    stream = portaudio.streams[-1]
    stream.feed(frame(0.3))
    stream.unplug()
    assert mic.error == audio.MSG_DISCONNECTED
    np.testing.assert_array_equal(mic.read(1.0), frame(0.3))  # captured audio survives
    started = time.monotonic()
    assert mic.read(5.0) is None
    assert time.monotonic() - started < 1.0
    mic.stop()
    join_background_closes()
    assert stream.closed
    assert audio._portaudio_users == 0


@pytest.mark.parametrize("delivered_first", [0, 1])
def test_a_silent_stream_is_treated_as_lost(
    portaudio: FakePortAudio, monkeypatch: pytest.MonkeyPatch, delivered_first: int
) -> None:
    monkeypatch.setattr(audio, "_FIRST_FRAME_TIMEOUT_S", 0.2)
    monkeypatch.setattr(audio, "_STALL_TIMEOUT_S", 0.2)
    mic = MicrophoneStream(None)
    mic.start()
    stream = portaudio.streams[-1]
    for _ in range(delivered_first):
        stream.feed(frame(0.2))
    assert len(read_all(mic)) == delivered_first
    deadline = time.monotonic() + 3.0
    while mic.error is None and time.monotonic() < deadline:
        assert mic.read(0.05) is None
    assert mic.error == audio.MSG_STALLED
    started = time.monotonic()
    assert mic.read(5.0) is None
    assert time.monotonic() - started < 1.0
    mic.stop()
    join_background_closes()
    assert stream.aborted and stream.closed
    assert audio._portaudio_users == 0


# -- the real microphone ----------------------------------------------------------------


@pytest.mark.desktop
def test_real_microphone_records_frames_and_levels() -> None:
    devices = list_input_devices()
    assert devices, "no microphone found"
    assert sum(d.is_default for d in devices) <= 1
    levels: list[float] = []
    mic = MicrophoneStream(None, on_level=levels.append)
    mic.start()
    frames: list[np.ndarray] = []
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            item = mic.read(0.2)
            if item is not None:
                frames.append(item)
    finally:
        mic.stop()
    frames += read_all(mic)
    assert mic.error is None
    assert mic.device_name
    assert 25 <= len(frames) <= 45  # about 31 frames a second
    assert all(f.shape == (FRAME_SAMPLES,) and f.dtype == np.float32 for f in frames)
    assert 1 <= len(levels) <= 40
    assert all(0.0 <= v <= 1.0 for v in levels)


@pytest.mark.desktop
def test_real_microphone_opens_through_wasapi_on_a_worker_thread() -> None:
    # WASAPI can't start a stream on a thread without COM. Inside pytest, Qt quietly provides it to every
    # thread, so this runs in a fresh process without Qt, the way the pipeline thread starts out.
    script = textwrap.dedent(
        """
        import logging, sys, threading
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
        from whisp.core.audio import MicrophoneStream

        def record():
            mic = MicrophoneStream(None)
            mic.start()
            try:
                print("frame" if mic.read(2.0) is not None else "no frame")
            finally:
                mic.stop()

        worker = threading.Thread(target=record)
        worker.start()
        worker.join()
        """
    )
    src = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "via Windows WASAPI" in proc.stdout, proc.stdout
    assert "frame" in proc.stdout.splitlines()


@pytest.mark.desktop
def test_real_microphone_opens_a_listed_device_by_name() -> None:
    device = list_input_devices()[0]
    mic = MicrophoneStream(device.id)
    mic.start()
    try:
        assert mic.read(2.0) is not None
    finally:
        mic.stop()
    assert mic.device_name == device.id
    assert mic.error is None
