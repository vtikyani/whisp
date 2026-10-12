"""Short UI sounds, synthesised with numpy at startup (no audio files).

Five cues: "start" (soft rising two-note chime), "stop" (gentle falling pair),
"lock" (hands-free engaged: a light double tick), "cancel" (a soft low blip)
and "error" (two low, muted tones). Each is 60-220 ms of warm sine/triangle
blends with a few soft harmonics and exponential decays, shaped with
raised-cosine attack and release so nothing clicks, and normalised to peak at
-16 dBFS before the user's volume is applied.

``play()`` never blocks and never raises. Requests go to a small worker thread
(``whisp-sounds``) that opens the output stream on first use and closes it
after a few seconds of silence, so an idle Whisp holds no audio device and
burns no CPU. Mixing happens in PortAudio's callback, so overlapping sounds
blend instead of cutting each other off. If no output device works, the
problem is logged once and sounds stay silent (with a retry now and then).

Microphone re-scans re-initialise PortAudio, which would close every open
stream, so the stream's whole lifetime (from the device query to close) is
wrapped in ``core.audio.hold_portaudio()``.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

from whisp.core.audio import hold_portaudio
from whisp.storage.settings import SettingsStore

log = logging.getLogger(__name__)

SOUND_NAMES = ("start", "stop", "lock", "cancel", "error")
PEAK_DBFS = -16.0
PEAK = 10.0 ** (PEAK_DBFS / 20.0)
DEFAULT_SAMPLE_RATE = 48_000
LOUDNESS_EXPONENT = 1.66
"""Volume slider to gain: v ** 1.66 makes each step sound equally louder (loudness doubles per +10 dB)."""
IDLE_CLOSE_SECONDS = 8.0
"""Close the output stream after this long without sounds."""
RETRY_SECONDS = 30.0
"""After the output device fails to open, wait this long before trying again."""
STALE_SECONDS = 0.4
"""A UI sound that could not start within this long is dropped: late feedback is worse than none."""
RELEASE_SECONDS = 0.012


# -- synthesis -------------------------------------------------------------------


def _raised_cosine(n: int) -> np.ndarray:
    """0 -> 1 over ``n`` samples with zero slope at both ends."""
    if n <= 1:
        return np.ones(max(n, 0))
    return 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))


def _tone(
    rate: int,
    freq: float,
    seconds: float,
    *,
    decay: float,
    glide_to: float | None = None,
    triangle: float = 0.18,
    harmonics: tuple[tuple[int, float], ...] = ((2, 0.16), (3, 0.05)),
    attack: float = 0.004,
) -> np.ndarray:
    """One warm note: sine/triangle blend, soft harmonics that fade faster than the fundamental, exponential decay."""
    n = max(1, round(rate * seconds))
    t = np.arange(n) / rate
    if glide_to is None:
        phase = 2.0 * np.pi * freq * t
    else:
        # Exponential glide, integrated so the pitch moves smoothly without a phase jump.
        k = np.log(glide_to / freq) / seconds
        phase = 2.0 * np.pi * freq * np.expm1(k * t) / k
    sine = np.sin(phase)
    tri = (2.0 / np.pi) * np.arcsin(sine)
    wave = (1.0 - triangle) * sine + triangle * tri
    for order, amount in harmonics:
        wave += amount * np.sin(order * phase) * np.exp(-t * 10.0 * (order - 1))
    envelope = np.exp(-t / decay)
    attack_n = min(n, max(2, round(rate * attack)))
    envelope[:attack_n] *= _raised_cosine(attack_n)
    return wave * envelope


def _place(total: np.ndarray, part: np.ndarray, at: float, rate: int, gain: float = 1.0) -> None:
    start = round(at * rate)
    end = min(len(total), start + len(part))
    total[start:end] += gain * part[: end - start]


def _finish(signal: np.ndarray, rate: int) -> np.ndarray:
    """Release fade to exact silence, remove any DC offset, normalise the peak to -16 dBFS."""
    signal = signal - float(np.mean(signal))
    release = min(len(signal), round(rate * RELEASE_SECONDS))
    signal[-release:] *= _raised_cosine(release)[::-1]
    signal[0] = 0.0
    signal[-1] = 0.0
    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal *= PEAK / peak
    return signal.astype(np.float32)


def synthesize(name: str, rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Build one cue as mono float32 samples at ``rate``. Raises KeyError for unknown names."""
    if name == "start":
        # E5 then B5: a soft, rising fifth.
        out = np.zeros(round(rate * 0.2))
        _place(out, _tone(rate, 659.25, 0.2, decay=0.07), 0.0, rate)
        _place(out, _tone(rate, 987.77, 0.145, decay=0.06), 0.055, rate, 0.85)
    elif name == "stop":
        # B5 then E5: the same interval, falling and a touch quieter.
        out = np.zeros(round(rate * 0.18))
        _place(out, _tone(rate, 987.77, 0.18, decay=0.05), 0.0, rate, 0.8)
        _place(out, _tone(rate, 659.25, 0.13, decay=0.06), 0.05, rate)
    elif name == "lock":
        # Two light ticks, the second a step higher: "locked in".
        out = np.zeros(round(rate * 0.11))
        tick = {"decay": 0.014, "triangle": 0.0, "harmonics": ((2, 0.08),), "attack": 0.003}
        _place(out, _tone(rate, 1567.98, 0.035, **tick), 0.0, rate, 0.9)
        _place(out, _tone(rate, 2093.0, 0.035, **tick), 0.07, rate)
    elif name == "cancel":
        # A soft low blip that sinks a little.
        out = _tone(rate, 392.0, 0.13, decay=0.045, glide_to=330.0, triangle=0.3, harmonics=((2, 0.1),))
    elif name == "error":
        # Two low, muted tones a minor third apart: noticeable, never harsh.
        out = np.zeros(round(rate * 0.21))
        muted = {"decay": 0.05, "triangle": 0.45, "harmonics": ((2, 0.06),), "attack": 0.006}
        _place(out, _tone(rate, 311.13, 0.11, **muted), 0.0, rate)
        _place(out, _tone(rate, 261.63, 0.11, **muted), 0.1, rate)
    else:
        raise KeyError(name)
    return _finish(out, rate)


def synthesize_all(rate: int = DEFAULT_SAMPLE_RATE) -> dict[str, np.ndarray]:
    return {name: synthesize(name, rate) for name in SOUND_NAMES}


# -- playback --------------------------------------------------------------------


@dataclass(eq=False)
class _Voice:
    samples: np.ndarray
    gain: float
    pos: int = field(default=0)


class SoundPlayer:
    """Plays the UI cues. ``play`` may be called from any thread; it never blocks or raises."""

    def __init__(self, settings: SettingsStore) -> None:
        self._settings = settings
        self._rate = DEFAULT_SAMPLE_RATE
        self._sounds = synthesize_all(self._rate)
        self._requests: queue.SimpleQueue[tuple[str, float, float]] = queue.SimpleQueue()
        self._pending: deque[_Voice] = deque()  # worker -> audio callback
        self._active: list[_Voice] = []  # touched only by the audio callback while a stream is open
        self._stream: sd.OutputStream | None = None
        self._portaudio_hold: contextlib.ExitStack | None = None
        self._last_activity = 0.0
        self._retry_after = 0.0
        self._failure_logged = False
        self._closed = False
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name="whisp-sounds", daemon=True)
        self._thread.start()

    def play(self, name: str) -> None:
        if self._closed:
            return
        if name not in SOUND_NAMES:
            log.warning("Unknown UI sound %r", name)
            return
        audio = self._settings.current.audio
        if not audio.sounds_enabled or audio.sounds_volume <= 0.0:
            return
        gain = float(audio.sounds_volume) ** LOUDNESS_EXPONENT
        self._requests.put((name, gain, time.monotonic()))
        self._wake.set()

    def shutdown(self) -> None:
        """Stop the worker and close the output stream. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._stopping.set()
        self._wake.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            log.warning("Sound worker did not stop within 2 s")

    # -- worker thread ---------------------------------------------------------------

    def _run(self) -> None:
        while not self._stopping.is_set():
            self._wake.wait(timeout=1.0 if self._stream is not None else None)
            self._wake.clear()
            if self._stopping.is_set():
                break
            try:
                self._serve_requests()
                self._close_if_idle()
            except Exception:
                log.exception("Sound worker error")
        self._close_stream()

    def _serve_requests(self) -> None:
        while True:
            try:
                name, gain, requested = self._requests.get_nowait()
            except queue.Empty:
                return
            if not self._ensure_stream():
                continue
            if time.monotonic() - requested > STALE_SECONDS:
                continue
            self._pending.append(_Voice(self._sounds[name], gain))
            self._last_activity = time.monotonic()

    def _close_if_idle(self) -> None:
        if self._stream is None:
            return
        if self._stream_broken():
            self._close_stream()
            return
        busy = bool(self._pending) or bool(self._active)
        if not busy and time.monotonic() - self._last_activity > IDLE_CLOSE_SECONDS:
            self._close_stream()

    def _stream_broken(self) -> bool:
        try:
            return self._stream is not None and not self._stream.active
        except sd.PortAudioError:
            return True

    def _ensure_stream(self) -> bool:
        if self._stream is not None and not self._stream_broken():
            return True
        self._close_stream()
        now = time.monotonic()
        if now < self._retry_after:
            return False
        # Hold PortAudio before touching it: a microphone re-scan in progress finishes first, and none can
        # start (and close our stream) until the stream is closed again.
        hold = contextlib.ExitStack()
        hold.enter_context(hold_portaudio())
        try:
            info = sd.query_devices(kind="output")
            rate = int(info.get("default_samplerate") or DEFAULT_SAMPLE_RATE)
            if rate != self._rate:
                self._sounds = synthesize_all(rate)
                self._rate = rate
            self._stream = self._open_stream(rate)
            self._portaudio_hold = hold
        except Exception as exc:  # PortAudioError, OSError, ValueError: any device trouble means "stay silent"
            hold.close()
            self._stream = None
            self._retry_after = now + RETRY_SECONDS
            if not self._failure_logged:
                self._failure_logged = True
                log.warning("Sound output is unavailable, so UI sounds are off for now: %s", exc)
            return False
        if self._failure_logged:
            log.info("Sound output is working again")
            self._failure_logged = False
        self._last_activity = time.monotonic()
        return True

    def _open_stream(self, rate: int) -> sd.OutputStream:
        options = {"samplerate": rate, "dtype": "float32", "latency": "low", "callback": self._callback}
        try:
            stream = sd.OutputStream(channels=1, **options)
        except sd.PortAudioError:
            # Some drivers only accept stereo; the callback copies the mono mix to every channel.
            stream = sd.OutputStream(channels=2, **options)
        try:
            stream.start()
        except Exception:
            stream.close(ignore_errors=True)
            raise
        return stream

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        hold, self._portaudio_hold = self._portaudio_hold, None
        try:
            if stream is not None:
                stream.close(ignore_errors=True)
        except Exception:
            log.debug("Closing the sound stream failed", exc_info=True)
        finally:
            if hold is not None:
                hold.close()  # microphone re-scans may re-initialise PortAudio again
        # The callback has stopped, so nothing else touches the voice lists now.
        self._pending.clear()
        self._active = []

    # -- audio thread -------------------------------------------------------------------

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        """Mix every playing voice into the output buffer. Runs on PortAudio's thread; never blocks."""
        outdata.fill(0.0)
        while self._pending:
            try:
                self._active.append(self._pending.popleft())
            except IndexError:
                break
        if not self._active:
            return
        mix = outdata[:, 0]
        still_playing: list[_Voice] = []
        for voice in self._active:
            count = min(frames, len(voice.samples) - voice.pos)
            if count > 0:
                mix[:count] += voice.samples[voice.pos : voice.pos + count] * voice.gain
                voice.pos += count
            if voice.pos < len(voice.samples):
                still_playing.append(voice)
        self._active = still_playing
        np.clip(mix, -1.0, 1.0, out=mix)
        for channel in range(1, outdata.shape[1]):
            outdata[:, channel] = mix
