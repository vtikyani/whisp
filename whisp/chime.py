import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100


def _generate_chime(freq_start, freq_end, duration=0.15, volume=0.3):
    """Generate a smooth sine-wave chime with frequency sweep and fade envelope."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)

    # Linear frequency sweep
    freqs = np.linspace(freq_start, freq_end, len(t))
    phase = 2 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
    wave = np.sin(phase)

    # Fade envelope: quick attack, smooth release
    attack = np.minimum(t / 0.01, 1.0)  # 10ms attack
    release = np.minimum((duration - t) / 0.05, 1.0)  # 50ms release
    envelope = attack * release

    return (wave * envelope * volume).astype(np.float32)


# Pre-generate chimes
_chime_activate = _generate_chime(523, 784, duration=0.15, volume=0.25)  # C5 -> G5 rising
_chime_deactivate = _generate_chime(784, 523, duration=0.15, volume=0.25)  # G5 -> C5 falling


def play_activate():
    sd.play(_chime_activate, samplerate=SAMPLE_RATE)


def play_deactivate():
    sd.play(_chime_deactivate, samplerate=SAMPLE_RATE)
