import logging
import queue
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger("whisp")

SAMPLE_RATE = 16000
BLOCK_SIZE = 512  # ~32ms at 16kHz


class AudioCapture(QObject):
    """Captures audio from the default microphone and emits FFT magnitudes for waveform."""

    fft_ready = pyqtSignal(np.ndarray)  # FFT magnitudes per bar

    def __init__(self, num_bars=7):
        super().__init__()
        self.num_bars = num_bars
        self.audio_queue = queue.Queue()  # consumed by VAD thread
        self._stream = None

        # Precompute FFT bin ranges for logarithmically-spaced frequency bands
        # Speech range: ~100Hz to ~4000Hz
        freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)
        lo, hi = 100.0, 4000.0
        log_edges = np.logspace(np.log10(lo), np.log10(hi), num_bars + 1)
        self._bin_ranges = []
        for i in range(num_bars):
            lo_bin = np.searchsorted(freqs, log_edges[i])
            hi_bin = np.searchsorted(freqs, log_edges[i + 1])
            hi_bin = max(hi_bin, lo_bin + 1)  # at least one bin
            self._bin_ranges.append((lo_bin, hi_bin))

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio status: %s", status)

        mono = indata[:, 0].copy()
        self.audio_queue.put(mono)

        # Compute FFT magnitudes for waveform
        spectrum = np.abs(np.fft.rfft(mono))
        magnitudes = np.zeros(self.num_bars)
        for i, (lo, hi) in enumerate(self._bin_ranges):
            magnitudes[i] = np.mean(spectrum[lo:hi]) if lo < hi else 0.0

        # Normalize to 0-1 range (empirical scaling)
        magnitudes = np.clip(magnitudes / 5.0, 0.0, 1.0)
        self.fft_ready.emit(magnitudes)

    def start(self):
        if self._stream is not None:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BLOCK_SIZE,
                dtype=np.float32,
                callback=self._audio_callback,
                latency="low",
            )
            self._stream.start()
            log.info("Audio capture started")
        except sd.PortAudioError as e:
            log.error("Failed to start audio capture: %s", e)
            raise

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            log.info("Audio capture stopped")
