import logging
import os
import queue
import numpy as np
import onnxruntime as ort  # must be imported before PyQt6 (done in main.py)
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger("whisp")

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms at 16kHz


def _find_vad_model():
    """Find the bundled ONNX model from silero_vad package."""
    try:
        import silero_vad
        pkg_dir = os.path.dirname(os.path.abspath(silero_vad.__file__))
        path = os.path.join(pkg_dir, "data", "silero_vad.onnx")
        if os.path.exists(path):
            return path
    except ImportError:
        pass
    raise FileNotFoundError("Silero VAD ONNX model not found. Install: pip install silero-vad")


MODEL_PATH = _find_vad_model()


class VADThread(QThread):
    """Processes audio chunks through Silero VAD (ONNX), emits complete phrases."""

    phrase_ready = pyqtSignal(np.ndarray)  # complete phrase audio (float32, 16kHz)
    speech_started = pyqtSignal()  # fired when speech begins

    def __init__(self, audio_queue: queue.Queue, silence_ms=300, threshold=0.5, max_phrase_ms=3000):
        super().__init__()
        self.audio_queue = audio_queue
        self.silence_ms = silence_ms
        self.threshold = threshold
        self._running = False

        # Silence duration in chunks (512 samples = 32ms per chunk)
        self._silence_chunks_needed = max(1, int(silence_ms / 32))
        # Max speech chunks before intermediate flush (0 = disabled)
        self._max_speech_chunks = max(1, int(max_phrase_ms / 32)) if max_phrase_ms > 0 else 0

    def run(self):
        self._running = True
        log.info("VAD thread started (ONNX)")

        # Load ONNX model
        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)

        # Context buffer: Silero VAD ONNX model requires 64-sample context
        # prepended to each 512-sample chunk (total input: 576 samples).
        # See OnnxWrapper.__call__ in silero_vad/utils_vad.py.
        context_size = 64
        context = np.zeros(context_size, dtype=np.float32)

        speech_chunks = []
        in_speech = False
        silence_count = 0

        while self._running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Run VAD inference with context prepend
            audio_input = chunk.astype(np.float32)
            full_input = np.concatenate([context.reshape(1, context_size), audio_input.reshape(1, CHUNK_SIZE)], axis=1)
            outputs = sess.run(None, {"input": full_input, "state": state, "sr": sr})
            context = full_input[0, -context_size:]
            speech_prob = float(outputs[0][0][0])
            state = outputs[1]

            if not in_speech and speech_prob >= self.threshold:
                # Speech started
                in_speech = True
                silence_count = 0
                speech_chunks.clear()
                speech_chunks.append(chunk)
                self.speech_started.emit()
                log.debug("Speech started (prob=%.3f)", speech_prob)

            elif in_speech:
                speech_chunks.append(chunk)

                if speech_prob < self.threshold:
                    silence_count += 1
                    if silence_count >= self._silence_chunks_needed:
                        # Speech ended
                        if speech_chunks:
                            phrase_audio = np.concatenate(speech_chunks)
                            log.info(
                                "Phrase complete: %d samples (%.1fs)",
                                len(phrase_audio),
                                len(phrase_audio) / SAMPLE_RATE,
                            )
                            self.phrase_ready.emit(phrase_audio)
                        speech_chunks.clear()
                        in_speech = False
                        silence_count = 0
                        # Reset ONNX state and context for next phrase
                        state = np.zeros((2, 1, 128), dtype=np.float32)
                        context = np.zeros(context_size, dtype=np.float32)
                else:
                    silence_count = 0

                # Intermediate flush: emit accumulated speech without waiting for silence
                if in_speech and self._max_speech_chunks > 0 and len(speech_chunks) >= self._max_speech_chunks:
                    phrase_audio = np.concatenate(speech_chunks)
                    speech_chunks.clear()
                    log.info(
                        "Intermediate flush: %d samples (%.1fs)",
                        len(phrase_audio),
                        len(phrase_audio) / SAMPLE_RATE,
                    )
                    self.phrase_ready.emit(phrase_audio)

        log.info("VAD thread stopped")

    def stop(self):
        self._running = False
