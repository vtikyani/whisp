"""Benchmark Whisp's speech models on real human speech: accuracy, speed and GPU memory.

    python scripts/benchmark_asr.py [--models-dir DIR] [--data-dir DIR] [--only MODEL:VARIANT:DEVICE ...]

Speech comes from ``hf-internal-testing/librispeech_asr_dummy``: 73 utterances
of LibriSpeech dev-clean (CC BY 4.0) with their reference transcripts, fetched
once through the Hugging Face datasets server into ``.local/test-data``.
Utterances are joined, with a quarter second of silence between them, into
clips of about 3, 10, 30 and 60 seconds.

Every model and device combination runs in its own process, so the first
recognition really is the first one in the process (CUDA, cuDNN and the memory
arenas start cold) and GPU memory is measured in isolation. For each it reports:

* load: seconds to load the model (plus warm-up, in brackets);
* cold: the first recognition of a 10-second clip by a freshly loaded model
  without warm-up, which is what Whisp's warm-up spares the first dictation;
* WER per clip length: word error rate after normalising both sides (lower
  case, no punctuation, numbers spelt out, common abbreviations expanded);
* latency per clip length: median recognition time once warmed up;
* VRAM: GPU memory taken while the model is loaded.

Models must already be installed in ``--models-dir`` (the model tests and the
app both install them; the default is the tests' shared cache).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATASET = "hf-internal-testing/librispeech_asr_dummy"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_API = f"https://huggingface.co/api/datasets/{DATASET}"
EXPECTED_UTTERANCES = 73
SAMPLE_RATE = 16_000
GAP_S = 0.25
BUCKETS: dict[str, tuple[float, float]] = {
    "3s": (2.0, 4.5),
    "10s": (8.0, 12.0),
    "30s": (25.0, 29.5),
    "60s": (50.0, 59.5),
}
"""Clip length buckets: (shortest, longest) in seconds. 30 s and 60 s stay just under Whisper's and Parakeet's
single-pass limits, so long single passes are measured, not the splitting."""

CONFIGS = (
    ("parakeet-tdt-0.6b-v2", "gpu", "gpu"),
    ("parakeet-tdt-0.6b-v2", "cpu", "cpu"),
    ("parakeet-tdt-0.6b-v2", "gpu", "cpu"),
    ("parakeet-tdt-0.6b-v3", "gpu", "gpu"),
    ("parakeet-tdt-0.6b-v3", "cpu", "cpu"),
    ("whisper-large-v3-turbo", "gpu", "gpu"),
    ("whisper-large-v3-turbo", "cpu", "cpu"),
)
"""(model, variant whose files are loaded, device). ``gpu`` files on the ``cpu`` device is the CUDA fallback."""


@dataclass(frozen=True)
class Utterance:
    id: str
    text: str
    audio: np.ndarray

    @property
    def seconds(self) -> float:
        return self.audio.size / SAMPLE_RATE


# -- data ------------------------------------------------------------------------------------------


def load_librispeech(data_dir: Path) -> list[Utterance]:
    """The 73 utterances with their transcripts, downloaded on first use."""
    folder = data_dir / "librispeech-dummy"
    meta_path, audio_path = folder / "utterances.json", folder / "audio.npz"
    if not (meta_path.is_file() and audio_path.is_file()):
        _download_librispeech(folder)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(audio_path) as audio:
        return [Utterance(u["id"], u["text"], audio[u["id"]].astype(np.float32)) for u in meta["utterances"]]


def _download_librispeech(folder: Path) -> None:
    import httpx
    import soundfile

    print(f"Downloading {DATASET} into {folder} ...", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:

        def get(url: str, **params: object) -> httpx.Response:
            # The datasets server answers 502 now and then; a short retry is enough.
            for attempt in range(5):
                response = client.get(url, params=params or None)
                if response.status_code < 500:
                    return response.raise_for_status()
                time.sleep(2 * (attempt + 1))
            return response.raise_for_status()

        revision = get(DATASET_API).json()["sha"]
        rows: list[dict] = []
        while len(rows) < EXPECTED_UTTERANCES:
            page = get(ROWS_URL, dataset=DATASET, config="clean", split="validation", offset=len(rows), length=25)
            batch = page.json()["rows"]
            if not batch:
                break
            rows.extend(batch)
        if len(rows) != EXPECTED_UTTERANCES:
            raise SystemExit(f"expected {EXPECTED_UTTERANCES} utterances, the server returned {len(rows)}")
        arrays: dict[str, np.ndarray] = {}
        utterances = []
        for row in rows:
            item = row["row"]
            data = get(item["audio"][0]["src"]).content
            samples, rate = soundfile.read(io.BytesIO(data), dtype="float32")
            if rate != SAMPLE_RATE or samples.ndim != 1:
                raise SystemExit(f"{item['id']}: unexpected audio format ({rate} Hz, shape {samples.shape})")
            arrays[item["id"]] = samples
            utterances.append({"id": item["id"], "text": item["text"], "seconds": round(samples.size / rate, 3)})
    np.savez_compressed(folder / "audio.npz", **arrays)
    source = {"dataset": DATASET, "revision": revision, "license": "CC-BY-4.0 (LibriSpeech)"}
    (folder / "utterances.json").write_text(
        json.dumps({"source": source, "utterances": utterances}, indent=1), encoding="utf-8"
    )


def build_clips(utterances: list[Utterance], shortest: float, longest: float) -> list[list[Utterance]]:
    """Consecutive utterances joined until a clip is at least ``shortest`` and at most ``longest`` seconds."""
    clips: list[list[Utterance]] = []
    current: list[Utterance] = []
    for utterance in utterances:
        if utterance.seconds > longest:
            continue
        if current and _clip_seconds([*current, utterance]) > longest:
            current = []
        current.append(utterance)
        if _clip_seconds(current) >= shortest:
            clips.append(current)
            current = []
    return clips


def _clip_seconds(parts: list[Utterance]) -> float:
    return sum(p.seconds for p in parts) + GAP_S * (len(parts) - 1)


def clip_audio(parts: list[Utterance]) -> np.ndarray:
    gap = np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, part in enumerate(parts):
        if index:
            pieces.append(gap)
        pieces.append(part.audio)
    return np.concatenate(pieces)


# -- scoring -----------------------------------------------------------------------------------------

_ABBREVIATIONS = {"mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint", "jr": "junior", "vs": "versus"}
_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def number_words(n: int) -> str:
    """Cardinal number in words, as LibriSpeech transcripts spell numbers."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))
    for size, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if n >= size:
            rest = n % size
            return number_words(n // size) + " " + name + ("" if rest == 0 else " " + number_words(rest))
    raise AssertionError("unreachable")


def normalize(text: str) -> list[str]:
    text = text.lower().replace("-", " ").replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    words: list[str] = []
    for word in text.split():
        word = word.strip("'")
        if not word:
            continue
        if word.isdigit():
            words.extend(number_words(int(word)).split())
        else:
            words.append(_ABBREVIATIONS.get(word, word))
    return words


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Word-level edit distance (substitutions + deletions + insertions)."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1]


# -- one configuration (runs in a child process) --------------------------------------------------


def run_config(models_dir: Path, data_dir: Path, model_id: str, variant: str, device: str) -> dict[str, object]:
    from whisp.core.asr.engine import EngineSpec, OnnxAsrEngine
    from whisp.core.asr.manager import ModelManager
    from whisp.core.hardware import nvidia_free_vram_mb

    manager = ModelManager(models_dir)
    if not manager.is_installed(model_id, variant):
        return {"skipped": f"{model_id} ({variant}) isn't installed in {models_dir}"}
    utterances = load_librispeech(data_dir)
    clips = {name: build_clips(utterances, lo, hi) for name, (lo, hi) in BUCKETS.items()}
    free_before = nvidia_free_vram_mb() if device == "gpu" else None
    spec = EngineSpec(manager.model_info(model_id), {variant: manager.model_path(model_id, variant)}, device == "gpu")
    started = time.perf_counter()
    engine = OnnxAsrEngine(spec)
    load_s = time.perf_counter() - started
    if engine.device != ("cuda" if device == "gpu" else "cpu"):
        return {"skipped": f"ran on {engine.device} instead of {device}: {engine.notice}"}
    cold_clip = clip_audio(clips["10s"][0])
    started = time.perf_counter()
    engine.transcribe(cold_clip, "en")
    cold_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    engine.warm_up()
    warm_up_s = time.perf_counter() - started
    buckets: dict[str, dict[str, object]] = {}
    for name, bucket in clips.items():
        errors = words = 0
        timings: list[float] = []
        for parts in bucket:
            audio = clip_audio(parts)
            started = time.perf_counter()
            text, _ = engine.transcribe(audio, "en")
            timings.append((time.perf_counter() - started) * 1000)
            reference = normalize(" ".join(p.text for p in parts))
            errors += word_errors(reference, normalize(text))
            words += len(reference)
        buckets[name] = {
            "clips": len(bucket),
            "seconds": round(statistics.mean(_clip_seconds(p) for p in bucket), 1),
            "wer": errors / max(1, words),
            "median_ms": statistics.median(timings),
            "max_ms": max(timings),
        }
    vram_mb = free_before - (nvidia_free_vram_mb() or 0) if free_before is not None else None
    engine.close()
    return {
        "model": model_id,
        "variant": variant,
        "device": device,
        "device_name": engine.device_name,
        "load_s": load_s,
        "warm_up_s": warm_up_s,
        "cold_ms": cold_ms,
        "vram_mb": vram_mb,
        "buckets": buckets,
    }


# -- driver ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models-dir", type=Path, default=ROOT / ".local" / "test-models")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".local" / "test-data")
    parser.add_argument("--only", nargs="*", default=None, help="MODEL:VARIANT:DEVICE combinations to run")
    parser.add_argument("--json", type=Path, default=None, help="also write the raw results here")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        model_id, variant, device = args.child.split(":")
        print(json.dumps(run_config(args.models_dir, args.data_dir, model_id, variant, device)))
        return 0

    utterances = load_librispeech(args.data_dir)
    total = sum(u.seconds for u in utterances)
    print(f"{len(utterances)} LibriSpeech utterances, {total / 60:.1f} minutes of speech")
    for name, (lo, hi) in BUCKETS.items():
        print(f"  {name}: {len(build_clips(utterances, lo, hi))} clips")
    configs = [tuple(c.split(":")) for c in args.only] if args.only else list(CONFIGS)
    # Whisp's GPU check looks for its runtime folder; keep the benchmark away from the real user profile.
    env = {**os.environ, "WHISP_HOME": os.environ.get("WHISP_HOME") or str(args.data_dir / "whisp-home")}
    results = []
    for model_id, variant, device in configs:
        print(f"Running {model_id} ({variant} files) on {device} ...", flush=True)
        command = [sys.executable, __file__, "--models-dir", str(args.models_dir), "--data-dir", str(args.data_dir)]
        proc = subprocess.run(
            [*command, "--child", f"{model_id}:{variant}:{device}"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        if proc.returncode != 0 or not lines:
            print(f"  failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
            continue
        result = json.loads(lines[-1])
        if "skipped" in result:
            print(f"  skipped: {result['skipped']}")
            continue
        results.append(result)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print()
    print(format_table(results))
    return 0


def format_table(results: list[dict]) -> str:
    names = list(BUCKETS)
    header = (
        "| Model | Files | Device | Load (+warm-up) | Cold 10 s | "
        + " | ".join(f"WER {n}" for n in names)
        + " | "
        + " | ".join(f"Latency {n}" for n in names)
        + " | VRAM |"
    )
    lines = [header, "|" + "---|" * (6 + 2 * len(names))]
    for r in results:
        buckets = r["buckets"]
        wer = " | ".join(f"{buckets[n]['wer'] * 100:.1f}%" for n in names)
        latency = " | ".join(f"{buckets[n]['median_ms']:.0f} ms" for n in names)
        vram = f"{r['vram_mb']:,} MB" if r["vram_mb"] is not None else "-"
        lines.append(
            f"| {r['model']} | {r['variant']} | {r['device']} | {r['load_s']:.1f} s (+{r['warm_up_s']:.1f} s) | "
            f"{r['cold_ms']:.0f} ms | {wer} | {latency} | {vram} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
