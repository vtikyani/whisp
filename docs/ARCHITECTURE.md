# Whisp architecture

Whisp is a local voice-dictation app for Windows. You hold a hotkey, speak,
release, and clean text appears at your cursor in any app. Everything runs on
your machine.

This document is the contract between Whisp's modules. If you change a public
interface described here, update this file in the same change.

## Principles

1. **Never lose words.** Audio captured before the user presses stop is always
   transcribed and delivered — even if the model is still loading, even if the
   user stops mid-word.
2. **Never block the GUI thread.** The Qt main thread only reacts to signals and
   updates UI. No `wait()`, `join()`, `sleep()`, network, disk-heavy or model
   work runs on it.
3. **Private by default.** Transcript text is never written to logs. History is
   local, optional, and auto-expires. Nothing leaves the machine unless the
   user configures an external AI server.
4. **Fail loudly, recover quietly.** Every error is logged with a traceback and
   surfaced to the user in plain words; the app keeps running.
5. **One source of truth.** Defaults live in `storage/settings.py`. Shared data
   types live in `core/types.py`.

## Layout

```
src/whisp/
  __init__.py            version, app name
  __main__.py            `python -m whisp`
  core/
    types.py             shared dataclasses and enums (no behaviour)
    onnx_runtime.py      ONNX Runtime with its telemetry off; Whisp imports it only from here
    net.py               the HTTPS certificate check (certifi + Windows' store), TLS error helpers
    hardware.py          GPUs (DXGI), NVIDIA driver/VRAM (NVML), CPU core layout
    audio.py             microphone capture, device listing, levels
    vad.py               Silero VAD (bundled ONNX model)
    segmenter.py         turns VAD frames into speech segments (pure logic)
    session.py           SessionPipeline: mic -> VAD -> segments -> ASR -> Transcript
    asr/
      catalog.py         speech models we support (pinned downloads)
      downloads.py       resumable, verified HTTP downloads with progress
      manager.py         ModelManager: install / delete / locate models
      gpu.py             GPU detection and NVIDIA runtime library setup
      engine.py          OnnxAsrEngine: load + transcribe with onnx-asr
      service.py         AsrService: worker thread owning the loaded model
    text/
      context.py         TextContext, app categorisation
      rules.py           filler removal, voice commands, dictionary, snippets, styles
      spacing.py         smart spacing between consecutive dictations
      scripts.py         writing systems without spaces (Chinese, Japanese) and joining text
      polish.py          Polisher: optional LLM rewrite (cleanup + command mode)
      prompts.py         system prompts and few-shot examples
    llm/
      runtime.py         LlmRuntime: manages the built-in llama.cpp server
      catalog.py         local LLMs we support (pinned downloads)
  platform/win32/
    keys.py              chord parsing/formatting, virtual-key tables
    hotkeys.py           HotkeyService: global hotkeys (hold, tap, cancel, record)
    clipboard.py         clipboard snapshot/restore, history-excluded writes
    insert.py            TextInserter: paste or type into the focused app
    foreground.py        focused-window info, foreground change watcher
    autostart.py         "launch at sign-in" via the HKCU Run key
    audio_mute.py        mute other apps while dictating (crash-safe restore)
    instance.py          single-instance guard + activation message
    window.py            DWM/Win32 window styling helpers
    system.py            reduced-motion, theme, DPI and misc system queries
  storage/
    paths.py             app folders (%APPDATA%\Whisp, %LOCALAPPDATA%\Whisp)
    settings.py          Settings schema + SettingsStore
    secrets.py           DPAPI encryption for API keys
    history.py           HistoryStore: SQLite history + stats
  ui/
    theme.py             design tokens for Qt-painted UI (mirrors docs/DESIGN.md)
    overlay/             the floating "pill" shown while dictating
    main_window/         the app window: Qt shell + web UI (HTML/CSS/JS)
  app/
    main.py              entry point: startup, wiring, shutdown
    services.py          AppServices container
    controller.py        AppController: the dictation state machine
    delivery.py          DeliveryService: text processing -> insertion -> history
    sounds.py            short UI sounds (synthesised, no audio files)
    tray.py              system tray icon and menu
    logging_setup.py     rotating logs, crash capture
  assets/
    models/silero_vad.onnx (+ licence)
    brand/               logo and icons
tests/                   pytest suite mirroring src/whisp/
```

## Threads

| Thread | Owner | Does |
|---|---|---|
| GUI (main) | Qt | Controller state machine, overlay, tray, main window. Never blocks. |
| PortAudio callback | `core/audio.py` | Copies frames into a queue, computes the level. Nothing else. |
| `whisp-pipeline` | `SessionPipeline` | Reads mic frames, runs VAD + segmenter, submits segments, assembles transcripts. |
| `whisp-asr` | `AsrService` | Owns the model. Loads/unloads it and runs every transcription, one at a time. |
| `whisp-delivery` | `DeliveryService` | Text rules, optional polish, insertion, history — strictly in session order. |
| `whisp-hotkeys` | `HotkeyService` | Win32 message loop for `RegisterHotKey`, the low-level hook and raw input. |
| `whisp-foreground` | `ForegroundWatcher` | Win32 message loop for the foreground-change event hook. |
| `whisp-mute` | `MuteService` | Mutes and restores other apps' audio sessions (Core Audio COM). |
| download workers | `ModelManager`, `LlmRuntime` | One short-lived thread per download. |

Cross-thread communication uses Qt signals (queued automatically when emitted
from a worker thread) or thread-safe queues. Worker callbacks never touch
widgets.

## A dictation, end to end

1. `HotkeyService` emits `pressed("dictate")`.
2. `AppController` captures the focused window (`foreground.get_foreground_target()`),
   creates a session id, calls `SessionPipeline.start(id)`, enables Esc-to-cancel,
   shows the overlay and plays the start sound. Recording mode is `HOLD_PENDING`.
3. `SessionPipeline` opens the microphone on its own thread. Frames flow through
   `SileroVad` and `Segmenter`. Each completed segment (at a natural pause) goes to
   `AsrService.submit()` right away, so long dictations are mostly transcribed
   before the user stops. With live preview on, the in-progress tail is sent to
   `AsrService.submit_preview()` about once a second; results update the overlay.
4. On release: if the key was held shorter than `hotkeys.tap_to_lock_ms`, the
   session switches to `HANDS_FREE` and keeps recording. Otherwise the controller
   calls `SessionPipeline.stop(id)`.
5. `stop()` closes the mic, then flushes everything the segmenter still holds as the
   final segment. When every segment's result is back, the pipeline emits
   `finished(Transcript)` (or `no_speech(id)` if nothing was said).
6. `AppController` hands `SessionInfo` + `Transcript` to `DeliveryService.submit()`.
7. `DeliveryService` (on its own thread, in order): build `TextContext` → apply
   rules → optional `Polisher.polish()` (falls back to the rule output on any
   failure or timeout) → smart spacing → `TextInserter.insert()` → `HistoryStore.add()`
   → emits `delivered(DeliveryResult)`.
8. `AppController` shows the outcome on the overlay and updates the tray.

The controller can start a new recording while earlier sessions are still being
processed. Their text is inserted in the order the sessions started.

**Command mode** is the same flow with `SessionKind.COMMAND`: at start the
controller asks `TextInserter.capture_selection()` (on a worker) for the selected
text. At delivery, `Polisher.rewrite(selection, instruction)` produces the
replacement, which is pasted over the still-active selection.

## Module contracts

Signatures are Python; `->` names the return type. "Any thread" means the call is
thread-safe.

### core/hardware.py

```python
@dataclass(frozen=True)
class GpuAdapter:
    vendor: str          # "nvidia" | "amd" | "intel" | "other"
    name: str
    vram_mb: int

@dataclass(frozen=True)
class NvidiaInfo:
    driver_version: str  # "617.42"
    driver_major: int
    name: str
    vram_total_mb: int
    vram_free_mb: int

@dataclass(frozen=True)
class Hardware:
    gpus: tuple[GpuAdapter, ...]
    nvidia: NvidiaInfo | None
    logical_cpus: int
    performance_cores: int   # P-cores on hybrid CPUs, else physical cores
    best_gpu: GpuAdapter | None          # property
    has_nvidia: bool                     # property
    has_discrete_non_nvidia: bool        # property (AMD/Intel worth using via Vulkan)

def detect() -> Hardware          # cached; never raises; any thread
def refresh() -> Hardware
def nvidia_free_vram_mb() -> int | None   # live, uncached
```

### core/audio.py

```python
class AudioError(Exception): ...          # message is user-facing

def list_input_devices() -> list[AudioDevice]
    # Fresh scan (re-initialises PortAudio when no stream is open).

class MicrophoneStream:
    def __init__(self, device: str | None, on_level: Callable[[float], None] | None = None): ...
    def start(self) -> None               # raises AudioError; initialises COM on the calling thread (WASAPI needs it)
    def read(self, timeout: float) -> np.ndarray | None
        # One FRAME_SAMPLES float32 frame at 16 kHz, or None on timeout. Any thread.
    def stop(self) -> None                # idempotent; frames already captured stay readable
    @property
    def device_name(self) -> str
    @property
    def error(self) -> str | None         # user-facing reason the stream ended by itself (unplugged, stalled)

@contextmanager
def hold_portaudio() -> Iterator[None]
    # Device re-scans restart PortAudio, which closes every open stream in the process.
    # Wrap the lifetime of any sounddevice stream opened outside this module in this.

def perceptual_level(mean_square: float) -> float   # -60 dBFS -> 0, -12 dBFS -> 1
MSG_NO_DEVICE, MSG_OPEN_FAILED, MSG_DISCONNECTED, MSG_STALLED, MSG_SCAN_FAILED   # user-facing messages
```

`on_level(level)` is called from the audio thread at most ~30 times a second with
a perceptual 0–1 level. Unknown or unplugged devices fall back to the Windows
default with a logged warning. Frames nobody reads are dropped after about a
minute, so a stream used only as a level meter stays small.

### core/vad.py

```python
class SileroVad:
    def __init__(self, model_path: Path | None = None): ...   # default: bundled model
    def reset(self) -> None
    def __call__(self, frame: np.ndarray) -> float            # speech probability, one 512-sample frame
```

Not thread-safe; one instance per thread.

### core/segmenter.py (pure, no threads, no Qt)

```python
@dataclass(frozen=True)
class SegmenterConfig:
    start_threshold: float = 0.5      # speech starts at or above this probability
    end_threshold: float = 0.35       # ...and continues until it falls below this
    min_speech_ms: int = 200          # shorter blips (clicks, coughs) are not speech
    commit_silence_ms: int = 900      # a pause this long ends a segment while recording
    pre_roll_ms: int = 300            # audio kept before speech onset
    post_roll_ms: int = 300           # audio kept after speech ends
    max_segment_s: float = 28.0       # longer speech is cut at the quietest recent frame

@dataclass(frozen=True)
class SegmentAudio:
    samples: np.ndarray
    speech_ms: int

class Segmenter:
    def __init__(self, config: SegmenterConfig | None = None): ...   # None = defaults
    def push(self, frame: np.ndarray, speech_prob: float) -> list[SegmentAudio]
    def flush(self) -> list[SegmentAudio]     # on stop: everything still held
    def pending(self) -> np.ndarray | None    # in-progress speech (for live preview)
    @property
    def total_speech_ms(self) -> int
```

### core/session.py

```python
class SessionPipeline(QObject):
    level = Signal(float)           # mic level while recording
    preview = Signal(str, str)      # session_id, text so far
    finished = Signal(object)       # Transcript
    no_speech = Signal(str)         # session_id
    failed = Signal(str, str)       # session_id, user-facing message

    def __init__(self, settings: SettingsStore, asr: AsrService, *,
                 stream_factory: StreamFactory = MicrophoneStream, parent: QObject | None = None): ...
    def start(self, session_id: str) -> None    # returns at once; mic opens on the pipeline thread
    def stop(self, session_id: str) -> None     # flush and finish
    def cancel(self, session_id: str) -> None   # discard everything
    def shutdown(self) -> None                  # waits up to 2 s
```

- `finished` is emitted in session start order; `no_speech` and `failed` immediately.
- A session can end without `stop()`: if the microphone is unplugged or stalls, the
  pipeline finishes it (with what was captured) and emits `finished`, `no_speech` or
  `failed`. The controller treats that as the end of the recording.
- Empty recognised text with no errors gives `no_speech`, unless the microphone sent
  at least a second of nothing but digital silence: then `failed` says the microphone
  is muted, switched off or blocked. Silence never ends a session early (noise-gated
  microphones send it between words).

### core/asr

```python
# catalog.py
@dataclass(frozen=True)
class ModelFile:
    path: str           # relative path inside the repo
    size: int           # bytes
    sha256: str

@dataclass(frozen=True)
class ModelVariant:
    key: str            # "gpu" or "cpu"
    quantization: str | None
    files: tuple[ModelFile, ...]

@dataclass(frozen=True)
class ModelInfo:
    id: str             # stable id used in settings
    onnx_asr_name: str  # onnx-asr model type
    name: str           # display name
    description: str
    languages: tuple[str, ...]   # ISO codes; ("en",) for English-only
    accuracy: int       # 1-5, for UI
    speed: int          # 1-5, for UI
    license: str
    license_url: str
    repo_id: str
    revision: str       # pinned commit
    variants: tuple[ModelVariant, ...]
    recommended: bool
    attribution: str = ""          # credit line for the About page (CC-BY models)
    def variant(self, key: str) -> ModelVariant    # raises KeyError
    multilingual: bool             # property

CATALOG: tuple[ModelInfo, ...]
    # Parakeet TDT 0.6B v2 (English, default), Parakeet TDT 0.6B v3 (25 European languages),
    # Whisper Large v3 Turbo (99 languages). "gpu" variants are fp32 (Whisper: fp16), "cpu" variants int8.
def get_model(model_id: str) -> ModelInfo        # raises KeyError

# downloads.py — resumable (HTTP Range), sha256-verified, cancellable, atomic; no huggingface_hub.
#   Certificates are checked with core.net.tls_context(); one that can't be verified fails at once
#   (no retries) with advice for work networks. llm/runtime.py downloads the same way.
class DownloadError(Exception): ...     # str() is user-facing
class DownloadCancelled(Exception): ...
def download_files(items, *, progress=None, cancel=None, client=None, check_space=True) -> None
def format_size(num_bytes: int) -> str

# manager.py
class ModelManager(QObject):
    progress = Signal(str, object, object)   # model_id, bytes_done, bytes_total (Python ints)
    changed = Signal()                       # something was installed, removed, or failed
    download_failed = Signal(str, str)       # model_id, user-facing message
    def __init__(self, models_dir: Path, *, catalog=CATALOG, base_url=HF_BASE_URL, client_factory=..., parent=None): ...
    def is_installed(self, model_id: str, variant: str) -> bool        # any thread; KeyError for unknown ids
    def model_path(self, model_id: str, variant: str) -> Path
    def model_info(self, model_id: str) -> ModelInfo
    def installed_variants(self, model_id: str) -> list[str]
    def install_state(self) -> dict[str, dict]   # {installed, downloading, variant, progress, in_use} per model
    def download(self, model_id: str, variant: str) -> None            # async
    def is_downloading(self, model_id: str) -> bool
    def cancel(self, model_id: str) -> None
    def delete(self, model_id: str) -> None      # raises ModelInUseError (an OSError) while loaded
    def retain(self, model_id: str, variant: str) -> bool   # the engine holds it
    def release(self, model_id: str) -> None
    def in_use(self, model_id: str) -> bool
```

A model is installed only when its manifest — written last, after every file's
size and sha256 check — is present, so a half-finished folder is never loaded.

```python
# gpu.py
@dataclass(frozen=True)
class GpuInfo:
    has_nvidia: bool
    name: str | None
    vram_mb: int | None
    cuda_ready: bool          # runtime libraries load and a CUDA session really runs
    detail: str               # plain-language explanation for the UI
    runtime_download_size: int = 0   # bytes the GPU runtime pack would download; 0 = not needed/won't help
def detect_gpu() -> GpuInfo   # cached after first call; any thread
def refresh_gpu() -> GpuInfo
def prepare_cuda() -> bool    # make CUDA/cuDNN loadable (pip wheels in dev, the runtime pack when frozen)
def install_cuda_runtime(progress=None, cancel=None, *, directory=None, client=None) -> None   # blocking
def runtime_installed(directory: Path | None = None) -> bool
def cuda_provider_options() -> dict[str, Any]   # cudnn_conv_algo_search=HEURISTIC

# engine.py
@dataclass(frozen=True)
class EngineSpec:
    model: ModelInfo
    folders: dict[str, Path]  # installed variants
    use_cuda: bool
    cpu_threads: int = 0      # 0 = performance cores
def load_engine(spec: EngineSpec) -> AsrEngine     # warmed up; raises EngineError
class AsrEngine(Protocol):
    device: str; device_name: str; variant: str; notice: str | None
    def transcribe(self, samples: np.ndarray, language: str) -> tuple[str, str | None]
    def close(self) -> None

# service.py
class AsrService(QObject):
    status_changed = Signal(object)          # EngineStatus
    gpu_runtime_progress = Signal(object, object)   # bytes_done, bytes_total
    gpu_runtime_finished = Signal(object)           # GpuInfo; the model then reloads onto the GPU
    gpu_runtime_failed = Signal(str)                # message, "" if the user cancelled
    def __init__(self, settings: SettingsStore, models: ModelManager, *, engine_factory=load_engine,
                 gpu_info=detect_gpu, gpu_installer=install_cuda_runtime, gpu_refresh=refresh_gpu,
                 clock=time.monotonic, parent=None): ...
    def start(self) -> None                  # starts worker; loads the configured model if installed
    def shutdown(self) -> None
    @property
    def status(self) -> EngineStatus
    def ensure_loaded(self) -> None          # async; re-checks installs; no-op if ready or loading
    def reload(self) -> None                 # choose model/variant/device again (e.g. after the GPU pack)
    def submit(self, segment: Segment, callback: Callable[[AsrResult], None]) -> None
        # Any thread. Exactly one callback per job, normally on whisp-asr. Jobs wait while
        # the model loads. If no model can be loaded, the callback gets AsrResult(error=...).
    def submit_preview(self, session_id: str, samples: np.ndarray,
                       callback: Callable[[str], None]) -> None
        # Lowest priority; only the newest preview per session is kept; called at most once, only with text.
    def discard(self, session_id: str) -> None   # drop queued work for a cancelled session
    def install_gpu_runtime(self) -> bool        # async; False if already running
    def cancel_gpu_runtime(self) -> None
    gpu_runtime_installing: bool                 # property
```

`AsrService` reloads on its own when `transcription.model_id` or
`transcription.device` change (`transcription.language` is read per job), and
unloads after `transcription.unload_after_minutes` of idleness (0 = never).
Parakeet runs through onnx-asr's model classes with a subclass that clamps the
encoder's reported lengths (onnx-asr issue #138: it otherwise crashes on some
clip lengths). Whisper output of three words or fewer is double-checked with the
VAD (it invents "you" on silence), and a loop guard cuts repeated sentences.

If the GPU fails mid-recognition (a driver reset, out of graphics memory), the
service moves to the CPU and recognises that segment again there, at most once,
so no sentence is lost. A model that loads on the GPU but fails its warm-up run
falls back to the CPU with a notice. Submitting a segment drops its session's
queued preview, which would be ignored anyway.

Whisper writes at most 444 tokens per pass, and at most 64 per second of audio
(English needs about 4, Burmese about 55). A few seconds of Burmese, Malayalam,
Georgian or Telugu can fill a pass; when one fills without finishing, its audio
is cut at the quietest moment in its middle third and each part is recognised
on its own with the same language, halving at most `MAX_SPLITS` (3) times.

Recognised pieces (cut parts, segments) are joined by `text.scripts.join_spoken`:
with a space, except next to Chinese or Japanese, which are written without spaces.

### core/text

```python
# context.py
@dataclass(frozen=True)
class TextContext:
    category: AppCategory
    style: str                     # "verbatim" | "clean" | "casual" | "formal"
    app_name: str
    language: str | None
    dictionary: tuple[DictionaryEntry, ...]
    snippets: tuple[Snippet, ...]
    remove_fillers: bool
    voice_commands: bool
    custom_instructions: str
    now: datetime | None = None    # filled at delivery time, for snippet variables

def categorize(process_name: str, title: str, overrides: Mapping[str, str]) -> AppCategory
def build_context(settings: Settings, target: TargetWindow | None, language: str | None) -> TextContext

# rules.py
def apply_rules(raw: str, ctx: TextContext) -> str     # deterministic, fast, pure
def expand_variables(text: str, now: datetime) -> str  # snippet {date} {time} {weekday} {year}; {{ }} = literal
    # Spoken punctuation ("comma", "period"…) only works in English text: the language comes from the
    # setting, an English-only model or Whisper's detection, else from the text's common words.
    # "new line", "new paragraph" and "scratch that" work in every language.

# scripts.py
def is_spaceless(char: str) -> bool             # Chinese or Japanese: written without spaces between words
def join_spoken(parts: Iterable[str]) -> str    # strip, drop empties, space between unless spaceless

# spacing.py
class SpacingMemory:
    def __init__(self, *, clock=time.monotonic, window_s: float = 600,
                 activity: Callable[[int], KeyActivity | None] | None = None): ...
        # activity(hwnd) = HotkeyService.typing_activity: the KIND of the user's last key in that
        # window, never the key itself
    def remember(self, target: TargetWindow | None, text: str) -> None   # target None forgets every window
    def prefix_for(self, target: TargetWindow | None, text: str, mode: str) -> str   # "" or " "
    def forget(self) -> None

# polish.py
@dataclass(frozen=True)
class PolishResult:
    text: str
    used: bool             # False when we fell back to the input (error set) or skipped on purpose (error None)
    error: str | None
    elapsed_ms: int

class Polisher:
    def __init__(self, settings: SettingsStore, runtime: LlmRuntime): ...
        # runtime: anything with base_url() / api_key() / installed_models()
    def is_ready(self) -> bool                                          # any thread
    def polish(self, text: str, ctx: TextContext) -> PolishResult        # blocking; never raises
    def rewrite(self, selection: str, instruction: str, ctx: TextContext) -> PolishResult
    def test(self) -> tuple[bool, str]                                   # blocking
    def list_models(self, provider: str) -> list[str]                    # blocking
```

For an AI server on another machine, an address typed without a scheme gets
`https://` (this PC gets `http://`), and `test()` warns about plain `http://`.
When https fails because the server only speaks http, the message says to type
`http://` in front. A user name or password in an address never appears in
messages or logs. Only the dictionary terms that occur in the text are sent.

To guess whether a long text will finish within the time limit, the polisher
keeps the server's writing speed. It uses the server's own timings when it
reports them (llama-server `timings`, Ollama `eval_count`/`eval_duration`), which
leave out model loading. Otherwise it uses the clock, but needs two answers and
takes the median of the last five, so one cold start can't make it skip polish.

### core/llm/runtime.py

```python
class LlmRuntime(QObject):
    status_changed = Signal(object)          # LlmStatus(state, model_id, base_url, message, backend, installing)
    progress = Signal(str, object, object)   # item, bytes_done, bytes_total
    install_finished = Signal(str)           # model_id
    install_failed = Signal(str, str)        # model_id, user-facing message
    def __init__(self, settings: SettingsStore, runtime_dir: Path, models_dir: Path,
                 parent: QObject | None = None, *, flavor: str | None = None): ...
        # models_dir is the LLM models folder itself (paths.models_dir / "llm")
    def catalog(self) -> list[LlmModelInfo]
    def install(self, model_id: str) -> None       # async: server binary + model, sha256-verified
    def cancel_install(self) -> None
    def uninstall(self, model_id: str) -> None
    def is_installed(self, model_id: str) -> bool
    def installed_models(self) -> list[str]
    def model_dir(self, model_id: str) -> Path
    def ensure_running(self) -> None               # async; starts the server if installed + enabled
    def base_url(self) -> str | None               # any thread; None if not running
    def api_key(self) -> str | None                # per-launch key the server requires
    @property
    def status(self) -> LlmStatus                  # LlmState: off | not_installed | starting | ready | error
    def stop(self) -> None
    def shutdown(self) -> None
```

The server is a pinned llama.cpp `llama-server.exe` build (CUDA 13 / CUDA 12 /
Vulkan / CPU, chosen from `core.hardware`), run as a child process on 127.0.0.1
with a random per-launch API key, inside a Windows Job Object so it can never
outlive Whisp. Start it after the speech model has loaded, so it sizes itself to
the GPU memory that's really free. The server gets only the environment it needs
(Windows and temp folders, app-data folders, the CUDA device variables, a `PATH`
of its own folder and Windows', and the key). Only `llama-server.exe`, the
libraries it loads and their licences are unpacked from the release zip.
`install` and `uninstall` refuse ids that aren't in the catalog, and uninstall
only ever deletes a folder directly inside the models folder.

### platform/win32

```python
# keys.py
@dataclass(frozen=True)
class Chord: ...                                  # modifiers + optional trigger key
def parse_chord(text: str) -> Chord               # ValueError with a friendly message
def format_chord(chord: Chord) -> str             # canonical: "ctrl+shift+space"
def display_chord(text: str) -> list[str]         # key-cap labels: ["Ctrl", "Shift", "Space"]
def key_kind(vk: int, *, ctrl=False, alt=False, win=False, shift=False) -> KeyKind | None
    # For smart spacing; never the key itself. None for keys that leave the text and cursor alone,
    # including Alt+Tab, any Win shortcut, and Ctrl+C/S/B/I/U. Pass AltGr as neither Ctrl nor Alt.

# hotkeys.py
class HotkeyService(QObject):
    pressed = Signal(str)        # action: "dictate" | "command" | "paste_last" | "copy_last" | "hands_free" | "cancel"
    released = Signal(str)       # every pressed gets exactly one released, even after interrupted
    interrupted = Signal(str)    # a modifier-only chord was combined with another key
    recorded = Signal(str)       # chord captured by start_recording(); "" if the user pressed Esc
    binding_error = Signal(str, str)   # action, message
    session_locked = Signal()    # the PC was locked or the session disconnected (recording stops)
    def __init__(self, parent: QObject | None = None): ...
    def start(self) -> None
    def stop(self) -> None
    def set_bindings(self, bindings: Mapping[str, str]) -> dict[str, str]   # returns {action: error}
        # Waits up to 2 s for the hotkey thread. If it's busy (Windows can be slow right after sign-in),
        # returns the validation errors only: the shortcuts are applied when it's free and any problem
        # arrives as binding_error; ones still not applied after 20 s are reported as not responding.
    def binding_errors(self) -> dict[str, str]
    def set_cancel_active(self, active: bool) -> None    # Esc -> pressed("cancel"), swallowed
    def start_recording(self) -> None                    # next chord -> recorded(), not dispatched
    def stop_recording(self) -> None
    def set_paused(self, paused: bool) -> None
    def typing_activity(self, hwnd: int) -> KeyActivity | None
        # Kind of the last key typed into that window (the 256 most recent windows are kept);
        # None if unknown (no hook). After the hook starts or is reinstalled, every window reports
        # a fresh baseline: keys may have gone unseen.
```

Action names are any `[a-z][a-z0-9_]*`; "cancel" is reserved for Esc.

Chords with a regular key (`ctrl+shift+space`, `f8`) use `RegisterHotKey`, so
Windows swallows them and reports conflicts; release is detected by polling key
state. Windows keeps many Win combinations for itself (`ctrl+win+space` switches
input languages); a Win chord that Windows refuses to register is handled by the
hook instead, which swallows its key while exactly the chord's modifiers are
held. Modifier-only chords (`ctrl+win`, `rctrl`) use a low-level keyboard hook
on `whisp-hotkeys`, watched by a raw-input health check that reinstalls the
hook if Windows drops it. Holding the Windows key as part of a chord must not
open the Start menu. When a longer chord follows a shorter one
(`ctrl+win` → `ctrl+win+space`), `interrupted` for the shorter one is emitted
before `pressed` for the longer one.

```python
# foreground.py
def get_foreground_target() -> TargetWindow | None      # any thread
def is_own_hidden_window(hwnd: int) -> bool              # an invisible window of Whisp's (e.g. the tray menu's)
def focused_control(hwnd: int) -> int                    # the control with the keyboard focus in hwnd's thread
class ForegroundWatcher(QObject):
    changed = Signal(object)                             # TargetWindow | None
    def start(self) -> None
    def stop(self) -> None
    is_watching: bool                                    # property; True once its hook is installed

# insert.py
class TextInserter:
    def insert(self, text: str, *, target: TargetWindow | None, method: str,
               restore_clipboard: bool, on_focus_change: str) -> InsertionResult
        # Blocking; call from a worker thread. Waits for modifier keys to be released.
        # Leaves the text on the clipboard instead when no window, or one of Whisp's own hidden
        # windows, has the focus, or the app runs as administrator and Whisp doesn't.
        # "type" pastes instead when the text has emoji and the focused text box is a Qt app's
        # (Telegram, Zoom…): Qt delivers a typed emoji ahead of the keys typed before it.
    def copy_to_clipboard(self, text: str) -> bool
    def capture_selection(self, timeout_ms: int = 500) -> str | None
        # Blocking; waits for modifiers to be released, copies the selection via Ctrl+C,
        # restores the clipboard. Returns None if nothing was selected.
    def default_method_for(self, process_name: str) -> str | None   # built-in table (terminals)
    def restore_delay_for(self, process_name: str) -> float
        # How long an app needs to read a paste before the clipboard is restored
        # (0.6 s; 3 s for Outlook/Word/WhatsApp; 5 s for remote desktops).
        # insert() returns as soon as the text is in; the restore waits on a short-lived
        # "whisp-clipboard-restore" thread that holds the clipboard lock until it's done.

# autostart.py
def autostart_status(value_name: str = "Whisp") -> str
    # "enabled" | "disabled" | "blocked" (turned off in Windows settings)
def set_autostart(enabled: bool, value_name: str = "Whisp") -> None   # raises AutostartError
def refresh_autostart_command(value_name: str = "Whisp") -> bool
    # Installed builds only: point an existing entry at this copy (keeps Windows' on/off switch).

# audio_mute.py
class MuteService:                      # worker thread "whisp-mute"; COM (MTA) on that thread
    def __init__(self, muter: AudioMuter | None = None, *, extra_exclusions: Iterable[str] = ()): ...
    def mute(self) -> None              # non-blocking, any thread
    def restore(self) -> None           # non-blocking, any thread
    def set_extra_exclusions(self, names: Iterable[str]) -> None
    def shutdown(self, timeout: float = 2.0) -> None   # final restore, then join
```

Muting skips Whisp itself, the system-sounds session, call and chat apps
(`COMMUNICATION_APPS`) and any app currently using a microphone. Before muting,
the targets are written to `muted-sessions.json` in the data folder; if Whisp
crashes, the next start unmutes them. Windows remembers an app's mute across
restarts, so entries for apps that quit while muted stay pending until the app's
next session appears, then get unmuted. Sessions the user un-muted by hand during
a dictation are left alone. The controller mutes 250 ms into a recording (so an
accidental shortcut never makes music skip) and restores as soon as recording ends.

```python
# instance.py
class SingleInstance(QObject):
    def __init__(self, name: str | None = None, parent: QObject | None = None): ...
    message = Signal(str)          # sent by a second launch, e.g. "show"
    def acquire(self) -> bool      # False if another instance owns the lock
    def send(self, message: str) -> bool
    def release(self) -> None

# window.py
def style_overlay_window(widget: QWidget) -> None   # no-activate, tool window, topmost
def set_click_through(widget: QWidget, enabled: bool) -> None
def style_app_window(widget: QWidget, *, dark: bool) -> None   # dark title bar, caption colour
```

### storage

```python
# settings.py — see the module; SettingsStore.current is an immutable snapshot.
#   update()/reset() save first and raise OSError if that fails (nothing changes then).
#   save_problem: a message if settings couldn't be written at start-up (Whisp runs on them in memory).

# paths.py
def display_path(value: Path | str) -> str     # the user's folder shown as "~" in logs and messages

# secrets.py
def protect(plain: str) -> str      # DPAPI (current user) + base64; raises OSError if Windows refuses
def unprotect(token: str) -> str    # "" if it can't be decrypted

# history.py
@dataclass(frozen=True)
class HistoryEntry:
    id: int
    created_at: float              # time.time()
    kind: str                      # "dictation" | "command"
    app_name: str
    process_name: str
    category: str
    raw_text: str                  # "" when history text is disabled
    final_text: str
    word_count: int
    audio_ms: int
    speech_ms: int
    latency_ms: int
    model_id: str
    polished: bool
    outcome: str

class HistoryStore(QObject):
    added = Signal(object)          # HistoryEntry
    changed = Signal()              # deletions, clears, pruning
    cleared = Signal()              # everything was cleared (delivery forgets the last text too)
    def __init__(self, db_path: Path, settings: SettingsStore): ...
    def add(self, entry: HistoryEntry) -> HistoryEntry       # any thread; returns it with id
    def list(self, *, query: str = "", app: str = "", limit: int = 50, offset: int = 0) -> list[HistoryEntry]
    def get(self, entry_id: int) -> HistoryEntry | None
    def delete(self, entry_id: int) -> None
    def clear(self) -> None
    def prune(self, *, now: float | None = None) -> int
    def start_maintenance(self) -> None   # prune now and every 24 h (and when privacy settings change)
    def last_text(self) -> str | None
    def stats(self, *, now: float | None = None) -> dict     # see "Stats" below
    def apps(self) -> list[str]
    def close(self) -> None
    persistent: bool                 # False if the database couldn't open: history is in memory this session
    search_uses_fts: bool

def count_words(text: str) -> int
```

Text fields are hidden from `repr()` so a stray log line can't print them. With
history turned off, new rows keep word counts and timings (for stats) but no text.

**Stats** (`HistoryStore.stats()`): `total_words`, `total_dictations`,
`words_today`, `words_this_week`, `avg_wpm` (words ÷ speaking minutes),
`time_saved_minutes` (vs typing at 40 wpm, minus speaking time),
`streak_days`, `best_streak_days`, `top_apps` (`[{app, words}]`, top 5),
`daily` (last 30 days, `[{date: "YYYY-MM-DD", words}]`).

### ui

```python
# overlay
class Overlay(QObject):
    stop_clicked = Signal()
    cancel_clicked = Signal()
    idle_clicked = Signal()
    def __init__(self, settings: SettingsStore): ...
    def show_recording(self, kind: SessionKind, mode: RecordingMode) -> None
    def set_mode(self, mode: RecordingMode) -> None
    def set_level(self, level: float) -> None
    def set_preview(self, text: str) -> None
    def set_note(self, text: str | None) -> None       # e.g. "Loading model…" while recording
    def show_processing(self) -> None
    def show_outcome(self, outcome: Outcome, message: str | None = None) -> None
    def show_notice(self, text: str, kind: str = "info") -> None   # "info" | "warning" | "error"
    def set_target(self, target: TargetWindow | None) -> None      # monitor + fullscreen awareness
    def set_hotkey_hint(self, keys: list[str]) -> None
    def shutdown(self) -> None

# main_window
def register_web_scheme() -> None    # must run before QApplication is created
class MainWindow(QWidget):
    def __init__(self, services: AppServices, *, adapters: BridgeAdapters | None = None): ...
    def show_page(self, page: str, section: str | None = None) -> None
        # "home" | "history" | "dictionary" | "snippets" | "styles" | "settings" | "onboarding"
```

The web UI (Preact + htm, vendored, no build step) is served from a custom
`whisp://app/` scheme under a strict Content-Security-Policy. It talks to Python
through one QWebChannel object, `bridge`, with a single RPC slot
`call(method: str, params_json: str) -> str` (JSON reply `{"ok": value}`,
`{"error": message}`, or `{"pending": id}` for slow calls whose result follows as
a `call.result` event) and one signal `event(name: str, payload_json: str)`. The
method and event tables live in the `ui/main_window/bridge.py` docstring. Stored
API keys never travel back to the page.

### app

```python
# services.py
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

# sounds.py
class SoundPlayer:
    def __init__(self, settings: SettingsStore): ...
    def play(self, name: str) -> None    # "start" | "stop" | "lock" | "cancel" | "error"; non-blocking
    def shutdown(self) -> None

# delivery.py
class DeliveryService(QObject):
    delivered = Signal(object)            # DeliveryResult
    def __init__(self, settings, history, polisher, inserter, *, text_tools=None, get_target=None,
                 typing_activity=None, parent=None): ...
    def submit(self, info: SessionInfo, transcript: Transcript) -> None
    def paste_last(self) -> None
    def copy_last(self) -> None
    def shutdown(self, timeout: float = 1.5) -> None

# controller.py
class AppController(QObject):
    state_changed = Signal()
    notification = Signal(str, str, str)          # title, message, kind
    open_window_requested = Signal(str, str)      # page, section
    def start(self) -> None
    def shutdown(self) -> None
    def start_hands_free(self, *, from_menu: bool = False) -> None
    def toggle_paused(self) -> None
    def paste_last(self) -> None
    def copy_last(self) -> None
    def on_foreground_changed(self, target: TargetWindow | None) -> None
    def hotkey_keys(self, action: str) -> list[str]
    def status_message(self) -> str
    def problem(self) -> str | None
    is_recording: bool; recording_mode: RecordingMode | None; is_paused: bool
    binding_errors: dict[str, str]; last_result: DeliveryResult | None; state: AppState
```

## Recording state machine (AppController)

| Event | No active recording | `HOLD_PENDING` | `HOLD` | `HANDS_FREE` |
|---|---|---|---|---|
| dictate pressed | start (HOLD_PENDING) | — | — | stop |
| dictate released | — | held < tap threshold → HANDS_FREE; else stop | stop | — |
| hands_free pressed | start (HANDS_FREE) | — | — | stop |
| cancel (Esc / ✕) | — | cancel | cancel | cancel |
| interrupted (< 1 s in) | — | cancel silently | cancel silently | — |
| overlay stop | — | — | — | stop |
| max length reached | — | stop | stop | stop |
| PC locked / session disconnected | — | stop | stop | stop |

`command` follows the same table with `SessionKind.COMMAND`. Starting a session
never waits for the previous one to finish processing.

Longer shortcuts that begin with the dictate shortcut need care, because the
shorter chord fires first:

- **Starting.** Ctrl+Win+Space starts as Ctrl+Win (a dictation), then Space
  interrupts it. An interruption doesn't cancel at once: if `hands_free` (or
  `command`) is pressed within `INTERRUPT_GRACE_MS`, the same recording is kept
  and becomes hands-free (or a command), audio included. Otherwise it's cancelled
  silently. Either arrival order works, and the longer shortcut is adopted within
  the first second whether the recording is still HOLD_PENDING or already HOLD.
  Letting go during the grace period neither locks hands-free nor stops, and other
  apps aren't muted until it's decided.
- **Stopping.** When a press stops hands-free, a longer shortcut that completes
  within `STOP_GESTURE_S` is the rest of the same keypress and is ignored, so
  pressing Ctrl+Win+Space to stop doesn't start a new recording.
- **Command mode** reads the selected text when recording stops (the command
  shortcut is still held while it starts, and Ctrl+C under held Alt/Win isn't a copy).
- One minute before `audio.max_recording_minutes`, a notice and a soft sound warn
  that recording will stop; at the limit everything said so far is kept.

## Errors

- User-facing messages are short, plain and actionable ("Microphone not found —
  check Settings → Microphone").
- Every caught exception is logged with `log.exception(...)`.
- `logging_setup` installs `sys.excepthook`, `threading.excepthook`, a Qt message
  handler and `faulthandler` (native crashes go to `logs/crash.log`).
- Never log transcript text, selections, clipboard contents or window titles.
  Log lengths and counts instead. Dataclass fields that hold them use
  `field(repr=False)`, so logging an object can't leak them either.

## Testing

`pytest` from the repo root. Tests must not touch the user's real settings,
history or models: the `conftest.py` fixture points `WHISP_HOME` at a temp
folder. Tests that need a GPU, downloaded models, or an interactive desktop are
marked `gpu`, `models` or `desktop` and are skipped unless enabled with
`-m`. Any test that uses the clipboard saves and restores it. Desktop tests take
the focus only from the app that started them, only while nobody touches the
keyboard or mouse, and keep it until the run ends: Windows lets an app take the
focus only while it has it, so handing it back between tests would lock them out.

## Conventions

- Python 3.12, `from __future__ import annotations`, type hints everywhere.
- `ruff` for lint and format (config in `pyproject.toml`), line length 120.
- Logging: `log = logging.getLogger(__name__)`.
- Win32 calls go through `ctypes` with explicit `argtypes`/`restype`.
- No placeholder code, no mock data in the app. Tests may use fakes.
