"""Local language models Whisp can download for AI polish and command mode.

Every file is pinned to a Hugging Face commit and checked against its
SHA-256 (the ``lfs.oid`` from ``GET /api/models/<repo>/tree/<revision>``), so a
download is exactly the file that was tested. Verified 2026-10-12 against
llama.cpp build b11429.

Measured on an RTX 4070 with ``-c 8192`` and MTP on: Qwen3.5-4B used 3.6 GB
of VRAM and generated ~185 tokens/s; Qwen3.5-2B 1.7 GB and ~285 tokens/s;
Gemma 4 E4B 3.4 GB and ~275 tokens/s.

"MTP" (multi-token prediction) means the model drafts a few tokens at once
and then checks them, which roughly doubles generation speed here. The
Qwen3.5 MTP files carry their draft head inside the main file; Gemma 4 needs
a separate small draft file.
"""

from __future__ import annotations

from dataclasses import dataclass

_HF = "https://huggingface.co"


@dataclass(frozen=True)
class LlmFile:
    """One pinned file in a Hugging Face repository."""

    repo_id: str
    revision: str
    """Commit SHA the file is pinned to."""
    path: str
    """Path inside the repository; also the local file name."""
    size: int
    """Bytes."""
    sha256: str

    @property
    def url(self) -> str:
        return f"{_HF}/{self.repo_id}/resolve/{self.revision}/{self.path}"

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class LlmModelInfo:
    id: str
    """Stable id used in settings (``polish.builtin_model``)."""
    name: str
    """Display name."""
    description: str
    """One line for the UI."""
    model_file: LlmFile
    draft_file: LlmFile | None
    """Separate speculative-decoding draft model, when the model needs one."""
    license: str
    license_url: str
    multilingual: bool
    vram_mb: int
    """Measured graphics memory with Whisp's server settings (8K context), rounded up."""
    ram_mb: int
    """Approximate memory when running on the CPU."""
    server_args: tuple[str, ...]
    """Extra llama-server flags this model needs (``{draft}`` is replaced with the draft file's path)."""
    recommended: bool = False

    @property
    def files(self) -> tuple[LlmFile, ...]:
        return (self.model_file,) if self.draft_file is None else (self.model_file, self.draft_file)

    @property
    def download_size(self) -> int:
        return sum(f.size for f in self.files)


_MTP_ARGS = ("--spec-type", "draft-mtp", "--spec-draft-n-max", "4")

CATALOG: tuple[LlmModelInfo, ...] = (
    LlmModelInfo(
        id="qwen3.5-4b",
        name="Qwen3.5 4B",
        description="Best quality for most PCs. Polishes a paragraph in about half a second on a recent graphics card.",
        model_file=LlmFile(
            repo_id="unsloth/Qwen3.5-4B-MTP-GGUF",
            revision="86835bf9949e4d14d6860f7910b1340ad4f271a9",
            path="Qwen3.5-4B-Q4_K_M.gguf",
            size=2_834_975_040,
            sha256="3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630",
        ),
        draft_file=None,
        license="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        multilingual=True,
        vram_mb=3_700,
        ram_mb=3_800,
        server_args=_MTP_ARGS,
        recommended=True,
    ),
    LlmModelInfo(
        id="qwen3.5-2b",
        name="Qwen3.5 2B",
        description="Fastest and lightest. A good choice without a graphics card; a little less careful with wording.",
        model_file=LlmFile(
            repo_id="unsloth/Qwen3.5-2B-MTP-GGUF",
            revision="e05864f8066d874d5f85aaff007ae57a2a7d1efe",
            path="Qwen3.5-2B-Q4_K_M.gguf",
            size=1_329_851_808,
            sha256="18c92228fc1498a02506f64593f7cc8cc02f406f732a217c6ba99ad1254d4f0b",
        ),
        draft_file=None,
        license="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE",
        multilingual=True,
        vram_mb=1_800,
        ram_mb=2_000,
        server_args=_MTP_ARGS,
    ),
    LlmModelInfo(
        id="gemma-4-e4b",
        name="Gemma 4 E4B",
        description="Larger download with strong results in many languages. Needs a graphics card with 6 GB or more.",
        model_file=LlmFile(
            repo_id="ggml-org/gemma-4-E4B-it-GGUF",
            revision="b8093469224f83f5c38f691eb906c380e9e63114",
            path="gemma-4-E4B-it-Q4_0.gguf",
            size=4_590_807_392,
            sha256="a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2",
        ),
        draft_file=LlmFile(
            repo_id="ggml-org/gemma-4-E4B-it-GGUF",
            revision="b8093469224f83f5c38f691eb906c380e9e63114",
            path="mtp-gemma-4-E4B-it-Q8_0.gguf",
            size=98_653_280,
            sha256="f38ae62962657c7a6303c49bbb147e9ae23634e911cfa532fac0818c2e18b665",
        ),
        license="Apache-2.0",
        license_url="https://huggingface.co/google/gemma-4-E4B-it",
        multilingual=True,
        vram_mb=3_500,
        ram_mb=5_600,
        server_args=("--model-draft", "{draft}", *_MTP_ARGS),
    ),
)

_BY_ID = {m.id: m for m in CATALOG}


def get_model(model_id: str) -> LlmModelInfo:
    """The catalog entry for ``model_id``. Raises ``KeyError`` for unknown ids."""
    return _BY_ID[model_id]


def recommended_model() -> LlmModelInfo:
    return next(m for m in CATALOG if m.recommended)
