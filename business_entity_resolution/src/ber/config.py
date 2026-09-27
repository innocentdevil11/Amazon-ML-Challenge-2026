"""Central, portable configuration.

Everything that differs between "my Windows laptop", "Kaggle notebook" and
"Colab notebook" is decided here: data/output/cache directories, whether a
GPU is available, and how many CPU workers to use. No other module hardcodes
a path or a device string.

Override via environment variables (``BER_DATA_DIR`` etc.) or CLI flags in
``run_pipeline.py``. Nothing here reads argv directly so it also works when
pasted into a notebook cell.
"""
from __future__ import annotations

import dataclasses
import io
import os
import sys
from pathlib import Path


def _disable_tokenizer_parallelism() -> None:
    """Prevents a real deadlock: HuggingFace's fast (Rust/Rayon) tokenizer
    keeps its own internal thread pool once used, and hangs -- with no error,
    no CPU usage, indistinguishable from a stalled process -- if joblib later
    starts more threads or processes in the same interpreter (confirmed by
    reproduction: loading the embedding model, then calling
    ``pair_cosine_chunked``, hung under BOTH the loky and threading joblib
    backends). Must be set before ``sentence_transformers``/``transformers``
    is imported anywhere, so it lives here in the first module the pipeline
    imports, not in embed.py where the import actually happens.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_disable_tokenizer_parallelism()


def _force_utf8_stdio() -> None:
    """Windows consoles default to cp1252; force UTF-8 so native-script
    business names (Devanagari, Bengali, ...) never crash a print()."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() != "utf-8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_stdio()


def _detect_data_dir() -> Path:
    env = os.environ.get("BER_DATA_DIR")
    if env:
        return Path(env)
    kaggle_input = Path("/kaggle/input")
    kaggle_dataset_dirs = sorted(kaggle_input.glob("*/dataset")) if kaggle_input.exists() else []
    candidates = [
        # Kaggle: /kaggle/input/<dataset-slug>/dataset
        *kaggle_dataset_dirs,
        Path("/kaggle/input/dataset"),
        Path("/content/dataset"),  # Colab, if the zip was extracted there
        Path(__file__).resolve().parents[2] / "student_resource" / "dataset",  # local repo layout
        Path("student_resource/dataset"),
        Path("dataset"),
    ]
    for c in candidates:
        if (c / "train" / "train_source1.tsv").exists():
            return c
    # Fall back to the local repo default even if it doesn't exist yet, so the
    # error message downstream points at a sensible path.
    return Path(__file__).resolve().parents[2] / "student_resource" / "dataset"


def _detect_gpu() -> bool:
    if os.environ.get("BER_USE_GPU") is not None:
        return os.environ["BER_USE_GPU"].lower() in ("1", "true", "yes")
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_n_jobs() -> int:
    env = os.environ.get("BER_N_JOBS")
    if env:
        return int(env)
    cpu = os.cpu_count() or 4
    return max(1, cpu - 1)


@dataclasses.dataclass
class Config:
    data_dir: Path = dataclasses.field(default_factory=_detect_data_dir)
    output_dir: Path = dataclasses.field(
        default_factory=lambda: Path(os.environ.get("BER_OUTPUT_DIR", "output"))
    )
    cache_dir: Path = dataclasses.field(
        default_factory=lambda: Path(os.environ.get("BER_CACHE_DIR", "cache"))
    )
    use_gpu: bool = dataclasses.field(default_factory=_detect_gpu)
    n_jobs: int = dataclasses.field(default_factory=_detect_n_jobs)
    seed: int = int(os.environ.get("BER_SEED", "42"))

    # blocking / embedding knobs (overridable per-run for the benchmark stage)
    embed_model: str = os.environ.get(
        "BER_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    embed_batch_size: int = int(os.environ.get("BER_EMBED_BATCH", "256"))
    embed_max_seq_len: int = int(os.environ.get("BER_EMBED_MAXLEN", "64"))
    # Embed native-script text (pre-transliteration) alongside the ASCII
    # text for ANN blocking, taking the effective max similarity across both
    # representations. Diagnosed as fixing a real problem (mangled anyascii
    # transliteration defeating both phonetic and embedding blocking for
    # India's native-script records) but measured on the dry-run subsample
    # to be a net negative once its knock-on effect on hard-negative
    # sampling composition is included, without closing India's gap the way
    # it was expected to -- default False (disabled) pending a fix that
    # isolates the intended effect without that side effect. See
    # blocking._EmbedIndex.
    use_native_embedding: bool = os.environ.get("BER_USE_NATIVE_EMBED", "0").lower() in ("1", "true", "yes")
    k_ann: int = int(os.environ.get("BER_K_ANN", "20"))
    k_ann_reverse: int = int(os.environ.get("BER_K_ANN_REVERSE", "3"))
    k_phonetic: int = int(os.environ.get("BER_K_PHONETIC", "10"))
    max_phonetic_bucket: int = int(os.environ.get("BER_MAX_PHONETIC_BUCKET", "500"))
    k_housenum: int = int(os.environ.get("BER_K_HOUSENUM", "10"))
    max_housenum_bucket: int = int(os.environ.get("BER_MAX_HOUSENUM_BUCKET", "500"))
    sn_window: int = int(os.environ.get("BER_SN_WINDOW", "5"))
    k_final: int = int(os.environ.get("BER_K_FINAL", "0"))  # 0 = auto-pick from recall sweep
    hard_neg_per_s1: int = int(os.environ.get("BER_HARD_NEG", "12"))
    val_frac: float = float(os.environ.get("BER_VAL_FRAC", "0.15"))
    calib_frac: float = float(os.environ.get("BER_CALIB_FRAC", "0.05"))
    train_s1_sample: int | None = (
        int(os.environ["BER_TRAIN_S1_SAMPLE"])
        if os.environ.get("BER_TRAIN_S1_SAMPLE")
        else 600_000
    )
    allow_long: bool = os.environ.get("BER_ALLOW_LONG", "0").lower() in ("1", "true", "yes")
    long_stage_minutes: float = 15.0

    def paths(self, split: str) -> "SplitPaths":
        return SplitPaths(self, split)

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass
class SplitPaths:
    cfg: Config
    split: str  # "train" | "test"

    @property
    def dir(self) -> Path:
        d = self.cfg.data_dir / self.split
        return d

    def raw(self, source: str) -> Path:
        return self.dir / f"{self.split}_{source}.tsv"

    def ground_truth(self) -> Path:
        return self.dir / f"{self.split}_ground_truth.tsv"

    @property
    def cache(self) -> Path:
        d = self.cfg.cache_dir / self.split
        d.mkdir(parents=True, exist_ok=True)
        return d


def get_config() -> Config:
    cfg = Config()
    cfg.ensure_dirs()
    return cfg
