"""Sentence-embedding stage: multilingual MiniLM embeddings for blocking and
for the "embedding cosine similarity" feature.

Resumable by design (the embedding stage is the one that can run for hours):
texts are deduplicated, sorted by length for efficient batching, and encoded
in fixed-size chunks written to disk as soon as each chunk finishes. Re-running
skips any chunk whose ``.done`` marker already exists, so a killed/interrupted
run picks back up instead of starting over.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import Config

_MODEL_CACHE: dict[str, object] = {}


def get_device(cfg: Config) -> str:
    return "cuda" if cfg.use_gpu else "cpu"


def _load_model(cfg: Config):
    key = f"{cfg.embed_model}:{get_device(cfg)}"
    if key not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg.embed_model, device=get_device(cfg))
        model.max_seq_length = cfg.embed_max_seq_len
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def embedding_dim(cfg: Config) -> int:
    model = _load_model(cfg)
    return model.get_sentence_embedding_dimension()


def benchmark_embedding_throughput(cfg: Config, sample_texts: list[str]) -> dict:
    """Encode a small sample and return texts/sec, used to project total wall
    time for the full corpus before committing to a long run."""
    model = _load_model(cfg)
    # warmup (first batch pays for CUDA kernel compilation / graph capture)
    _ = model.encode(sample_texts[: min(32, len(sample_texts))], batch_size=cfg.embed_batch_size, show_progress_bar=False)
    t0 = time.time()
    model.encode(sample_texts, batch_size=cfg.embed_batch_size, show_progress_bar=False, convert_to_numpy=True)
    elapsed = time.time() - t0
    rate = len(sample_texts) / max(elapsed, 1e-6)
    return {"device": get_device(cfg), "n_sample": len(sample_texts), "seconds": elapsed, "texts_per_sec": rate}


def _chunk_path(cache_dir: Path, tag: str, chunk_id: int) -> Path:
    return cache_dir / f"emb_{tag}_chunk{chunk_id:04d}.npy"


def _meta_path(cache_dir: Path, tag: str) -> Path:
    return cache_dir / f"emb_{tag}_meta.json"


def _unique_emb_path(cache_dir: Path, tag: str) -> Path:
    return cache_dir / f"emb_{tag}_unique.fp16.dat"


def _inverse_path(cache_dir: Path, tag: str) -> Path:
    return cache_dir / f"emb_{tag}_inverse.npy"


def compute_embeddings(
    texts: list[str],
    cache_dir: Path,
    cfg: Config,
    tag: str,
    chunk_size: int = 200_000,
    force: bool = False,
) -> tuple[np.memmap, np.ndarray]:
    """Return (unique_embeddings[n_unique, dim] fp16 memmap, inverse[len(texts)] int32)
    such that ``unique_embeddings[inverse[i]]`` is the embedding of ``texts[i]``.

    Resumable: safe to Ctrl-C and re-run with the same ``tag``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = _meta_path(cache_dir, tag)
    unique_path = _unique_emb_path(cache_dir, tag)
    inverse_path = _inverse_path(cache_dir, tag)

    if not force and meta_path.exists() and unique_path.exists() and inverse_path.exists():
        meta = json.loads(meta_path.read_text())
        n_unique, dim = meta["n_unique"], meta["dim"]
        emb = np.memmap(unique_path, dtype=np.float16, mode="r", shape=(n_unique, dim))
        inverse = np.load(inverse_path)
        if len(inverse) == len(texts):
            return emb, inverse

    # 1. Dedupe, preserving first-occurrence order for the public `inverse` map.
    uniq_texts, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    uniq_texts = uniq_texts.tolist()
    n_unique = len(uniq_texts)
    dim = embedding_dim(cfg)

    # 2. Sort unique texts by length for efficient batching (short sequences
    #    padded together), independent of the public index order.
    order = sorted(range(n_unique), key=lambda i: len(uniq_texts[i]))
    sorted_texts = [uniq_texts[i] for i in order]

    n_chunks = (n_unique + chunk_size - 1) // chunk_size
    model = _load_model(cfg)
    for c in range(n_chunks):
        out_path = _chunk_path(cache_dir, tag, c)
        done_marker = out_path.with_suffix(".done")
        if done_marker.exists() and not force:
            continue
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, n_unique)
        batch_texts = sorted_texts[lo:hi]
        vecs = model.encode(
            batch_texts,
            batch_size=cfg.embed_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float16)
        np.save(out_path, vecs)
        done_marker.write_text("ok")
        print(f"  [embed:{tag}] chunk {c + 1}/{n_chunks} done ({hi - lo} texts)", flush=True)

    # 3. Reassemble into unique_emb, in the *original* uniq_texts order (i.e.
    #    unsort), as a disk-backed memmap so we never hold 12M vectors in RAM.
    unique_emb = np.memmap(unique_path, dtype=np.float16, mode="w+", shape=(n_unique, dim))
    for c in range(n_chunks):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, n_unique)
        vecs = np.load(_chunk_path(cache_dir, tag, c))
        dest_idx = order[lo:hi]
        unique_emb[dest_idx] = vecs
    unique_emb.flush()

    np.save(inverse_path, inverse.astype(np.int32))
    meta_path.write_text(json.dumps({"n_unique": n_unique, "dim": dim}))

    # cleanup per-chunk files now that they're merged
    for c in range(n_chunks):
        _chunk_path(cache_dir, tag, c).unlink(missing_ok=True)
        _chunk_path(cache_dir, tag, c).with_suffix(".done").unlink(missing_ok=True)

    emb_ro = np.memmap(unique_path, dtype=np.float16, mode="r", shape=(n_unique, dim))
    return emb_ro, inverse.astype(np.int32)


def load_embeddings(cache_dir: Path, tag: str) -> tuple[np.memmap, np.ndarray]:
    cache_dir = Path(cache_dir)
    meta = json.loads(_meta_path(cache_dir, tag).read_text())
    emb = np.memmap(_unique_emb_path(cache_dir, tag), dtype=np.float16, mode="r", shape=(meta["n_unique"], meta["dim"]))
    inverse = np.load(_inverse_path(cache_dir, tag))
    return emb, inverse


def row_vectors(unique_emb: np.memmap, inverse: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    """Gather full-precision (float32) vectors for a subset of original rows."""
    return np.asarray(unique_emb[inverse[row_indices]], dtype=np.float32)
