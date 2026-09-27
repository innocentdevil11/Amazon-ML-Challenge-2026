"""Deterministic, hash-based split of S1 entities into val / calib / gbm-train
pools. Hashing (not ``random.sample``) makes the split reproducible across
runs and processes without needing to persist an index, and is stable to
``PYTHONHASHSEED`` (unlike the builtin ``hash()``).

Validation-split ground truth is never touched by training, calibration, or
IDF fitting -- every consumer of the split takes an explicit ``pool`` name.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .config import Config


def _stable_frac(entity_id: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{entity_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_pools(s1_ids: list[str], cfg: Config) -> dict[str, str]:
    """Return {entity_id: "val" | "calib" | "train" | "train_unused"}.

    "train" is the (possibly subsampled) pool the GBM actually trains on;
    "train_unused" are entities in the remaining 80% not selected by
    ``train_s1_sample`` (kept out of everything, simplest way to bound
    training-set size without touching val/calib).
    """
    fracs = {sid: _stable_frac(sid, cfg.seed) for sid in s1_ids}
    pools: dict[str, str] = {}
    train_pool = []
    for sid, f in fracs.items():
        if f < cfg.val_frac:
            pools[sid] = "val"
        elif f < cfg.val_frac + cfg.calib_frac:
            pools[sid] = "calib"
        else:
            train_pool.append(sid)

    if cfg.train_s1_sample is not None and len(train_pool) > cfg.train_s1_sample:
        train_pool_sorted = sorted(train_pool, key=lambda sid: fracs[sid])
        selected = set(train_pool_sorted[: cfg.train_s1_sample])
    else:
        selected = set(train_pool)

    for sid in train_pool:
        pools[sid] = "train" if sid in selected else "train_unused"
    return pools
