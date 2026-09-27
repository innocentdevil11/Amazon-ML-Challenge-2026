"""Candidate generation (blocking): the stage that determines the recall
ceiling of the whole pipeline. Combines three independent blockers --
embedding ANN, phonetic bucketing, sorted neighborhood -- unions their
output per S1 entity, then re-ranks the union with TF-IDF cosine and keeps
the top ``k_final``.

All three blockers, and the re-rank, are scoped **within country** (matches
never cross countries in the ground truth, verified in
:func:`check_no_cross_country_links`), which keeps every per-country pool
small enough for exact TF-IDF re-rank and for an exact-enough ANN index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from . import ann as ann_mod
from . import embed as embed_mod
from . import tfidf as tfidf_mod
from .config import Config

CANDIDATE_COLUMNS = [
    "s1_entity_id", "cand_entity_id", "source", "country",
    "from_ann", "from_phonetic", "from_sn", "from_housenum",
    "ann_sim", "ann_rank", "tfidf_sim",
]
BLOCKER_FLAGS = ("from_ann", "from_phonetic", "from_sn", "from_housenum")


def check_no_cross_country_links(s1n: pd.DataFrame, s2n: pd.DataFrame, s3n: pd.DataFrame, gt: pd.DataFrame) -> float:
    """Sanity check used by the plan's recall report: fraction of ground-truth
    links whose S1 and matched-record countries disagree. Non-zero means the
    per-country scoping below is dropping real matches."""
    s1_country = s1n.set_index("entity_id")["country"]
    cand_country = pd.concat([s2n, s3n]).set_index("entity_id")["country"]
    rows = []
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if not ids:
            continue
        c1 = s1_country.get(s1_id)
        for cid in ids.split(","):
            c2 = cand_country.get(cid)
            rows.append(c1 == c2 if c2 is not None else True)
    return 1 - (sum(rows) / len(rows)) if rows else 0.0


class _EmbedIndex:
    """Maps each (source, local row) to one or two positions in the shared
    embedding array: always its transliterated ``embed_text``, plus, for
    records with native-script content, a *second* row embedding the
    original script (``embed_text_native``). ANN search treats these as
    independent query/candidate rows and the caller's max-aggregation over
    duplicate (s1, cand) pairs (see ``generate_candidates``) then naturally
    keeps whichever representation scored higher -- the "take the max
    similarity across both representations" this module's docstring
    promises, without needing a special two-vector search path in FAISS.
    """

    def __init__(self, s1n: pd.DataFrame, s2n: pd.DataFrame, s3n: pd.DataFrame, use_native: bool = True):
        self._offsets: dict[str, tuple[int, int, np.ndarray]] = {}  # source -> (ascii_offset, native_offset, native_mask)
        texts: list[str] = []
        ascii_blocks = {}
        for source, df in (("S1", s1n), ("S2", s2n), ("S3", s3n)):
            ascii_blocks[source] = df["embed_text"].tolist()
        native_blocks = {}
        for source, df in (("S1", s1n), ("S2", s2n), ("S3", s3n)):
            if use_native:
                native = df["embed_text_native"].fillna("")
                mask = (native.str.len() > 0).to_numpy()
                native_blocks[source] = (mask, native.to_numpy()[mask].tolist())
            else:
                native_blocks[source] = (np.zeros(len(df), dtype=bool), [])

        offset = 0
        ascii_offset = {}
        for source in ("S1", "S2", "S3"):
            ascii_offset[source] = offset
            texts.extend(ascii_blocks[source])
            offset += len(ascii_blocks[source])
        native_offset = {}
        for source in ("S1", "S2", "S3"):
            mask, native_texts = native_blocks[source]
            native_offset[source] = offset
            texts.extend(native_texts)
            offset += len(native_texts)
            self._offsets[source] = (ascii_offset[source], native_offset[source], mask)

        self.texts = texts

    def positions(self, source: str, local_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Global embedding-array positions and their owning entry, for
        every ascii row in ``local_idx`` plus every native row among them
        that has one. ``owner`` indexes into ``local_idx`` itself (0-based
        position within the caller's country-filtered subset), matching
        what callers use as ``s1_local``/``cand_local`` -- *not* the global
        dataframe row numbers ``local_idx`` holds.
        """
        ascii_off, native_off, native_mask = self._offsets[source]
        ascii_pos = ascii_off + local_idx
        owner = np.arange(len(local_idx))
        sub_mask = native_mask[local_idx]
        if sub_mask.any():
            # position of each True entry within the FULL native block for
            # this source == count of True entries before it in native_mask
            native_rank = np.cumsum(native_mask) - 1
            native_pos = native_off + native_rank[local_idx[sub_mask]]
            return np.concatenate([ascii_pos, native_pos]), np.concatenate([owner, owner[sub_mask]])
        return ascii_pos, owner


def _embed_all(s1n, s2n, s3n, cfg: Config, cache_dir: Path, tag: str) -> tuple[np.memmap, np.ndarray, _EmbedIndex]:
    embed_index = _EmbedIndex(s1n, s2n, s3n, use_native=cfg.use_native_embedding)
    # suffix keeps the two embedding modes from colliding in the resume cache
    cache_tag = f"{tag}_all" if cfg.use_native_embedding else f"{tag}_all_asciionly"
    unique_emb, inverse = embed_mod.compute_embeddings(embed_index.texts, cache_dir, cfg, tag=cache_tag)
    return unique_emb, inverse, embed_index


def _ann_block_country(
    unique_emb, inverse, embed_index: _EmbedIndex, s1_idx, s2_idx, s3_idx, cfg: Config
) -> list[pd.DataFrame]:
    frames = []
    s1_glob, s1_owner = embed_index.positions("S1", s1_idx)
    s1_vec = embed_mod.row_vectors(unique_emb, inverse, s1_glob)
    s1_index = ann_mod.build_index(s1_vec, use_gpu=cfg.use_gpu) if len(s1_glob) else None
    for source, cand_idx in (("S2", s2_idx), ("S3", s3_idx)):
        if len(cand_idx) == 0 or len(s1_glob) == 0:
            continue
        cand_glob, cand_owner = embed_index.positions(source, cand_idx)
        cand_vec = embed_mod.row_vectors(unique_emb, inverse, cand_glob)
        index = ann_mod.build_index(cand_vec, use_gpu=cfg.use_gpu)
        sims, ids = ann_mod.search(index, s1_vec, cfg.k_ann)
        rows_s1, ranks = np.nonzero(ids >= 0)
        cand_local = cand_owner[ids[rows_s1, ranks]]  # map hit row back through native/ascii duplicates to its owning local_idx
        frames.append(pd.DataFrame({
            "s1_local": s1_owner[rows_s1], "source": source, "cand_local": cand_local,
            "from_ann": True, "ann_sim": sims[rows_s1, ranks], "ann_rank": ranks,
        }))
        # reverse search: each candidate -> its nearest S1s, catches S1
        # entities with many matches that a per-S1 top-k forward search alone
        # could truncate away.
        rsims, rids = ann_mod.search(s1_index, cand_vec, cfg.k_ann_reverse)
        rows_cand, rranks = np.nonzero(rids >= 0)
        frames.append(pd.DataFrame({
            "s1_local": s1_owner[rids[rows_cand, rranks]], "source": source, "cand_local": cand_owner[rows_cand],
            "from_ann": True, "ann_sim": rsims[rows_cand, rranks], "ann_rank": 999,
        }))
    return frames


def _phonetic_block_country(s1_sub, cand_sub_by_source, cfg: Config) -> list[pd.DataFrame]:
    """Bucket by (state, metaphone-of-first-token), then keep the
    ``k_phonetic`` best RapidFuzz matches within each S1's bucket.

    A common metaphone code (generic first words like "medical" or
    "enterprises") can bucket many thousands of candidates; scoring an S1
    against every one of them would make phonetic blocking cost scale with
    corpus size *twice over* (more S1 rows to score AND a bigger bucket to
    score each one against). ``max_phonetic_bucket`` bounds the second factor
    by scoring against a capped, deterministic sample of an oversized bucket
    instead of the whole thing -- ANN and sorted-neighborhood cover the
    resulting recall gap on pathologically generic buckets.
    """
    frames = []
    rng = np.random.RandomState(0)
    s1_key = s1_sub["state_code"].fillna("") + "|" + s1_sub["metaphone_key"].fillna("")
    for source, cand_sub in cand_sub_by_source.items():
        if len(cand_sub) == 0:
            continue
        cand_key = cand_sub["state_code"].fillna("") + "|" + cand_sub["metaphone_key"].fillna("")
        buckets: dict[str, list[int]] = {}
        for i, k in enumerate(cand_key.values):
            buckets.setdefault(k, []).append(i)
        for k, positions in buckets.items():
            if len(positions) > cfg.max_phonetic_bucket:
                buckets[k] = list(rng.choice(positions, size=cfg.max_phonetic_bucket, replace=False))
        pairs_s1, pairs_cand = [], []
        for i, k in enumerate(s1_key.values):
            cand_positions = buckets.get(k)
            if not cand_positions:
                continue
            if len(cand_positions) > cfg.k_phonetic:
                names = cand_sub["name_core"].values[cand_positions]
                scored = process.extract(
                    s1_sub["name_core"].values[i], names, scorer=fuzz.token_set_ratio,
                    limit=cfg.k_phonetic,
                )
                chosen = [cand_positions[idx] for _, _, idx in scored]
            else:
                chosen = cand_positions
            pairs_s1.extend([i] * len(chosen))
            pairs_cand.extend(chosen)
        if pairs_s1:
            frames.append(pd.DataFrame({
                "s1_local": pairs_s1, "source": source, "cand_local": pairs_cand,
                "from_phonetic": True, "ann_sim": np.nan, "ann_rank": -1,
            }))
    return frames


def _housenum_block_country(s1_sub, cand_sub_by_source, cfg: Config) -> list[pd.DataFrame]:
    """Bucket by (state, house/building number) -- already scoped to one
    country by the caller -- then keep the ``k_housenum`` best RapidFuzz
    address matches within each S1's bucket. Records with no extracted house
    number are excluded from both sides -- bucketing on an empty key would
    otherwise create one enormous, meaningless bucket out of every
    addressless record.

    State is folded into the key for the same reason it is in phonetic
    blocking: a low house number ("1", "100") is extremely common on its own
    -- measured on this project's data, house-number-only bucketing was
    >25x slower than phonetic bucketing at the same sample size -- but
    combined with state it discriminates about as well as the phonetic key
    does. ``max_housenum_bucket`` remains as a hard cap for whatever
    (state, house-number) combinations are still oversized.
    """
    frames = []
    rng = np.random.RandomState(1)
    s1_state_num = s1_sub["state_code"].fillna("") + "|" + s1_sub["house_number"].fillna("")
    s1_has_num = s1_sub["house_number"].fillna("") != ""
    for source, cand_sub in cand_sub_by_source.items():
        if len(cand_sub) == 0 or not s1_has_num.any():
            continue
        cand_state_num = cand_sub["state_code"].fillna("") + "|" + cand_sub["house_number"].fillna("")
        cand_has_num = cand_sub["house_number"].fillna("") != ""
        if not cand_has_num.any():
            continue
        buckets: dict[str, list[int]] = {}
        for i, (k, has) in enumerate(zip(cand_state_num.values, cand_has_num.values)):
            if has:
                buckets.setdefault(k, []).append(i)
        for k, positions in buckets.items():
            if len(positions) > cfg.max_housenum_bucket:
                buckets[k] = list(rng.choice(positions, size=cfg.max_housenum_bucket, replace=False))
        pairs_s1, pairs_cand = [], []
        for i, (k, has) in enumerate(zip(s1_state_num.values, s1_has_num.values)):
            if not has:
                continue
            cand_positions = buckets.get(k)
            if not cand_positions:
                continue
            if len(cand_positions) > cfg.k_housenum:
                addrs = cand_sub["addr_norm"].values[cand_positions]
                scored = process.extract(
                    s1_sub["addr_norm"].values[i], addrs, scorer=fuzz.token_set_ratio,
                    limit=cfg.k_housenum,
                )
                chosen = [cand_positions[idx] for _, _, idx in scored]
            else:
                chosen = cand_positions
            pairs_s1.extend([i] * len(chosen))
            pairs_cand.extend(chosen)
        if pairs_s1:
            frames.append(pd.DataFrame({
                "s1_local": pairs_s1, "source": source, "cand_local": pairs_cand,
                "from_housenum": True, "ann_sim": np.nan, "ann_rank": -1,
            }))
    return frames


def _sn_block_country(s1_sub, cand_sub_by_source, cfg: Config, key_col: str) -> list[pd.DataFrame]:
    frames = []
    W = cfg.sn_window
    for source, cand_sub in cand_sub_by_source.items():
        if len(cand_sub) == 0:
            continue
        combined_keys = np.concatenate([s1_sub[key_col].values, cand_sub[key_col].values])
        n1 = len(s1_sub)
        is_s1 = np.concatenate([np.ones(n1, dtype=bool), np.zeros(len(cand_sub), dtype=bool)])
        local_idx = np.concatenate([np.arange(n1), np.arange(len(cand_sub))])
        order = np.argsort(combined_keys, kind="stable")
        sorted_is_s1 = is_s1[order]
        sorted_local = local_idx[order]
        n = len(order)
        pairs_s1, pairs_cand = [], []
        s1_positions = np.flatnonzero(sorted_is_s1)
        for p in s1_positions:
            lo, hi = max(0, p - W), min(n, p + W + 1)
            for q in range(lo, hi):
                if q != p and not sorted_is_s1[q]:
                    pairs_s1.append(sorted_local[p])
                    pairs_cand.append(sorted_local[q])
        if pairs_s1:
            frames.append(pd.DataFrame({
                "s1_local": pairs_s1, "source": source, "cand_local": pairs_cand,
                "from_sn": True, "ann_sim": np.nan, "ann_rank": -1,
            }))
    return frames


def generate_candidates(
    s1n: pd.DataFrame, s2n: pd.DataFrame, s3n: pd.DataFrame,
    cfg: Config, cache_dir: Path, tag: str,
) -> pd.DataFrame:
    """Full blocking pipeline. Returns a long table, one row per (S1, candidate)
    pair *before* k_final truncation, with a ``tfidf_sim`` re-rank score and
    per-blocker provenance flags -- callers truncate with
    :func:`truncate_to_k`."""
    unique_emb, inverse, embed_index = _embed_all(s1n, s2n, s3n, cfg, cache_dir, tag)
    s1n = s1n.reset_index(drop=True)
    s2n = s2n.reset_index(drop=True)
    s3n = s3n.reset_index(drop=True)

    all_frames = []
    countries = s1n["country"].dropna().unique()
    for country in countries:
        s1_idx = np.flatnonzero(s1n["country"].values == country)
        s2_idx = np.flatnonzero(s2n["country"].values == country)
        s3_idx = np.flatnonzero(s3n["country"].values == country)
        if len(s1_idx) == 0:
            continue
        s1_sub = s1n.iloc[s1_idx].reset_index(drop=True)
        s2_sub = s2n.iloc[s2_idx].reset_index(drop=True)
        s3_sub = s3n.iloc[s3_idx].reset_index(drop=True)
        cand_sub_by_source = {"S2": s2_sub, "S3": s3_sub}

        country_frames = []
        country_frames += _ann_block_country(
            unique_emb, inverse, embed_index, s1_idx, s2_idx, s3_idx, cfg
        )
        country_frames += _phonetic_block_country(s1_sub, cand_sub_by_source, cfg)
        country_frames += _housenum_block_country(s1_sub, cand_sub_by_source, cfg)
        country_frames += _sn_block_country(s1_sub, cand_sub_by_source, cfg, "sn_key_a")
        country_frames += _sn_block_country(s1_sub, cand_sub_by_source, cfg, "sn_key_b")
        if not country_frames:
            continue
        union = pd.concat(country_frames, ignore_index=True)
        for flag in BLOCKER_FLAGS:
            if flag not in union.columns:
                union[flag] = False
            union[flag] = union[flag].fillna(False).astype(bool)
        agg = union.groupby(["s1_local", "source", "cand_local"], as_index=False).agg(
            from_ann=("from_ann", "max"),
            from_phonetic=("from_phonetic", "max"),
            from_sn=("from_sn", "max"),
            from_housenum=("from_housenum", "max"),
            ann_sim=("ann_sim", "max"),
            ann_rank=("ann_rank", "min"),
        )
        agg["s1_entity_id"] = s1_sub["entity_id"].values[agg["s1_local"].values]
        cand_entity = np.empty(len(agg), dtype=object)
        mask_s2 = agg["source"].values == "S2"
        if mask_s2.any():
            cand_entity[mask_s2] = s2_sub["entity_id"].values[agg["cand_local"].values[mask_s2]]
        if (~mask_s2).any():
            cand_entity[~mask_s2] = s3_sub["entity_id"].values[agg["cand_local"].values[~mask_s2]]
        agg["cand_entity_id"] = cand_entity
        agg["country"] = country
        all_frames.append(agg[[
            "s1_entity_id", "cand_entity_id", "source", "country",
            "from_ann", "from_phonetic", "from_sn", "from_housenum", "ann_sim", "ann_rank",
        ]])
        print(f"  [block] {country}: s1={len(s1_sub)} union_pairs={len(agg)}", flush=True)

    if not all_frames:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    candidates = pd.concat(all_frames, ignore_index=True)
    return candidates


def add_tfidf_rerank(
    candidates: pd.DataFrame, s1n: pd.DataFrame, s2n: pd.DataFrame, s3n: pd.DataFrame,
    vectorizer, idf, n_jobs: int = 1,
) -> pd.DataFrame:
    """Attach a row-aligned TF-IDF cosine (``tfidf_sim``) to every candidate
    pair, computed on ``name_core + ' ' + addr_core``."""
    text_by_id = pd.concat([
        s1n.set_index("entity_id")["embed_text"],
        s2n.set_index("entity_id")["embed_text"],
        s3n.set_index("entity_id")["embed_text"],
    ])
    a_texts = text_by_id.reindex(candidates["s1_entity_id"]).fillna("").tolist()
    b_texts = text_by_id.reindex(candidates["cand_entity_id"]).fillna("").tolist()
    candidates = candidates.copy()
    candidates["tfidf_sim"] = tfidf_mod.pair_cosine_chunked(vectorizer, idf, a_texts, b_texts, n_jobs=n_jobs)
    return candidates


def add_competition_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate and per-S1 rank/competition features (Section 2 of the
    plan): how many S1s want this same S2/S3 record, and where this pair
    ranks among that record's suitors and among this S1's own candidates."""
    c = candidates.copy()
    c["score_for_rank"] = c["tfidf_sim"].fillna(0) + c["ann_sim"].fillna(0)
    g_cand = c.groupby(["source", "cand_entity_id"])["score_for_rank"]
    c["n_competitors"] = g_cand.transform("size")
    c["rank_for_cand"] = g_cand.rank(ascending=False, method="first")
    c["best_for_cand"] = g_cand.transform("max")
    c["gap_to_best_for_cand"] = c["best_for_cand"] - c["score_for_rank"]

    g_s1 = c.groupby("s1_entity_id")["score_for_rank"]
    c["rank_for_s1"] = g_s1.rank(ascending=False, method="first")
    c["best_for_s1"] = g_s1.transform("max")
    c["gap_to_best_for_s1"] = c["best_for_s1"] - c["score_for_rank"]
    return c.drop(columns=["score_for_rank", "best_for_cand", "best_for_s1"])


def truncate_to_k(candidates: pd.DataFrame, k_final: int) -> pd.DataFrame:
    if k_final <= 0:
        return candidates
    ranked = candidates.copy()
    ranked["_rank"] = ranked.groupby("s1_entity_id")["tfidf_sim"].rank(ascending=False, method="first")
    return ranked[ranked["_rank"] <= k_final].drop(columns=["_rank"])


def recall_sweep(
    candidates: pd.DataFrame, gt: pd.DataFrame, val_s1_ids: set,
    ks=(5, 10, 15, 20, 25, 30, 40, 50),
) -> pd.DataFrame:
    """For each k, report pair recall and the resulting F0.5 *ceiling*
    (a perfect classifier applied to the top-k candidates) on ``val_s1_ids``.
    """
    from .metrics import f05_macro

    gt_map = {
        row.source1_entity_id: set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
        for row in gt.itertuples() if row.source1_entity_id in val_s1_ids
    }
    val_cand = candidates[candidates["s1_entity_id"].isin(val_s1_ids)].copy()
    val_cand["_rank"] = val_cand.groupby("s1_entity_id")["tfidf_sim"].rank(ascending=False, method="first")

    rows = []
    full_pairs = set(zip(val_cand["s1_entity_id"], val_cand["cand_entity_id"]))
    total_true = sum(len(v) for v in gt_map.values())

    def _ceiling_at(sub: pd.DataFrame) -> dict:
        pred_map = sub.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()
        tp = sum(len(pred_map.get(s1, set()) & gt_map.get(s1, set())) for s1 in gt_map)
        recall = tp / total_true if total_true else 1.0
        preds_for_score = {s1: pred_map.get(s1, set()) & gt_map.get(s1, set()) for s1 in gt_map}
        f05 = f05_macro(preds_for_score, gt_map)
        return dict(recall=recall, ceiling_f05=f05, mean_candidates=sub.groupby("s1_entity_id").size().mean() if len(sub) else 0.0, total_pairs=len(sub))

    for k in ks:
        sub = val_cand[val_cand["_rank"] <= k]
        rows.append({"k": k, **_ceiling_at(sub)})
    rows.append({"k": "full_union", **_ceiling_at(val_cand)})

    for flag in BLOCKER_FLAGS:
        sub = val_cand[val_cand[flag]]
        r = _ceiling_at(sub)
        rows.append({"k": f"{flag[5:]}_only", **r})

    return pd.DataFrame(rows)


def candidate_count_stats(candidates: pd.DataFrame, all_s1_ids) -> dict:
    """Per-S1 candidate-count distribution (mean/median/P95/max), including
    S1 entities with *zero* candidates -- otherwise the distribution looks
    healthier than it is by silently excluding blocking's worst failures."""
    counts = candidates.groupby("s1_entity_id").size()
    counts = counts.reindex(list(all_s1_ids), fill_value=0)
    return {
        "n_s1": int(len(counts)),
        "n_s1_zero_candidates": int((counts == 0).sum()),
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "p95": float(counts.quantile(0.95)),
        "max": int(counts.max()) if len(counts) else 0,
    }
