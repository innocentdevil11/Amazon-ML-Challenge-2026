"""Pairwise feature engineering for (S1, candidate) pairs.

Consumes the candidate table produced by :mod:`ber.blocking` (which already
carries ``ann_sim``, ``tfidf_sim``, per-blocker flags and the competition
features) and joins in the normalized name/address fields from
:mod:`ber.normalize` to compute the remaining string-similarity and
structured-field features the matching model trains on.

String-similarity metrics (Levenshtein ratio, Jaro-Winkler, token set/sort)
are computed pairwise with RapidFuzz (C implementation) in worker
processes -- this, not blocking or embedding, is the stage where a plain
Python loop over tens of millions of pairs would dominate runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import partial_ratio, ratio, token_set_ratio, token_sort_ratio

RECORD_FIELDS = [
    "name_norm", "name_core", "name_alt", "legal_forms", "is_domain", "translit",
    "addr_norm", "addr_core", "pincode", "house_number", "numeric_tokens",
    "state_code", "landmark", "country", "business_name", "business_address",
]


def _lookup_table(s1n: pd.DataFrame, s2n: pd.DataFrame, s3n: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([s1n, s2n, s3n], ignore_index=True).set_index("entity_id")[RECORD_FIELDS]


def _jaccard_str(a: str, b: str) -> float:
    sa = a.split() if a else []
    sb = b.split() if b else []
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    sa, sb = set(sa), set(sb)
    return len(sa & sb) / len(sa | sb)


def _row_jaccard(a_list: list, b_list: list) -> np.ndarray:
    out = np.empty(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        out[i] = _jaccard_str(a or "", b or "")
    return out


def _chunked_jaccard(a_list: list, b_list: list, n_jobs: int, chunk_size: int = 200_000) -> np.ndarray:
    """Token-set Jaccard, computed straight from the two strings inside one
    tight loop instead of pandas ``.apply(frozenset)`` into a persistent
    Series first: building and keeping a full-length Series of Python
    ``frozenset`` objects for every jaccard-based feature at once (name
    tokens, address tokens, numeric tokens, landmark tokens -- 4+ such
    Series alive simultaneously) is what pushed memory into swapping on a
    multi-million-row candidate table. Here each row's two sets exist only
    for that iteration and are immediately collectible, and the loop itself
    is chunked across processes the same way RapidFuzz scoring already is.
    """
    n = len(a_list)
    if n_jobs <= 1 or n < chunk_size * 2:
        return _row_jaccard(a_list, b_list)
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_row_jaccard)(a_list[i : i + chunk_size], b_list[i : i + chunk_size])
        for i in range(0, n, chunk_size)
    )
    return np.concatenate(results)


def _row_legal_flags(a_list: list, b_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(a_list)
    match = np.zeros(n, dtype=np.int8)
    conflict = np.zeros(n, dtype=np.int8)
    missing = np.zeros(n, dtype=np.int8)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        sa = set(a.split()) if a else set()
        sb = set(b.split()) if b else set()
        if not sa or not sb:
            missing[i] = 1
        elif sa == sb:
            match[i] = 1
        else:
            conflict[i] = 1
    return match, conflict, missing


def _chunked_legal_flags(a_list: list, b_list: list, n_jobs: int, chunk_size: int = 200_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(a_list)
    if n_jobs <= 1 or n < chunk_size * 2:
        return _row_legal_flags(a_list, b_list)
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_row_legal_flags)(a_list[i : i + chunk_size], b_list[i : i + chunk_size])
        for i in range(0, n, chunk_size)
    )
    return (
        np.concatenate([r[0] for r in results]),
        np.concatenate([r[1] for r in results]),
        np.concatenate([r[2] for r in results]),
    )


def _row_pair_scores(names_a: list, names_b: list) -> dict[str, np.ndarray]:
    n = len(names_a)
    lev = np.empty(n, dtype=np.float32)
    jw = np.empty(n, dtype=np.float32)
    tset = np.empty(n, dtype=np.float32)
    tsort = np.empty(n, dtype=np.float32)
    part = np.empty(n, dtype=np.float32)
    for i, (a, b) in enumerate(zip(names_a, names_b)):
        a = a or ""
        b = b or ""
        lev[i] = ratio(a, b)
        jw[i] = JaroWinkler.similarity(a, b) * 100
        tset[i] = token_set_ratio(a, b)
        tsort[i] = token_sort_ratio(a, b)
        part[i] = partial_ratio(a, b)
    return {"lev": lev / 100, "jw": jw / 100, "tset": tset / 100, "tsort": tsort / 100, "part": part / 100}


def _chunked_pair_scores(names_a: list, names_b: list, n_jobs: int, chunk_size: int = 100_000) -> dict[str, np.ndarray]:
    n = len(names_a)
    if n_jobs <= 1 or n < chunk_size * 2:
        return _row_pair_scores(names_a, names_b)
    from joblib import Parallel, delayed

    idx_chunks = range(0, n, chunk_size)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_row_pair_scores)(names_a[i : i + chunk_size], names_b[i : i + chunk_size])
        for i in idx_chunks
    )
    out = {}
    for key in results[0]:
        out[key] = np.concatenate([r[key] for r in results])
    return out


def compute_features(
    candidates: pd.DataFrame, s1n: pd.DataFrame, s2n: pd.DataFrame, s3n: pd.DataFrame, n_jobs: int = 1,
) -> pd.DataFrame:
    lookup = _lookup_table(s1n, s2n, s3n)
    a = lookup.reindex(candidates["s1_entity_id"]).reset_index(drop=True)
    b = lookup.reindex(candidates["cand_entity_id"]).reset_index(drop=True)
    out = candidates.reset_index(drop=True).copy()

    # --- name features -----------------------------------------------------
    scores = _chunked_pair_scores(a["name_core"].tolist(), b["name_core"].tolist(), n_jobs)
    out["name_lev"] = scores["lev"]
    out["name_jarowinkler"] = scores["jw"]
    out["name_token_set"] = scores["tset"]
    out["name_token_sort"] = scores["tsort"]
    out["name_partial"] = scores["part"]

    out["name_exact_core"] = (a["name_core"].values == b["name_core"].values).astype(np.int8)
    out["name_exact_norm"] = (a["name_norm"].values == b["name_norm"].values).astype(np.int8)
    a_first = a["name_core"].str.split().str[0].fillna("")
    b_first = b["name_core"].str.split().str[0].fillna("")
    out["name_first_token_match"] = (a_first == b_first).astype(np.int8)

    name_core_a, name_core_b = a["name_core"].fillna("").tolist(), b["name_core"].fillna("").tolist()
    out["name_token_jaccard"] = _chunked_jaccard(name_core_a, name_core_b, n_jobs)

    legal_match, legal_conflict, legal_missing = _chunked_legal_flags(
        a["legal_forms"].fillna("").tolist(), b["legal_forms"].fillna("").tolist(), n_jobs
    )
    out["legal_form_match"] = legal_match
    out["legal_form_conflict"] = legal_conflict
    out["legal_form_missing"] = legal_missing

    out["either_is_domain"] = (a["is_domain"].values | b["is_domain"].values).astype(np.int8)
    out["either_translit"] = (a["translit"].values | b["translit"].values).astype(np.int8)

    name_len_a = a["name_core"].str.len().replace(0, np.nan)
    name_len_b = b["name_core"].str.len().replace(0, np.nan)
    out["name_len_ratio"] = (
        np.minimum(name_len_a, name_len_b) / np.maximum(name_len_a, name_len_b)
    ).fillna(0.0)

    # optional alt-name (dba) boost: best of core-vs-core and alt-vs-core
    has_alt = (a["name_alt"].str.len() > 0) | (b["name_alt"].str.len() > 0)
    if has_alt.any():
        alt_scores = _chunked_pair_scores(
            (a["name_alt"].where(a["name_alt"].str.len() > 0, a["name_core"])).tolist(),
            (b["name_alt"].where(b["name_alt"].str.len() > 0, b["name_core"])).tolist(),
            n_jobs,
        )
        out["name_alt_token_set"] = np.maximum(out["name_token_set"].values, alt_scores["tset"])
    else:
        out["name_alt_token_set"] = out["name_token_set"]

    # --- address features ---------------------------------------------------
    addr_scores = _chunked_pair_scores(a["addr_norm"].tolist(), b["addr_norm"].tolist(), n_jobs)
    out["addr_lev"] = addr_scores["lev"]
    out["addr_token_set"] = addr_scores["tset"]

    out["addr_token_jaccard"] = _chunked_jaccard(
        a["addr_core"].fillna("").tolist(), b["addr_core"].fillna("").tolist(), n_jobs
    )
    # numeric_tokens is stored as a sorted, space-joined string (see
    # normalize.py), so it's already jaccard-ready without parsing back into
    # a set first.
    out["addr_numeric_jaccard"] = _chunked_jaccard(
        a["numeric_tokens"].fillna("").tolist(), b["numeric_tokens"].fillna("").tolist(), n_jobs
    )

    def _match_flag(sa, sb):
        va, vb = sa.values, sb.values
        out_arr = np.full(len(va), -1, dtype=np.int8)
        both = (va != "") & (vb != "")
        out_arr[both] = (va[both] == vb[both]).astype(np.int8)
        return out_arr

    out["house_number_match"] = _match_flag(a["house_number"], b["house_number"])
    out["pincode_match"] = _match_flag(a["pincode"], b["pincode"])
    out["state_match"] = _match_flag(a["state_code"], b["state_code"])

    landmark_present = (a["landmark"].str.len() > 0) | (b["landmark"].str.len() > 0)
    landmark_jaccard = _chunked_jaccard(a["landmark"].fillna("").tolist(), b["landmark"].fillna("").tolist(), n_jobs)
    out["landmark_overlap"] = np.where(landmark_present, landmark_jaccard, -1.0)

    out["addr_empty_a"] = (a["business_address"].fillna("").str.len() == 0).astype(np.int8)
    out["addr_empty_b"] = (b["business_address"].fillna("").str.len() == 0).astype(np.int8)

    # An interaction feature for "high name confidence, address absent" was
    # tried here (either_addr_missing flag + name_token_set gated on it) to
    # address a real diagnosed problem: a near-perfect name match crashing to
    # near-zero calibrated probability whenever one side's address is simply
    # empty. Measured on the dry-run subsample, it made things *worse* --
    # F0.5 0.9390 -> 0.9249 with both new columns; ablation showed dropping
    # either one alone (keeping the other) was worse still (0.9227 / 0.9217)
    # than dropping both (0.9358), pointing to a harmful interaction between
    # two highly-correlated columns rather than one bad feature -- and even
    # "both dropped" didn't fully recover the original score, most likely
    # because these two nearly-redundant columns and the reshuffled hard
    # negatives they subtly caused (candidate composition also shifted from
    # the native-embedding fix) change which splits a small ~5%-of-data GBM
    # settles on. Reverted rather than shipped, in favor of the option (b)
    # this feature's diagnosis called out but didn't try: fixing this via
    # hard-negative sampling instead of a new column, once there's a
    # properly-sized (full-scale) run to validate a fix like that against --
    # a change this order-sensitive isn't safe to validate on a 5% sample.

    addr_len_a = a["addr_core"].str.len().replace(0, np.nan)
    addr_len_b = b["addr_core"].str.len().replace(0, np.nan)
    out["addr_len_ratio"] = (
        np.minimum(addr_len_a, addr_len_b) / np.maximum(addr_len_a, addr_len_b)
    ).fillna(0.0)

    # --- other ---------------------------------------------------------------
    out["country_match"] = (a["country"].values == b["country"].values).astype(np.int8)
    out["source_is_s2"] = (out["source"].values == "S2").astype(np.int8)
    for col in ("from_ann", "from_phonetic", "from_sn", "from_housenum"):
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(np.int8)
    out["ann_sim"] = out.get("ann_sim", pd.Series(np.nan, index=out.index)).fillna(-1.0)
    out["ann_rank"] = out.get("ann_rank", pd.Series(-1, index=out.index)).fillna(-1)
    out["tfidf_sim"] = out.get("tfidf_sim", pd.Series(0.0, index=out.index)).fillna(0.0)

    return out


FEATURE_COLUMNS = [
    "name_lev", "name_jarowinkler", "name_token_set", "name_token_sort", "name_partial",
    "name_exact_core", "name_exact_norm", "name_first_token_match", "name_token_jaccard",
    "legal_form_match", "legal_form_conflict", "legal_form_missing",
    "either_is_domain", "either_translit", "name_len_ratio", "name_alt_token_set",
    "addr_lev", "addr_token_set", "addr_token_jaccard", "addr_numeric_jaccard",
    "house_number_match", "pincode_match", "state_match", "landmark_overlap",
    "addr_empty_a", "addr_empty_b", "addr_len_ratio",
    "country_match", "source_is_s2",
    "from_ann", "from_phonetic", "from_sn", "from_housenum", "ann_sim", "ann_rank", "tfidf_sim",
    "n_competitors", "rank_for_cand", "gap_to_best_for_cand",
    "rank_for_s1", "gap_to_best_for_s1",
]
