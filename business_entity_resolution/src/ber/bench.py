"""``--stage benchmark``: measures real throughput on a small slice of the
actual data for every expensive stage and extrapolates to the full split, so
a long run is never started blind. Every number here is a projection, not a
promise -- it exists to catch "this will take 6 hours on this laptop" before
those 6 hours are spent, not to be precise to the minute.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import embed as embed_mod
from . import normalize as normalize_mod
from . import tfidf as tfidf_mod
from .config import Config
from .io import read_tsv

SAMPLE_S1 = 20_000
ASSUMED_CANDIDATES_PER_S1 = 120  # rough union size before k_final truncation; see README
STAGE_ORDER = ("normalize", "embed", "ann_build_search", "phonetic_sn_block", "tfidf_rerank", "featurize", "train_xgb")


def _row_counts(paths) -> dict[str, int]:
    counts = {}
    for source in ("source1", "source2", "source3"):
        p = paths.raw(source)
        with open(p, encoding="utf-8") as f:
            counts[source] = sum(1 for _ in f) - 1  # minus header
    return counts


def _proportional_sample(paths, source: str, frac: float, min_rows: int = 2000) -> pd.DataFrame:
    total = None
    df = read_tsv(paths.raw(source), nrows=max(min_rows, int(1 / max(frac, 1e-9))))
    return df


def run_benchmark(paths, cfg: Config) -> dict:
    print("Benchmarking on a data slice (this itself should take under a minute)...")
    counts = _row_counts(paths)
    total_records = sum(counts.values())
    ratio = min(1.0, SAMPLE_S1 / max(counts["source1"], 1))

    s1_sample = read_tsv(paths.raw("source1"), nrows=SAMPLE_S1)
    s2_sample = read_tsv(paths.raw("source2"), nrows=max(2000, int(counts["source2"] * ratio)))
    s3_sample = read_tsv(paths.raw("source3"), nrows=max(2000, int(counts["source3"] * ratio)))

    results = {}

    # --- normalize ---
    # chunk_size forced small so this exercises the same multi-process path
    # the full run uses (its default 20k chunks wouldn't split a 20k-row
    # sample at all, making the benchmark measure single-core performance).
    norm_chunk = max(500, SAMPLE_S1 // cfg.n_jobs)
    t0 = time.time()
    s1n = normalize_mod.normalize_dataframe(s1_sample, n_jobs=cfg.n_jobs, chunk_size=norm_chunk)
    s2n = normalize_mod.normalize_dataframe(s2_sample, n_jobs=cfg.n_jobs, chunk_size=norm_chunk)
    s3n = normalize_mod.normalize_dataframe(s3_sample, n_jobs=cfg.n_jobs, chunk_size=norm_chunk)
    elapsed = time.time() - t0
    sample_records = len(s1_sample) + len(s2_sample) + len(s3_sample)
    results["normalize"] = elapsed * (total_records / sample_records) / 60

    # --- embedding throughput (real model, real text) ---
    sample_texts = pd.concat([s1n["embed_text"], s2n["embed_text"], s3n["embed_text"]]).tolist()
    bench = embed_mod.benchmark_embedding_throughput(cfg, sample_texts[: min(2000, len(sample_texts))])
    results["embed_device"] = bench["device"]
    results["embed_texts_per_sec"] = bench["texts_per_sec"]
    results["embed"] = (total_records / max(bench["texts_per_sec"], 1e-6)) / 60

    # --- ANN build+search on synthetic vectors sized like the real corpus ---
    # Build (k-means training) and search have very different cost drivers --
    # build scales with the *training sample* (capped, ~independent of total
    # corpus size once large enough; see ann.build_index's docstring) while
    # search scales with the number of queries issued -- so they're timed and
    # extrapolated separately rather than folded into one multiplier.
    dim = embed_mod.embedding_dim(cfg)
    rng = np.random.RandomState(0)
    from . import ann as ann_mod

    bench_corpus_n = 200_000  # kept small and fixed so the benchmark itself stays fast; extrapolated below
    synth = rng.randn(bench_corpus_n, dim).astype("float32")
    synth /= np.linalg.norm(synth, axis=1, keepdims=True)
    t0 = time.time()
    index = ann_mod.build_index(synth, use_gpu=cfg.use_gpu)
    build_elapsed = time.time() - t0
    n_query_sample = min(5000, bench_corpus_n)
    queries = synth[:n_query_sample]
    t0 = time.time()
    ann_mod.search(index, queries, cfg.k_ann)
    search_elapsed = time.time() - t0
    print(
        f"  [bench] ann raw: build={build_elapsed:.2f}s search={search_elapsed:.2f}s "
        f"corpus={bench_corpus_n} queries={n_query_sample}", flush=True,
    )

    # ~3 index builds per country (S2 pool, S3 pool, S1 pool for reverse
    # search), covering every S1/S2/S3 record once each in total.
    total_indexed = counts["source1"] + counts["source2"] + counts["source3"]
    build_time_full = build_elapsed * (total_indexed / bench_corpus_n)
    # forward search: every S1 queries both the S2 and S3 index (k_ann each);
    # reverse search: every S2/S3 record queries the S1 index (k_ann_reverse).
    total_queries = 2 * counts["source1"] + counts["source2"] + counts["source3"]
    search_time_full = search_elapsed * (total_queries / n_query_sample)
    results["ann_build_search"] = (build_time_full + search_time_full) / 60

    # --- phonetic + sorted-neighborhood blocking (real code path, real bucket sizes) ---
    from . import blocking as blocking_mod

    cand_sub_by_source = {"S2": s2n, "S3": s3n}
    t0 = time.time()
    blocking_mod._phonetic_block_country(s1n, cand_sub_by_source, cfg)
    phonetic_elapsed = time.time() - t0
    t0 = time.time()
    blocking_mod._housenum_block_country(s1n, cand_sub_by_source, cfg)
    housenum_elapsed = time.time() - t0
    t0 = time.time()
    blocking_mod._sn_block_country(s1n, cand_sub_by_source, cfg, "sn_key_a")
    blocking_mod._sn_block_country(s1n, cand_sub_by_source, cfg, "sn_key_b")
    sn_elapsed = time.time() - t0
    print(f"  [bench] phonetic raw: {phonetic_elapsed:.2f}s  housenum raw: {housenum_elapsed:.2f}s  "
          f"sorted-neighborhood raw: {sn_elapsed:.2f}s (on {len(s1n)} S1 rows, mixed countries)", flush=True)
    # The bucket caps (max_phonetic_bucket / max_housenum_bucket) make both
    # blockers' cost scale with the number of S1 rows processed, not with
    # corpus size, so this extrapolates linearly on S1 count;
    # sorted-neighborhood is similarly linear (a fixed window per record).
    results["phonetic_sn_block"] = (phonetic_elapsed + housenum_elapsed + sn_elapsed) * (counts["source1"] / len(s1n)) / 60

    # --- TF-IDF re-rank throughput (parallelized, matching production) ---
    vec = tfidf_mod.build_hashing_vectorizer()
    idf = tfidf_mod.fit_idf(vec, sample_texts)
    # Sized large enough (300k, not 50k) that process-pool startup overhead
    # -- a fixed cost paid once regardless of workload -- doesn't dominate
    # the measurement and understate real per-core throughput.
    n_pairs_sample = min(300_000, len(sample_texts) * 2)
    a_texts = (sample_texts * ((n_pairs_sample // max(len(sample_texts), 1)) + 1))[:n_pairs_sample]
    b_texts = a_texts[::-1]
    tfidf_chunk_size = max(2000, n_pairs_sample // cfg.n_jobs)
    t0 = time.time()
    tfidf_mod.pair_cosine_chunked(vec, idf, a_texts, b_texts, n_jobs=cfg.n_jobs, chunk_size=tfidf_chunk_size)
    elapsed = time.time() - t0
    total_pairs_est = counts["source1"] * ASSUMED_CANDIDATES_PER_S1
    print(f"  [bench] tfidf raw: elapsed={elapsed:.2f}s over n_pairs_sample={n_pairs_sample} (n_jobs={cfg.n_jobs})", flush=True)
    results["tfidf_rerank"] = elapsed * (total_pairs_est / n_pairs_sample) / 60

    # --- featurize (rapidfuzz) throughput ---
    from .features import _chunked_pair_scores

    names_a = (s1n["name_core"].tolist() * ((n_pairs_sample // max(len(s1n), 1)) + 1))[:n_pairs_sample]
    names_b = names_a[::-1]
    featurize_chunk_size = max(2000, n_pairs_sample // cfg.n_jobs)
    t0 = time.time()
    _chunked_pair_scores(names_a, names_b, cfg.n_jobs, chunk_size=featurize_chunk_size)
    elapsed = time.time() - t0
    results["featurize"] = elapsed * (total_pairs_est / n_pairs_sample) / 60 * 2  # ~2 similarity passes (name+addr)

    # --- XGBoost training throughput ---
    import xgboost as xgb

    n_feat = 30
    X = rng.randn(20_000, n_feat).astype("float32")
    y = (rng.rand(20_000) < 0.15).astype(int)
    t0 = time.time()
    clf = xgb.XGBClassifier(
        tree_method="hist", device="cuda" if cfg.use_gpu else "cpu", n_estimators=400, max_depth=6,
    )
    clf.fit(X, y)
    elapsed = time.time() - t0
    est_train_rows = min(cfg.train_s1_sample or counts["source1"], counts["source1"]) * (1 + cfg.hard_neg_per_s1)
    results["train_xgb"] = elapsed * max(1.0, est_train_rows / 20_000) / 60

    total_minutes = sum(v for k, v in results.items() if isinstance(v, (int, float)) and k not in ("embed_texts_per_sec",))
    results["TOTAL_ESTIMATE_MIN"] = total_minutes
    results["row_counts"] = counts
    return results


def print_benchmark_table(results: dict, cfg: Config) -> None:
    print("\n=== BENCHMARK: projected wall time for the FULL split ===")
    print(f"  device: embed={results['embed_device']}  gpu_configured={cfg.use_gpu}")
    print(f"  row counts: {results['row_counts']}")
    for stage in STAGE_ORDER:
        minutes = results[stage]
        flag = "  <-- LONG (>15min)" if minutes > cfg.long_stage_minutes else ""
        print(f"  {stage:20s}: {minutes:8.1f} min{flag}")
    print(f"  {'TOTAL (rough)':20s}: {results['TOTAL_ESTIMATE_MIN']:8.1f} min")
    print(
        "  NOTE: these are projections from a small real-data slice (normalize/embed/phonetic+SN/tfidf/"
        "featurize) plus synthetic-vector timings (ANN) and synthetic-feature timings (XGBoost); treat as "
        "an order-of-magnitude guide, not an exact ETA."
    )
    long_stages = [s for s in STAGE_ORDER if results[s] > cfg.long_stage_minutes]
    if long_stages and not cfg.allow_long:
        print(
            f"\n  Stage(s) projected over {cfg.long_stage_minutes:.0f} min: {long_stages}. "
            "Re-run with --allow-long (or BER_ALLOW_LONG=1) once you've reviewed this table."
        )
