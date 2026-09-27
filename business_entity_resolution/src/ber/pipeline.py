"""Stage orchestration. Each stage reads its inputs from the previous
stage's cached parquet (under ``cache_dir/<split>/``) and writes its own, so
a run can be stopped and resumed stage-by-stage (this matters most for the
embedding stage, which can run for hours) -- pass ``force=True`` to redo a
stage whose cache is already there.

Model artifacts (the trained classifier, the isotonic calibrator, the
TF-IDF IDF weights, the chosen decision threshold) live under
``cache_dir/shared/`` because they are fit once on the *train* split and
then reused, unchanged, for the *test* split -- refitting any of them on
test data would leak test-time information into the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline as baseline_mod
from . import blocking, features as features_mod, model as model_mod, normalize as normalize_mod
from . import output as output_mod
from . import split as split_mod
from . import tfidf as tfidf_mod
from . import tune as tune_mod
from .config import Config
from .io import read_ground_truth, read_source, write_tsv
from .metrics import f05_macro


def _shared_dir(cfg: Config) -> Path:
    d = cfg.cache_dir / "shared"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Stage 1: normalize
# --------------------------------------------------------------------------
def stage_normalize(paths, cfg: Config, force: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for source in ("source1", "source2", "source3"):
        cache_file = paths.cache / f"{source}_norm.parquet"
        if cache_file.exists() and not force:
            out[source] = pd.read_parquet(cache_file)
            continue
        raw = read_source(paths, paths.split, source)
        normed = normalize_mod.normalize_dataframe(raw, n_jobs=cfg.n_jobs)
        normed.to_parquet(cache_file, index=False)
        out[source] = normed
        print(f"  [normalize] {paths.split}/{source}: {len(normed)} rows -> {cache_file.name}")
    return out


# --------------------------------------------------------------------------
# Stage 2: blocking
# --------------------------------------------------------------------------
def _idf_path(cfg: Config) -> Path:
    return _shared_dir(cfg) / "tfidf_idf.pkl"


def _blocking_config_path(cfg: Config) -> Path:
    return _shared_dir(cfg) / "blocking_config.json"


def stage_block(paths, cfg: Config, force: bool = False) -> pd.DataFrame:
    cache_file = paths.cache / "candidates.parquet"
    if cache_file.exists() and not force:
        return pd.read_parquet(cache_file)

    normed = stage_normalize(paths, cfg, force=False)
    s1n, s2n, s3n = normed["source1"], normed["source2"], normed["source3"]

    vectorizer = tfidf_mod.build_hashing_vectorizer()
    idf_path = _idf_path(cfg)
    if paths.split == "train" or not idf_path.exists():
        sample_texts = pd.concat([
            s1n["embed_text"], s2n["embed_text"].sample(min(len(s2n), 1_500_000), random_state=cfg.seed),
            s3n["embed_text"].sample(min(len(s3n), 1_500_000), random_state=cfg.seed),
        ]).tolist()
        idf = tfidf_mod.fit_idf(vectorizer, sample_texts)
        tfidf_mod.save_idf(idf, idf_path)
    else:
        idf = tfidf_mod.load_idf(idf_path)

    full_union = blocking.generate_candidates(s1n, s2n, s3n, cfg, paths.cache, tag=paths.split)
    full_union = blocking.add_tfidf_rerank(full_union, s1n, s2n, s3n, vectorizer, idf, n_jobs=cfg.n_jobs)

    all_s1_ids = s1n["entity_id"].tolist()
    union_stats = blocking.candidate_count_stats(full_union, all_s1_ids)
    print(
        f"  [block] candidate count per S1, full union (pre-truncation): "
        f"mean={union_stats['mean']:.1f} median={union_stats['median']:.1f} "
        f"p95={union_stats['p95']:.1f} max={union_stats['max']} "
        f"zero-candidate S1s={union_stats['n_s1_zero_candidates']}/{union_stats['n_s1']}"
    )

    k_final = cfg.k_final
    if paths.split == "train":
        gt = read_ground_truth(paths)
        cross_frac = blocking.check_no_cross_country_links(s1n, s2n, s3n, gt)
        print(f"  [block] cross-country ground-truth link fraction: {cross_frac:.4f} (should be ~0)")
        pools = split_mod.assign_pools(s1n["entity_id"].tolist(), cfg)
        val_ids = {sid for sid, p in pools.items() if p == "val"}
        sweep = blocking.recall_sweep(full_union, gt, val_ids)
        print("\n  [block] recall / ceiling-F0.5 sweep on the validation split:")
        print(sweep.to_string(index=False))
        sweep.to_csv(paths.cache / "recall_sweep.csv", index=False)

        if k_final <= 0:
            numeric = sweep[sweep["k"].apply(lambda x: isinstance(x, (int, np.integer)))]
            full_row = sweep[sweep["k"] == "full_union"].iloc[0]
            target = full_row["ceiling_f05"] - 0.002
            good = numeric[numeric["ceiling_f05"] >= target]
            k_final = int(good["k"].min()) if len(good) else int(numeric["k"].max())
        print(f"  [block] chosen k_final = {k_final}")
        _blocking_config_path(cfg).write_text(json.dumps({"k_final": k_final}))
        (paths.cache / "pools.json").write_text(json.dumps(pools))
    else:
        if k_final <= 0:
            if not _blocking_config_path(cfg).exists():
                raise RuntimeError("No shared blocking_config.json found -- run the TRAIN split's block stage first.")
            k_final = json.loads(_blocking_config_path(cfg).read_text())["k_final"]

    candidates = blocking.truncate_to_k(full_union, k_final)
    candidates = blocking.add_competition_features(candidates)
    candidates.to_parquet(cache_file, index=False)
    final_stats = blocking.candidate_count_stats(candidates, all_s1_ids)
    print(
        f"  [block] candidate count per S1, final (post k_final={k_final}): "
        f"mean={final_stats['mean']:.1f} median={final_stats['median']:.1f} "
        f"p95={final_stats['p95']:.1f} max={final_stats['max']} "
        f"zero-candidate S1s={final_stats['n_s1_zero_candidates']}/{final_stats['n_s1']}"
    )
    print(f"  [block] {paths.split}: {len(candidates)} final candidate pairs -> {cache_file.name}")
    return candidates


# --------------------------------------------------------------------------
# Stage 3: featurize
# --------------------------------------------------------------------------
def stage_featurize(paths, cfg: Config, force: bool = False) -> pd.DataFrame:
    cache_file = paths.cache / "features.parquet"
    if cache_file.exists() and not force:
        return pd.read_parquet(cache_file)

    candidates = stage_block(paths, cfg, force=False)
    normed = stage_normalize(paths, cfg, force=False)
    s1n, s2n, s3n = normed["source1"], normed["source2"], normed["source3"]
    feats = features_mod.compute_features(candidates, s1n, s2n, s3n, n_jobs=cfg.n_jobs)

    if paths.split == "train":
        gt = read_ground_truth(paths)
        gt_map = model_mod.build_gt_map(gt)
        feats = model_mod.label_candidates(feats, gt_map)
        pools = json.loads((paths.cache / "pools.json").read_text())
        feats["pool"] = feats["s1_entity_id"].map(pools).fillna("train_unused")

    feats.to_parquet(cache_file, index=False)
    print(f"  [featurize] {paths.split}: {len(feats)} rows x {len(features_mod.FEATURE_COLUMNS)} features -> {cache_file.name}")
    return feats


# --------------------------------------------------------------------------
# Stage 4: train (train split only)
# --------------------------------------------------------------------------
def _model_path(cfg: Config) -> Path:
    return _shared_dir(cfg) / "model"


def stage_train(paths, cfg: Config, force: bool = False):
    if paths.split != "train":
        raise ValueError("stage_train only runs on the train split.")
    model_file = _model_path(cfg).with_suffix(".xgb.json")
    if model_file.exists() and not force:
        return model_mod.load_model(_model_path(cfg))

    feats = stage_featurize(paths, cfg, force=False)
    train_df = feats[feats["pool"] == "train"]
    train_df = model_mod.sample_hard_negatives(train_df, cfg.hard_neg_per_s1)
    calib_df = feats[feats["pool"] == "calib"]
    print(f"  [train] gbm-train rows (after hard-neg sampling): {len(train_df)}  positives: {(train_df['label']==1).sum()}")

    clf = model_mod.train_model(train_df, cfg, eval_df=calib_df)
    raw_proba = model_mod.predict_proba(clf, calib_df)
    iso = model_mod.fit_calibration(raw_proba, calib_df["label"].values)
    model_mod.save_model(clf, iso, _model_path(cfg))
    print(f"  [train] saved model + isotonic calibrator -> {_model_path(cfg)}.*")
    return clf, iso


# --------------------------------------------------------------------------
# Score + assign (shared by train-validation and test-prediction)
# --------------------------------------------------------------------------
def score_and_assign(paths, cfg: Config, force: bool = False) -> pd.DataFrame:
    cache_file = paths.cache / "scored_assigned.parquet"
    if cache_file.exists() and not force:
        return pd.read_parquet(cache_file)

    feats = stage_featurize(paths, cfg, force=False)
    clf, iso = model_mod.load_model(_model_path(cfg)) if paths.split == "test" else stage_train(
        paths.cfg.paths("train"), cfg, force=False
    )
    raw_proba = model_mod.predict_proba(clf, feats)
    feats = feats.copy()
    feats["calib_proba"] = model_mod.apply_calibration(iso, raw_proba)
    assigned = tune_mod.assign_one_owner(feats, prob_col="calib_proba")

    keep_cols = [
        "s1_entity_id", "cand_entity_id", "source", "country", "calib_proba",
        "ann_sim", "tfidf_sim",
    ]
    if "label" in assigned.columns:
        keep_cols.append("label")
    if "pool" in assigned.columns:
        keep_cols.append("pool")
    assigned[keep_cols].to_parquet(cache_file, index=False)
    return assigned[keep_cols]


# --------------------------------------------------------------------------
# Stage 5: threshold tuning (train split only)
# --------------------------------------------------------------------------
def _decision_config_path(cfg: Config) -> Path:
    return _shared_dir(cfg) / "decision_config.json"


def stage_tune(paths, cfg: Config, force: bool = False) -> dict:
    if paths.split != "train":
        raise ValueError("stage_tune only runs on the train split.")
    decision_path = _decision_config_path(cfg)
    if decision_path.exists() and not force:
        return json.loads(decision_path.read_text())

    # `force` must reach score_and_assign too: its own cache (scored_assigned.parquet)
    # holds predictions from whatever model existed when it last ran, so a
    # `--force`-retrained model here would otherwise get silently re-scored
    # with stale predictions from the OLD model (confirmed: retraining without
    # this fix reproduced the pre-retrain numbers exactly, because tune read
    # score_and_assign's stale cache instead of the just-retrained model).
    assigned = score_and_assign(paths, cfg, force=force)
    gt = read_ground_truth(paths)
    gt_map = model_mod.build_gt_map(gt)
    pools = json.loads((paths.cache / "pools.json").read_text())
    val_ids = {sid for sid, p in pools.items() if p == "val"}

    best_t, sweep_df = tune_mod.sweep_global_threshold(assigned, gt_map, val_ids)
    pred_a = tune_mod._pred_map_from_threshold(assigned, best_t, "calib_proba")
    eval_a = tune_mod.evaluate(pred_a, gt_map, val_ids)

    pred_b = tune_mod.per_entity_expected_f05(assigned, val_ids)
    eval_b = tune_mod.evaluate(pred_b, gt_map, val_ids)

    if eval_a["f05"] >= eval_b["f05"]:
        mode, chosen_eval = "global_threshold", eval_a
        decision = {"mode": "global_threshold", "threshold": float(best_t)}
    else:
        mode, chosen_eval = "per_entity", eval_b
        decision = {"mode": "per_entity", "floor": 0.02}

    cross_fit = tune_mod.cross_fit_threshold_score(assigned, gt_map, val_ids) if mode == "global_threshold" else chosen_eval["f05"]

    decision_path.write_text(json.dumps(decision))

    # Simple interpretable weighted-sum baseline (0.40 name + 0.35 address +
    # 0.15 house-number + 0.10 country), reported for the methodology
    # writeup only -- never used for the actual submission. Goes through the
    # same one-owner assignment and the same threshold sweep as the GBM so
    # the two F0.5 numbers are directly comparable.
    feats_full = stage_featurize(paths, cfg, force=False)
    feats_full["baseline_score"] = baseline_mod.compute_baseline_score(feats_full)
    baseline_assigned = tune_mod.assign_one_owner(feats_full, prob_col="baseline_score")
    best_t_base, _ = tune_mod.sweep_global_threshold(baseline_assigned, gt_map, val_ids, prob_col="baseline_score")
    pred_base = tune_mod._pred_map_from_threshold(baseline_assigned, best_t_base, "baseline_score")
    eval_base = tune_mod.evaluate(pred_base, gt_map, val_ids)

    recall_sweep_path = paths.cache / "recall_sweep.csv"
    recall_row = None
    if recall_sweep_path.exists():
        rs = pd.read_csv(recall_sweep_path)
        recall_row = rs[rs["k"] == "full_union"].iloc[0].to_dict()

    print("\n" + "=" * 70)
    print(f"VALIDATION RESULTS  ({paths.split} split, {len(val_ids)} held-out S1 entities)")
    print("=" * 70)
    if recall_row is not None:
        print(f"  BLOCKING RECALL CEILING (perfect classifier on final candidates):")
        print(f"    pair recall = {recall_row['recall']:.4f}   ceiling F0.5 = {recall_row['ceiling_f05']:.4f}")
    print(f"  Decision mode chosen: {mode}  ({decision})")
    print(f"  FINAL precision (micro/pair-level): {chosen_eval['precision_micro']:.4f}")
    print(f"  FINAL recall    (micro/pair-level): {chosen_eval['recall_micro']:.4f}")
    print(f"  FINAL precision (macro/per-entity): {chosen_eval['precision_macro']:.4f}")
    print(f"  FINAL recall    (macro/per-entity): {chosen_eval['recall_macro']:.4f}")
    print(f"  FINAL F0.5 (macro-avg, in-sample tuned): {chosen_eval['f05']:.4f}")
    print(f"  FINAL F0.5 (macro-avg, 2-fold cross-fit, honest estimate): {cross_fit:.4f}")
    print(f"  --- comparison: simple weighted baseline (0.40 name + 0.35 addr + 0.15 housenum + 0.10 country) ---")
    print(f"  Baseline F0.5 (macro-avg, in-sample tuned threshold={best_t_base:.3f}): {eval_base['f05']:.4f}"
          f"   (GBM improvement: {chosen_eval['f05'] - eval_base['f05']:+.4f})")
    print("=" * 70 + "\n")

    (paths.cache / "val_report.json").write_text(json.dumps({
        "decision": decision, "eval": chosen_eval, "cross_fit_f05": cross_fit,
        "baseline_eval": eval_base, "baseline_threshold": float(best_t_base),
        "recall_ceiling": recall_row,
    }, default=float))
    return decision


# --------------------------------------------------------------------------
# Stage 6: predict + output
# --------------------------------------------------------------------------
def stage_predict(paths, cfg: Config, force: bool = False) -> None:
    decision = stage_tune(paths.cfg.paths("train"), cfg, force=False)
    assigned = score_and_assign(paths, cfg, force=force)

    s1_raw = read_source(paths, paths.split, "source1")
    required_ids = s1_raw["entity_id"].tolist()

    if paths.split == "test":
        matching_path = cfg.output_dir / "matching_results.tsv"
        candidate_path = cfg.output_dir / "candidate_pairs.tsv"
    else:
        pools = json.loads((paths.cache / "pools.json").read_text())
        val_ids = [sid for sid, p in pools.items() if p == "val"]
        required_ids = val_ids
        matching_path = paths.cache / "val_matching_results.tsv"
        candidate_path = paths.cache / "val_candidate_pairs.tsv"
        write_tsv(pd.DataFrame({"entity_id": val_ids}), paths.cache / "val_s1_ids.txt")

    # Assignment (score_and_assign) already resolved every candidate against
    # ALL S1 entities in this split, matching real one-owner competition; the
    # decision rule is then only evaluated for the entities we actually need
    # to emit a row for.
    if decision["mode"] == "global_threshold":
        pred_map = tune_mod._pred_map_from_threshold(assigned, decision["threshold"], "calib_proba")
    else:
        pred_map = tune_mod.per_entity_expected_f05(assigned, set(required_ids), floor=decision.get("floor", 0.02))

    cand_map = output_mod.candidates_df_to_map(assigned)

    output_mod.write_matching_results(pred_map, required_ids, matching_path)
    output_mod.write_candidate_pairs(cand_map, required_ids, candidate_path)
    print(f"  [predict] wrote {matching_path} and {candidate_path}")

    if paths.split == "test":
        merged = s1_raw[["entity_id", "country"]].copy()
        merged["has_match"] = merged["entity_id"].map(lambda i: bool(pred_map.get(i)))
        print("\n  Test-set match-rate sanity check by country (France cannot be validated -- no labels exist):")
        print(merged.groupby("country")["has_match"].mean().to_string())


def stage_validate(paths, cfg: Config, validator_script: Path) -> None:
    """Format-validate the actual submission files.

    The organizers' validator hardcodes ``test_source{1,2,3}.tsv`` as its
    expected filenames (it's built to check the real test-set submission),
    so it can only ever run against ``--split test``. For ``--split train``
    there is no such file to compare against (our held-out val split uses
    ``train_source*.tsv``), so this cross-checks the val output's own score
    with the standalone scorer (:mod:`validate_predictions`, run via
    ``ber.metrics`` directly here) instead of the format validator.
    """
    if paths.split != "test":
        gt = read_ground_truth(paths)
        gt_map = model_mod.build_gt_map(gt)
        pools = json.loads((paths.cache / "pools.json").read_text())
        val_ids = {sid for sid, p in pools.items() if p == "val"}
        pred_map = {
            row.source1_entity_id: set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
            for row in pd.read_csv(paths.cache / "val_matching_results.tsv", sep="\t", dtype=str, keep_default_na=False).itertuples()
        }
        f05 = f05_macro(pred_map, {k: v for k, v in gt_map.items() if k in val_ids})
        print(f"  [validate] cross-check via ber.metrics.f05_macro on val_matching_results.tsv: F0.5 = {f05:.4f}")
        print(
            "  [validate] the organizers' validate_submission.py only applies to --split test "
            "(it hardcodes test_source1/2/3.tsv); run it directly on output/ once the test split is predicted."
        )
        return

    matching_path = cfg.output_dir / "matching_results.tsv"
    candidate_path = cfg.output_dir / "candidate_pairs.tsv"
    code, out = output_mod.run_official_validator(matching_path, candidate_path, paths.dir, validator_script, check_ids=True)
    print(out)
    print("PASS" if code == 0 else "FAIL")


STAGES = ["normalize", "block", "featurize", "train", "tune", "predict", "validate"]


def run_all(paths, cfg: Config, validator_script: Path, force: bool = False) -> None:
    stage_normalize(paths, cfg, force=force)
    stage_block(paths, cfg, force=force)
    stage_featurize(paths, cfg, force=force)
    if paths.split == "train":
        stage_train(paths, cfg, force=force)
        stage_tune(paths, cfg, force=force)
    stage_predict(paths, cfg, force=force)
    stage_validate(paths, cfg, validator_script)
