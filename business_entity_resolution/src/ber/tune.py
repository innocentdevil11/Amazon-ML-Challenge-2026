"""Assignment (one S2/S3 record can only ever be claimed by one S1 entity,
verified true of every ground-truth pair in the training data) and decision
threshold tuning, optimizing F_0.5 specifically.

Assignment happens *before* thresholding: each candidate goes to whichever
S1 scored it highest (ties broken by embedding cosine). Because the true
labeling never gives one candidate to two different S1s, handing a
candidate to its top scorer can only ever remove a false positive from a
losing S1, never remove a true positive -- so this ordering is at least as
good as thresholding first and resolving collisions after, and it lets the
whole threshold sweep run as vectorized array ops instead of a per-threshold
bipartite match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import f05_macro, precision_recall_macro, precision_recall_micro
from .split import _stable_frac


def assign_one_owner(df: pd.DataFrame, prob_col: str = "calib_proba") -> pd.DataFrame:
    """Keep, for every (source, cand_entity_id), only the row with the
    highest ``prob_col`` (ties broken by ann_sim then tfidf_sim)."""
    d = df.sort_values(
        [prob_col, "ann_sim", "tfidf_sim"], ascending=False
    ).drop_duplicates(subset=["source", "cand_entity_id"], keep="first")
    return d


def _pred_map_from_threshold(df: pd.DataFrame, threshold: float, prob_col: str) -> dict[str, set]:
    kept = df[df[prob_col] >= threshold]
    return kept.groupby("s1_entity_id")["cand_entity_id"].apply(set).to_dict()


def sweep_global_threshold(
    assigned_df: pd.DataFrame, gt_map: dict, s1_ids: set, prob_col: str = "calib_proba",
    thresholds=None,
) -> tuple[float, pd.DataFrame]:
    """Sweep candidate thresholds and return the F0.5-maximizing one.

    Computed with ``np.bincount`` over a once-factorized, once-filtered
    array rather than one ``groupby(...).apply(set)`` per threshold: the
    naive version re-groups and rebuilds a Python ``set`` per S1 for every
    one of ~190 thresholds, which on a multi-million-row assigned table
    became the dominant cost of the whole tuning stage (confirmed: on this
    project's 4.4M-row dry run, this loop -- run 4 times over, for the GBM
    sweep, both cross-fit folds, and the baseline sweep -- was slower than
    training the model itself). This computes the exact same per-entity
    F0.5 formula (:func:`ber.metrics.f05_pair`), just without materializing
    a set of matched IDs at every threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.991, 0.005)
    gt_sub = {k: v for k, v in gt_map.items() if k in s1_ids}
    n_total = len(gt_sub)
    if n_total == 0:
        return float(thresholds[0]), pd.DataFrame({"threshold": thresholds, "f05": np.nan})

    sub = assigned_df[assigned_df["s1_entity_id"].isin(gt_sub.keys())]
    codes, uniques = pd.factorize(sub["s1_entity_id"].to_numpy())
    n_present = len(uniques)
    proba = sub[prob_col].to_numpy(dtype=np.float64)
    cand_ids = sub["cand_entity_id"].to_numpy()
    s1_arr = sub["s1_entity_id"].to_numpy()
    is_true = np.fromiter(
        (cid in gt_sub.get(s1, ()) for s1, cid in zip(s1_arr, cand_ids)), dtype=bool, count=len(s1_arr)
    )
    true_count = np.array([len(gt_sub[u]) for u in uniques], dtype=np.float64)
    is_singleton = true_count == 0

    # S1 entities with zero assigned candidate rows never appear in `sub` at
    # all; they always score as an empty prediction (1.0 if a true
    # singleton, else 0.0), independent of threshold.
    missing_true_counts = [len(gt_sub[m]) for m in (set(gt_sub) - set(uniques))]
    missing_f_sum = float(sum(1.0 for c in missing_true_counts if c == 0))

    rows = []
    best_t, best_f = float(thresholds[0]), -1.0
    for t in thresholds:
        pred_mask = proba >= t
        tp_counts = np.bincount(codes[pred_mask & is_true], minlength=n_present).astype(np.float64)
        pred_counts = np.bincount(codes[pred_mask], minlength=n_present).astype(np.float64)

        f = np.where(pred_counts[is_singleton] == 0, 1.0, 0.0) if is_singleton.any() else np.array([])
        ns = ~is_singleton
        tp, pc, tc = tp_counts[ns], pred_counts[ns], true_count[ns]
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.divide(tp, pc, out=np.zeros_like(tp), where=pc > 0)
            recall = tp / tc
            denom = 0.25 * precision + recall
            f_ns = np.where((tp > 0) & (denom > 0), 1.25 * precision * recall / np.where(denom == 0, 1, denom), 0.0)

        f05 = (f.sum() + f_ns.sum() + missing_f_sum) / n_total
        rows.append({"threshold": t, "f05": f05})
        if f05 > best_f:
            best_f, best_t = f05, t
    return best_t, pd.DataFrame(rows)


def per_entity_expected_f05(
    assigned_df: pd.DataFrame, s1_ids: set, prob_col: str = "calib_proba", floor: float = 0.02,
) -> dict[str, set]:
    """Parameter-free per-entity decision rule: for each S1, sort its
    candidates by calibrated probability and pick the prefix length m
    (including m=0, i.e. singleton) that maximizes an *expected* F0.5 where
    each candidate's probability of being a true match is taken to be its
    calibrated score. This needs no threshold and adapts to how many
    plausible matches each entity actually has."""
    pred_map: dict[str, set] = {}
    sub = assigned_df[assigned_df["s1_entity_id"].isin(s1_ids) & (assigned_df[prob_col] >= floor)]
    for s1, group in sub.groupby("s1_entity_id"):
        p = np.sort(group[prob_col].values)[::-1]
        cand_ids = group.sort_values(prob_col, ascending=False)["cand_entity_id"].values
        cum = np.cumsum(p)
        total = cum[-1]
        singleton_score = np.prod(1 - p)  # P(no true match among candidates)
        best_score, best_m = singleton_score, 0
        for m in range(1, len(p) + 1):
            precision = cum[m - 1] / m
            recall = cum[m - 1] / total if total > 0 else 0.0
            denom = 0.25 * precision + recall
            f = 1.25 * precision * recall / denom if denom > 0 else 0.0
            if f > best_score:
                best_score, best_m = f, m
        if best_m > 0:
            pred_map[s1] = set(cand_ids[:best_m])
        else:
            pred_map[s1] = set()
    for s1 in s1_ids:
        pred_map.setdefault(s1, set())
    return pred_map


def evaluate(pred_map: dict, gt_map: dict, s1_ids: set) -> dict:
    gt_sub = {k: v for k, v in gt_map.items() if k in s1_ids}
    f05 = f05_macro(pred_map, gt_sub)
    p_micro, r_micro = precision_recall_micro(pred_map, gt_sub)
    p_macro, r_macro = precision_recall_macro(pred_map, gt_sub)
    return dict(f05=f05, precision_micro=p_micro, recall_micro=r_micro, precision_macro=p_macro, recall_macro=r_macro)


def cross_fit_threshold_score(
    assigned_df: pd.DataFrame, gt_map: dict, s1_ids: set, prob_col: str = "calib_proba", seed: int = 7,
) -> float:
    """Honest 2-fold estimate for the global-threshold mode: tune the
    threshold on one half of the validation S1s, score it on the other half,
    swap, and average -- so the reported number isn't inflated by tuning and
    evaluating on the exact same rows."""
    ids = sorted(s1_ids)
    fold_a = {i for i in ids if _stable_frac(i, seed) < 0.5}
    fold_b = set(ids) - fold_a
    scores = []
    for tune_fold, eval_fold in ((fold_a, fold_b), (fold_b, fold_a)):
        t, _ = sweep_global_threshold(assigned_df, gt_map, tune_fold, prob_col)
        pred_map = _pred_map_from_threshold(assigned_df, t, prob_col)
        scores.append(evaluate(pred_map, gt_map, eval_fold)["f05"])
    return float(np.mean(scores))
