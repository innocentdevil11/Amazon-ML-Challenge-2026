"""Labeling, the XGBoost matching classifier, and isotonic calibration.

Model: XGBoost (Apache-2.0 licensed, well under the 8B-parameter constraint --
a few hundred shallow trees). Trains on GPU (``device="cuda"``) when
available and falls back to CPU automatically; the algorithm (``hist``) is
identical either way, so results are consistent across machines.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import Config
from .features import FEATURE_COLUMNS


def build_gt_map(gt: pd.DataFrame) -> dict[str, set]:
    return {
        row.source1_entity_id: set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()
        for row in gt.itertuples()
    }


def label_candidates(df: pd.DataFrame, gt_map: dict[str, set]) -> pd.DataFrame:
    df = df.copy()
    df["label"] = [
        1 if cand in gt_map.get(s1, ()) else 0
        for s1, cand in zip(df["s1_entity_id"], df["cand_entity_id"])
    ]
    return df


def sample_hard_negatives(df: pd.DataFrame, hard_neg_per_s1: int) -> pd.DataFrame:
    """Keep every positive plus, per S1, the top ``hard_neg_per_s1`` *hardest*
    negatives -- non-matching candidates the blocking/re-rank stage ranked
    highest -- instead of a random sample. These are the pairs the model
    actually needs to learn to reject."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0].copy()
    neg["_r"] = neg.groupby("s1_entity_id")["rank_for_s1"].rank(method="first")
    neg = neg[neg["_r"] <= hard_neg_per_s1].drop(columns=["_r"])
    return pd.concat([pos, neg], ignore_index=True)


def train_model(train_df: pd.DataFrame, cfg: Config, eval_df: pd.DataFrame | None = None):
    import xgboost as xgb

    X = train_df[FEATURE_COLUMNS].astype(np.float32).values
    y = train_df["label"].astype(np.int32).values
    params = dict(
        tree_method="hist",
        device="cuda" if cfg.use_gpu else "cpu",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        eval_metric="aucpr",
        early_stopping_rounds=30 if eval_df is not None else None,
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
    )
    clf = xgb.XGBClassifier(**{k: v for k, v in params.items() if v is not None})
    eval_set = None
    if eval_df is not None and len(eval_df):
        X_eval = eval_df[FEATURE_COLUMNS].astype(np.float32).values
        y_eval = eval_df["label"].astype(np.int32).values
        eval_set = [(X_eval, y_eval)]
    try:
        clf.fit(X, y, eval_set=eval_set, verbose=False)
    except Exception as exc:  # pragma: no cover - GPU unavailable at fit time
        if cfg.use_gpu:
            print(f"  [model] CUDA training failed ({exc}); falling back to CPU.")
            params["device"] = "cpu"
            clf = xgb.XGBClassifier(**{k: v for k, v in params.items() if v is not None})
            clf.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            raise
    return clf


def predict_proba(clf, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS].astype(np.float32).values
    return clf.predict_proba(X)[:, 1]


def fit_calibration(raw_proba: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_proba, labels)
    return iso


def apply_calibration(iso: IsotonicRegression, raw_proba: np.ndarray) -> np.ndarray:
    return iso.predict(raw_proba)


def save_model(clf, iso: IsotonicRegression, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(path.with_suffix(".xgb.json")))
    path.with_suffix(".iso.pkl").write_bytes(pickle.dumps(iso))


def load_model(path: str | Path):
    import xgboost as xgb

    path = Path(path)
    clf = xgb.XGBClassifier()
    clf.load_model(str(path.with_suffix(".xgb.json")))
    iso = pickle.loads(path.with_suffix(".iso.pkl").read_bytes())
    return clf, iso
