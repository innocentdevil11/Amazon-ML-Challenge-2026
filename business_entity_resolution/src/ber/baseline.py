"""A simple, fully interpretable weighted-sum baseline score, reported
alongside the GBM in ``stage_tune`` purely as a comparison point for the
methodology writeup -- it is never used for the actual submission.

    score = 0.40 * name_similarity + 0.35 * address_similarity
          + 0.15 * house_number_match + 0.10 * country_match

Every input is already a 0-1 (or {0,1}) feature computed in
:mod:`ber.features`, so the weighted sum lands in roughly [0, 1] and can be
thresholded/assigned exactly like the calibrated GBM probability -- it goes
through the same one-owner assignment and the same threshold sweep, so the
two numbers are comparable on equal footing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"name": 0.40, "addr": 0.35, "housenum": 0.15, "country": 0.10}


def compute_baseline_score(df: pd.DataFrame) -> np.ndarray:
    name_sim = df["name_token_set"].to_numpy(dtype=np.float64)
    addr_sim = df["addr_token_set"].to_numpy(dtype=np.float64)
    # house_number_match is coded 1/0/-1 (match/mismatch/missing, see
    # features.py); a missing house number is treated as "no evidence" (0),
    # not as a penalty.
    housenum = np.clip(df["house_number_match"].to_numpy(dtype=np.float64), 0, 1)
    country = df["country_match"].to_numpy(dtype=np.float64)
    return (
        WEIGHTS["name"] * name_sim
        + WEIGHTS["addr"] * addr_sim
        + WEIGHTS["housenum"] * housenum
        + WEIGHTS["country"] * country
    )
