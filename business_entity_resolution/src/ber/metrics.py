"""The exact scoring formula from the problem statement (README.md, section
"Evaluation Criteria"), implemented once here and shared by the pipeline's
self-reported validation score and by the standalone ``validate_predictions.py``,
so the two can never silently drift apart.

F_0.5 = 1.25 * P * R / (0.25 * P + R), computed **per S1 entity** and then
macro-averaged over every entity in the evaluation set. A singleton (empty
ground truth) scores 1.0 for an empty prediction and 0.0 for any non-empty
prediction.
"""
from __future__ import annotations


def f05_pair(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return 1.25 * precision * recall / denom


def f05_macro(pred_map: dict, gt_map: dict) -> float:
    """Macro-average F0.5 over every key in ``gt_map`` (the evaluation set).
    An S1 entity absent from ``pred_map`` is treated as an empty prediction."""
    if not gt_map:
        return float("nan")
    scores = [f05_pair(pred_map.get(s1, set()), true) for s1, true in gt_map.items()]
    return sum(scores) / len(scores)


def precision_recall_micro(pred_map: dict, gt_map: dict) -> tuple[float, float]:
    """Pair-level (micro) precision/recall, for the diagnostic printout
    alongside the macro F0.5 that is actually scored."""
    tp = fp = fn = 0
    for s1, true in gt_map.items():
        pred = pred_map.get(s1, set())
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall


def precision_recall_macro(pred_map: dict, gt_map: dict) -> tuple[float, float]:
    precisions, recalls = [], []
    for s1, true in gt_map.items():
        pred = pred_map.get(s1, set())
        tp = len(pred & true)
        precisions.append(tp / len(pred) if pred else (1.0 if not true else 0.0))
        recalls.append(tp / len(true) if true else 1.0)
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)
