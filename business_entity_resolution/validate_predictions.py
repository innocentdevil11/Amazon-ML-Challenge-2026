#!/usr/bin/env python3
"""Standalone F_0.5 macro-average scorer against a held-out ground-truth
split -- the same formula (:mod:`ber.metrics`) the pipeline itself prints,
run independently so you can double check a submission's own self-reported
score, or score an old run's output without re-running the pipeline.

Usage
-----
    python validate_predictions.py \\
        --pred cache/train/val_matching_results.tsv \\
        --gt student_resource/dataset/train/train_ground_truth.tsv \\
        --ids cache/train/val_s1_ids.txt \\
        --candidates cache/train/val_candidate_pairs.tsv   # optional: also reports the blocking recall ceiling
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ber.io import read_tsv  # noqa: E402
from ber.metrics import f05_macro, precision_recall_macro, precision_recall_micro  # noqa: E402


def _load_id_list_file(path: Path, id_col: str, list_col: str) -> dict[str, set]:
    df = read_tsv(path)
    return {
        row[id_col]: set(row[list_col].split(",")) if row[list_col] else set()
        for _, row in df.iterrows()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="matching_results.tsv-format predictions to score")
    ap.add_argument("--gt", required=True, help="ground truth TSV (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--ids", required=True, help="text file, one S1 entity_id per line, restricting evaluation to this set")
    ap.add_argument("--candidates", default=None, help="optional candidate_pairs.tsv, to report the recall ceiling too")
    args = ap.parse_args()

    ids = {line.strip() for line in Path(args.ids).read_text(encoding="utf-8").splitlines() if line.strip()}
    pred_map = _load_id_list_file(Path(args.pred), "source1_entity_id", "matched_entity_ids")
    gt_map = _load_id_list_file(Path(args.gt), "source1_entity_id", "matched_entity_ids")
    gt_map = {k: v for k, v in gt_map.items() if k in ids}

    missing = ids - set(pred_map)
    if missing:
        print(f"WARNING: {len(missing)} evaluated IDs missing from predictions; treated as empty predictions.")

    f05 = f05_macro(pred_map, gt_map)
    p_micro, r_micro = precision_recall_micro(pred_map, gt_map)
    p_macro, r_macro = precision_recall_macro(pred_map, gt_map)

    print(f"Evaluated S1 entities: {len(gt_map)}")
    print(f"Precision (micro/pair-level): {p_micro:.4f}")
    print(f"Recall    (micro/pair-level): {r_micro:.4f}")
    print(f"Precision (macro/per-entity): {p_macro:.4f}")
    print(f"Recall    (macro/per-entity): {r_macro:.4f}")
    print(f"F0.5 (macro-average, the scored metric): {f05:.4f}")

    if args.candidates:
        cand_map = _load_id_list_file(Path(args.candidates), "source1_entity_id", "candidate_entity_ids")
        total_true = sum(len(v) for v in gt_map.values())
        tp = sum(len(cand_map.get(s1, set()) & true) for s1, true in gt_map.items())
        ceiling_pred = {s1: cand_map.get(s1, set()) & true for s1, true in gt_map.items()}
        ceiling_f05 = f05_macro(ceiling_pred, gt_map)
        print(f"\nBlocking recall ceiling: pair recall = {tp / total_true if total_true else 1.0:.4f}  "
              f"ceiling F0.5 (perfect classifier on these candidates) = {ceiling_f05:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
