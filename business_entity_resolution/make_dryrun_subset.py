#!/usr/bin/env python3
"""Builds a small, self-contained subsample of the TRAIN split for a fast
end-to-end dry run of the pipeline (normalize -> block -> featurize -> train
-> tune -> validate) in minutes instead of hours.

Sampling strategy:
  1. Sample ``--frac`` of S1 entities (seeded, so it's reproducible).
  2. Look up their ground-truth matches -- these S2/S3 records are always
     kept, so the dry run's blocking recall ceiling is meaningful instead of
     being artificially near-zero from randomly dropping true matches.
  3. Additionally sample ``--frac`` of ALL S2/S3 records (seeded) as
     "distractor" background pool, at roughly the same unlinked-record ratio
     as the full dataset, so blocking has to do real work (not just find the
     one true match in an otherwise-empty pool).
  4. Union (2) and (3), write out matching train_source{1,2,3}.tsv and
     train_ground_truth.tsv, restricted to the sampled S1 entities.

Usage:
    python make_dryrun_subset.py --frac 0.05 --data-dir ../student_resource/dataset --out-dir dryrun_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ber.io import read_tsv, write_tsv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="../student_resource/dataset")
    ap.add_argument("--out-dir", default="dryrun_data")
    args = ap.parse_args()

    src_train = Path(args.data_dir) / "train"
    out_train = Path(args.out_dir) / "train"
    out_train.mkdir(parents=True, exist_ok=True)

    print(f"Reading full train split from {src_train} ...")
    s1 = read_tsv(src_train / "train_source1.tsv")
    gt = read_tsv(src_train / "train_ground_truth.tsv")
    s2 = read_tsv(src_train / "train_source2.tsv")
    s3 = read_tsv(src_train / "train_source3.tsv")
    print(f"  full sizes: S1={len(s1)} S2={len(s2)} S3={len(s3)} GT={len(gt)}")

    s1_sample = s1.sample(frac=args.frac, random_state=args.seed)
    sampled_ids = set(s1_sample["entity_id"])
    gt_sample = gt[gt["source1_entity_id"].isin(sampled_ids)].copy()

    true_match_ids = set()
    for ids in gt_sample["matched_entity_ids"]:
        if ids:
            true_match_ids.update(ids.split(","))
    true_s2 = {i for i in true_match_ids if i.startswith("S2-")}
    true_s3 = {i for i in true_match_ids if i.startswith("S3-")}

    s2_bg = s2.sample(frac=args.frac, random_state=args.seed)
    s3_bg = s3.sample(frac=args.frac, random_state=args.seed)

    s2_keep_ids = true_s2 | set(s2_bg["entity_id"])
    s3_keep_ids = true_s3 | set(s3_bg["entity_id"])
    s2_out = s2[s2["entity_id"].isin(s2_keep_ids)]
    s3_out = s3[s3["entity_id"].isin(s3_keep_ids)]

    print(f"  dry-run sizes: S1={len(s1_sample)} S2={len(s2_out)} S3={len(s3_out)} GT={len(gt_sample)}")
    print(f"  true matches preserved: S2={len(true_s2)} (found {len(true_s2 & set(s2_out.entity_id))}), "
          f"S3={len(true_s3)} (found {len(true_s3 & set(s3_out.entity_id))})")
    assert true_s2 <= set(s2_out["entity_id"]), "lost a true S2 match during subsampling"
    assert true_s3 <= set(s3_out["entity_id"]), "lost a true S3 match during subsampling"

    write_tsv(s1_sample, out_train / "train_source1.tsv")
    write_tsv(s2_out, out_train / "train_source2.tsv")
    write_tsv(s3_out, out_train / "train_source3.tsv")
    write_tsv(gt_sample, out_train / "train_ground_truth.tsv")
    print(f"Wrote dry-run train split to {out_train}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
