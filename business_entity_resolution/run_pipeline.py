#!/usr/bin/env python3
"""CLI entry point for the business-entity-resolution pipeline.

Examples
--------
    # 1. ALWAYS run this first and read the printed table before anything else.
    python run_pipeline.py --split train --stage benchmark

    # 2. Once the benchmark looks acceptable (or you've accepted the risk):
    python run_pipeline.py --split train --stage all --allow-long

    # 3. Generate the submission files from the trained model:
    python run_pipeline.py --split test --stage all --allow-long

Stages: benchmark, normalize, block, featurize, train, tune, predict, validate, all
(train/tune only run for --split train; predict/validate run for either split --
on --split train they (re)produce the held-out validation files instead of the
submission files).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ber import pipeline as pl  # noqa: E402
from ber.bench import print_benchmark_table, run_benchmark  # noqa: E402
from ber.config import get_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
VALIDATOR_SCRIPT = REPO_ROOT.parent / "student_resource" / "utils" / "validate_submission.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--stage",
        choices=["benchmark", "normalize", "block", "featurize", "train", "tune", "predict", "validate", "all"],
        required=True,
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached artifacts and redo this stage.")
    parser.add_argument("--allow-long", action="store_true", help="Proceed even if the benchmark flagged a stage as long.")
    parser.add_argument("--validator-script", default=str(VALIDATOR_SCRIPT))
    args = parser.parse_args()

    cfg = get_config()
    if args.allow_long:
        cfg.allow_long = True
    paths = cfg.paths(args.split)

    print(f"Config: data_dir={cfg.data_dir}  output_dir={cfg.output_dir}  cache_dir={cfg.cache_dir}")
    print(f"        use_gpu={cfg.use_gpu}  n_jobs={cfg.n_jobs}  allow_long={cfg.allow_long}")

    if args.stage == "benchmark":
        results = run_benchmark(paths, cfg)
        print_benchmark_table(results, cfg)
        return 0

    if not cfg.allow_long:
        print(
            "NOTE: --allow-long was not passed. Long stages (embedding especially) will still run when "
            "invoked directly, but 'all' stops after any stage it hasn't benchmarked. Run "
            "'--stage benchmark' first, review the table, then re-run with --allow-long.",
        )
        if args.stage == "all":
            print("Refusing to run --stage all without --allow-long. Run --stage benchmark first.")
            return 1

    dispatch = {
        "normalize": lambda: pl.stage_normalize(paths, cfg, force=args.force),
        "block": lambda: pl.stage_block(paths, cfg, force=args.force),
        "featurize": lambda: pl.stage_featurize(paths, cfg, force=args.force),
        "train": lambda: pl.stage_train(paths, cfg, force=args.force),
        "tune": lambda: pl.stage_tune(paths, cfg, force=args.force),
        "predict": lambda: pl.stage_predict(paths, cfg, force=args.force),
        "validate": lambda: pl.stage_validate(paths, cfg, Path(args.validator_script)),
        "all": lambda: pl.run_all(paths, cfg, Path(args.validator_script), force=args.force),
    }
    dispatch[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
