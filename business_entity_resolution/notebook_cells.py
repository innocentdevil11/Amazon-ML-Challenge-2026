# %% [1] SETUP
# Paste each "# %% [...]" block into its own Colab/Kaggle cell (or run
# `python make_notebook.py` to generate pipeline.ipynb from this file
# automatically). Nothing here hardcodes a local path -- BER_DATA_DIR /
# BER_OUTPUT_DIR / BER_CACHE_DIR (or the auto-detection in ber/config.py)
# decide where data comes from and artifacts go.
"""
!pip install -q sentence-transformers==5.2.2 rapidfuzz==3.14.3 jellyfish==1.2.1 \
    anyascii==0.3.3 faiss-cpu==1.15.1
# On a Kaggle/Colab GPU runtime, swap the line above's faiss-cpu for:
#   faiss-gpu-cu12==1.9.0.post1
"""
import os
import sys

# If you uploaded/cloned this project's `business_entity_resolution/` folder
# as-is, point this at its `src/` directory (Kaggle: /kaggle/working/..., or
# wherever you extracted the submission zip's code/business_entity_resolution/).
PROJECT_SRC = "/kaggle/working/business_entity_resolution/src"  # <-- EDIT ME on Kaggle/Colab
if PROJECT_SRC not in sys.path and os.path.isdir(PROJECT_SRC):
    sys.path.insert(0, PROJECT_SRC)

# Uncomment and edit if auto-detection (ber/config.py) doesn't find your data:
# os.environ["BER_DATA_DIR"] = "/kaggle/input/<your-dataset-slug>/dataset"
# os.environ["BER_OUTPUT_DIR"] = "/kaggle/working/output"
# os.environ["BER_CACHE_DIR"] = "/kaggle/working/cache"
# os.environ["BER_USE_GPU"] = "1"   # force on/off instead of auto-detecting

from ber import pipeline as pl
from ber.bench import print_benchmark_table, run_benchmark
from ber.config import get_config

cfg = get_config()
paths_train = cfg.paths("train")
paths_test = cfg.paths("test")
print(f"data_dir={cfg.data_dir}  use_gpu={cfg.use_gpu}  n_jobs={cfg.n_jobs}")

# %% [2] BENCHMARK -- always run this before the long stages below
bench_results = run_benchmark(paths_train, cfg)
print_benchmark_table(bench_results, cfg)
# If "embed" (or anything else) is flagged LONG, either switch this runtime to
# a GPU, lower BER_EMBED_BATCH / accept the wait, then set:
cfg.allow_long = True  # <-- set only after reading the table above

# %% [3] NORMALIZE (train)
normed_train = pl.stage_normalize(paths_train, cfg)

# %% [4] BLOCK (train) -- prints the recall/ceiling-F0.5 sweep table
candidates_train = pl.stage_block(paths_train, cfg)

# %% [5] FEATURIZE (train)
features_train = pl.stage_featurize(paths_train, cfg)

# %% [6] TRAIN (GBM + isotonic calibration)
clf, iso = pl.stage_train(paths_train, cfg)

# %% [7] TUNE (threshold sweep on the held-out validation split)
# Prints: blocking recall ceiling, final precision/recall, final F0.5 (tuned
# and 2-fold cross-fit) -- your honest estimated leaderboard score.
decision = pl.stage_tune(paths_train, cfg)

# %% [8] OUTPUT (test predictions)
normed_test = pl.stage_normalize(paths_test, cfg)
candidates_test = pl.stage_block(paths_test, cfg)
features_test = pl.stage_featurize(paths_test, cfg)
pl.stage_predict(paths_test, cfg)

# %% [9] VALIDATE
from pathlib import Path

VALIDATOR_SCRIPT = "/kaggle/working/student_resource/utils/validate_submission.py"  # <-- EDIT ME
pl.stage_validate(paths_test, cfg, Path(VALIDATOR_SCRIPT))
