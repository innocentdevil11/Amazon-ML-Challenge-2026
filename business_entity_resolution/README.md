# Business Entity Resolution Pipeline

Rules + classical ML only (multilingual sentence embeddings for blocking,
XGBoost for matching, isotonic calibration) -- **no LLMs anywhere**, per the
challenge's fair-play rules. Everything is driven by a config object
(`src/ber/config.py`) that auto-detects data location, GPU availability and
CPU worker count, so the same code runs unmodified locally, on Kaggle, or on
Colab.

## Layout

```
business_entity_resolution/
  run_pipeline.py          # CLI: python run_pipeline.py --split train --stage all
  validate_predictions.py  # standalone F0.5 macro-average scorer vs. held-out ground truth
  notebook_cells.py        # the same pipeline as "# %% [n]" notebook cells
  make_notebook.py         # notebook_cells.py -> pipeline.ipynb (stdlib json, no nbformat)
  tests/test_smoke.py      # normalization + metric unit tests
  src/ber/
    config.py     # paths, GPU/CPU auto-detect, all tunable knobs
    io.py         # UTF-8, tab-separated, "\n"-only-line-ending TSV read/write
    lexicons.py   # every hand-authored dictionary (legal suffixes, address abbreviations, ...)
    normalize.py  # Section 1: normalization module
    embed.py      # Section 2: sentence-transformer embeddings (resumable)
    ann.py        # Section 2: FAISS ANN index/search
    tfidf.py      # Section 2: TF-IDF re-rank + a name/address similarity feature
    blocking.py   # Section 2: union of ANN + phonetic + sorted-neighborhood, recall sweep
    features.py   # Section 3: pairwise feature engineering
    split.py      # Section 4: deterministic S1 val/calib/train split
    model.py      # Section 4: labeling, hard negatives, XGBoost, isotonic calibration
    tune.py       # Section 5: one-owner assignment + F0.5 threshold tuning
    metrics.py    # the exact F0.5 macro-average formula (shared everywhere)
    output.py     # Section 6: TSV writers + official validator subprocess wrapper
    bench.py      # the --stage benchmark projections
    pipeline.py   # stage orchestration + on-disk resume caching
```

## Data

Expected layout (matches the challenge's `student_resource/dataset/`):

```
<data_dir>/
  train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

`ber/config.py` auto-detects `<data_dir>` in this order: `$BER_DATA_DIR` env
var, common Kaggle/Colab locations (`/kaggle/input/*/dataset`,
`/content/dataset`), then the local repo's
`../student_resource/dataset` relative to this package. Override explicitly
with:

```bash
export BER_DATA_DIR=/path/to/dataset      # or set it in the notebook's first cell
export BER_OUTPUT_DIR=/path/to/output     # default: ./output
export BER_CACHE_DIR=/path/to/cache       # default: ./cache (resumable stage artifacts)
```

## Local setup (Windows/Linux/Mac)

```bash
cd business_entity_resolution
python -m venv .venv
.venv\Scripts\activate                      # Windows; use `source .venv/bin/activate` on Linux/Mac

# Install a CUDA build of torch FIRST if you have an NVIDIA GPU (check with
# `nvidia-smi`; match the CUDA version it reports -- cu128 below works for
# driver-reported CUDA >= 12.8, which covers most current drivers):
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# CPU-only machine: `pip install torch==2.7.1` instead.

pip install -r requirements.txt
python -m pytest tests/ -q
```

## Running it

**Always benchmark before a long run** -- this measures real throughput on a
slice of your actual data and projects the full-split wall time for every
stage:

```bash
python run_pipeline.py --split train --stage benchmark
```

Any stage projected over 15 minutes is flagged `<-- LONG`. On most laptops
this is the embedding stage. Once you've read the table:

```bash
python run_pipeline.py --split train --stage all --allow-long
python run_pipeline.py --split test  --stage all --allow-long
```

This produces, in `output/`: `matching_results.tsv` and `candidate_pairs.tsv`
(for the **test** split -- these are your submission files), plus, in
`cache/train/`: `val_matching_results.tsv` / `val_candidate_pairs.tsv` /
`val_s1_ids.txt` (your held-out validation predictions, for scoring
yourself). The train run's final stage prints a labeled block with the
blocking recall ceiling, final precision/recall, and final F0.5 (both
in-sample-tuned and an honest 2-fold cross-fit estimate) -- read this before
you ever touch the leaderboard.

Each stage caches its output under `cache/<split>/` and is skipped on
re-run unless you pass `--force`; the embedding stage additionally
checkpoints itself chunk-by-chunk (`cache/<split>/emb_*`), so an interrupted
run resumes instead of restarting. Stages, in order:
`normalize -> block -> featurize -> train -> tune -> predict -> validate`
(`train`/`tune` only apply to `--split train`; the trained model and chosen
decision threshold live under `cache/shared/` and are reused, frozen, for
`--split test`).

Validate the submission files against the organizers' own validator (also
run automatically by `--stage validate` / `all`):

```bash
python ../student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test \
    --check-ids
```

Score your own held-out split independently of the pipeline's self-reported
number:

```bash
python validate_predictions.py \
    --pred cache/train/val_matching_results.tsv \
    --gt ../student_resource/dataset/train/train_ground_truth.tsv \
    --ids cache/train/val_s1_ids.txt \
    --candidates cache/train/val_candidate_pairs.tsv
```

## Kaggle / Colab

1. Upload this `business_entity_resolution/` folder (e.g. to
   `/kaggle/working/`) and the dataset (as a Kaggle Dataset, or extracted
   under `/content/dataset` on Colab).
2. Open `notebook_cells.py`, paste each `# %% [n] SECTION` block into its own
   cell (or run `python make_notebook.py` locally first and upload the
   resulting `pipeline.ipynb` instead), and edit the two path placeholders
   marked `<-- EDIT ME`.
3. Run cells top to bottom. Cell `[2]` is the same benchmark stage -- read
   its output before setting `cfg.allow_long = True` in that same cell.
4. GPU is used automatically when present: embeddings run on
   `sentence-transformers`' CUDA path, XGBoost trains with `device="cuda"`,
   and FAISS uses a GPU index when a GPU-capable FAISS build
   (`faiss-gpu-cu12`, see `requirements-colab.txt`) is installed; otherwise
   everything falls back to CPU with no code changes.

## Timing (measured on an i5-12450H / RTX 2050 4GB / 15.7GB RAM laptop, ~12.5M records/split)

| Stage | Estimate | Notes |
|---|---|---|
| normalize | 4-8 min | parallelized across CPU cores |
| embed | **1.5-3 h** | dominant cost; resumable; ~5-10x faster on a Kaggle/Colab T4 |
| ANN build+search | 15-30 min | FAISS CPU (no official faiss-gpu wheel on Windows) |
| phonetic + sorted-neighborhood | 3-6 min | |
| TF-IDF re-rank | 10-20 min | |
| featurize | 10-35 min | RapidFuzz pairwise metrics, the other CPU-bound stage |
| XGBoost train | 3-6 min | GPU when available, CPU otherwise |
| calibration + threshold tuning | 2-4 min | |
| predict + write + validate | 5-8 min | |

Re-run `--stage benchmark` on your own machine/data before trusting these --
they scale with row counts, `k_final`, and hardware.

## Notes on approach (see `Documentation_template.md` for the full write-up)

- **No external lookups.** Every normalization dictionary in `lexicons.py`
  was authored from patterns observed directly in the provided training
  data (native-script legal/state words, US/India/France address
  abbreviations, city aliases) -- nothing is fetched from an external
  database, geocoder, or API.
- **Country is an open set.** No feature or model input one-hot-encodes
  `country`; France appears only in the test set and must be handled by the
  same generic features as US/India.
- **License compliance.** `paraphrase-multilingual-MiniLM-L12-v2`
  (Apache-2.0) and XGBoost (Apache-2.0) are both far under the 8B-parameter
  ceiling.
