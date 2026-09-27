#!/bin/bash
# Full train+test run, single self-contained sequential script (no separate
# watcher process -- a prior attempt using a cross-process `kill -0 $PID`
# watcher proved unreliable on this Git-Bash/Windows setup and caused a
# duplicate-launch race). Configuration: BER_USE_NATIVE_EMBED=0 (embedding
# fix reverted) + missing-address feature already removed from features.py
# -- the known-good baseline that scored F0.5=0.9390 on the dry run before
# either fix was attempted.
cd "/c/Users/Asus/Downloads/6ab10eb3b23ba_student_resource/business_entity_resolution"
export BER_DATA_DIR="../student_resource/dataset" BER_CACHE_DIR="cache" BER_OUTPUT_DIR="output" BER_USE_NATIVE_EMBED=0
PYEXE="./.venv/Scripts/python.exe"

run_stage_with_retry() {
  local split=$1
  "$PYEXE" -u run_pipeline.py --split "$split" --stage all --allow-long
  local code=$?
  if [ $code -ne 0 ]; then
    echo "[run_full] --split $split --stage all exited $code -- retrying once (cached stages resume, so this is cheap)"
    sleep 10
    "$PYEXE" -u run_pipeline.py --split "$split" --stage all --allow-long
    code=$?
    if [ $code -ne 0 ]; then
      echo "[run_full] --split $split --stage all failed again on retry, exit $code -- giving up on this split"
    fi
  fi
  return $code
}

run_stage_with_retry train
echo "TRAIN_EXIT_CODE=$?"

run_stage_with_retry test
echo "TEST_EXIT_CODE=$?"

"$PYEXE" -u ../student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir ../student_resource/dataset/test \
  --check-ids
echo "OFFICIAL_VALIDATOR_EXIT_CODE=$?"

echo "ALL_DONE"
