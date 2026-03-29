#!/bin/bash
#SBATCH --job-name=bottleneck
#SBATCH --output=logs/learning/eval_bottleneck_slices_%j.out
#SBATCH --error=logs/learning/eval_bottleneck_slices_%j.err
#SBATCH --time=00:45:00
#SBATCH --mem=8G
set -euo pipefail

# Evaluate on bottleneck slices (multiple floats, lower/upper cues, multi-entity).
# Example: sbatch batch/learning/eval_bottleneck_slices.sbatch
# With runs: RUN_DIRS="rule_baseline run0 multitask_run0" sbatch batch/learning/eval_bottleneck_slices.sbatch

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs/bottleneck_slices

DATA_DIR="${DATA_DIR:-$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/learning_runs}"
SPLIT="${SPLIT:-test}"
RUN_DIRS="${RUN_DIRS:-rule_baseline}"

python -m src.learning.eval_bottleneck_slices \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --split "$SPLIT" \
  --run_dirs $RUN_DIRS
