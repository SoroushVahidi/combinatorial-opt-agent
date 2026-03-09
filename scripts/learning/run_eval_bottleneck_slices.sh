#!/usr/bin/env bash
# Evaluate on bottleneck slices.
# Usage: ./scripts/learning/run_eval_bottleneck_slices.sh [--sbatch] [run_dirs...]
# Example: ./scripts/learning/run_eval_bottleneck_slices.sh rule_baseline run0 multitask_run0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p artifacts/learning_runs/bottleneck_slices

RUNS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch)
      RUN_DIRS="${RUN_DIRS:-rule_baseline}"
      sbatch batch/learning/eval_bottleneck_slices.sbatch
      exit 0
      ;;
    *) RUNS+=("$1"); shift ;;
  esac
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
  RUNS=(rule_baseline)
fi
python -m src.learning.eval_bottleneck_slices \
  --data_dir "$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp" \
  --out_dir "$REPO_ROOT/artifacts/learning_runs" \
  --split test \
  --run_dirs "${RUNS[@]}"
