#!/usr/bin/env bash
# Gather Stage 3 results and write summary + comparison table.
# Usage: ./scripts/learning/run_collect_stage3_results.sh [--sbatch]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p artifacts/learning_runs logs/learning

if [[ "${1:-}" = "--sbatch" ]]; then
  sbatch batch/learning/collect_stage3_results.sbatch
  exit 0
fi

python -m src.learning.collect_stage3_results \
  --config "$REPO_ROOT/configs/learning/experiment_matrix_stage3.json" \
  --runs_dir "$REPO_ROOT/artifacts/learning_runs" \
  --deterministic_dir "$REPO_ROOT/results/paper"
