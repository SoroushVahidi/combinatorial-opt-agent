#!/bin/bash
#SBATCH --job-name=stage3_collect
#SBATCH --output=logs/learning/collect_stage3_results_%j.out
#SBATCH --error=logs/learning/collect_stage3_results_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G
set -euo pipefail

# Gather Stage 3 results (learned metrics, bottleneck slices, deterministic baselines).
# Example: sbatch batch/learning/collect_stage3_results.sbatch
# Run after run_stage3_experiments (or after eval) to produce summary + comparison table.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs

CONFIG="${CONFIG:-$REPO_ROOT/configs/learning/experiment_matrix_stage3.json}"
RUNS_DIR="${RUNS_DIR:-$REPO_ROOT/artifacts/learning_runs}"
DET_DIR="${DET_DIR:-$REPO_ROOT/results/paper}"

python -m src.learning.collect_stage3_results \
  --config "$CONFIG" \
  --runs_dir "$RUNS_DIR" \
  --deterministic_dir "$DET_DIR"
