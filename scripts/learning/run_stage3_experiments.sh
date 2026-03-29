#!/usr/bin/env bash
# Run Stage 3 experiment round (local or sbatch).
# Usage: ./scripts/learning/run_stage3_experiments.sh [print_only|local|sbatch] [--force]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p artifacts/learning_runs logs/learning

MODE="${1:-print_only}"
FORCE=""
[[ "${2:-}" = "--force" ]] && FORCE="--force"

if [[ "$MODE" = "sbatch" ]]; then
  [[ -n "$FORCE" ]] && export FORCE=1
  sbatch batch/learning/run_stage3_experiments.sbatch
  exit 0
fi

python -m src.learning.run_stage3_experiments --config "$REPO_ROOT/configs/learning/experiment_matrix_stage3.json" --mode "$MODE" $FORCE
