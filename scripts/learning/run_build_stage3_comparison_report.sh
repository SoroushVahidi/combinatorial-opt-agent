#!/usr/bin/env bash
# Build Stage 3 paper comparison report (requires stage3_results_summary.json).
# Usage: ./scripts/learning/run_build_stage3_comparison_report.sh [--sbatch]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p artifacts/learning_runs logs/learning

if [[ "${1:-}" = "--sbatch" ]]; then
  sbatch batch/learning/build_stage3_comparison_report.sbatch
  exit 0
fi

python -m src.learning.build_stage3_comparison_report \
  --summary_json "$REPO_ROOT/artifacts/learning_runs/stage3_results_summary.json" \
  --out_path "$REPO_ROOT/artifacts/learning_runs/stage3_paper_comparison.md"
