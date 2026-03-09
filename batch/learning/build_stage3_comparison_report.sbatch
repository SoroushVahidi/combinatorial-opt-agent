#!/bin/bash
#SBATCH --job-name=stage3_report
#SBATCH --output=logs/learning/build_stage3_comparison_report_%j.out
#SBATCH --error=logs/learning/build_stage3_comparison_report_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=1G
set -euo pipefail

# Build Stage 3 paper comparison report (requires stage3_results_summary.json).
# Example: sbatch batch/learning/build_stage3_comparison_report.sbatch
# Run after collect_stage3_results.py.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs

SUMMARY_JSON="${SUMMARY_JSON:-$REPO_ROOT/artifacts/learning_runs/stage3_results_summary.json}"
OUT_PATH="${OUT_PATH:-$REPO_ROOT/artifacts/learning_runs/stage3_paper_comparison.md}"

python -m src.learning.build_stage3_comparison_report \
  --summary_json "$SUMMARY_JSON" \
  --out_path "$OUT_PATH"
