#!/usr/bin/env bash
# Submit the first VALID NLP4LP learning run (clean train/dev/test, no test-as-train).
# Usage: ./scripts/learning/submit_valid_first_run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs artifacts/learning_corpus artifacts/learning_ranker_data/nlp4lp

echo "Submitting valid first learning run (benchmark-safe splits)..."
JOB_ID=$(sbatch batch/learning/train_nlp4lp_valid_first_run.sbatch | awk '{print $4}')
echo "Submitted job: $JOB_ID"
echo "Logs: logs/learning/train_nlp4lp_valid_first_run_${JOB_ID}.out"
echo "Output: artifacts/learning_runs/valid_first_run/"
echo "Record: docs/learning_runs/valid_first_learning_run_record.md"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel: scancel $JOB_ID"
