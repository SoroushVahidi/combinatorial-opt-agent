#!/usr/bin/env bash
# Submit the first NLP4LP learning run to SLURM.
# Usage: ./scripts/learning/submit_train_first_run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs

echo "Submitting first learning run..."
JOB_ID=$(sbatch batch/learning/train_nlp4lp_first_run.sbatch | awk '{print $4}')
echo "Submitted job: $JOB_ID"
echo "Logs: logs/learning/train_nlp4lp_first_run_${JOB_ID}.out"
echo "Checkpoint: artifacts/learning_runs/first_learning_run/checkpoint.pt"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel: scancel $JOB_ID"
