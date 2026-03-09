#!/bin/bash
#SBATCH --job-name=learn_eval
#SBATCH --output=logs/learning/eval_nlp4lp_ranker_%j.out
#SBATCH --error=logs/learning/eval_nlp4lp_ranker_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
set -euo pipefail

# Evaluate NLP4LP pairwise ranker (uses rule baseline if no checkpoint).
# Example: sbatch batch/learning/eval_nlp4lp_ranker.sbatch
# With checkpoint: RUN_DIR=artifacts/learning_runs/run0 sbatch batch/learning/eval_nlp4lp_ranker.sbatch

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning

DATA_DIR="${DATA_DIR:-$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp}"
RUN_DIR="${RUN_DIR:-}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/artifacts/learning_runs/eval_out}"
SPLIT="${SPLIT:-test}"
DECODER="${DECODER:-argmax}"

OPTS="--data_dir $DATA_DIR --split $SPLIT --decoder $DECODER --out_dir $OUT_DIR"
[[ -n "$RUN_DIR" ]] && OPTS="$OPTS --run_dir $RUN_DIR"

python -m src.learning.eval_nlp4lp_pairwise_ranker $OPTS
