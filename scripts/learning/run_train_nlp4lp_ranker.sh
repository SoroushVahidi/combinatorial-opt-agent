#!/usr/bin/env bash
# Train NLP4LP pairwise ranker.
# Usage: ./scripts/learning/run_train_nlp4lp_ranker.sh [--sbatch] [--run_name NAME] [--encoder NAME]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs

RUN_NAME=run0
ENCODER=distilroberta-base
USE_FEATURES=""
USE_SBATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch) USE_SBATCH=1; shift ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --encoder) ENCODER="$2"; shift 2 ;;
    --use_features) USE_FEATURES=1; shift ;;
    *) shift ;;
  esac
done

if [[ "${USE_SBATCH:-0}" = "1" ]]; then
  export RUN_NAME ENCODER
  [[ -n "$USE_FEATURES" ]] && export USE_FEATURES=1
  sbatch batch/learning/train_nlp4lp_ranker.sbatch
else
  OPTS="--run_name $RUN_NAME --encoder $ENCODER"
  [[ -n "$USE_FEATURES" ]] && OPTS="$OPTS --use_features"
  python -m src.learning.train_nlp4lp_pairwise_ranker $OPTS
fi
