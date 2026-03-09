#!/usr/bin/env bash
# Evaluate NLP4LP pairwise ranker.
# Usage: ./scripts/learning/run_eval_nlp4lp_ranker.sh [--sbatch] [--run_dir PATH]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p artifacts/learning_runs/eval_out

RUN_DIR=""
USE_SBATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch) USE_SBATCH=1; shift ;;
    --run_dir) RUN_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [[ "$USE_SBATCH" = "1" ]]; then
  [[ -n "$RUN_DIR" ]] && export RUN_DIR
  sbatch batch/learning/eval_nlp4lp_ranker.sbatch
else
  OPTS="--split test"
  [[ -n "$RUN_DIR" ]] && OPTS="$OPTS --run_dir $RUN_DIR"
  python -m src.learning.eval_nlp4lp_pairwise_ranker $OPTS
fi
