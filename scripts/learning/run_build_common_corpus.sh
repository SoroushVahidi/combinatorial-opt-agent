#!/usr/bin/env bash
# Wrapper to build common corpus (run locally or submit via sbatch).
# Usage: ./scripts/learning/run_build_common_corpus.sh [--sbatch] [--dataset DATASET] [--split SPLIT] [--max_examples N]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_corpus

USE_SBATCH=0
DATASET=all
SPLIT=all
MAX_EXAMPLES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch) USE_SBATCH=1; shift ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --split)   SPLIT="$2"; shift 2 ;;
    --max_examples) MAX_EXAMPLES="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [[ "$USE_SBATCH" = "1" ]]; then
  export DATASET SPLIT
  [[ -n "$MAX_EXAMPLES" ]] && export MAX_EXAMPLES
  sbatch batch/learning/build_common_corpus.sbatch
else
  OPTS="--dataset $DATASET --split $SPLIT --output_dir $REPO_ROOT/artifacts/learning_corpus --seed 42"
  [[ -n "$MAX_EXAMPLES" ]] && OPTS="$OPTS --max_examples $MAX_EXAMPLES"
  python -m src.learning.build_common_grounding_corpus $OPTS
fi
