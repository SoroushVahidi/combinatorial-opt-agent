#!/usr/bin/env bash
# Build NLP4LP pairwise ranker data from corpus.
# Usage: ./scripts/learning/run_build_nlp4lp_ranker_data.sh [--sbatch]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p artifacts/learning_ranker_data/nlp4lp

if [[ "${1:-}" = "--sbatch" ]]; then
  sbatch batch/learning/build_nlp4lp_ranker_data.sbatch
else
  python -m src.learning.build_nlp4lp_pairwise_ranker_data \
    --corpus_dir "$REPO_ROOT/artifacts/learning_corpus" \
    --output_dir "$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp" \
    --seed 42
fi
