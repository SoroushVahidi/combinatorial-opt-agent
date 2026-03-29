#!/usr/bin/env bash
# Validate and summarize learning corpus.
# Usage: ./scripts/learning/run_validate_corpus.sh [--sbatch]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/learning

if [[ "${1:-}" = "--sbatch" ]]; then
  sbatch batch/learning/validate_common_corpus.sbatch
else
  python -m src.learning.validate_common_grounding_corpus --corpus_dir "$REPO_ROOT/artifacts/learning_corpus" --verbose
  python -m src.learning.summarize_common_grounding_corpus --corpus_dir "$REPO_ROOT/artifacts/learning_corpus"
fi
