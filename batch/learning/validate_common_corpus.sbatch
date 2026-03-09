#!/bin/bash
#SBATCH --job-name=learn_validate
#SBATCH --output=logs/learning/validate_common_corpus_%j.out
#SBATCH --error=logs/learning/validate_common_corpus_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=2G
set -euo pipefail

# Validate and summarize learning corpus.
# Example: sbatch batch/learning/validate_common_corpus.sbatch

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning

CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/artifacts/learning_corpus}"

python -m src.learning.validate_common_grounding_corpus --corpus_dir "$CORPUS_DIR" --verbose
python -m src.learning.summarize_common_grounding_corpus --corpus_dir "$CORPUS_DIR"
