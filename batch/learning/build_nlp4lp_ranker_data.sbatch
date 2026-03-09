#!/bin/bash
#SBATCH --job-name=learn_ranker_data
#SBATCH --output=logs/learning/build_nlp4lp_ranker_data_%j.out
#SBATCH --error=logs/learning/build_nlp4lp_ranker_data_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=2G
set -euo pipefail

# Build NLP4LP pairwise ranker data from common corpus.
# Example: sbatch batch/learning/build_nlp4lp_ranker_data.sbatch

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_ranker_data/nlp4lp

CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/artifacts/learning_corpus}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp}"
SEED="${SEED:-42}"

python -m src.learning.build_nlp4lp_pairwise_ranker_data \
  --corpus_dir "$CORPUS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED"
