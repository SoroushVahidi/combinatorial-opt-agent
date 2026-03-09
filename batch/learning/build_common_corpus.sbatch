#!/bin/bash
#SBATCH --job-name=learn_corpus
#SBATCH --output=logs/learning/build_common_corpus_%j.out
#SBATCH --error=logs/learning/build_common_corpus_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
set -euo pipefail

# Build common learning corpus (NLP4LP, NL4Opt, TAT-QA, FinQA).
# Example: sbatch batch/learning/build_common_corpus.sbatch
# With env: DATASET=all SPLIT=all MAX_EXAMPLES=500 sbatch batch/learning/build_common_corpus.sbatch

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_corpus

DATASET="${DATASET:-all}"
SPLIT="${SPLIT:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/learning_corpus}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
SEED="${SEED:-42}"
VERBOSE="${VERBOSE:-0}"

OPTS="--dataset $DATASET --split $SPLIT --output_dir $OUTPUT_DIR --seed $SEED"
[[ -n "$MAX_EXAMPLES" ]] && OPTS="$OPTS --max_examples $MAX_EXAMPLES"
[[ "$VERBOSE" = "1" ]] && OPTS="$OPTS --verbose"

python -m src.learning.build_common_grounding_corpus $OPTS
