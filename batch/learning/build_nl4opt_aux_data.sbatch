#!/bin/bash
#SBATCH --job-name=nl4opt_aux
#SBATCH --output=logs/learning/build_nl4opt_aux_data_%j.out
#SBATCH --error=logs/learning/build_nl4opt_aux_data_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
set -euo pipefail

# Build NL4Opt auxiliary training data (entity, bound, role tasks).
# Example: sbatch batch/learning/build_nl4opt_aux_data.sbatch
# With limit: MAX_EXAMPLES=100 sbatch batch/learning/build_nl4opt_aux_data.sbatch

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_aux_data/nl4opt

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/learning_aux_data/nl4opt}"
SEED="${SEED:-42}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

OPTS="--output_dir $OUTPUT_DIR --seed $SEED"
[[ -n "$MAX_EXAMPLES" ]] && OPTS="$OPTS --max_examples $MAX_EXAMPLES"

python -m src.learning.build_nl4opt_aux_data $OPTS
