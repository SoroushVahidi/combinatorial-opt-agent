#!/usr/bin/env bash
# Build NL4Opt auxiliary data (entity, bound, role tasks).
# Usage: ./scripts/learning/run_build_nl4opt_aux_data.sh [--sbatch]
# With limit: MAX_EXAMPLES=100 ./scripts/learning/run_build_nl4opt_aux_data.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p artifacts/learning_aux_data/nl4opt

if [[ "${1:-}" = "--sbatch" ]]; then
  sbatch batch/learning/build_nl4opt_aux_data.sbatch
  exit 0
fi

OPTS="--output_dir $REPO_ROOT/artifacts/learning_aux_data/nl4opt --seed 42"
[[ -n "${MAX_EXAMPLES:-}" ]] && OPTS="$OPTS --max_examples $MAX_EXAMPLES"

python -m src.learning.build_nl4opt_aux_data $OPTS
