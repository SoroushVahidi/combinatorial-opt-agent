#!/usr/bin/env bash
# Run training environment check (local or sbatch).
# Usage: ./scripts/learning/run_check_training_env.sh [local|sbatch]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/learning

MODE="${1:-local}"
if [[ "$MODE" = "sbatch" ]]; then
  sbatch batch/learning/check_training_env.sbatch
  exit 0
fi

# Local: use venv if available
if [[ -x "$REPO_ROOT/venv/bin/python" ]] && "$REPO_ROOT/venv/bin/python" -c 'import torch' 2>/dev/null; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv"
fi
python -m src.learning.check_training_env
