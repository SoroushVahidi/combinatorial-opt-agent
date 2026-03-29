#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --output=logs/learning/check_training_env_%j.out
#SBATCH --error=logs/learning/check_training_env_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning

# Activate env with torch (same pattern as run_stage3_experiments)
if [[ -x "$REPO_ROOT/venv/bin/python" ]] && "$REPO_ROOT/venv/bin/python" -c 'import torch' 2>/dev/null; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv: $REPO_ROOT/venv"
elif ! python -c 'import torch' 2>/dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || true
  if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    for env in ${CONDA_ENV:-pt pytorch base}; do
      if conda activate "$env" 2>/dev/null && python -c 'import torch' 2>/dev/null; then
        echo "Using conda env: $env"
        break
      fi
    done
  fi
fi

echo "which python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
python -m src.learning.check_training_env
echo "Exit code: $?"
