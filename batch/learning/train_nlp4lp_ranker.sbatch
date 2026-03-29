#!/bin/bash
#SBATCH --job-name=learn_train
#SBATCH --output=logs/learning/train_nlp4lp_ranker_%j.out
#SBATCH --error=logs/learning/train_nlp4lp_ranker_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
set -euo pipefail

# Train NLP4LP pairwise ranker.
# Example: sbatch batch/learning/train_nlp4lp_ranker.sbatch
# With env: RUN_NAME=run1 ENCODER=roberta-base sbatch batch/learning/train_nlp4lp_ranker.sbatch

REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs

# Activate env with torch (repo venv first, else conda)
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

echo "=== Environment diagnostics ==="
echo "which python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
python -c 'import torch; print("torch:", torch.__version__)' 2>/dev/null || echo "torch: FAILED TO IMPORT"
python -c 'import transformers; print("transformers:", transformers.__version__)' 2>/dev/null || echo "transformers: FAILED TO IMPORT"
python -c 'import torch; print("torch.cuda.is_available():", torch.cuda.is_available())' 2>/dev/null || true
echo "=== End diagnostics ==="

if ! python -c 'import torch' 2>/dev/null; then
  echo "ERROR: torch required. Activate repo venv or a conda env with torch and resubmit."
  exit 1
fi
if ! python -c 'import transformers' 2>/dev/null; then
  echo "ERROR: transformers required. Activate an env with transformers and resubmit."
  exit 1
fi

RUN_NAME="${RUN_NAME:-run0}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/artifacts/learning_runs}"
ENCODER="${ENCODER:-distilroberta-base}"
USE_FEATURES="${USE_FEATURES:-0}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
LR="${LR:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"

OPTS="--run_name $RUN_NAME --data_dir $DATA_DIR --save_dir $SAVE_DIR --encoder $ENCODER --epochs $EPOCHS --max_steps $MAX_STEPS --lr $LR --batch_size $BATCH_SIZE"
[[ "$USE_FEATURES" = "1" ]] && OPTS="$OPTS --use_features"

python -m src.learning.train_nlp4lp_pairwise_ranker $OPTS
