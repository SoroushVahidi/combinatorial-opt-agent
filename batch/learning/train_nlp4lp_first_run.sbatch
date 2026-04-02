#!/bin/bash
#SBATCH --job-name=nlp4lp_first
#SBATCH --output=logs/learning/train_nlp4lp_first_run_%j.out
#SBATCH --error=logs/learning/train_nlp4lp_first_run_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#
# First learning run: NLP4LP pairwise mention-slot ranker (text-only).
# Task: binary classification of (slot, mention) pairs for number-to-slot grounding.
# Dataset: artifacts/learning_ranker_data/nlp4lp (train.jsonl, test.jsonl).
# Output: artifacts/learning_runs/first_learning_run/checkpoint.pt
#
# Submit: sbatch batch/learning/train_nlp4lp_first_run.sbatch
# Or:     ./scripts/learning/submit_train_first_run.sh
#
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs

# --- Environment: prefer repo venv (has torch/transformers) ---
if [[ -x "$REPO_ROOT/venv/bin/python" ]] && "$REPO_ROOT/venv/bin/python" -c 'import torch' 2>/dev/null; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv: $REPO_ROOT/venv"
elif ! python -c 'import torch' 2>/dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || true
  if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    for env in ${CONDA_ENV:-pt pytorch combinatorial-train base}; do
      if conda activate "$env" 2>/dev/null && python -c 'import torch' 2>/dev/null; then
        echo "Using conda env: $env"
        break
      fi
    done
  fi
fi

# --- Preflight checks ---
echo "=== Preflight diagnostics ==="
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "cwd: $(pwd)"
echo "which python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
python -c 'import torch; print("torch:", torch.__version__)' 2>/dev/null || echo "torch: FAILED"
python -c 'import transformers; print("transformers:", transformers.__version__)' 2>/dev/null || echo "transformers: FAILED"
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || true
nvidia-smi 2>/dev/null || echo "nvidia-smi: not in PATH"
echo "=== End preflight ==="

if ! python -c 'import torch' 2>/dev/null; then
  echo "ERROR: torch required. Activate repo venv or conda env with torch and resubmit."
  exit 1
fi
if ! python -c 'import transformers' 2>/dev/null; then
  echo "ERROR: transformers required."
  exit 1
fi

# --- Data check ---
DATA_DIR="${DATA_DIR:-$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp}"
TRAIN_FILE="$DATA_DIR/train.jsonl"
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "ERROR: Train data not found: $TRAIN_FILE"
  echo "Run: python -m src.learning.build_common_grounding_corpus --dataset nlp4lp --split all"
  echo "Then: python -m src.learning.build_nlp4lp_pairwise_ranker_data"
  exit 1
fi
echo "Train data: $TRAIN_FILE ($(wc -l < "$TRAIN_FILE") lines)"

# --- Training ---
RUN_NAME="${RUN_NAME:-first_learning_run}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/artifacts/learning_runs}"
OUT_DIR="$SAVE_DIR/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "=== Starting training ==="
echo "Run name: $RUN_NAME"
echo "Output dir: $OUT_DIR"
echo "Encoder: distilroberta-base"
echo "Max steps: 200"
echo "Batch size: 8"
echo ""

python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name "$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --encoder distilroberta-base \
  --epochs 1 \
  --max_steps 200 \
  --lr 2e-5 \
  --batch_size 8

echo ""
echo "=== Training complete ==="
ls -la "$OUT_DIR" 2>/dev/null || true
if [[ -f "$OUT_DIR/checkpoint.pt" ]]; then
  echo "SUCCESS: checkpoint.pt written to $OUT_DIR"
else
  echo "WARNING: checkpoint.pt not found; check logs for errors"
fi
