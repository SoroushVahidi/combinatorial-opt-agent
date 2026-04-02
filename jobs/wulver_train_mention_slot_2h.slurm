#!/bin/bash
#SBATCH --job-name=mention_slot
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/mention_slot_%%j.out
#SBATCH --error=logs/mention_slot_%%j.err

set -euo pipefail

# Repo path on Wulver (GPFS). Override with PROJECT_ROOT if needed.
PROJECT_ROOT="${PROJECT_ROOT:-/mmfs1/home/sv96/combinatorial-opt-agent}"
if [[ ! -d "$PROJECT_ROOT" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH:-}"

# Prefer repo venv (has torch/transformers); else try conda env.
if [[ -x "$PROJECT_ROOT/venv/bin/python" ]] && "$PROJECT_ROOT/venv/bin/python" -c 'import torch' 2>/dev/null; then
  export PATH="$PROJECT_ROOT/venv/bin:${PATH}"
  echo "Using project venv: $PROJECT_ROOT/venv"
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
hostname
date
pwd
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || true
nvidia-smi 2>/dev/null || echo "nvidia-smi not found"
echo "=== End diagnostics ==="

mkdir -p logs
mkdir -p results/mention_slot_scorer

DATA_DIR="$PROJECT_ROOT/data/processed"
PAIRS="$DATA_DIR/mention_slot_pairs.jsonl"
PAIRS_DEV="$DATA_DIR/mention_slot_pairs_dev.jsonl"

if [[ ! -f "$PAIRS" ]]; then
  echo "Generating mention-slot pairs..."
  python -m training.generate_mention_slot_pairs --eval "$DATA_DIR/nlp4lp_eval_orig.jsonl" --out "$PAIRS" || {
    echo "Data generation failed. Running smoke test."
    python -m training.train_mention_slot_scorer --smoke --output-dir "$PROJECT_ROOT/results/mention_slot_scorer"
    exit 0
  }
fi

if ! python -c 'import torch' 2>/dev/null; then
  echo "PyTorch not available after conda attempts. Running smoke test only."
  python -m training.train_mention_slot_scorer --smoke --output-dir "$PROJECT_ROOT/results/mention_slot_scorer" || true
  echo "Done (smoke only). Install torch in your conda env or set CONDA_ENV=your_env and resubmit."
  exit 0
fi

echo "Starting training..."
python -m training.train_mention_slot_scorer \
  --data "$PAIRS" \
  --data-dev "$PAIRS_DEV" \
  --output-dir "$PROJECT_ROOT/results/mention_slot_scorer" \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5

echo "Done."
