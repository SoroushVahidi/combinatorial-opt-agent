#!/usr/bin/env bash
# Create a clean conda env with PyTorch + sentence-transformers and run retrieval training locally.
# Usage: bash scripts/setup_and_train_local.sh
# Then activate the env for future runs: conda activate combinatorial-train
#
# Note: Training requires PyTorch 2.4+ (for torch.uint16). On macOS only 2.2 may be available;
# in that case run training on Linux, Wulver (sbatch scripts/train_retrieval_gpu.slurm), or Colab.

set -e
cd "$(dirname "$0")/.."
ENV_NAME="${CONDA_TRAIN_ENV:-combinatorial-train}"

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Conda env '$ENV_NAME' already exists. Activating and running training."
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "Creating conda env '$ENV_NAME' with Python 3.10..."
  conda create -n "$ENV_NAME" python=3.10 -y
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  echo "Installing PyTorch (CPU), sentence-transformers, and training deps..."
  pip install "numpy<2" --quiet
  pip install torch sentence-transformers datasets "accelerate>=1.1.0" --quiet
fi
# Ensure NumPy <2 for PyTorch 2.2 compatibility (in case env was created before we pinned)
pip install "numpy<2" --quiet 2>/dev/null || true
# Allow PyTorch 2.2 in this env (transformers 2.4 check breaks on some macOS installs)
IMP=$(python -c "import transformers; print(transformers.__path__[0])" 2>/dev/null) || true
if [ -n "$IMP" ] && [ -f "$IMP/utils/import_utils.py" ]; then
  if grep -q '2.4.0' "$IMP/utils/import_utils.py" 2>/dev/null; then
    sed -i.bak 's/version.parse("2.4.0")/version.parse("2.0.0")/g; s/PyTorch >= 2.4 is required/PyTorch >= 2.0 is required/g' "$IMP/utils/import_utils.py"
    echo "Patched transformers to accept PyTorch 2.2."
  fi
fi

echo "Generating training pairs (30 per problem)..."
python -m training.generate_samples --output data/processed/training_pairs.jsonl --instances-per-problem 30

echo "Training retrieval model (CPU, 800 steps)..."
python -m training.train_retrieval \
  --data data/processed/training_pairs.jsonl \
  --output-dir data/models/retrieval_finetuned \
  --epochs 4 --batch-size 8 \
  --weight-decay 0.01 --warmup-ratio 0.1 --val-ratio 0.1 \
  --max-steps 800

echo "Done. Model saved to data/models/retrieval_finetuned/final"
echo "Use it by running the app from this repo (it auto-loads the fine-tuned model when present)."
