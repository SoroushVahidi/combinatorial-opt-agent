#!/bin/bash
#SBATCH --job-name=train_retrieval
#SBATCH --output=train_retrieval_%j.out
#SBATCH --error=train_retrieval_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

# Fine-tune the retrieval model on synthetic (query, passage) samples. Uses GPU.
# Usage: sbatch scripts/train_retrieval_gpu.slurm
# Then use the saved model in data/models/retrieval_finetuned/final

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HUB_DISABLE_XET=1

module load python/3.10 2>/dev/null || true
if [ -d "venv" ]; then
  source venv/bin/activate
else
  source "${HOME}/.local/bin/activate" 2>/dev/null || true
fi

echo "Installing training deps (datasets, accelerate)..."
pip install --no-cache-dir datasets 'accelerate>=1.1.0' --quiet
python -c "import accelerate; print('accelerate', accelerate.__version__)"

echo "Generating training samples (100 instances per problem)..."
python -m training.generate_samples --output data/processed/training_pairs.jsonl --instances-per-problem 100

echo "Starting fine-tuning (GPU, batch_size=16 for stability)..."
python -u -m training.train_retrieval \
  --data data/processed/training_pairs.jsonl \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir data/models/retrieval_finetuned \
  --epochs 2 \
  --batch-size 16

echo "Done. Model saved to data/models/retrieval_finetuned/final"
