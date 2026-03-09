#!/bin/bash
#SBATCH --job-name=train_retrieval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

# Fine-tune the retrieval model on synthetic (query, passage) samples. Uses GPU.
# Usage: sbatch scripts/train_retrieval_gpu.slurm
# Then use the saved model in data/models/retrieval_finetuned/final

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/data/models}"
mkdir -p "${OUTPUT_BASE}"

echo "========================================"
echo "SLURM job:   ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Repo root:   ${REPO_ROOT}"
echo "========================================"

cd "${REPO_ROOT}"
export HF_HUB_DISABLE_XET=1

# ---------------------------------------------------------------------------
# Environment activation (robust: conda env → venv → module)
# ---------------------------------------------------------------------------
echo ""
echo "--- Activating environment ---"

_activate_conda() {
    local env_name="$1"
    if conda env list 2>/dev/null | grep -q "^${env_name} "; then
        eval "$(conda shell.bash hook 2>/dev/null)" || true
        conda activate "${env_name}" && echo "Activated conda env: ${env_name}" && return 0
    fi
    return 1
}

ENV_ACTIVATED=0

if [ -n "${CONDA_ENV:-}" ]; then
    _activate_conda "${CONDA_ENV}" && ENV_ACTIVATED=1 || true
fi

if [ "${ENV_ACTIVATED}" -eq 0 ]; then
    for env in combinatorial-train pt pytorch; do
        _activate_conda "${env}" && ENV_ACTIVATED=1 && break || true
    done
fi

if [ "${ENV_ACTIVATED}" -eq 0 ] && [ -d "${REPO_ROOT}/venv" ]; then
    source "${REPO_ROOT}/venv/bin/activate" && echo "Activated project venv" && ENV_ACTIVATED=1
fi

if [ "${ENV_ACTIVATED}" -eq 0 ]; then
    module load python/3.10 2>/dev/null && echo "Loaded module python/3.10" || true
fi

# ---------------------------------------------------------------------------
# Pre-flight checks — fail early if packages are missing
# ---------------------------------------------------------------------------
echo ""
echo "--- Pre-flight checks ---"
echo "which python: $(which python)"
python --version

python -c "import torch; print('torch:', torch.__version__)" \
    || { echo "ERROR: torch not importable. Activate an env with torch."; exit 1; }
python -c "import sentence_transformers; print('sentence_transformers:', sentence_transformers.__version__)" \
    || { echo "ERROR: sentence_transformers not importable."; exit 1; }
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# Install runtime deps if needed
# ---------------------------------------------------------------------------
echo ""
echo "Installing training deps (datasets, accelerate)..."
pip install --no-cache-dir datasets 'accelerate>=1.1.0' --quiet
python -c "import accelerate; print('accelerate', accelerate.__version__)"

# ---------------------------------------------------------------------------
# Generate training samples and train
# ---------------------------------------------------------------------------
echo ""
echo "Generating training samples (extended catalog, 100 instances per problem)..."
python -m training.generate_samples --output data/processed/training_pairs.jsonl --instances-per-problem 100

echo ""
echo "Starting fine-tuning (GPU, regularization + val-based best model)..."
python -u -m training.train_retrieval \
  --data data/processed/training_pairs.jsonl \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir "${OUTPUT_BASE}/retrieval_finetuned" \
  --epochs 4 \
  --batch-size 16 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --val-ratio 0.1

echo "Done. Model saved to ${OUTPUT_BASE}/retrieval_finetuned/final (best by val loss)"

