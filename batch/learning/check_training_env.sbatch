#!/bin/bash
#SBATCH --job-name=check_train_env
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00

# check_training_env.sbatch — Verify the training environment on a GPU compute node.
#
# Submit from repo root:
#   sbatch batch/learning/check_training_env.sbatch
#
# Outputs are written to the job log.  A non-zero exit code means the environment
# is not ready for training.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Repo root and writable output base
# ---------------------------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/logs/learning}"
mkdir -p "${OUTPUT_BASE}"

echo "========================================"
echo "SLURM job:       ${SLURM_JOB_ID:-local}"
echo "Node:            $(hostname)"
echo "Repo root:       ${REPO_ROOT}"
echo "Output base:     ${OUTPUT_BASE}"
echo "========================================"

# ---------------------------------------------------------------------------
# 2. Change to repo root
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 3. Activate environment
#    Strategy (mirrors train_retrieval_gpu.slurm pattern):
#      a) try CONDA_ENV if set
#      b) try conda env named 'pt', 'pytorch', or 'combinatorial-train'
#      c) fall back to project venv
#      d) fall back to module load
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

# a) User-supplied env
if [ -n "${CONDA_ENV:-}" ]; then
    _activate_conda "${CONDA_ENV}" && ENV_ACTIVATED=1
fi

# b) Known env names
if [ "${ENV_ACTIVATED}" -eq 0 ]; then
    for env in combinatorial-train pt pytorch; do
        _activate_conda "${env}" && ENV_ACTIVATED=1 && break || true
    done
fi

# c) Project venv
if [ "${ENV_ACTIVATED}" -eq 0 ] && [ -d "${REPO_ROOT}/venv" ]; then
    source "${REPO_ROOT}/venv/bin/activate" && echo "Activated project venv" && ENV_ACTIVATED=1
fi

# d) Module system
if [ "${ENV_ACTIVATED}" -eq 0 ]; then
    module load python/3.10 2>/dev/null && echo "Loaded module python/3.10" || true
fi

# ---------------------------------------------------------------------------
# 4. Print basic Python diagnostics
# ---------------------------------------------------------------------------
echo ""
echo "--- Python diagnostics ---"
echo "which python: $(which python 2>/dev/null || echo 'not found')"
python --version 2>&1 || { echo "ERROR: python not found on PATH"; exit 1; }

echo ""
echo "--- Package checks ---"
python -c "import torch; print('torch:', torch.__version__)" \
    || echo "WARN: torch not importable"
python -c "import transformers; print('transformers:', transformers.__version__)" \
    || echo "WARN: transformers not importable"
python -c "import torch; print('cuda available:', torch.cuda.is_available())" \
    || echo "WARN: could not check CUDA (torch not importable)"

# ---------------------------------------------------------------------------
# 5. Run the full environment check script
# ---------------------------------------------------------------------------
echo ""
echo "--- Full environment check ---"
python src/learning/check_training_env.py
CHECK_EXIT=$?

# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
echo ""
if [ "${CHECK_EXIT}" -eq 0 ]; then
    echo "Environment check PASSED. Ready for training."
else
    echo "Environment check FAILED (exit ${CHECK_EXIT}). See above for details."
    echo "Fix the environment before submitting training jobs."
fi
exit "${CHECK_EXIT}"
