#!/bin/bash
#SBATCH --job-name=train_nlp4lp_ranker
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --time=2:00:00

# train_nlp4lp_ranker.sbatch — Fine-tune the NLP4LP pairwise text ranker.
#
# Submit from repo root:
#   sbatch batch/learning/train_nlp4lp_ranker.sbatch
#
# Override the run name:
#   RUN_NAME=nlp4lp_pairwise_text_only sbatch batch/learning/train_nlp4lp_ranker.sbatch
#
# Outputs:
#   artifacts/runs/<RUN_NAME>/checkpoint/
#   artifacts/runs/<RUN_NAME>/metrics.json
#   artifacts/runs/<RUN_NAME>/metrics.md

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/artifacts/runs}"
RUN_NAME="${RUN_NAME:-nlp4lp_pairwise_text_only}"
RUN_DIR="${OUTPUT_BASE}/${RUN_NAME}"

mkdir -p "${RUN_DIR}"

echo "========================================"
echo "SLURM job:   ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Repo root:   ${REPO_ROOT}"
echo "Run dir:     ${RUN_DIR}"
echo "Run name:    ${RUN_NAME}"
echo "========================================"

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 2. Environment activation
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
# 3. Fail early if required packages are missing
# ---------------------------------------------------------------------------
echo ""
echo "--- Pre-flight checks ---"
echo "which python: $(which python)"
python --version

python -c "import torch; print('torch:', torch.__version__)" \
    || { echo "ERROR: torch not importable. Activate an env with torch."; exit 1; }
python -c "import transformers; print('transformers:', transformers.__version__)" \
    || { echo "ERROR: transformers not importable."; exit 1; }
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# 4. Check required input data
# ---------------------------------------------------------------------------
echo ""
echo "--- Input data check ---"
DATA_DIR="${REPO_ROOT}/artifacts/learning_ranker_data/nlp4lp"
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Input data directory not found: ${DATA_DIR}"
    echo "Run the data-preparation step first (training/generate_mention_slot_pairs.py or equivalent)."
    exit 1
fi
echo "Input data: ${DATA_DIR} — OK"

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
echo ""
echo "--- Training: ${RUN_NAME} ---"
python -u src/learning/train_nlp4lp_pairwise_ranker.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${RUN_DIR}" \
    --run-name "${RUN_NAME}" \
    2>&1 | tee "${RUN_DIR}/train.log"
TRAIN_EXIT="${PIPESTATUS[0]}"

# ---------------------------------------------------------------------------
# 6. Post-training evaluation
# ---------------------------------------------------------------------------
if [ "${TRAIN_EXIT}" -eq 0 ]; then
    echo ""
    echo "--- Post-training evaluation ---"
    python -u src/learning/eval_nlp4lp_pairwise_ranker.py \
        --run-dir "${RUN_DIR}" 2>&1 | tee -a "${RUN_DIR}/train.log" || true
    python -u src/learning/eval_bottleneck_slices.py \
        --run-dir "${RUN_DIR}" 2>&1 | tee -a "${RUN_DIR}/train.log" || true
fi

echo ""
if [ "${TRAIN_EXIT}" -eq 0 ]; then
    echo "Training SUCCEEDED. Outputs: ${RUN_DIR}"
else
    echo "Training FAILED (exit ${TRAIN_EXIT}). See ${RUN_DIR}/train.log"
fi
exit "${TRAIN_EXIT}"
