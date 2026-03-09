#!/bin/bash
#SBATCH --job-name=stage3_experiments
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
#SBATCH --time=8:00:00

# run_stage3_experiments.sbatch — Orchestrate the full Stage-3 experiment matrix.
#
# Reads configs/learning/experiment_matrix_stage3.json and runs each listed
# experiment (pairwise ranker variants + multitask grounder variants).
#
# Submit from repo root:
#   sbatch batch/learning/run_stage3_experiments.sbatch
#
# To run only specific experiments, set a comma-separated filter:
#   EXPERIMENTS=nlp4lp_pairwise_text_only sbatch batch/learning/run_stage3_experiments.sbatch

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/artifacts/runs}"
EXPERIMENT_MATRIX="${REPO_ROOT}/configs/learning/experiment_matrix_stage3.json"
EXPERIMENT_FILTER="${EXPERIMENTS:-}"   # comma-separated; empty = run all

mkdir -p "${OUTPUT_BASE}"
LOG_DIR="${REPO_ROOT}/logs/learning"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "SLURM job:          ${SLURM_JOB_ID:-local}"
echo "Node:               $(hostname)"
echo "Repo root:          ${REPO_ROOT}"
echo "Output base:        ${OUTPUT_BASE}"
echo "Experiment matrix:  ${EXPERIMENT_MATRIX}"
echo "Filter:             ${EXPERIMENT_FILTER:-<all>}"
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
# 4. Confirm experiment matrix exists
# ---------------------------------------------------------------------------
echo ""
echo "--- Experiment matrix ---"
if [ ! -f "${EXPERIMENT_MATRIX}" ]; then
    echo "ERROR: Experiment matrix not found: ${EXPERIMENT_MATRIX}"
    exit 1
fi
echo "Matrix: ${EXPERIMENT_MATRIX}"
python -c "
import json, sys
with open('${EXPERIMENT_MATRIX}') as f:
    m = json.load(f)
exps = m.get('experiments', [])
print(f'  Total experiments: {len(exps)}')
for e in exps:
    print(f'    - {e[\"name\"]} (type={e.get(\"type\",\"?\")})')
"

# ---------------------------------------------------------------------------
# 5. Run experiment matrix
# ---------------------------------------------------------------------------
echo ""
echo "--- Running experiments ---"
FILTER_ARG=""
if [ -n "${EXPERIMENT_FILTER}" ]; then
    FILTER_ARG="--filter ${EXPERIMENT_FILTER}"
fi
bash scripts/learning/run_stage3_experiments.sh \
    --matrix "${EXPERIMENT_MATRIX}" \
    --output-base "${OUTPUT_BASE}" \
    ${FILTER_ARG} \
    2>&1 | tee "${LOG_DIR}/stage3_${SLURM_JOB_ID:-local}.log"
STAGE3_EXIT="${PIPESTATUS[0]}"

echo ""
if [ "${STAGE3_EXIT}" -eq 0 ]; then
    echo "Stage-3 experiments COMPLETED. Outputs: ${OUTPUT_BASE}"
else
    echo "Stage-3 experiments FAILED (exit ${STAGE3_EXIT}). See logs."
fi
exit "${STAGE3_EXIT}"
