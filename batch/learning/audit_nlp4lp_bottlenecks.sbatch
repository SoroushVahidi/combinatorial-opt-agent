#!/bin/bash
#SBATCH --job-name=audit_nlp4lp_bottlenecks
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=low
#SBATCH --account=ikoutis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00

# audit_nlp4lp_bottlenecks.sbatch — CPU-only bottleneck audit for NLP4LP grounding.
#
# Submit from repo root:
#   sbatch batch/learning/audit_nlp4lp_bottlenecks.sbatch
#
# Override output directory:
#   OUTPUT_BASE=/path/to/outputs sbatch batch/learning/audit_nlp4lp_bottlenecks.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/artifacts/learning_audit}"
LOG_DIR="${REPO_ROOT}/logs/learning"

mkdir -p "${OUTPUT_BASE}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "SLURM job:   ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Repo root:   ${REPO_ROOT}"
echo "Output base: ${OUTPUT_BASE}"
echo "========================================"

cd "${REPO_ROOT}"

# --- Optional environment activation ---
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

echo "which python: $(which python)"
python --version

LOG_FILE="${LOG_DIR}/audit_nlp4lp_bottlenecks_${SLURM_JOB_ID:-$(date +%Y%m%dT%H%M%S)}.log"
echo "Log: ${LOG_FILE}"

python src/learning/audit_nlp4lp_bottlenecks.py \
    --out "${OUTPUT_BASE}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Bottleneck audit complete. Outputs: ${OUTPUT_BASE}"
