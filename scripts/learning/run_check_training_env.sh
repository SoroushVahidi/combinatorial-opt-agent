#!/usr/bin/env bash
# run_check_training_env.sh — Local runner for the training environment check.
#
# On Wulver (Slurm): use the sbatch script instead.
#   sbatch batch/learning/check_training_env.sbatch
#
# Local usage (no Slurm):
#   bash scripts/learning/run_check_training_env.sh
#   CONDA_ENV=my_env bash scripts/learning/run_check_training_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/logs/learning}"

mkdir -p "${OUTPUT_BASE}"

echo "========================================"
echo "Repo root:   ${REPO_ROOT}"
echo "Output base: ${OUTPUT_BASE}"
echo "========================================"

cd "${REPO_ROOT}"

# --- Optional environment activation (mirrors sbatch script) ---
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
    source "${REPO_ROOT}/venv/bin/activate"
    echo "Activated project venv"
fi

echo ""
echo "which python: $(which python 2>/dev/null || echo 'not found')"
python --version 2>&1

LOG_FILE="${OUTPUT_BASE}/check_training_env_$(date +%Y%m%dT%H%M%S).log"
echo ""
echo "Running check (output also tee'd to ${LOG_FILE}) ..."
python src/learning/check_training_env.py 2>&1 | tee "${LOG_FILE}"
EXIT_CODE="${PIPESTATUS[0]}"

echo ""
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "Environment check PASSED. Log: ${LOG_FILE}"
else
    echo "Environment check FAILED (exit ${EXIT_CODE}). Log: ${LOG_FILE}"
fi
exit "${EXIT_CODE}"
