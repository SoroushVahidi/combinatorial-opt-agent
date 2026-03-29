#!/usr/bin/env bash
# run_audit_nlp4lp_bottlenecks.sh — Local runner for the NLP4LP bottleneck audit.
#
# On Wulver (Slurm): use the sbatch script instead.
#   sbatch batch/learning/audit_nlp4lp_bottlenecks.sbatch
#
# Local usage (no Slurm):
#   bash scripts/learning/run_audit_nlp4lp_bottlenecks.sh
#   OUTPUT_BASE=/my/dir bash scripts/learning/run_audit_nlp4lp_bottlenecks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/artifacts/learning_audit}"
LOG_DIR="${REPO_ROOT}/logs/learning"

mkdir -p "${OUTPUT_BASE}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Repo root:   ${REPO_ROOT}"
echo "Output base: ${OUTPUT_BASE}"
echo "========================================"

cd "${REPO_ROOT}"

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
    source "${REPO_ROOT}/venv/bin/activate" && echo "Activated project venv"
fi

echo ""
echo "which python: $(which python 2>/dev/null || echo 'not found')"
python --version 2>&1

LOG_FILE="${LOG_DIR}/audit_nlp4lp_bottlenecks_$(date +%Y%m%dT%H%M%S).log"
echo "Log: ${LOG_FILE}"
echo ""

python src/learning/audit_nlp4lp_bottlenecks.py \
    --out "${OUTPUT_BASE}" \
    2>&1 | tee "${LOG_FILE}"
EXIT_CODE="${PIPESTATUS[0]}"

echo ""
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "Bottleneck audit PASSED. Outputs: ${OUTPUT_BASE}  Log: ${LOG_FILE}"
else
    echo "Bottleneck audit FAILED (exit ${EXIT_CODE}). Log: ${LOG_FILE}"
fi
exit "${EXIT_CODE}"
