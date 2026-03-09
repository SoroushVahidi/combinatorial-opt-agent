#!/usr/bin/env bash
# run_stage3_experiments.sh — Submit or run all Stage-3 learning experiments.
#
# On Wulver (Slurm) — submit the full matrix as a single GPU job:
#   sbatch batch/learning/run_stage3_experiments.sbatch
#
# Local sequential run (no GPU, for smoke-testing):
#   bash scripts/learning/run_stage3_experiments.sh --dry-run
#
# Options:
#   --dry-run          Print experiment list without running.
#   --filter NAME,...  Comma-separated list of experiment names to run.
#   --matrix PATH      Path to experiment matrix JSON (default: configs/learning/experiment_matrix_stage3.json).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/artifacts/runs}"
MATRIX="${REPO_ROOT}/configs/learning/experiment_matrix_stage3.json"
FILTER=""
DRY_RUN=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1; shift ;;
        --filter)      FILTER="$2"; shift 2 ;;
        --matrix)      MATRIX="$2"; shift 2 ;;
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "${REPO_ROOT}"

echo "========================================"
echo "Repo root:   ${REPO_ROOT}"
echo "Output base: ${OUTPUT_BASE}"
echo "Matrix:      ${MATRIX}"
echo "Filter:      ${FILTER:-<all>}"
echo "Dry run:     ${DRY_RUN}"
echo "========================================"

if [ ! -f "${MATRIX}" ]; then
    echo "ERROR: Experiment matrix not found: ${MATRIX}"
    exit 1
fi

# Read experiment names from matrix JSON
EXPERIMENT_NAMES=$(python3 -c "
import json
with open('${MATRIX}') as f:
    m = json.load(f)
for e in m.get('experiments', []):
    print(e['name'])
")

echo ""
echo "Experiments in matrix:"
echo "${EXPERIMENT_NAMES}" | while read -r name; do echo "  - ${name}"; done

if [ "${DRY_RUN}" -eq 1 ]; then
    echo ""
    echo "Dry run: not executing any experiments."
    exit 0
fi

# Run each experiment
FAILED=()
SUCCESS=()
echo ""
echo "--- Running experiments ---"
while IFS= read -r exp_name; do
    # Apply filter if set
    if [ -n "${FILTER}" ]; then
        if ! echo "${FILTER}" | tr ',' '\n' | grep -qx "${exp_name}"; then
            echo "Skipping (filtered out): ${exp_name}"
            continue
        fi
    fi

    echo ""
    echo ">>> ${exp_name}"
    RUN_DIR="${OUTPUT_BASE}/${exp_name}"
    mkdir -p "${RUN_DIR}"

    # Look up experiment type from matrix JSON
    EXP_TYPE=$(python3 -c "
import json
with open('${MATRIX}') as f:
    m = json.load(f)
for e in m.get('experiments', []):
    if e['name'] == '${exp_name}':
        print(e.get('type', 'pairwise_ranker'))
        break
" 2>/dev/null || echo "pairwise_ranker")

    # Dispatch to the appropriate training script based on type
    case "${EXP_TYPE}" in
        pairwise_ranker)
            TRAIN_CMD="RUN_NAME=${exp_name} OUTPUT_BASE=${OUTPUT_BASE} bash batch/learning/train_nlp4lp_ranker.sbatch"
            ;;
        multitask_grounder)
            TRAIN_CMD="RUN_NAME=${exp_name} OUTPUT_BASE=${OUTPUT_BASE} bash batch/learning/train_multitask_grounder.sbatch"
            ;;
        *)
            echo "Unknown experiment type '${EXP_TYPE}' for ${exp_name}; skipping."
            FAILED+=("${exp_name}")
            continue
            ;;
    esac

    if eval "${TRAIN_CMD}" 2>&1 | tee "${RUN_DIR}/train.log"; then
        SUCCESS+=("${exp_name}")
        echo "<<< ${exp_name} — SUCCEEDED"
    else
        FAILED+=("${exp_name}")
        echo "<<< ${exp_name} — FAILED (see ${RUN_DIR}/train.log)"
    fi
done <<< "${EXPERIMENT_NAMES}"

echo ""
echo "=== Summary ==="
echo "Succeeded (${#SUCCESS[@]}): ${SUCCESS[*]:-none}"
echo "Failed    (${#FAILED[@]}): ${FAILED[*]:-none}"

[ "${#FAILED[@]}" -eq 0 ]
