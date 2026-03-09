#!/bin/bash
#SBATCH --job-name=stage3_exp
#SBATCH --output=logs/learning/run_stage3_experiments_%j.out
#SBATCH --error=logs/learning/run_stage3_experiments_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# Wulver requires an explicit QoS (see scripts/wulver_qos_queries.sh). Override: sbatch --qos=low ...
#SBATCH --qos=standard
set -euo pipefail

# Run full Stage 3 experiment round (corpus/ranker/aux if needed, train all runs, eval, bottleneck slices).
# Example: sbatch batch/learning/run_stage3_experiments.sbatch
# Force rerun: FORCE=1 sbatch batch/learning/run_stage3_experiments.sbatch
# On clusters where repo/home is read-only on compute nodes, the script uses SCRATCH or TMPDIR for logs/artifacts.
# To get logs on shared storage, set OUTPUT_BASE before submitting: OUTPUT_BASE=$SCRATCH/yourproject sbatch ...

# Use submit dir so we're in the repo even when $0 is relative on the compute node
REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT" || { echo "Cannot cd to REPO_ROOT=$REPO_ROOT"; exit 1; }

# Use writable base for logs and artifacts (repo may be read-only on compute nodes, e.g. Wulver).
OUTPUT_BASE="${OUTPUT_BASE:-}"
if [[ -n "$OUTPUT_BASE" ]]; then
  WRITABLE_BASE="$OUTPUT_BASE"
  mkdir -p "$WRITABLE_BASE/logs/learning" "$WRITABLE_BASE/artifacts"
elif mkdir -p logs/learning artifacts/learning_runs 2>/dev/null; then
  WRITABLE_BASE="$REPO_ROOT"
else
  WRITABLE_BASE="${SCRATCH:-${TMPDIR:-/tmp}}/stage3_${SLURM_JOB_ID:-$$}"
  mkdir -p "$WRITABLE_BASE/logs/learning" "$WRITABLE_BASE/artifacts" || { echo "Cannot create $WRITABLE_BASE"; exit 1; }
  touch "$WRITABLE_BASE/logs/learning/.write_test" && rm -f "$WRITABLE_BASE/logs/learning/.write_test" || { echo "Cannot write to $WRITABLE_BASE"; exit 1; }
fi
# Redirect stdout/stderr to writable location when not using repo (so logs are in SCRATCH/TMPDIR)
if [[ "$WRITABLE_BASE" != "$REPO_ROOT" ]]; then
  echo "Using writable base: $WRITABLE_BASE (logs and artifacts here)"
  echo "SCRATCH=${SCRATCH:-<unset>} TMPDIR=${TMPDIR:-<unset>}"
  LOGFILE="$WRITABLE_BASE/logs/learning/run_stage3_experiments_${SLURM_JOB_ID:-$$}.out"
  exec >"$LOGFILE" 2>&1
fi

CONFIG="${CONFIG:-$REPO_ROOT/configs/learning/experiment_matrix_stage3.json}"
FORCE="${FORCE:-0}"

# Activate env with torch (repo venv first, else conda) so launcher and subprocesses use same Python
if [[ -x "$REPO_ROOT/venv/bin/python" ]] && "$REPO_ROOT/venv/bin/python" -c 'import torch' 2>/dev/null; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv: $REPO_ROOT/venv"
elif ! python -c 'import torch' 2>/dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || true
  if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    for env in ${CONDA_ENV:-pt pytorch base}; do
      if conda activate "$env" 2>/dev/null && python -c 'import torch' 2>/dev/null; then
        echo "Using conda env: $env"
        break
      fi
    done
  fi
fi

echo "=== Environment diagnostics ==="
echo "which python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
python -c 'import torch; print("torch:", torch.__version__)' 2>/dev/null || echo "torch: FAILED TO IMPORT"
python -c 'import transformers; print("transformers:", transformers.__version__)' 2>/dev/null || echo "transformers: FAILED TO IMPORT"
python -c 'import torch; print("torch.cuda.is_available():", torch.cuda.is_available())' 2>/dev/null || true
echo "=== End diagnostics ==="

if ! python -c 'import torch' 2>/dev/null; then
  echo "ERROR: torch required for Stage 3 training. Activate repo venv or a conda env with torch and resubmit."
  exit 1
fi
if ! python -c 'import transformers' 2>/dev/null; then
  echo "ERROR: transformers required for Stage 3 training. Activate an env with transformers and resubmit."
  exit 1
fi

OPTS="--config $CONFIG --mode local --artifacts_base $WRITABLE_BASE/artifacts"
[[ "$FORCE" = "1" ]] && OPTS="$OPTS --force"

set +e
python -m src.learning.run_stage3_experiments $OPTS
R=$?
echo "Python exit code: $R"
# Try to copy log back to repo so it's visible from login node (if repo is writable)
if [[ "$WRITABLE_BASE" != "$REPO_ROOT" && -n "${LOGFILE:-}" && -f "$LOGFILE" ]]; then
  mkdir -p "$REPO_ROOT/logs/learning" 2>/dev/null && cp -f "$LOGFILE" "$REPO_ROOT/logs/learning/" 2>/dev/null || true
fi
exit $R
