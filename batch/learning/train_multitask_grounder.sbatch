#!/bin/bash
#SBATCH --job-name=multitask_gr
#SBATCH --output=logs/learning/train_multitask_grounder_%j.out
#SBATCH --error=logs/learning/train_multitask_grounder_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
set -euo pipefail

# Train multitask grounder: NL4Opt auxiliary + NLP4LP pairwise.
# Example (pretrain then finetune with all aux): sbatch batch/learning/train_multitask_grounder.sbatch
# Example (joint): MODE=joint sbatch batch/learning/train_multitask_grounder.sbatch
# With aux flags: USE_ENTITY=1 USE_BOUND=1 USE_ROLE=1 sbatch batch/learning/train_multitask_grounder.sbatch

REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs

# Activate env with torch (repo venv first, else conda)
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
  echo "ERROR: torch required. Activate repo venv or a conda env with torch and resubmit."
  exit 1
fi
if ! python -c 'import transformers' 2>/dev/null; then
  echo "ERROR: transformers required. Activate an env with transformers and resubmit."
  exit 1
fi

RUN_NAME="${RUN_NAME:-multitask_run0}"
MODE="${MODE:-pretrain_then_finetune}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp}"
AUX_DIR="${AUX_DIR:-$REPO_ROOT/artifacts/learning_aux_data/nl4opt}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/artifacts/learning_runs}"
ENCODER="${ENCODER:-distilroberta-base}"
USE_ENTITY="${USE_ENTITY:-0}"
USE_BOUND="${USE_BOUND:-0}"
USE_ROLE="${USE_ROLE:-0}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-50}"
FINETUNE_STEPS="${FINETUNE_STEPS:-100}"
JOINT_STEPS="${JOINT_STEPS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-5}"

OPTS="--mode $MODE --run_name $RUN_NAME --nlp4lp_data_dir $DATA_DIR --nl4opt_aux_dir $AUX_DIR --save_dir $SAVE_DIR --encoder $ENCODER"
OPTS="$OPTS --pretrain_steps $PRETRAIN_STEPS --finetune_steps $FINETUNE_STEPS --joint_steps $JOINT_STEPS --batch_size $BATCH_SIZE --lr $LR"
[[ "$USE_ENTITY" = "1" ]] && OPTS="$OPTS --use_nl4opt_entity"
[[ "$USE_BOUND" = "1" ]] && OPTS="$OPTS --use_nl4opt_bound"
[[ "$USE_ROLE" = "1" ]] && OPTS="$OPTS --use_nl4opt_role"

python -m src.learning.train_multitask_grounder $OPTS
