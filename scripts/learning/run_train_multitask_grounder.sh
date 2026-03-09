#!/usr/bin/env bash
# Train multitask grounder (NL4Opt aux + NLP4LP pairwise).
# Usage: ./scripts/learning/run_train_multitask_grounder.sh [--sbatch] [--joint] [--entity] [--bound] [--role]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p artifacts/learning_runs

MODE="pretrain_then_finetune"
USE_ENTITY="" USE_BOUND="" USE_ROLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch)
      export RUN_NAME="${RUN_NAME:-multitask_run0}"
      export MODE="${MODE:-pretrain_then_finetune}"
      export USE_ENTITY="${USE_ENTITY:-0}"; export USE_BOUND="${USE_BOUND:-0}"; export USE_ROLE="${USE_ROLE:-0}"
      sbatch batch/learning/train_multitask_grounder.sbatch
      exit 0
      ;;
    --joint) MODE="joint"; shift ;;
    --entity) USE_ENTITY="1"; shift ;;
    --bound) USE_BOUND="1"; shift ;;
    --role) USE_ROLE="1"; shift ;;
    *) shift ;;
  esac
done

OPTS="--mode $MODE --run_name ${RUN_NAME:-multitask_run0} --encoder distilroberta-base"
OPTS="$OPTS --nlp4lp_data_dir $REPO_ROOT/artifacts/learning_ranker_data/nlp4lp"
OPTS="$OPTS --nl4opt_aux_dir $REPO_ROOT/artifacts/learning_aux_data/nl4opt"
OPTS="$OPTS --save_dir $REPO_ROOT/artifacts/learning_runs"
OPTS="$OPTS --pretrain_steps 50 --finetune_steps 100 --joint_steps 150 --batch_size 8 --lr 2e-5"
[[ -n "$USE_ENTITY" ]] && OPTS="$OPTS --use_nl4opt_entity"
[[ -n "$USE_BOUND" ]] && OPTS="$OPTS --use_nl4opt_bound"
[[ -n "$USE_ROLE" ]] && OPTS="$OPTS --use_nl4opt_role"

python -m src.learning.train_multitask_grounder $OPTS
