#!/bin/bash
#SBATCH --job-name=nlp4lp_real
#SBATCH --output=logs/learning/real_data_only_learning_check_%j.out
#SBATCH --error=logs/learning/real_data_only_learning_check_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#
# Real-data-only learning check: largest valid NLP4LP split, no synthetic/GAMS aux.
# Builds full corpus (with NLP4LP_GOLD_CACHE), split 70/15/15, train, eval, rule baseline on same test.
#
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs artifacts/learning_corpus artifacts/learning_ranker_data/nlp4lp

CORPUS_DIR="$REPO_ROOT/artifacts/learning_corpus"
RANKER_DATA_DIR="$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp"
SAVE_DIR="$REPO_ROOT/artifacts/learning_runs"
RUN_NAME="real_data_only_learning_check"
OUT_DIR="$SAVE_DIR/$RUN_NAME"
GOLD_CACHE="${NLP4LP_GOLD_CACHE:-$REPO_ROOT/results/paper/nlp4lp_gold_cache.json}"

# Use gold cache so we get full corpus when building (330 instances -> 230/50/50 split)
export NLP4LP_GOLD_CACHE="$GOLD_CACHE"

# --- Environment ---
if [[ -x "$REPO_ROOT/venv/bin/python" ]] && "$REPO_ROOT/venv/bin/python" -c 'import torch' 2>/dev/null; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv"
elif ! python -c 'import torch' 2>/dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || true
  if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    for env in pt pytorch combinatorial-train base; do
      if conda activate "$env" 2>/dev/null && python -c 'import torch' 2>/dev/null; then break; fi
    done
  fi
fi
if ! python -c 'import torch' 2>/dev/null; then echo "ERROR: torch required."; exit 1; fi
if ! python -c 'import transformers' 2>/dev/null; then echo "ERROR: transformers required."; exit 1; fi

echo "=== 1) Build full NLP4LP corpus (test) then split 70/15/15 ==="
if [[ ! -f "$GOLD_CACHE" ]]; then
  echo "WARNING: NLP4LP_GOLD_CACHE not found at $GOLD_CACHE; corpus may be small or empty."
fi
python -m src.learning.build_common_grounding_corpus --dataset nlp4lp --split test --output_dir "$CORPUS_DIR"
echo "Corpus test records: $(wc -l < "$CORPUS_DIR/nlp4lp_test.jsonl")"
python -m src.learning.split_nlp4lp_corpus_for_benchmark --corpus_dir "$CORPUS_DIR" --seed 42 --verbose
python -m src.learning.build_nlp4lp_pairwise_ranker_data --corpus_dir "$CORPUS_DIR" --output_dir "$RANKER_DATA_DIR"
echo "Train/Dev/Test pairs: $(wc -l < "$RANKER_DATA_DIR/train.jsonl") / $(wc -l < "$RANKER_DATA_DIR/dev.jsonl") / $(wc -l < "$RANKER_DATA_DIR/test.jsonl")"

echo "=== 2) Split integrity ==="
if ! python -m src.learning.verify_split_integrity --data_dir "$RANKER_DATA_DIR"; then
  echo "ERROR: Split integrity failed."; exit 1
fi

echo "=== 3) Train (real data only, no aux) ==="
mkdir -p "$OUT_DIR"
git rev-parse HEAD > "$OUT_DIR/git_rev.txt" 2>/dev/null || true
python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name "$RUN_NAME" \
  --data_dir "$RANKER_DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --encoder distilroberta-base \
  --seed 42 --epochs 1 --max_steps 500 --lr 2e-5 --batch_size 8

echo "=== 4) Evaluate learned model on held-out test ==="
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir "$RANKER_DATA_DIR" --run_dir "$OUT_DIR" --split test --out_dir "$OUT_DIR"

echo "=== 5) Rule (deterministic) baseline on SAME test split ==="
RULE_DIR="$SAVE_DIR/rule_baseline_same_test"
mkdir -p "$RULE_DIR"
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir "$RANKER_DATA_DIR" --split test --out_dir "$RULE_DIR"
echo "Rule baseline metrics: $RULE_DIR/metrics.json"

echo "=== Done ==="
ls -la "$OUT_DIR" "$RULE_DIR" 2>/dev/null || true
