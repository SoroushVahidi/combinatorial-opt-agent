#!/usr/bin/env bash
# Real-data-only learning check: build full NLP4LP split, train, eval, rule baseline on same test.
# Requires: NLP4LP_GOLD_CACHE set (e.g. results/paper/nlp4lp_gold_cache.json), torch + transformers.
# Usage: ./scripts/learning/run_real_data_only_learning_check.sh
#   or:  NLP4LP_GOLD_CACHE=/path/to/nlp4lp_gold_cache.json ./scripts/learning/run_real_data_only_learning_check.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs artifacts/learning_corpus artifacts/learning_ranker_data/nlp4lp

CORPUS_DIR="$REPO_ROOT/artifacts/learning_corpus"
RANKER_DATA_DIR="$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp"
SAVE_DIR="$REPO_ROOT/artifacts/learning_runs"
RUN_NAME="real_data_only_learning_check"
OUT_DIR="$SAVE_DIR/$RUN_NAME"
GOLD_CACHE="${NLP4LP_GOLD_CACHE:-$REPO_ROOT/results/paper/nlp4lp_gold_cache.json}"

export NLP4LP_GOLD_CACHE="$GOLD_CACHE"

echo "=== 1) Build full NLP4LP corpus then split 70/15/15 ==="
python -m src.learning.build_common_grounding_corpus --dataset nlp4lp --split test --output_dir "$CORPUS_DIR"
echo "Corpus test records: $(wc -l < "$CORPUS_DIR/nlp4lp_test.jsonl")"
python -m src.learning.split_nlp4lp_corpus_for_benchmark --corpus_dir "$CORPUS_DIR" --seed 42 --verbose
python -m src.learning.build_nlp4lp_pairwise_ranker_data --corpus_dir "$CORPUS_DIR" --output_dir "$RANKER_DATA_DIR"
echo "Train/Dev/Test pairs: $(wc -l < "$RANKER_DATA_DIR/train.jsonl") / $(wc -l < "$RANKER_DATA_DIR/dev.jsonl") / $(wc -l < "$RANKER_DATA_DIR/test.jsonl")"

echo "=== 2) Split integrity ==="
python -m src.learning.verify_split_integrity --data_dir "$RANKER_DATA_DIR"

echo "=== 3) Train (real data only) ==="
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

echo "=== 5) Rule baseline on SAME test split ==="
RULE_DIR="$SAVE_DIR/rule_baseline_same_test"
mkdir -p "$RULE_DIR"
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir "$RANKER_DATA_DIR" --split test --out_dir "$RULE_DIR"
echo "Rule baseline metrics: $RULE_DIR/metrics.json"

echo "=== Done. Fill docs/learning_runs/real_data_only_learning_check.md Section 4 and 6 from the metrics. ==="
