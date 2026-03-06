#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/results/paper_run_${TS}"
LATEST_DIR="$ROOT/results/paper_run_latest"
mkdir -p "$RUN_DIR"

SEED_SPLITS=42
SEED_EVAL=999
SEED_TRAIN=42
TOP_K=10
NUM_EVAL=500

# Check torch availability once
if python - << 'PYCHK'
try:
    import torch  # noqa: F401
    ok = True
except Exception:
    ok = False
import sys
sys.exit(0 if ok else 1)
PYCHK
then
  TORCH_OK=1
  echo "[run_paper_experiments] torch available."
else
  TORCH_OK=0
  echo "[run_paper_experiments] torch NOT available; skipping SBERT-based steps."
fi

# (a) build catalog
python pipeline/run_collection.py

# (b) splits
python -m training.splits \
  --splits-out data/processed/splits.json \
  --seed "${SEED_SPLITS}"

# (c) training pairs (train split only)
python -m training.generate_samples \
  --splits data/processed/splits.json \
  --split train \
  --output data/processed/training_pairs.jsonl \
  --seed "${SEED_TRAIN}" \
  --instances-per-problem 100

cp data/processed/splits.json "$RUN_DIR/splits.json"
cp data/processed/training_pairs.jsonl "$RUN_DIR/training_pairs.jsonl"

# (d) train retrieval (only if torch is available)
if [ "$TORCH_OK" -eq 1 ]; then
  python -m training.train_retrieval \
    --data data/processed/training_pairs.jsonl \
    --output-dir data/models/retrieval_finetuned \
    --epochs 2 \
    --batch-size 32
else
  echo "[run_paper_experiments] Skipping fine-tuning (no torch)."
fi

# (e) eval dev + test (only if torch is available)
if [ "$TORCH_OK" -eq 1 ]; then
  python -m training.evaluate_retrieval \
    --splits data/processed/splits.json \
    --split dev \
    --regenerate \
    --num "${NUM_EVAL}" \
    --seed "${SEED_EVAL}" \
    --results-dir "$RUN_DIR" \
    --top-k "${TOP_K}"

  python -m training.evaluate_retrieval \
    --splits data/processed/splits.json \
    --split test \
    --regenerate \
    --num "${NUM_EVAL}" \
    --seed "${SEED_EVAL}" \
    --results-dir "$RUN_DIR" \
    --top-k "${TOP_K}"
else
  echo "[run_paper_experiments] Skipping evaluate_retrieval (no torch)."
fi

# (f) baselines on test
if [ "$TORCH_OK" -eq 1 ]; then
  BASELINES=(bm25 tfidf sbert sbert_finetuned)
else
  BASELINES=(bm25 tfidf)
fi

python -m training.run_baselines \
  --splits data/processed/splits.json \
  --split test \
  --eval-file data/processed/eval_test.jsonl \
  --k "${TOP_K}" \
  --baselines "${BASELINES[@]}" \
  --results-dir "$RUN_DIR"

# (g) catalog validation
python -m formulation.run_verify_catalog \
  --catalog data/processed/all_problems.json \
  --results-dir "$RUN_DIR"

# (h) baseline plots + LaTeX table
python -m training.plot_baselines \
  --csv "$RUN_DIR/baselines_test.csv" \
  --results-dir "$RUN_DIR"

# config snapshot
python - <<EOF2
import json
cfg = {
  "timestamp": "${TS}",
  "seeds": {"splits": ${SEED_SPLITS}, "train": ${SEED_TRAIN}, "eval": ${SEED_EVAL}},
  "top_k": ${TOP_K},
  "num_eval": ${NUM_EVAL},
  "command": "scripts/run_paper_experiments.sh",
  "torch_available": bool(${TORCH_OK}),
}
out = "${RUN_DIR}/config.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print("Wrote", out)
EOF2

rm -f "$LATEST_DIR"
ln -s "paper_run_${TS}" "$LATEST_DIR"

