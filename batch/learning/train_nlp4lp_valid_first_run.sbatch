#!/bin/bash
#SBATCH --job-name=nlp4lp_valid
#SBATCH --output=logs/learning/train_nlp4lp_valid_first_run_%j.out
#SBATCH --error=logs/learning/train_nlp4lp_valid_first_run_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#
# First VALID learning run: NLP4LP pairwise mention-slot ranker.
# - Builds test corpus from local eval_orig, then splits by instance_id (70/15/15)
#   into train/dev/test so splits are distinct (no test-as-train fallback).
# - Runs split-integrity check; fails if train/test or dev/test are identical.
# - Trains with fixed seed (42), logs loss, saves config with counts.
# - Evaluates on held-out test only.
#
# Submit: sbatch batch/learning/train_nlp4lp_valid_first_run.sbatch
#
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs artifacts/learning_corpus artifacts/learning_ranker_data/nlp4lp

CORPUS_DIR="$REPO_ROOT/artifacts/learning_corpus"
RANKER_DATA_DIR="$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp"
RUN_NAME="valid_first_run"
OUT_DIR="$REPO_ROOT/artifacts/learning_runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
git rev-parse HEAD > "$OUT_DIR/git_rev.txt" 2>/dev/null || true

# --- Environment ---
if [[ -x "$REPO_ROOT/venv/bin/python" ]] && "$REPO_ROOT/venv/bin/python" -c 'import torch' 2>/dev/null; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv: $REPO_ROOT/venv"
elif ! python -c 'import torch' 2>/dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || true
  if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    for env in ${CONDA_ENV:-pt pytorch combinatorial-train base}; do
      if conda activate "$env" 2>/dev/null && python -c 'import torch' 2>/dev/null; then
        echo "Using conda env: $env"
        break
      fi
    done
  fi
fi

echo "=== Preflight diagnostics ==="
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "cwd: $(pwd)"
echo "which python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
python -c 'import torch; print("torch:", torch.__version__)' 2>/dev/null || echo "torch: FAILED"
python -c 'import transformers; print("transformers:", transformers.__version__)' 2>/dev/null || echo "transformers: FAILED"
python -c 'import torch; print("CUDA:", torch.cuda.is_available())' 2>/dev/null || true
echo "=== End preflight ==="

if ! python -c 'import torch' 2>/dev/null; then
  echo "ERROR: torch required."
  exit 1
fi
if ! python -c 'import transformers' 2>/dev/null; then
  echo "ERROR: transformers required."
  exit 1
fi

# --- 1) Build NLP4LP test corpus (local only; no HF on login) ---
if [[ ! -f "$CORPUS_DIR/nlp4lp_test.jsonl" ]]; then
  echo "=== Building NLP4LP test corpus from local eval_orig ==="
  python -m src.learning.build_common_grounding_corpus \
    --dataset nlp4lp --split test --output_dir "$CORPUS_DIR"
  if [[ ! -f "$CORPUS_DIR/nlp4lp_test.jsonl" ]]; then
    echo "ERROR: Failed to create nlp4lp_test.jsonl. Check data/processed/nlp4lp_eval_orig.jsonl and gold."
    exit 1
  fi
fi
echo "Corpus test lines: $(wc -l < "$CORPUS_DIR/nlp4lp_test.jsonl")"

# --- 2) Split into train/dev/test by instance_id (benchmark-safe) ---
echo "=== Splitting corpus into train/dev/test (seed 42, 70/15/15) ==="
python -m src.learning.split_nlp4lp_corpus_for_benchmark \
  --corpus_dir "$CORPUS_DIR" --seed 42 --verbose

# --- 3) Build pairwise ranker data from all three splits ---
echo "=== Building ranker train/dev/test JSONL ==="
python -m src.learning.build_nlp4lp_pairwise_ranker_data \
  --corpus_dir "$CORPUS_DIR" --output_dir "$RANKER_DATA_DIR"

# --- 4) Split integrity: fail if train==test or dev==test ---
echo "=== Split integrity check (no fallback; benchmark mode) ==="
if ! python -m src.learning.verify_split_integrity --data_dir "$RANKER_DATA_DIR"; then
  echo "ERROR: Split integrity failed. Do not train."
  exit 1
fi

# --- 5) Train with seed and logging ---
echo "=== Training (seed 42, max_steps 200) ==="
python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name "$RUN_NAME" \
  --data_dir "$RANKER_DATA_DIR" \
  --save_dir "$REPO_ROOT/artifacts/learning_runs" \
  --encoder distilroberta-base \
  --seed 42 \
  --epochs 1 \
  --max_steps 200 \
  --lr 2e-5 \
  --batch_size 8

# --- 6) Evaluate on held-out test ---
echo "=== Evaluation on held-out test ==="
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir "$RANKER_DATA_DIR" \
  --run_dir "$OUT_DIR" \
  --split test \
  --out_dir "$OUT_DIR"

echo ""
echo "=== Valid first run complete ==="
ls -la "$OUT_DIR" 2>/dev/null || true
if [[ -f "$OUT_DIR/checkpoint.pt" ]]; then
  echo "Checkpoint: $OUT_DIR/checkpoint.pt"
fi
if [[ -f "$OUT_DIR/metrics.json" ]]; then
  echo "Test metrics: $OUT_DIR/metrics.json"
fi
