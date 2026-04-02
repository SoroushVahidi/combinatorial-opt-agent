#!/bin/bash
#SBATCH --job-name=gams_aux_cmp
#SBATCH --output=logs/learning/gams_aux_comparison_%j.out
#SBATCH --error=logs/learning/gams_aux_comparison_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#
# Compare NLP4LP-only vs GAMS-aux pretrain + NLP4LP on held-out NLP4LP test.
# Baseline: 200 steps on NLP4LP. Aux: 50 steps on GAMS aux, then 200 steps on NLP4LP from that checkpoint.
# Benchmark validity applies only to evaluation on NLP4LP test; GAMS labels are weak/heuristic.
#
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/learning artifacts/learning_runs artifacts/learning_corpus artifacts/learning_ranker_data/nlp4lp artifacts/learning_ranker_data/gams_aux

CORPUS_DIR="$REPO_ROOT/artifacts/learning_corpus"
RANKER_DATA_DIR="$REPO_ROOT/artifacts/learning_ranker_data/nlp4lp"
GAMS_AUX_DIR="$REPO_ROOT/artifacts/learning_ranker_data/gams_aux"
SAVE_DIR="$REPO_ROOT/artifacts/learning_runs"

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

echo "=== 1) NLP4LP corpus + split + ranker data + integrity ==="
if [[ ! -f "$CORPUS_DIR/nlp4lp_test.jsonl" ]]; then
  python -m src.learning.build_common_grounding_corpus --dataset nlp4lp --split test --output_dir "$CORPUS_DIR"
fi
python -m src.learning.split_nlp4lp_corpus_for_benchmark --corpus_dir "$CORPUS_DIR" --seed 42
python -m src.learning.build_nlp4lp_pairwise_ranker_data --corpus_dir "$CORPUS_DIR" --output_dir "$RANKER_DATA_DIR"
if ! python -m src.learning.verify_split_integrity --data_dir "$RANKER_DATA_DIR"; then
  echo "ERROR: Split integrity failed."; exit 1
fi

echo "=== 2) GAMS auxiliary data ==="
python tools/build_gams_aux_ranker_data.py --output_dir "$GAMS_AUX_DIR"
echo "GAMS aux rows: $(wc -l < "$GAMS_AUX_DIR/train.jsonl")"

echo "=== 3) Baseline: NLP4LP-only (200 steps) ==="
BASELINE_DIR="$SAVE_DIR/nlp4lp_only_baseline"
mkdir -p "$BASELINE_DIR"
python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name nlp4lp_only_baseline \
  --data_dir "$RANKER_DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --encoder distilroberta-base \
  --seed 42 --epochs 1 --max_steps 200 --lr 2e-5 --batch_size 8

echo "=== 4) GAMS-aux pretrain (50 steps) ==="
AUX_PRETRAIN_DIR="$SAVE_DIR/gams_aux_pretrain"
mkdir -p "$AUX_PRETRAIN_DIR"
python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name gams_aux_pretrain \
  --data_dir "$GAMS_AUX_DIR" \
  --save_dir "$SAVE_DIR" \
  --encoder distilroberta-base \
  --seed 42 --epochs 1 --max_steps 50 --lr 2e-5 --batch_size 8

echo "=== 5) NLP4LP fine-tune from GAMS-aux checkpoint (200 steps) ==="
python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name gams_aux_then_nlp4lp \
  --data_dir "$RANKER_DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --encoder distilroberta-base \
  --seed 42 --epochs 1 --max_steps 200 --lr 2e-5 --batch_size 8 \
  --init_checkpoint "$AUX_PRETRAIN_DIR/checkpoint.pt"

echo "=== 6) Evaluate both on held-out NLP4LP test ==="
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir "$RANKER_DATA_DIR" --run_dir "$SAVE_DIR/nlp4lp_only_baseline" --split test --out_dir "$SAVE_DIR/nlp4lp_only_baseline"
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir "$RANKER_DATA_DIR" --run_dir "$SAVE_DIR/gams_aux_then_nlp4lp" --split test --out_dir "$SAVE_DIR/gams_aux_then_nlp4lp"

echo "=== 7) Write comparison report ==="
python -c "
import json
from pathlib import Path
base = Path('$SAVE_DIR')
b = base / 'nlp4lp_only_baseline' / 'metrics.json'
g = base / 'gams_aux_then_nlp4lp' / 'metrics.json'
out = base.resolve().parent.parent / 'docs' / 'learning_runs' / 'gams_aux_vs_nlp4lp_only.md'
out.parent.mkdir(parents=True, exist_ok=True)
lines = ['# GAMS aux vs NLP4LP-only comparison\n', 'Benchmark: held-out NLP4LP test. GAMS labels are weak/heuristic.\n']
if b.exists():
    mb = json.loads(b.read_text())
    lines.append('## NLP4LP-only baseline\n')
    lines.append(f\"- pairwise_accuracy: {mb.get('pairwise_accuracy')}\")
    lines.append(f\"- slot_selection_accuracy: {mb.get('slot_selection_accuracy')}\")
    lines.append(f\"- exact_slot_fill_accuracy: {mb.get('exact_slot_fill_accuracy')}\")
    lines.append('')
if g.exists():
    mg = json.loads(g.read_text())
    lines.append('## GAMS-aux then NLP4LP\n')
    lines.append(f\"- pairwise_accuracy: {mg.get('pairwise_accuracy')}\")
    lines.append(f\"- slot_selection_accuracy: {mg.get('slot_selection_accuracy')}\")
    lines.append(f\"- exact_slot_fill_accuracy: {mg.get('exact_slot_fill_accuracy')}\")
    lines.append('')
lines.append('(See artifacts/learning_runs/ for full metrics.json.)')
out.write_text('\n'.join(lines))
print('Wrote', out)
"

echo "=== Done ==="
ls -la "$SAVE_DIR/nlp4lp_only_baseline" "$SAVE_DIR/gams_aux_then_nlp4lp" 2>/dev/null || true
