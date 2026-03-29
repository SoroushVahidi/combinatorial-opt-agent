#!/usr/bin/env bash
# Check status of the retrieval training job and model.
# Run from project root: bash scripts/check_training_status.sh
# With --resubmit-if-failed: if no job running and no model, resubmit and print new job id.
# Or run periodically: watch -n 60 bash scripts/check_training_status.sh

RESUBMIT_IF_FAILED=
[ "${1:-}" = "--resubmit-if-failed" ] && RESUBMIT_IF_FAILED=1

set -e
cd "${BASH_SOURCE[0]%/*}/.."

echo "=== SLURM jobs (train_retrieval) ==="
squeue -u "$USER" --name=train_retrieval 2>/dev/null || true
if ! squeue -u "$USER" --name=train_retrieval 2>/dev/null | grep -q train_retrieval; then
  echo "(no train_retrieval job in queue)"
fi

echo ""
echo "=== Latest training log files ==="
for f in train_retrieval_*.out; do
  [ -f "$f" ] || continue
  echo "--- $f (last modified: $(stat -c %y "$f" 2>/dev/null || stat -f %Sm "$f" 2>/dev/null)) ---"
done
OUT=$(ls -t train_retrieval_*.out 2>/dev/null | head -1)
if [ -n "$OUT" ]; then
  echo "Tail of stdout ($OUT):"
  tail -30 "$OUT"
fi

ERR=$(ls -t train_retrieval_*.err 2>/dev/null | head -1)
if [ -n "$ERR" ]; then
  echo ""
  echo "Tail of stderr ($ERR):"
  tail -20 "$ERR"
fi

echo ""
echo "=== Fine-tuned model ==="
MODEL_OK=
if [ -d "data/models/retrieval_finetuned/final" ] && [ -f "data/models/retrieval_finetuned/final/config.json" ]; then
  echo "Present: data/models/retrieval_finetuned/final"
  ls -la data/models/retrieval_finetuned/final/
  MODEL_OK=1
else
  echo "Not yet: data/models/retrieval_finetuned/final (training may still be running or failed)"
fi

# Optional: resubmit if no job in queue and no success
if [ -n "$RESUBMIT_IF_FAILED" ]; then
  JOB_RUNNING=$(squeue -u "$USER" --name=train_retrieval -h 2>/dev/null | wc -l)
  if [ "$JOB_RUNNING" -eq 0 ] && [ -z "$MODEL_OK" ]; then
    echo ""
    echo ">>> No job running and no model; resubmitting..."
    sbatch scripts/train_retrieval_gpu.slurm
  fi
fi
