#!/usr/bin/env bash
# Automatically check training status every N seconds.
# Run from project root: bash scripts/watch_training.sh
# Optional: bash scripts/watch_training.sh 120   (check every 2 minutes)
# Optional: bash scripts/watch_training.sh 60 --stop-when-done   (exit when job finishes and model exists)
# Optional: bash scripts/watch_training.sh 60 --notify-when-stopped   (when job stops, write status file and tell you to inform the assistant)
# Run in background: nohup bash scripts/watch_training.sh 60 --notify-when-stopped > watch_training.log 2>&1 &

INTERVAL="${1:-60}"
STOP_WHEN_DONE=
NOTIFY_WHEN_STOPPED=
for arg in "${2:-}" "${3:-}"; do
  [ "$arg" = "--stop-when-done" ] && STOP_WHEN_DONE=1
  [ "$arg" = "--notify-when-stopped" ] && NOTIFY_WHEN_STOPPED=1
done

cd "${BASH_SOURCE[0]%/*}/.."

STATUS_FILE="training_latest_status.txt"

write_status_and_exit() {
  local status="$1"
  local job_id="$2"
  local out_file="$3"
  local err_file="$4"
  local model_present="$5"
  {
    echo "STATUS=$status"
    echo "JOB_ID=$job_id"
    echo "TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "OUT_FILE=$out_file"
    echo "ERR_FILE=$err_file"
    echo "MODEL_PRESENT=$model_present"
    echo ""
    echo "--- LAST 50 LINES OF STDOUT ---"
    [ -n "$out_file" ] && [ -f "$out_file" ] && tail -50 "$out_file"
    echo ""
    echo "--- LAST 40 LINES OF STDERR ---"
    [ -n "$err_file" ] && [ -f "$err_file" ] && tail -40 "$err_file"
  } > "$STATUS_FILE"
  echo ""
  echo "=============================================="
  echo "Training has stopped. Status written to: $STATUS_FILE"
  echo ""
  echo "  Copy this to your assistant (Cursor chat):"
  echo ""
  echo "  Training stopped - check training_latest_status.txt and do the next steps."
  echo ""
  echo "=============================================="
  exit 0
}

echo "Watching training every ${INTERVAL}s. Ctrl+C to stop."
[ -n "$NOTIFY_WHEN_STOPPED" ] && echo "When the job stops, status will be written to $STATUS_FILE and you'll be told to inform the assistant."
echo ""

while true; do
  echo "========== $(date) =========="
  bash scripts/check_training_status.sh
  echo ""

  JOB_RUNNING=$(squeue -u "$USER" --name=train_retrieval -h 2>/dev/null | wc -l)
  HAS_MODEL=
  [ -d "data/models/retrieval_finetuned/final" ] && [ -f "data/models/retrieval_finetuned/final/config.json" ] && HAS_MODEL=yes

  if [ "$JOB_RUNNING" -eq 0 ]; then
    # Job no longer in queue - training stopped
    OUT=$(ls -t train_retrieval_*.out 2>/dev/null | head -1)
    ERR=$(ls -t train_retrieval_*.err 2>/dev/null | head -1)
    JOB_ID=""
    [[ "$OUT" =~ train_retrieval_([0-9]+)\.out ]] && JOB_ID="${BASH_REMATCH[1]}"

    if [ -n "$NOTIFY_WHEN_STOPPED" ]; then
      if [ -n "$HAS_MODEL" ]; then
        write_status_and_exit "FINISHED_OK" "$JOB_ID" "$OUT" "$ERR" "yes"
      else
        # Check if last run crashed (crash.log or no "Done" in .out)
        if [ -f "data/models/retrieval_finetuned/crash.log" ]; then
          write_status_and_exit "FINISHED_FAILED" "$JOB_ID" "$OUT" "$ERR" "no"
        elif [ -n "$OUT" ] && grep -q "Done\. Model saved" "$OUT" 2>/dev/null; then
          write_status_and_exit "FINISHED_OK" "$JOB_ID" "$OUT" "$ERR" "yes"
        else
          write_status_and_exit "FINISHED_FAILED" "$JOB_ID" "$OUT" "$ERR" "no"
        fi
      fi
    fi

    if [ -n "$STOP_WHEN_DONE" ] && [ -n "$HAS_MODEL" ]; then
      echo "Training finished and model is saved. Exiting."
      exit 0
    fi
  fi

  sleep "$INTERVAL"
done
