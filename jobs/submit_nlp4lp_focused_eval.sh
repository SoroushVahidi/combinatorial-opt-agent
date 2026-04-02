#!/usr/bin/env bash
# Submit the focused eval + failure audit to run on compute nodes.
# Usage (from project root): bash jobs/submit_nlp4lp_focused_eval.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
jid=$(sbatch jobs/run_nlp4lp_focused_eval.slurm | tee /dev/stderr | sed -n 's/Submitted batch job \([0-9]*\)/\1/p')
echo "Job ID: $jid — monitor with: squeue -j $jid; tail -f logs/nlp4lp_focused_eval_${jid}.out"
