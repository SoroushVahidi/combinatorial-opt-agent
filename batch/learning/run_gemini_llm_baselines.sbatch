#!/bin/bash
#SBATCH --job-name=gemini_nlp4lp
#SBATCH --output=logs/learning/run_gemini_llm_baselines_%j.out
#SBATCH --error=logs/learning/run_gemini_llm_baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs/learning results/llm_baselines

# Optional: repo-local secrets (gitignored) — same pattern as README .env
if [[ -f "$REPO_ROOT/.env" ]]; then
  set +u
  set +a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set -a
  set -u
fi

# Load interactive-shell exports (including GEMINI_API_KEY).
# NOTE: ~/.bashrc often sources /etc/bashrc which references $BASHRCSOURCED before
# it is set; with `set -u` that aborts. Temporarily allow unset vars while sourcing.
if [[ -f "$HOME/.bashrc" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$HOME/.bashrc"
  set -u
fi

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set. Add it to ~/.bashrc and resubmit."
  exit 1
fi

# Prefer repo venv when available.
if [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv: $REPO_ROOT/venv"
fi

echo "which python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
python -c "import google.generativeai as g; print('google-generativeai import: OK')" 2>/dev/null || {
  echo "ERROR: google-generativeai is not available in this Python env."
  exit 1
}

for variant in orig noisy short; do
  echo "=== Running Gemini baseline for ${variant} ==="
  python tools/nlp4lp_downstream_utility.py \
    --variant "${variant}" \
    --method gemini
done

echo "Gemini baseline run completed."
echo "Summary CSV: results/paper/nlp4lp_downstream_summary.csv"
