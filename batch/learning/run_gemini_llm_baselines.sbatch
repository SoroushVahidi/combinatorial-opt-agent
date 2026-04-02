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

# Cap OpenMP/BLAS/NumExpr threads before the Gemini client loads (mitigates Wulver pthread_create warnings).
# Does not change API quota. Disable with: export GEMINI_LIMIT_RUNTIME_THREADS=0 before sbatch.
export GEMINI_LIMIT_RUNTIME_THREADS="${GEMINI_LIMIT_RUNTIME_THREADS:-1}"

# Prefer repo venv when available (preflight must use the same env as the run).
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

# Auto-pick the first model that passes a minimal probe (configs/llm_baselines.yaml fallbacks).
# Default ON — no manual export of GEMINI_MODEL required. Disable with GEMINI_AUTO_PICK_MODEL=0.
# Writes JSON to GEMINI_MODEL_SELECTED_FILE or results/llm_baselines/gemini_selected_model.json.
_gemini_auto_pick="${GEMINI_AUTO_PICK_MODEL:-1}"
if [[ "$_gemini_auto_pick" != "0" && "$_gemini_auto_pick" != "false" && "$_gemini_auto_pick" != "no" && "$_gemini_auto_pick" != "off" ]]; then
  ART="${GEMINI_MODEL_SELECTED_FILE:-$REPO_ROOT/results/llm_baselines/gemini_selected_model.json}"
  mkdir -p "$(dirname "$ART")"
  if ! PICKED=$(python tools/llm_baselines.py pick-model --write-selected "$ART"); then
    _pick_rc=$?
    if [[ $_pick_rc -eq 3 && "${GEMINI_SKIP_ON_ZERO_QUOTA:-}" == "1" ]]; then
      echo "Skipping Gemini baseline job: pick-model found no usable model (all candidates failed). See artifact: $ART"
      exit 0
    fi
    echo "ERROR: pick-model failed (exit $_pick_rc). Every candidate failed probes; see stderr and $ART"
    exit "$_pick_rc"
  fi
  export GEMINI_MODEL="$PICKED"
  echo "GEMINI_AUTO_PICK (on): using GEMINI_MODEL=$GEMINI_MODEL (artifact: $ART)"
fi

# Preflight: model must be list_models-visible and must not hit free-tier limit:0.
# Exit 2 = hard zero quota. If GEMINI_SKIP_ON_ZERO_QUOTA=1, exit 0 and skip this job.
python tools/llm_baselines.py preflight
_preflight_rc=$?
if [[ $_preflight_rc -eq 2 && "${GEMINI_SKIP_ON_ZERO_QUOTA:-}" == "1" ]]; then
  echo "Skipping Gemini baseline job (preflight: hard zero quota for this model/account)."
  exit 0
fi
if [[ $_preflight_rc -ne 0 ]]; then
  echo "ERROR: Gemini preflight failed (exit $_preflight_rc). See docs/gemini_api_quota.md"
  exit "$_preflight_rc"
fi
# Avoid repeating the same probe on every variant in this job.
export GEMINI_SKIP_PREFLIGHT=1

for variant in orig noisy short; do
  echo "=== Running Gemini baseline for ${variant} ==="
  python tools/nlp4lp_downstream_utility.py \
    --variant "${variant}" \
    --method gemini
done

echo "Gemini baseline run completed."
echo "Summary CSV: results/paper/nlp4lp_downstream_summary.csv"
