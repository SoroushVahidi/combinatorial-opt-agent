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

# Optional: load API key from a gitignored one-line file (avoids exporting in ~/.bashrc).
# Example: export GEMINI_API_KEY_FILE=$REPO_ROOT/.secrets/gemini_api_key  (chmod 600)
if [[ -z "${GEMINI_API_KEY:-}" && -n "${GEMINI_API_KEY_FILE:-}" ]]; then
  if [[ -f "$GEMINI_API_KEY_FILE" ]]; then
    export GEMINI_API_KEY="$(tr -d '\r\n' < "$GEMINI_API_KEY_FILE")"
  else
    echo "ERROR: GEMINI_API_KEY_FILE is set but not found: $GEMINI_API_KEY_FILE"
    exit 1
  fi
fi

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set. Export it, add to ~/.bashrc, use repo .env, or set GEMINI_API_KEY_FILE."
  exit 1
fi

# Cap OpenMP/BLAS/NumExpr threads before the Gemini client loads (mitigates Wulver pthread_create warnings).
# Does not change API quota. Disable with: export GEMINI_LIMIT_RUNTIME_THREADS=0 before sbatch.
export GEMINI_LIMIT_RUNTIME_THREADS="${GEMINI_LIMIT_RUNTIME_THREADS:-1}"

# Prefer repo venv when available (preflight must use the same env as the run).
if [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv: $REPO_ROOT/venv"
else
  echo "WARNING: $REPO_ROOT/venv/bin/python not found; using PATH python (install deps in this env)."
fi

# ── Batch banner (grep-friendly; see docs/gemini_api_quota.md) ───────────────
_run_id="${SLURM_JOB_ID:-manual}"
echo "=========================================="
echo "Gemini NLP4LP batch — run_id=${_run_id}"
echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname 2>/dev/null || true)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<n/a>}"
echo "REPO_ROOT=$REPO_ROOT"
echo "GEMINI_AUTO_PICK_MODEL=${GEMINI_AUTO_PICK_MODEL:-1}"
_skip_z="${GEMINI_SKIP_ON_ZERO_QUOTA:-}"
if [[ "$_skip_z" == "1" || "$_skip_z" == "true" || "$_skip_z" == "yes" || "$_skip_z" == "on" ]]; then
  echo "GEMINI_SKIP_ON_ZERO_QUOTA=enabled (soft-skip on hard limit:0 / pick-model total failure)"
else
  echo "GEMINI_SKIP_ON_ZERO_QUOTA=disabled (job fails on hard quota or pick-model exit 3)"
fi
_lim="${GEMINI_LIMIT_RUNTIME_THREADS:-}"
if [[ "$_lim" == "0" || "$_lim" == "false" || "$_lim" == "no" || "$_lim" == "off" ]]; then
  echo "GEMINI_LIMIT_RUNTIME_THREADS=0 (thread cap disabled in tools/llm_baselines.py)"
else
  echo "GEMINI_LIMIT_RUNTIME_THREADS=${GEMINI_LIMIT_RUNTIME_THREADS:-1} (OMP/BLAS/NUMEXPR capped in Python before google.genai import)"
fi
echo "GEMINI_MODEL_SELECTED_FILE=${GEMINI_MODEL_SELECTED_FILE:-$REPO_ROOT/results/llm_baselines/gemini_selected_model.json}"
echo "=========================================="

echo "which python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
python -c "from google import genai; print('google-genai import: OK')" 2>/dev/null || {
  echo "ERROR: google-genai is not available in this Python env (pip install google-genai)."
  exit 1
}

# Auto-pick the first model that passes a minimal probe (configs/llm_baselines.yaml fallbacks).
# Default ON — no manual export of GEMINI_MODEL required. Disable with GEMINI_AUTO_PICK_MODEL=0.
# Writes JSON to GEMINI_MODEL_SELECTED_FILE or results/llm_baselines/gemini_selected_model.json.
_gemini_auto_pick="${GEMINI_AUTO_PICK_MODEL:-1}"
if [[ "$_gemini_auto_pick" != "0" && "$_gemini_auto_pick" != "false" && "$_gemini_auto_pick" != "no" && "$_gemini_auto_pick" != "off" ]]; then
  ART="${GEMINI_MODEL_SELECTED_FILE:-$REPO_ROOT/results/llm_baselines/gemini_selected_model.json}"
  mkdir -p "$(dirname "$ART")"
  echo "pick-model: running (candidates from yaml + GEMINI_MODEL / GEMINI_MODEL_FALLBACKS) ..."
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
  echo "GEMINI_SELECTED_MODEL=$GEMINI_MODEL"
  echo "GEMINI_AUTO_PICK=on (pick-model succeeded; artifact: $ART)"
else
  echo "GEMINI_AUTO_PICK=off (using GEMINI_MODEL or configs/llm_baselines.yaml gemini.model only; no pick-model probe)"
  python - <<'PY'
from tools.llm_baselines import gemini_fallback_candidates, resolve_gemini_model_from_config
print("resolved_model:", resolve_gemini_model_from_config())
print("fallback_chain (same order pick-model would use):", ", ".join(gemini_fallback_candidates()))
PY
fi

# Preflight: model must be list_models-visible and must not hit free-tier limit:0.
# Exit 2 = hard zero quota. If GEMINI_SKIP_ON_ZERO_QUOTA=1, exit 0 and skip this job.
echo "preflight: running list_models + minimal generateContent probe ..."
python tools/llm_baselines.py preflight
_preflight_rc=$?
if [[ $_preflight_rc -eq 0 ]]; then
  echo "PREFLIGHT_PASSED=1"
else
  echo "PREFLIGHT_PASSED=0 (exit $_preflight_rc)"
fi
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
echo "GEMINI_SKIP_PREFLIGHT=1 (set for variant loop; downstream will not re-probe)"

# Small JSON record for log aggregation (cwd is REPO_ROOT).
python - <<'PY'
import json
import os
import datetime
import pathlib

root = pathlib.Path.cwd()
art = pathlib.Path(os.environ.get("GEMINI_MODEL_SELECTED_FILE") or (root / "results/llm_baselines/gemini_selected_model.json"))
out = root / "results" / "llm_baselines" / f"gemini_sbatch_run_{os.environ.get('SLURM_JOB_ID', 'manual')}.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "run_id": os.environ.get("SLURM_JOB_ID", "manual"),
    "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gemini_model": (os.environ.get("GEMINI_MODEL") or "").strip(),
    "preflight_passed": True,
    "gemini_auto_pick": os.environ.get("GEMINI_AUTO_PICK_MODEL", "1"),
    "gemini_skip_on_zero_quota": os.environ.get("GEMINI_SKIP_ON_ZERO_QUOTA", ""),
    "gemini_limit_runtime_threads": os.environ.get("GEMINI_LIMIT_RUNTIME_THREADS", "1"),
    "gemini_skip_preflight_after": "1",
    "model_artifact": str(art),
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Wrote batch status JSON: {out}")
PY

for variant in orig noisy short; do
  echo "=== Running Gemini baseline for ${variant} ==="
  python tools/nlp4lp_downstream_utility.py \
    --variant "${variant}" \
    --method gemini
done

echo "Gemini baseline run completed."
echo "Summary CSV: results/paper/nlp4lp_downstream_summary.csv"
