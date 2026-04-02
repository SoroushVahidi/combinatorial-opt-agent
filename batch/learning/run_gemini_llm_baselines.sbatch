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
mkdir -p logs/learning results/llm_baselines results/rerun/gemini

# ── Bootstrap: do NOT source ~/.bashrc under `set -u` (breaks on BASHRCSOURCED). ──
# Use only: repo .env, Slurm --export / --export-file, or GEMINI_API_KEY_FILE.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set +u
  set +a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set -a
  set -u
fi

if [[ -z "${GEMINI_API_KEY:-}" && -n "${GEMINI_API_KEY_FILE:-}" ]]; then
  if [[ -f "$GEMINI_API_KEY_FILE" ]]; then
    export GEMINI_API_KEY="$(tr -d '\r\n' < "$GEMINI_API_KEY_FILE")"
  else
    echo "ERROR: GEMINI_API_KEY_FILE is set but not found: $GEMINI_API_KEY_FILE"
    exit 1
  fi
fi

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  echo "Fix: use Slurm --export=ALL or --export-file=..., put GEMINI_API_KEY in repo .env, or set GEMINI_API_KEY_FILE."
  echo "Do not rely on sourcing ~/.bashrc in this batch script."
  exit 1
fi

export GEMINI_LIMIT_RUNTIME_THREADS="${GEMINI_LIMIT_RUNTIME_THREADS:-1}"

if [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv: $REPO_ROOT/venv"
else
  echo "WARNING: $REPO_ROOT/venv/bin/python not found; using PATH python."
fi

_run_id="${SLURM_JOB_ID:-manual}"
RERUN_ROOT="${GEMINI_RERUN_ROOT:-$REPO_ROOT/results/rerun/gemini/run_${_run_id}}"
export RERUN_ROOT
export NLP4LP_OUTPUT_DIR="${NLP4LP_OUTPUT_DIR:-$RERUN_ROOT/paper}"
mkdir -p "$RERUN_ROOT" "$NLP4LP_OUTPUT_DIR" logs/learning

echo "=========================================="
echo "Gemini NLP4LP batch — run_id=${_run_id}"
echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname 2>/dev/null || true)"
echo "REPO_ROOT=$REPO_ROOT"
echo "RERUN_ROOT=$RERUN_ROOT"
echo "NLP4LP_OUTPUT_DIR=$NLP4LP_OUTPUT_DIR"
echo "GEMINI_AUTO_PICK_MODEL=${GEMINI_AUTO_PICK_MODEL:-1}"
echo "GEMINI_SKIP_ON_ZERO_QUOTA=${GEMINI_SKIP_ON_ZERO_QUOTA:-}"
echo "GEMINI_SMOKE_TEST=${GEMINI_SMOKE_TEST:-0} (1 = orig only, --max-queries 5, then exit)"
echo "=========================================="

echo "which python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
python -c "from google import genai; print('google-genai import: OK')" 2>/dev/null || {
  echo "ERROR: google-genai is not available (pip install google-genai)."
  exit 1
}

# ── Model selection, then mandatory preflight (scripts/gemini_preflight.py) ───
_gemini_auto_pick="${GEMINI_AUTO_PICK_MODEL:-1}"
if [[ "$_gemini_auto_pick" != "0" && "$_gemini_auto_pick" != "false" && "$_gemini_auto_pick" != "no" && "$_gemini_auto_pick" != "off" ]]; then
  ART="${GEMINI_MODEL_SELECTED_FILE:-$REPO_ROOT/results/llm_baselines/gemini_selected_model.json}"
  mkdir -p "$(dirname "$ART")"
  echo "pick-model: probing yaml/env fallback chain (tools/llm_baselines.py) ..."
  if ! PICKED="$(python tools/llm_baselines.py pick-model --write-selected "$ART")"; then
    _pick_rc=$?
    if [[ $_pick_rc -eq 3 && "${GEMINI_SKIP_ON_ZERO_QUOTA:-}" == "1" ]]; then
      echo "Skipping Gemini baseline job: pick-model found no usable model. See $ART"
      exit 0
    fi
    echo "ERROR: pick-model failed (exit $_pick_rc). See stderr and $ART"
    exit "$_pick_rc"
  fi
  export GEMINI_MODEL="$PICKED"
  echo "GEMINI_MODEL=$GEMINI_MODEL (artifact $ART)"
else
  echo "GEMINI_AUTO_PICK_MODEL=off — using GEMINI_MODEL / configs/llm_baselines.yaml gemini.model only."
  python - <<'PY'
from tools.llm_baselines import gemini_fallback_candidates, resolve_gemini_model_from_config
print("resolved_model:", resolve_gemini_model_from_config())
print("fallback_chain:", ", ".join(gemini_fallback_candidates()))
PY
fi

_preflight_json="$RERUN_ROOT/preflight_${_run_id}.json"
echo "Mandatory Gemini preflight → $_preflight_json"
if ! python scripts/gemini_preflight.py \
  --output-json "$_preflight_json" \
  --estimate-calls \
  --estimate-queries 331 \
  --estimate-variants 3; then
  _pf_rc=$?
  if [[ $_pf_rc -eq 2 && "${GEMINI_SKIP_ON_ZERO_QUOTA:-}" == "1" ]]; then
    echo "Skipping job: hard zero quota (preflight exit 2)."
    exit 0
  fi
  echo "ERROR: gemini_preflight.py failed (exit $_pf_rc). See $_preflight_json and stderr above."
  exit "$_pf_rc"
fi

export GEMINI_SKIP_PREFLIGHT=1
echo "GEMINI_SKIP_PREFLIGHT=1 for nlp4lp_downstream_utility variant loop"

python - <<'PY'
import json
import os
import datetime
import pathlib

root = pathlib.Path.cwd()
rerun = pathlib.Path(os.environ.get("RERUN_ROOT", root / "results/rerun/gemini/run_manual"))
out = rerun / f"gemini_sbatch_meta_{os.environ.get('SLURM_JOB_ID', 'manual')}.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "run_id": os.environ.get("SLURM_JOB_ID", "manual"),
    "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gemini_model": (os.environ.get("GEMINI_MODEL") or "").strip(),
    "nlp4lp_output_dir": os.environ.get("NLP4LP_OUTPUT_DIR", ""),
    "rerun_root": str(rerun),
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out}")
PY

# ── Smoke test (optional): tiny orig run ──────────────────────────────────────
if [[ "${GEMINI_SMOKE_TEST:-0}" == "1" || "${GEMINI_SMOKE_TEST:-0}" == "true" || "${GEMINI_SMOKE_TEST:-0}" == "yes" ]]; then
  echo "=== GEMINI_SMOKE_TEST: orig, max 5 queries ==="
  python tools/nlp4lp_downstream_utility.py \
    --variant orig \
    --method gemini \
    --max-queries 5 \
    --output-dir "$NLP4LP_OUTPUT_DIR"
  echo "Smoke test finished OK (outputs under $NLP4LP_OUTPUT_DIR)."
  exit 0
fi

for variant in orig noisy short; do
  echo "=== Running Gemini baseline for ${variant} ==="
  _extra=()
  if [[ "${GEMINI_RESUME:-0}" == "1" || "${GEMINI_RESUME:-0}" == "true" || "${GEMINI_RESUME:-0}" == "yes" ]]; then
    _extra+=(--resume)
  fi
  python tools/nlp4lp_downstream_utility.py \
    --variant "${variant}" \
    --method gemini \
    --output-dir "$NLP4LP_OUTPUT_DIR" \
    "${_extra[@]}"
done

echo "Gemini baseline run completed."
echo "Per-variant artifacts: $NLP4LP_OUTPUT_DIR"
echo "Summary CSV (sidecar): $NLP4LP_OUTPUT_DIR/nlp4lp_downstream_summary.csv"
