#!/bin/bash
#SBATCH --job-name=mistral_nlp4lp
#SBATCH --output=logs/learning/run_mistral_llm_baselines_%j.out
#SBATCH --error=logs/learning/run_mistral_llm_baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}}"
cd "$REPO_ROOT"
mkdir -p logs/learning results/llm_baselines results/rerun/mistral

# ── Bootstrap: do NOT source ~/.bashrc under `set -u`. ──
if [[ -f "$REPO_ROOT/.env" ]]; then
  set +u
  set +a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set -a
  set -u
fi

if [[ -z "${MISTRAL_API_KEY:-}" && -n "${MISTRAL_API_KEY_FILE:-}" ]]; then
  if [[ -f "$MISTRAL_API_KEY_FILE" ]]; then
    export MISTRAL_API_KEY="$(tr -d '\r\n' < "$MISTRAL_API_KEY_FILE")"
  else
    echo "ERROR: MISTRAL_API_KEY_FILE is set but not found: $MISTRAL_API_KEY_FILE"
    exit 1
  fi
fi

if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
  echo "ERROR: MISTRAL_API_KEY is not set."
  echo "Fix: Slurm --export=ALL / --export-file=..., repo .env, or MISTRAL_API_KEY_FILE."
  exit 1
fi

if [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
  export PATH="$REPO_ROOT/venv/bin:${PATH}"
  echo "Using repo venv: $REPO_ROOT/venv"
else
  echo "WARNING: $REPO_ROOT/venv/bin/python not found; using PATH python."
fi

_run_id="${SLURM_JOB_ID:-manual}"
RERUN_ROOT="${MISTRAL_RERUN_ROOT:-$REPO_ROOT/results/rerun/mistral/run_${_run_id}}"
export RERUN_ROOT
export NLP4LP_OUTPUT_DIR="${NLP4LP_OUTPUT_DIR:-$RERUN_ROOT/paper}"
mkdir -p "$RERUN_ROOT" "$NLP4LP_OUTPUT_DIR" logs/learning

echo "=========================================="
echo "Mistral NLP4LP batch — run_id=${_run_id}"
echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname 2>/dev/null || true)"
echo "REPO_ROOT=$REPO_ROOT"
echo "RERUN_ROOT=$RERUN_ROOT"
echo "NLP4LP_OUTPUT_DIR=$NLP4LP_OUTPUT_DIR"
echo "MISTRAL_MODEL=${MISTRAL_MODEL:-}"
echo "MISTRAL_SMOKE_TEST=${MISTRAL_SMOKE_TEST:-0} (1 = orig only, --max-queries 5, then exit)"
echo "MISTRAL_RUN_ALL_VARIANTS=${MISTRAL_RUN_ALL_VARIANTS:-0} (1 = orig + noisy + short after smoke)"
echo "MISTRAL_RESUME=${MISTRAL_RESUME:-0}"
echo "=========================================="

python -c "from mistralai.client import Mistral; print('mistralai import: OK')" 2>/dev/null || {
  echo "ERROR: mistralai is not available (pip install mistralai)."
  exit 1
}

_preflight_json="$RERUN_ROOT/preflight_${_run_id}.json"
echo "Mandatory Mistral preflight → $_preflight_json"
_est_variants=1
if [[ "${MISTRAL_RUN_ALL_VARIANTS:-0}" == "1" || "${MISTRAL_RUN_ALL_VARIANTS:-0}" == "true" || "${MISTRAL_RUN_ALL_VARIANTS:-0}" == "yes" ]]; then
  _est_variants=3
fi
if ! python scripts/mistral_preflight.py \
  --output-json "$_preflight_json" \
  --estimate-calls \
  --estimate-queries 331 \
  --estimate-variants "$_est_variants"; then
  _pf_rc=$?
  echo "ERROR: mistral_preflight.py failed (exit $_pf_rc). See $_preflight_json and stderr above."
  exit "$_pf_rc"
fi

export MISTRAL_SKIP_PREFLIGHT=1

python - <<'PY'
import json
import os
import datetime
import pathlib

root = pathlib.Path.cwd()
rerun = pathlib.Path(os.environ.get("RERUN_ROOT", root / "results/rerun/mistral/run_manual"))
out = rerun / f"mistral_sbatch_meta_{os.environ.get('SLURM_JOB_ID', 'manual')}.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "run_id": os.environ.get("SLURM_JOB_ID", "manual"),
    "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "mistral_model": (os.environ.get("MISTRAL_MODEL") or "").strip(),
    "nlp4lp_output_dir": os.environ.get("NLP4LP_OUTPUT_DIR", ""),
    "rerun_root": str(rerun),
    "run_all_variants": os.environ.get("MISTRAL_RUN_ALL_VARIANTS", ""),
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out}")
PY

if [[ "${MISTRAL_SMOKE_TEST:-0}" == "1" || "${MISTRAL_SMOKE_TEST:-0}" == "true" || "${MISTRAL_SMOKE_TEST:-0}" == "yes" ]]; then
  echo "=== MISTRAL_SMOKE_TEST: orig, max 5 queries ==="
  python tools/nlp4lp_downstream_utility.py \
    --variant orig \
    --method mistral \
    --max-queries 5 \
    --output-dir "$NLP4LP_OUTPUT_DIR"
  echo "Smoke test finished OK (outputs under $NLP4LP_OUTPUT_DIR)."
  exit 0
fi

_extra=()
if [[ "${MISTRAL_RESUME:-0}" == "1" || "${MISTRAL_RESUME:-0}" == "true" || "${MISTRAL_RESUME:-0}" == "yes" ]]; then
  _extra+=(--resume)
fi

echo "=== Mistral baseline: orig (default first) ==="
python tools/nlp4lp_downstream_utility.py \
  --variant orig \
  --method mistral \
  --output-dir "$NLP4LP_OUTPUT_DIR" \
  "${_extra[@]}"

if [[ "${MISTRAL_RUN_ALL_VARIANTS:-0}" == "1" || "${MISTRAL_RUN_ALL_VARIANTS:-0}" == "true" || "${MISTRAL_RUN_ALL_VARIANTS:-0}" == "yes" ]]; then
  for variant in noisy short; do
    echo "=== Running Mistral baseline for ${variant} ==="
    python tools/nlp4lp_downstream_utility.py \
      --variant "${variant}" \
      --method mistral \
      --output-dir "$NLP4LP_OUTPUT_DIR" \
      "${_extra[@]}"
  done
else
  echo "MISTRAL_RUN_ALL_VARIANTS not set — skipping noisy/short (orig-only full run)."
fi

echo "Mistral baseline run completed."
echo "Per-variant artifacts: $NLP4LP_OUTPUT_DIR"
echo "Summary CSV (sidecar): $NLP4LP_OUTPUT_DIR/nlp4lp_downstream_summary.csv"
