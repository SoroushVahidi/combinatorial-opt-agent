# Gemini NLP4LP baseline — stabilization and rerun workflow

**Role:** Infrastructure and workflow for optional **Gemini** runs of `tools/nlp4lp_downstream_utility.py` — **not** a substitute for camera-ready paper tables in `results/paper/eaai_camera_ready_tables/`.  
**Public status summary:** [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) · **Headline metrics:** [`docs/RESULTS_PROVENANCE.md`](RESULTS_PROVENANCE.md)

> **Honest scope:** Code and Slurm wiring for Gemini have been stabilized (preflight, cache keys, partial save, `results/rerun/` layout). **A successful full Gemini benchmark rerun is not claimed here** unless you have committed artifacts under `results/rerun/gemini/…` and logs to match. OpenAI optional baseline artifacts under `results/paper/` reflect a completed historical run; Gemini rows in `nlp4lp_llm_baseline_comparison.csv` may remain empty until a rerun lands.

This document records the **code and workflow changes** for robust Gemini runs on Slurm, and the **verification performed in this workspace** (without fabricating API results).

## 1. Code and configuration changes

### 1.1 Slurm bootstrap (`batch/learning/run_gemini_llm_baselines.sbatch`)

- **Removed** sourcing of `~/.bashrc` under `set -u` (this caused `/etc/bashrc: BASHRCSOURCED: unbound variable` and aborted jobs before Python).
- **API key** must come from: repo `.env`, Slurm `--export` / `--export-file`, or `GEMINI_API_KEY_FILE`. Clear error if missing.
- **Order of operations**: `pick-model` (when `GEMINI_AUTO_PICK_MODEL` is on) → **mandatory** `scripts/gemini_preflight.py` (list + probe + call budget estimate) → variant loop.
- **Output layout**: defaults to `results/rerun/gemini/run_${SLURM_JOB_ID}/` via `RERUN_ROOT` / `NLP4LP_OUTPUT_DIR` (overridable env vars).
- **Smoke mode**: `GEMINI_SMOKE_TEST=1` runs `orig` only with `--max-queries 5`, then exits 0.
- **Resume**: set `GEMINI_RESUME=1` to pass `--resume` into `nlp4lp_downstream_utility.py` (skips `query_id`s already in the per-query CSV for that output dir).

### 1.2 Mandatory preflight (`scripts/gemini_preflight.py`)

- Validates `GEMINI_API_KEY`.
- Lists models with `generateContent`, checks configured model, runs a **minimal** `generateContent` (unless `--no-probe`).
- Writes JSON to `--output-json` for traceability.
- `--estimate-calls`: logs approximate `generateContent` count for full `orig` + `noisy` + `short` (331 queries × 2 stages × 3 variants ≈ **1986** calls).
- Exit codes: `0` ok, `1` config/model, `2` hard zero quota, `3` pick-all-failed (when using `--pick-instead`).

### 1.3 Client and cache (`tools/llm_baselines.py`)

- Gemini path already uses **`google.genai`** (`from google import genai`).
- **Cache key version 2**: disk cache entries now include **resolved model id**, temperature, and max output tokens so switching models does not reuse stale JSON.
- **`classify_gemini_quota_failure`**, **`is_gemini_transient_quota_or_rate_limit`**, and **`LlmBaselineRunInterrupted`** support fail-fast messaging and partial saves.

### 1.4 Downstream utility (`tools/nlp4lp_downstream_utility.py`)

- **`--output-dir`**: write artifacts outside `results/paper` (for reruns).
- **`--max-queries N`**: smoke / partial eval.
- **`--resume`**: skip completed `query_id`s from existing per-query CSV; merge rows; **skips per-type summary upsert** (documented in stderr) because types are not recomputed from CSV alone.
- **`--no-llm-save-partial-on-failure`**: disable partial CSV + `.interrupt.json` on quota errors (default: partial save **on** for Gemini/OpenAI).
- On LLM quota/API interrupt: writes partial per-query CSV + JSON aggregate + manifest; raises **`LlmBaselineRunInterrupted`** → CLI **exit 5**.

### 1.5 Fallback models (`configs/llm_baselines.yaml`)

- Removed default reliance on **`gemini-1.5-flash`** (404 on current API surfaces).
- Fallback order now includes **`gemini-2.5-flash`** and **`gemini-2.0-flash-lite`** before **`gemini-2.0-flash`**.

### 1.6 Git tracking (`.gitignore`)

- **`results/rerun/**`** is whitelisted so rerun bundles can be committed when desired.

## 2. OpenAI baseline

- **Not rerun** as part of this change set; the OpenAI path was already completing successfully and the interface remains compatible.

## 3. Verification in this workspace (no API calls)

| Step | Command | Result |
|------|---------|--------|
| Syntax / import | `python -m py_compile scripts/gemini_preflight.py tools/llm_baselines.py`; `python -c "import tools.nlp4lp_downstream_utility"` | OK |
| Preflight without key | `unset GEMINI_API_KEY; python scripts/gemini_preflight.py` | Exit **1**, clear `GEMINI_API_KEY` error |
| CLI | `python tools/nlp4lp_downstream_utility.py --help` | OK |

**Slurm job IDs**: none were submitted from this automation (no `GEMINI_API_KEY` in the execution environment used for verification). On the cluster, capture IDs via `sacct` or `echo $SLURM_JOB_ID` inside the batch script log banner.

## 4. Exact commands to run on the cluster (recommended)

```bash
cd /mmfs1/home/sv96/combinatorial-opt-agent

# 1) Smoke (5 queries, orig only)
export GEMINI_API_KEY=...   # or GEMINI_API_KEY_FILE, or rely on .env
export GEMINI_SMOKE_TEST=1
sbatch batch/learning/run_gemini_llm_baselines.sbatch

# 2) Full rerun (after smoke succeeds): unset smoke
unset GEMINI_SMOKE_TEST
sbatch batch/learning/run_gemini_llm_baselines.sbatch
```

Artifacts (default):

- `results/rerun/gemini/run_<JOBID>/preflight_<JOBID>.json`
- `results/rerun/gemini/run_<JOBID>/paper/` — per-query CSV, JSON, summary sidecars
- `logs/learning/run_gemini_llm_baselines_<JOBID>.out` / `.err`

## 5. Comparison to prior failures

| Prior failure | Mitigation |
|---------------|------------|
| `bashrc` + `set -u` | No `~/.bashrc` sourcing; explicit key/env |
| Stale model id (404) | Updated yaml fallbacks; `pick-model` + list_models preflight |
| `limit: 0` free tier | Preflight exit **2**; optional `GEMINI_SKIP_ON_ZERO_QUOTA=1` to skip job |
| Tiny daily cap mid-run | Call budget printed in preflight; on-disk **cache** (model-keyed); **partial** CSV + resume; conservative `--max-queries` / smoke |

## 6. Final status (this commit)

- **Implementation**: complete in repo.
- **Full Gemini orig / noisy / short reruns**: **not executed here** — requires your API key and quota on the cluster; use the commands above.
- **Cache / resume**: implemented; validate after smoke by checking `cache/llm_baselines/gemini/` and rerunning smoke (should hit cache for repeated queries).

Fill in after your cluster runs:

- Chosen `GEMINI_MODEL` (from `gemini_selected_model.json` or preflight JSON).
- Slurm job ID(s).
- Whether `orig` / `noisy` / `short` completed or were blocked by quota.
