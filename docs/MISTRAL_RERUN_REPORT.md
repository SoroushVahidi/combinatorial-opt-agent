# Mistral NLP4LP baseline — preflight and orig-first rerun workflow

**Role:** Infrastructure for optional **Mistral** runs of `tools/nlp4lp_downstream_utility.py` — **not** camera-ready paper tables unless you commit matching artifacts with provenance.  
**Public status:** [`docs/CURRENT_STATUS.md`](CURRENT_STATUS.md) · **Gemini parallel path:** [`docs/GEMINI_RERUN_REPORT.md`](GEMINI_RERUN_REPORT.md)

> **Honest scope:** This adds **provider wiring**, **`scripts/mistral_preflight.py`**, **`batch/learning/run_mistral_llm_baselines.sbatch`**, cache separation, and partial-save behavior consistent with Gemini/OpenAI. **No successful Mistral benchmark is claimed** until real outputs exist under `results/rerun/mistral/…` (or another output dir you document). **Cohere** is out of scope here; trial quotas are often too small for a full `orig + noisy + short` paper-style sweep—Mistral is the intended next provider to try after Gemini free-tier friction.

## Why Mistral first (vs Gemini free tier / Cohere)

- **Gemini** infrastructure is stable, but free-tier **429** / quota limits can block even small smoke tests on some accounts.
- **Mistral** API keys are available for this project; wiring matches the existing two-stage baseline (schema retrieval + JSON slot fill).
- **Cohere** is **not** implemented in this task; if added later, expect trial limits to suit **small slices** rather than full `331 × 2 stages × 3 variants` without a paid plan.

## 1. Components

### 1.1 Slurm batch (`batch/learning/run_mistral_llm_baselines.sbatch`)

- **No `~/.bashrc`** under `set -u` (same hardening as Gemini).
- **Key:** `MISTRAL_API_KEY`, or `MISTRAL_API_KEY_FILE`, or repo `.env`.
- **Order:** mandatory `scripts/mistral_preflight.py` → `MISTRAL_SKIP_PREFLIGHT=1` → downstream runs.
- **Outputs:** `MISTRAL_RERUN_ROOT` or default `results/rerun/mistral/run_${SLURM_JOB_ID}/` (`NLP4LP_OUTPUT_DIR` defaults to `…/paper`).
- **Smoke:** `MISTRAL_SMOKE_TEST=1` → `orig`, `--max-queries 5`, exit 0 after smoke.
- **Orig-first full run (default):** runs **`orig` only** unless `MISTRAL_RUN_ALL_VARIANTS=1` (then `noisy` and `short` follow).
- **Resume:** `MISTRAL_RESUME=1` → passes `--resume` to the utility.
- **Partition:** defaults to **`cpu`** (API-only workload; adjust `#SBATCH` for your site).

### 1.2 Preflight (`scripts/mistral_preflight.py`)

- Requires `MISTRAL_API_KEY`.
- One minimal `chat.complete` with JSON mode; writes JSON report to `--output-json`.
- `--estimate-calls`: logs approximate chat completions (331 × 2 × variants).
- Exit **1** on missing key or API failure (incl. 401/403/429).

### 1.3 Client and cache (`tools/llm_baselines.py`)

- Method **`mistral`**: `mistralai.client.Mistral`, `chat.complete`, `response_format={"type": "json_object"}`.
- **Disk cache:** `cache/llm_baselines/mistral/` (separate folder per provider).
- **Cache payload:** `provider`, resolved model, generation settings, and **`cache_key_version` 3** for Mistral (OpenAI/Gemini remain at v2) so blobs cannot collide.

### 1.4 Downstream (`tools/nlp4lp_downstream_utility.py`)

- **`--method mistral`** / **`--baseline mistral`**.
- Partial per-query CSV + interrupt manifest on **429 / 401 / 403** (Mistral HTTP errors); CLI exit **5** on `LlmBaselineRunInterrupted` (same as other LLMs).

### 1.5 Config (`configs/llm_baselines.yaml`)

- Block **`mistral:`** with default model **`mistral-small-latest`** (override with **`MISTRAL_MODEL`**).

## 2. Exact commands

**Local / login node — smoke (5 queries, orig):**

```bash
export MISTRAL_API_KEY=...   # or MISTRAL_API_KEY_FILE in sbatch
export MISTRAL_SMOKE_TEST=1
sbatch batch/learning/run_mistral_llm_baselines.sbatch
```

**Orig-only full run (331 queries, default batch behavior after unsetting smoke):**

```bash
unset MISTRAL_SMOKE_TEST
# optional: export MISTRAL_RESUME=1
sbatch batch/learning/run_mistral_llm_baselines.sbatch
```

**Optional — all three variants:**

```bash
export MISTRAL_RUN_ALL_VARIANTS=1
sbatch batch/learning/run_mistral_llm_baselines.sbatch
```

**Direct CLI (after preflight or with a valid key):**

```bash
python scripts/mistral_preflight.py --output-json /tmp/mistral_pf.json --estimate-calls --estimate-variants 1
python tools/nlp4lp_downstream_utility.py --variant orig --method mistral --max-queries 5 --output-dir results/rerun/mistral/manual_smoke/paper
```

## 3. What to inspect (success vs infrastructure-only)

| Signal | Meaning |
|--------|---------|
| `preflight_*.json` with `"ok": true` | Key + minimal API call succeeded |
| `nlp4lp_downstream_per_query_orig_mistral_typed.csv` with 331 rows | Orig variant completed for typed two-stage baseline |
| `nlp4lp_downstream_summary.csv` updated in the same output dir | Aggregates merged |
| `*.interrupt.json` + partial CSV | Run stopped mid-way (quota/auth); use `--resume` |

**Infrastructure-only:** code merged, preflight passes once, or Slurm job starts — **without** completed per-query CSVs — does **not** justify paper or headline benchmark claims.

## 4. Lightweight validation (no API benchmark)

```bash
python -m py_compile scripts/mistral_preflight.py tools/llm_baselines.py
python tools/nlp4lp_downstream_utility.py --help
unset MISTRAL_API_KEY; python scripts/mistral_preflight.py; echo exit=$?
```

Expected: preflight exits **1** with a clear missing-key message when the key is unset.
