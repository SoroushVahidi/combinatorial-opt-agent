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
- **Smoke:** `MISTRAL_SMOKE_TEST=1` → `orig`, `--max-queries 5`, exit 0 after smoke (still runs preflight first).
- **Full paper-style sweep (default):** runs **`orig` → `noisy` → `short`** (same as OpenAI/Gemini batches). Wall time **`24:00:00`** on `cpu` — increase `#SBATCH --time` if your site quota allows and runs hit timeouts.
- **Orig-only (legacy):** `MISTRAL_ORIG_ONLY=1` → skips `noisy`/`short` (preflight estimate uses 1 variant).
- **Resume:** `MISTRAL_RESUME=1` → passes `--resume` to the utility (partial per-query CSV must exist).
- **Partition / QoS (Wulver):** **`gpu`** + **`--gres=gpu:1`** + **`qos=standard`** — the cluster does not accept a `cpu` partition name for this workflow; the job is still **API-bound** (GPU unused). For other sites, edit `#SBATCH` lines in `batch/learning/run_mistral_llm_baselines.sbatch` to match local `sinfo`.

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

**Wulver — smoke (5 queries, orig):**

```bash
export MISTRAL_API_KEY=...   # or use repo .env / MISTRAL_API_KEY_FILE per batch script header
export MISTRAL_SMOKE_TEST=1
sbatch --export=ALL batch/learning/run_mistral_llm_baselines.sbatch
```

**Wulver — full run — all three variants (default; after smoke passes):**

```bash
unset MISTRAL_SMOKE_TEST MISTRAL_ORIG_ONLY
# optional: export MISTRAL_RESUME=1
sbatch --export=ALL batch/learning/run_mistral_llm_baselines.sbatch
```

**Wulver — orig-only (cheaper first full orig slice):**

```bash
unset MISTRAL_SMOKE_TEST
export MISTRAL_ORIG_ONLY=1
sbatch --export=ALL batch/learning/run_mistral_llm_baselines.sbatch
```

On sites where the scheduler forwards your login environment by default, plain `sbatch` may work; **Wulver submissions in Apr 2026 required `--export=ALL`** when the key was only in the interactive shell.

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

---

## 5. Execution register (honest status)

**Last registration update:** 2026-04-03 (Wulver `login01`). Detailed submission log: [`docs/provenance/mistral_wulver_submission_2026-04-03.md`](provenance/mistral_wulver_submission_2026-04-03.md).

| Field | Value |
|------|--------|
| **Latest classification** | **Blocked** — jobs exited before Mistral API preflight (no key in Slurm job environment). |
| **Job IDs (2026-04-03)** | **902367** — `sbatch` without `--export=ALL`; **902368** — `sbatch --export=ALL` (submitting shell still had no `MISTRAL_API_KEY`). |
| **Submission commands** | `export MISTRAL_ORIG_ONLY=1 && sbatch batch/learning/run_mistral_llm_baselines.sbatch` → 902367; same + `--export=ALL` → 902368. |
| **Run scope requested** | **Orig-only** (`MISTRAL_ORIG_ONLY=1`) — chosen as first real attempt (331 queries × 2 stages) to maximize chance of a **complete** outcome vs a 3-variant sweep. |
| **Preflight passed?** | **No** — batch script failed at key check. |
| **Model resolved** | N/A (no `preflight_*.json` written). Default would be `mistral-small-latest` from yaml / `MISTRAL_MODEL`. |
| **Slurm logs (gitignored)** | `logs/learning/run_mistral_llm_baselines_902367.out`, `_902368.out` — both report `ERROR: MISTRAL_API_KEY is not set.` |
| **Infrastructure fix same day** | Invalid Slurm partition `cpu` on Wulver → batch file updated to **`gpu`** + **`gres=gpu:1`** + **`qos=standard`** (matches OpenAI/Gemini batches). |
| **Smoke-test job ID** | *(not run in this batch; use `MISTRAL_SMOKE_TEST=1` if desired)* |
| **Full 3-variant job ID** | *(pending — submit with key present; omit `MISTRAL_ORIG_ONLY` for default orig→noisy→short)* |
| **Provider priority (unchanged)** | **1)** Mistral. **2)** If Mistral preflight fails (401/403/429), **OpenAI** or **Gemini** only with explicit doc update — **no silent switch**. **3)** Avoid Gemini free tier when logs show hard zero quota ([`GEMINI_RERUN_REPORT.md`](GEMINI_RERUN_REPORT.md)). |

**Next maintainer step:** On Wulver, export a real `MISTRAL_API_KEY` (or use `.env` / `MISTRAL_API_KEY_FILE`), then `sbatch --export=ALL …`. After a job **completes**, fill row counts and point `Artifact root` to `results/rerun/mistral/run_<JOBID>/`.

### 5.1 Security note

Use **`MISTRAL_API_KEY_FILE`** or Slurm **`--export-file`** for secrets; avoid echoing keys into shared shell transcripts. Rotate any key that may have leaked.

---

## 6. Provider fallback (if Mistral blocks)

1. Capture **`preflight_*.json`** and Slurm **`.err`** from the failed Mistral job.
2. If failure is **auth** (401/403) or **hard quota** (429 with no retry path), try **OpenAI** with the existing GPU batch script (requires `OPENAI_API_KEY` and HF access for NLP4LP).
3. If trying **Gemini**, run **`scripts/gemini_preflight.py`** first; if exit **2** (zero quota) or repeated hard 429, **stop** and document — do not claim a benchmark attempt succeeded.

Each provider attempt gets its own `results/rerun/<provider>/run_<JOBID>/` tree and a short note under the Gemini/Mistral report (or a future unified `LLM_RERUN_LOG.md` if you merge logs).
