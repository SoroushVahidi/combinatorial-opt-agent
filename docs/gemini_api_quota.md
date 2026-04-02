# Gemini API: quota, free tier, and this repo

## Recommended workflow (canonical)

1. **`preflight`** — `python tools/llm_baselines.py preflight` in the **same venv** you use for batch jobs. Confirms the resolved model (default **`gemini-2.5-flash-lite`** from `configs/llm_baselines.yaml`) is listed and passes a minimal `generateContent` probe.
2. **`pick-model`** (optional) — Run if **`preflight`** exits **2** (hard zero quota on the current model), or to write the selection artifact. **SLURM** users: `batch/learning/run_gemini_llm_baselines.sbatch` runs **`pick-model` before `preflight` by default** (`GEMINI_AUTO_PICK_MODEL` defaults to **on**; set **`GEMINI_AUTO_PICK_MODEL=0`** to use only yaml / `GEMINI_MODEL`).
3. **Batch / long run** — Submit **`run_gemini_llm_baselines.sbatch`** (or `tools/nlp4lp_downstream_utility.py`). After a successful **`preflight`** in the batch script, **`GEMINI_SKIP_PREFLIGHT=1`** avoids repeating probes per variant.

## Four kinds of messages (don’t confuse them)

| What you see | Kind | Notes |
|--------------|------|--------|
| API error with **`limit: 0`** (free-tier metric); preflight exit **2** | **Hard zero quota** | Not fixable by retries; change model, billing, or use **`GEMINI_SKIP_ON_ZERO_QUOTA=1`** to soft-skip. |
| **429** with a **positive** limit and “retry in Xs” / rate limit | **Transient** | `tools/llm_baselines.py` retries with backoff (hard zero is **not** retried). |
| Python **`FutureWarning`** / deprecated **`google.generativeai`** on import | **SDK deprecation** | Does not mean your key or model is wrong; see **[MIGRATION_GOOGLE_GENAI.md](MIGRATION_GOOGLE_GENAI.md)**. |
| **`pthread_create failed: Resource temporarily unavailable`** (e.g. Wulver) | **HPC / thread** | **`GEMINI_LIMIT_RUNTIME_THREADS=1`** (default in the sbatch script) caps OMP/BLAS threads **before** the client loads; **does not** change quota or API behavior. Preflight may still exit **0** despite warnings. |

### Wulver copy-paste (current defaults: `gemini-2.5-flash-lite`, auto-pick, thread cap, soft-skip optional)

```bash
cd ~/combinatorial-opt-agent   # or your clone path
source venv/bin/activate
module load python/3.10        # if you use modules
export GEMINI_API_KEY="..."    # AI Studio key
# Optional: if all models are limit:0, exit 0 instead of failing the Slurm job:
export GEMINI_SKIP_ON_ZERO_QUOTA=1
sbatch batch/learning/run_gemini_llm_baselines.sbatch
```

To **disable** auto-`pick-model` (yaml / `GEMINI_MODEL` only): `export GEMINI_AUTO_PICK_MODEL=0` before `sbatch`. To **disable** the thread cap: `export GEMINI_LIMIT_RUNTIME_THREADS=0`.

---

## Account-specific models — run preflight before Gemini jobs

- **Which model works is account-specific** (free tier, region, billing, model rollout). Defaults in `configs/llm_baselines.yaml` are a starting point only.
- **Always run** `python tools/llm_baselines.py preflight` (and/or `pick-model`) in the **same environment** as your SLURM or long baseline job **before** relying on unattended runs.

### Wulver-tested snapshot (one API key / account)

Runtime checks on a **single** tested environment reported:

| Model | Result |
|-------|--------|
| `gemini-2.0-flash` | Hard **zero** free-tier quota (`limit: 0` in API error) |
| `gemini-2.5-flash-lite` | **`preflight`** succeeded; **`pick-model`** resolves to this model |

**Your** Google account may differ—verify with `preflight` / `pick-model` locally.

## Standard-text models — default fallback order

Only **text / flash-style** model ids are listed in config (no image/video-only models). Order is defined in **`configs/llm_baselines.yaml`**:

1. **`gemini.model`** (default: **`gemini-2.5-flash-lite`**)
2. **`gemini.fallback_models`** in order: `gemini-2.0-flash-lite` → `gemini-2.0-flash` → `gemini-1.5-flash`

**`pick-model`** candidate order: **`GEMINI_MODEL`** (if set) → yaml **`gemini.model`** (if distinct) → **`fallback_models`** → **`GEMINI_MODEL_FALLBACKS`**.

## Wulver / HPC: `pthread_create failed` or thread warnings

Logs may show **`pthread_create failed: Resource temporarily unavailable`** while gRPC or client libraries start. **`preflight` can still exit 0**—the probe may succeed despite warnings.

**Mitigation (not quota):** `batch/learning/run_gemini_llm_baselines.sbatch` exports **`GEMINI_LIMIT_RUNTIME_THREADS=1`** by default. That enables `tools/llm_baselines.py` to set **`OMP_NUM_THREADS`**, **`OPENBLAS_NUM_THREADS`**, **`MKL_NUM_THREADS`**, **`NUMEXPR_MAX_THREADS`**, and **`TOKENIZERS_PARALLELISM`** *before* importing **`google.generativeai`**, and the same hook runs at **CLI entry** (`main()`), **`gemini_probe_minimal`**, and **`LLMTwoStageBaseline`** construction. It **does not** change free-tier limits, auth, or hard-zero detection.

Disable the cap for this job: `export GEMINI_LIMIT_RUNTIME_THREADS=0` before `sbatch` (or interactive Python).

## SDK migration (`google.generativeai` → `google.genai`)

The repo still uses the **`google.generativeai`** package (deprecated). A staged move to **`google.genai`** is documented in **[MIGRATION_GOOGLE_GENAI.md](MIGRATION_GOOGLE_GENAI.md)**.

---

## Default workflow

Pick **one** of these patterns (both are valid):

### A — Interactive: preflight → optional pick-model → baseline

Aligns with **Recommended workflow** above.

1. **`preflight`** on the default yaml model (or **`GEMINI_MODEL`**).
2. If exit **2** (hard zero) or you need a different id: **`pick-model`** (`--persist-selected` to write JSON).
3. **`preflight`** again on the chosen model.
4. Run **`nlp4lp_downstream_utility.py`** or **`sbatch`**.

### B — SLURM batch (defaults)

`run_gemini_llm_baselines.sbatch` runs **`pick-model`** (unless **`GEMINI_AUTO_PICK_MODEL=0`**), then **`preflight`**, then the variant loop with **`GEMINI_SKIP_PREFLIGHT=1`**. Matches “optional pick-model” in the sense that you can turn auto-pick off and rely on yaml only.

Creating another API key in the same AI Studio project **does not** change quota; use a model with nonzero limits, billing, or another project.

### Safest free-tier pipeline (soft-skip on zero quota)

Use this when you want **other experiment steps to continue** if every model hits **`limit: 0`**:

```bash
export GEMINI_API_KEY="..."   # already configured for your account

# 1–2: discover and persist a usable model (configure fallback_models in yaml first)
# Prints one model id on stdout; writes results/llm_baselines/gemini_selected_model.json (or $GEMINI_MODEL_SELECTED_FILE)
export GEMINI_MODEL="$(python tools/llm_baselines.py pick-model --persist-selected)"

# 3: verify (optional if you trust pick-model)
python tools/llm_baselines.py preflight

# 4: downstream — skips Gemini instead of failing if preflight would hit hard zero quota
export GEMINI_SKIP_ON_ZERO_QUOTA=1
python tools/nlp4lp_downstream_utility.py --method gemini --variant orig
```

For **SLURM**, you can set `GEMINI_SKIP_ON_ZERO_QUOTA=1` in the environment so **`preflight`** (exit **2**) or **`pick-model`** (exit **3**) causes the job to **exit 0** and skip the Gemini baseline (see `batch/learning/run_gemini_llm_baselines.sbatch`).

### Selection artifact (JSON)

- **`pick-model --write-selected PATH`** or **`--persist-selected`** (default path: **`$GEMINI_MODEL_SELECTED_FILE`**, else **`results/llm_baselines/gemini_selected_model.json`**) writes a small JSON file:
  - **`ok`**, **`model`**, **`selected_at`**, **`candidates_ordered`**, **`failures`** (per-candidate errors before success, or all failures if none worked), **`source`** (`pick-model`).
- On **total failure** (`pick-model` exit **3**), the same path is still written with **`ok: false`**, **`reason: all_candidates_failed`**, and a **`summary`** string so later steps and log parsers can see what happened.
- **`preflight --write-selected PATH`** (on success only) writes the same schema with **`source: preflight`** if you want a record without using `pick-model`.

`results/llm_baselines/` is under the repo’s generic `results/*` gitignore rule (except tracked paper paths); treat the artifact as a **local run log**, not something to commit.

### `pick-model` inside `sbatch` (default **on**)

**Default:** **`GEMINI_AUTO_PICK_MODEL`** is treated as **on** (`1`) unless you set **`0`**, **`false`**, **`no`**, or **`off`**. The script then:

1. Runs **`pick-model --write-selected "$ART"`** (`$GEMINI_MODEL_SELECTED_FILE` or **`results/llm_baselines/gemini_selected_model.json`**).
2. **`export GEMINI_MODEL=`** from stdout.
3. Runs **`preflight`**, then **`GEMINI_SKIP_PREFLIGHT=1`** for variants.

If **`pick-model` exits 3** and **`GEMINI_SKIP_ON_ZERO_QUOTA=1`**, the job **exits 0**. Otherwise exit **3** + failure artifact.

---

## What “invalid key” vs “quota” looks like

- A **working key** still returns **HTTP 429** with text like  
  `Quota exceeded for metric: ... generate_content_free_tier_requests, **limit: 0**`  
  for some models/paths. That means the request reached Google, but this **account/project/plan has no usable free-tier quota** for that metric—**not** an authentication bug.

- **Retries/backoff in code cannot bypass a hard `limit: 0`.**  
  The repo treats that as a **hard stop** (no retry loop). Transient 429s with a positive limit and a “retry in Xs” hint are still retried.

## What you can do (outside the repo)

1. In [Google AI Studio](https://aistudio.google.com/), check **usage & limits** for the Gemini API and whether **billing / paid** access is required for the model you use.
2. Use a **model that shows nonzero free-tier limits** for this account (see below), or switch to a **different project/account** with usable quota.
3. Official reference: [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## Free tier: usable only when limits are nonzero

**Free-tier Gemini is only reliable for models/paths where your account shows a nonzero limit** in AI Studio / API error text. If the API reports **`limit: 0`** for `generate_content_free_tier_requests`, no amount of code changes will make that model usable on that tier—you need billing, another model with quota, or another project.

## Repo tools (preflight & discovery)

| Command | Purpose |
|--------|---------|
| `python tools/llm_baselines.py preflight` | Check `GEMINI_API_KEY`, ensure the selected model appears in `list_models` with `generateContent`, run **one** minimal `generateContent` probe. **Exit 0** = OK; **exit 2** = hard zero quota; **exit 1** = other error. Optional **`--write-selected PATH`** on success. |
| `python tools/llm_baselines.py list-models` | Print models for this key that support `generateContent`. |
| `python tools/llm_baselines.py pick-model` | Try candidates in order until a minimal probe succeeds. **Exit 0** = success (prints model id on stdout); **exit 3** = **all** candidates failed (see stderr + artifact); **exit 1** = unexpected error. |
| `python tools/llm_baselines.py discover-usable --max-probes N` | Probe up to **N** listed models (expensive: one request per model) to see which return **ok** vs **limit_zero**. |

Configuration:

- `configs/llm_baselines.yaml` → `gemini.model`, optional `gemini.fallback_models`
- Env: **`GEMINI_MODEL`**, **`GEMINI_MODEL_FALLBACKS`** (comma-separated), **`GEMINI_MODEL_SELECTED_FILE`** (artifact path for `--persist-selected` / batch)

## Skipping Gemini when quota is zero (pipelines)

| Variable / flag | Effect |
|-----------------|--------|
| `GEMINI_SKIP_ON_ZERO_QUOTA=1` or `--gemini-skip-on-zero-quota` | If preflight would fail with hard zero quota, **skip** the Gemini baseline run (no new metrics rows) instead of erroring. Also maps **`pick-model` exit 3** to **exit 0** in **`run_gemini_llm_baselines.sbatch`**. |
| `GEMINI_SKIP_PREFLIGHT=1` or `--skip-gemini-preflight` | Skip list+probe (e.g. after a successful **`preflight`** in a batch script so each variant does not re-probe). |
| `GEMINI_AUTO_PICK_MODEL` | In **`run_gemini_llm_baselines.sbatch`**, **`1`** (default) runs **`pick-model`** before **`preflight`**; set **`0`** to skip auto-pick. |
| `GEMINI_LIMIT_RUNTIME_THREADS` | **`1`** (default in sbatch) enables OMP/BLAS thread caps before the Gemini client; **`0`** disables. |

SLURM `batch/learning/run_gemini_llm_baselines.sbatch` optionally runs **`pick-model`**, then **`preflight`**, then sets **`GEMINI_SKIP_PREFLIGHT=1`**. If preflight exits **2** and **`GEMINI_SKIP_ON_ZERO_QUOTA=1`**, the job exits **0** and skips the baseline.

## Model name (404 NOT_FOUND)

If you see **`NotFound`** for `models/...`, pick a model id that supports **`generateContent`** for your API version. Use `list-models` or AI Studio.

## Retries vs hard quota (implementation)

- **`tools/llm_baselines.py`** raises **`GeminiHardQuotaError`** when the API message indicates **`limit: 0`** on a quota error; the async retry loop **does not** retry that case.
- Other 429s may still use backoff and “retry in Xs” parsing.
