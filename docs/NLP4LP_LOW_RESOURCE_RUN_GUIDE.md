# NLP4LP 7-Method Pipeline: Low-Resource / Wulver Run Guide

This guide explains how to run the full 7-method NLP4LP evaluation pipeline on **Wulver** (or other constrained compute nodes) without subprocess or thread exhaustion. No modeling or scoring logic is changed—only execution is made robust.

## Problems addressed

- **BlockingIOError: [Errno 11] Resource temporarily unavailable** — Caused by forking subprocesses (e.g. `run_nlp4lp_focused_eval.py` launching `nlp4lp_downstream_utility.py` per method).
- **RuntimeError: can't start new thread** — Caused by HuggingFace `load_dataset` or tokenizers spawning too many threads on limited nodes.

## Safe mode (recommended on Wulver)

Use **`--safe`** on both scripts that touch subprocess/HF:

1. **`tools/run_nlp4lp_focused_eval.py`** — Runs the 7 downstream methods. With `--safe`:
   - No subprocess: all 7 methods run **in-process**, sequentially.
   - **Checkpointing**: If `results/paper/nlp4lp_downstream_summary.csv` already has a row for `(variant, baseline)`, that method is **skipped** (resume-friendly).
   - Low-resource env (thread limits) is set before any heavy imports.

2. **`tools/build_nlp4lp_per_instance_comparison.py`** — Builds per-instance comparison CSV. With `--safe`:
   - Thread env is set before importing the downstream utility (so HuggingFace/tokenizers see limits).
   - If **`NLP4LP_GOLD_CACHE`** is set and the cache file exists, gold data is **loaded from that file** instead of calling `load_dataset` (avoids HF thread pool).

## Exact commands (safe sequential mode)

Run from the **project root** (e.g. on Wulver after `cd combinatorial-opt-agent` and activating your venv).

### 1. Set environment (optional but recommended)

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
# Gold cache: step 1 writes it, step 2 reads it (so HF dataset loads only once)
export NLP4LP_GOLD_CACHE="$(pwd)/results/paper/nlp4lp_gold_cache.json"
```

### 2. Run 7 methods (sequential, in-process, with resume)

```bash
mkdir -p results/paper
python tools/run_nlp4lp_focused_eval.py --variant orig --safe
```

- Outputs: `results/paper/nlp4lp_downstream_summary.csv` (upserted), `nlp4lp_focused_eval_summary.csv`, and per-method per-query CSVs/JSONs.
- If a method already has a row in the summary CSV, it is skipped. To **rerun from scratch**, delete `results/paper/nlp4lp_downstream_summary.csv` (or remove the rows for the methods you want to rerun).

### 3. Build per-instance comparison

```bash
python tools/build_nlp4lp_per_instance_comparison.py --variant orig --safe
```

- If `NLP4LP_GOLD_CACHE` is set and the file from step 2 exists, gold is loaded from cache (no HF load).
- Output: `results/paper/nlp4lp_focused_per_instance_comparison.csv`.

### 4. Downstream analysis (no HF/subprocess; safe as-is)

```bash
python tools/analyze_nlp4lp_downstream_disagreements.py
python tools/build_nlp4lp_failure_audit.py
python tools/analyze_nlp4lp_three_bottlenecks.py
```

## Using the SLURM job script

From project root:

```bash
# Safe mode: set env so the job uses --safe and gold cache
export NLP4LP_SAFE=1
sbatch jobs/run_nlp4lp_focused_eval.slurm
```

The script already sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TOKENIZERS_PARALLELISM`, `HF_DATASETS_DISABLE_PROGRESS_BARS`, and `NLP4LP_GOLD_CACHE`. With `NLP4LP_SAFE=1` it passes `--safe` to both Python scripts.

## Reusing existing results

- **Summary CSV**: If `results/paper/nlp4lp_downstream_summary.csv` already contains rows for some `(variant, baseline)` pairs, `run_nlp4lp_focused_eval.py --safe` will **skip** those and only run missing methods. You can reuse any existing rows.
- **Per-instance comparison**: `build_nlp4lp_per_instance_comparison.py` overwrites `nlp4lp_focused_per_instance_comparison.csv` each run. It does not checkpoint by query; re-run only if you need a fresh comparison (e.g. after changing methods).
- **Artifacts in `results/paper/`**: All other outputs (disagreement labels, failure audit, three-bottlenecks) are derived from the summary and per-instance CSV. If those inputs are unchanged, you can keep or regenerate them.

## Resuming from partial progress

1. After step 2, if the job was killed mid-way, run the same command again:
   ```bash
   python tools/run_nlp4lp_focused_eval.py --variant orig --safe
   ```
   Already-completed methods (rows present in `nlp4lp_downstream_summary.csv`) are skipped; only missing methods run.
2. To force rerun of a specific method, open `nlp4lp_downstream_summary.csv`, delete the row for that `(variant, baseline)`, then run step 2 again.

## Confirmation: modeling unchanged

- **Scoring, assignment logic, and evaluation definitions** are unchanged. The same `run_setting`, `_score_mention_slot_*`, and aggregation logic run; only the **invocation** differs (in-process vs subprocess, single load of gold/catalog, thread env).
- Results in `results/paper/` produced with `--safe` should match those produced without `--safe` on a machine where subprocess and HF run successfully (modulo floating point / ordering where irrelevant).

## Files changed (reference)

| File | Change |
|------|--------|
| `tools/run_nlp4lp_focused_eval.py` | `--safe`: no subprocess; in-process `run_single_setting`; skip if summary row exists; set thread env. |
| `tools/nlp4lp_downstream_utility.py` | `_apply_low_resource_env()`; `_load_hf_gold(..., use_cache)` + `NLP4LP_GOLD_CACHE` read/write; `run_single_setting()` for single (variant, baseline, assignment_mode). |
| `tools/build_nlp4lp_per_instance_comparison.py` | `--safe`; parse args then set env **before** importing dutil; defer dutil import; call `_apply_low_resource_env()`. |
| `jobs/run_nlp4lp_focused_eval.slurm` | Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TOKENIZERS_PARALLELISM`, `HF_DATASETS_DISABLE_PROGRESS_BARS`, `NLP4LP_GOLD_CACHE`; add `NLP4LP_SAFE` and pass `--safe` when set. |
