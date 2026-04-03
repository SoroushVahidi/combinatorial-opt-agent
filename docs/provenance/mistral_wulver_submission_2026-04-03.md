# Mistral baseline — Wulver submission log (2026-04-03)

**Purpose:** Honest provenance for real `sbatch` attempts. **No Mistral benchmark metrics** were produced; jobs failed before preflight/API work.

## Environment

- **Host (submission):** `login01` (NJIT Wulver login node, path `/mmfs1/home/sv96/combinatorial-opt-agent`).
- **Automation note:** The non-interactive shell used for the first submission **did not** carry `MISTRAL_API_KEY`, so Slurm jobs inherited an empty key even when `sbatch --export=ALL` was used.

## Code fix (partition)

- **Issue:** `#SBATCH --partition=cpu` was rejected: `invalid partition specified: cpu`.
- **Change:** Aligned with OpenAI/Gemini LLM batches: `partition=gpu`, `gres=gpu:1`, `qos=standard` (API-bound workload; GPU unused). See commit touching `batch/learning/run_mistral_llm_baselines.sbatch`.

## Job 902367

| Field | Value |
|------|--------|
| **Command** | `export MISTRAL_ORIG_ONLY=1 && sbatch batch/learning/run_mistral_llm_baselines.sbatch` |
| **Scope intended** | Orig-only (`MISTRAL_ORIG_ONLY=1`), full 331 queries after preflight |
| **State** | `FAILED` (exit `1:0`), elapsed `00:00:00` |
| **Root cause** | `MISTRAL_API_KEY is not set` in batch job environment (stdout in gitignored `logs/learning/run_mistral_llm_baselines_902367.out`) |
| **Artifacts under `results/rerun/mistral/`** | None (script exited before `mkdir` usage of rerun tree with content) |

## Job 902368

| Field | Value |
|------|--------|
| **Command** | `export MISTRAL_ORIG_ONLY=1 && sbatch --export=ALL batch/learning/run_mistral_llm_baselines.sbatch` |
| **State** | `FAILED` (exit `1:0`), elapsed `00:00:00` |
| **Root cause** | Same — submitting shell had **no** `MISTRAL_API_KEY`, so `--export=ALL` exported no secret |
| **Log path (local, gitignored)** | `logs/learning/run_mistral_llm_baselines_902368.out` |

## Classification

**Blocked** — Slurm/bootstrap + **missing API key in the job environment** (not an auth failure at Mistral’s API).

## What a maintainer should do next

1. On a Wulver login session: `export MISTRAL_API_KEY=...` (or add key to repo `.env`, or `MISTRAL_API_KEY_FILE` + export that path).
2. `export MISTRAL_ORIG_ONLY=1` for a cheaper first completed run, or omit for full `orig/noisy/short`.
3. `sbatch --export=ALL batch/learning/run_mistral_llm_baselines.sbatch`
4. After completion, copy job ID, check `results/rerun/mistral/run_<JOBID>/preflight_*.json` and per-query CSV row counts; update [`docs/MISTRAL_RERUN_REPORT.md`](../MISTRAL_RERUN_REPORT.md) section 5.

**Do not** cite this file as evidence of a successful Mistral benchmark.
