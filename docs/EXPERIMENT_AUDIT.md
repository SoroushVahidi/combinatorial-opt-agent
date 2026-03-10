# Strict Experiment Audit — NLP4LP Downstream Revision

**Date of audit:** 2026-03-10  
**Branch audited:** `copilot/experiment-overview`  
**Auditor note:** This document is evidence-driven. No experiment is marked DONE unless concrete output artifacts with measured (non-placeholder, non-estimated) numbers exist.

---

## Executive Summary

| Category | Status |
|----------|--------|
| Retrieval-only (9 combinations) | ✅ DONE_AND_MEASURED |
| Full post-fix downstream benchmark (30 settings) | ❌ NOT_RUN — directory `results/eswa_revision/02_downstream_postfix/` does not exist |
| Pre-fix vs post-fix ablation (measured delta) | ❌ PLACEHOLDER_REPORT_ONLY — post-fix column is `~0.55–0.65` structural estimate, not a run |
| ESWA tables `postfix_main_metrics.csv` | ❌ PLACEHOLDER_REPORT_ONLY — every row has `source: placeholder-pre-fix-manuscript-era` |
| ESWA table `prefix_vs_postfix_ablation.csv` | ❌ PLACEHOLDER_REPORT_ONLY — post-fix column is `structural-estimate-not-measured` |
| Deterministic method comparison (pre-fix numbers) | ⚠️ PARTIALLY_DONE — pre-fix manuscript-era numbers only; post-fix TypeMatch unknown |
| Robustness across variants (downstream) | ⚠️ PARTIALLY_DONE — orig run from manuscripts; noisy/short downstream are N/A or structural |
| Learning appendix | ✅ DONE_AND_MEASURED — real GPU job 854626 |
| Runtime (retrieval half) | ✅ DONE_AND_MEASURED — locally timed |
| Runtime (downstream half) | ⚠️ PARTIALLY_DONE — estimated from docs, not timed |
| Error taxonomy | ⚠️ PARTIALLY_DONE — heuristic estimates from code audit, not end-to-end counts |

**Bottom line:** The two most important missing experiments are the **real post-fix downstream benchmark** and the **real pre-fix vs post-fix ablation**. Both require `HF_TOKEN` to load `udell-lab/NLP4LP` gold parameters and have not run. Every number in `postfix_main_metrics.csv` and `prefix_vs_postfix_ablation.csv` is a placeholder or structural estimate.

---

## Detailed Audit Table

| # | experiment_group | exact script / workflow expected to run it | expected outputs / artifacts | evidence found | status | notes |
|---|------------------|--------------------------------------------|------------------------------|----------------|--------|-------|
| A1 | Retrieval — orig variant (BM25, TF-IDF, LSA) | `training/run_baselines.py` (inline Python, see `retrieval_summary.md` for exact command) | `results/eswa_revision/01_retrieval/retrieval_results.json`, `retrieval_summary.md`, `results/eswa_revision/13_tables/retrieval_main.csv` | JSON file present with precise floats for all 9 variant×method combinations, 331 queries each; e.g. tfidf orig P@1=0.9094, elapsed_s=1.47. Summary MD timestamps 2026-03-10 and git commit e3fdaf4. | **DONE_AND_MEASURED** | Catalog has 335 entries vs manuscript's 331 (+4 entries). Delta ≤ 0.006 on most metrics; documented in `retrieval_summary.md`. |
| A2 | Retrieval — noisy variant (BM25, TF-IDF, LSA) | Same as A1 | Same JSON, noisy keys | noisy block present in `retrieval_results.json` with precise floats (e.g. tfidf P@1=0.9033, elapsed_s=0.26) | **DONE_AND_MEASURED** | See A1. |
| A3 | Retrieval — short variant (BM25, TF-IDF, LSA) | Same as A1 | Same JSON, short keys | short block present (e.g. tfidf P@1=0.7795, elapsed_s=0.24) | **DONE_AND_MEASURED** | See A1. |
| B1 | Full downstream post-fix benchmark — orig variant (all 10 methods) | `training/external/run_full_downstream_benchmark.py` triggered via `.github/workflows/nlp4lp.yml` (Phase 2) | `results/eswa_revision/02_downstream_postfix/` directory (30+ CSV/JSON files); `results/eswa_revision/13_tables/postfix_main_metrics.csv` rows with `source: measured`; `results/eswa_revision/14_reports/postfix_main_metrics.md` | Directory `results/eswa_revision/02_downstream_postfix/` **does not exist**. `postfix_main_metrics.csv` has 10 rows (orig only) each with `source: placeholder-pre-fix-manuscript-era`. `14_reports/postfix_main_metrics.md` **does not exist**. `experiment_manifest.json` records status `"BLOCKED — HF_TOKEN required"`. | **NOT_RUN** | `hf_access_check_runtime.md` says `⏳ AWAITING GITHUB ACTIONS TRIGGER`. DNS lookup for `huggingface.co` blocked in sandbox. No authenticated run of `run_full_downstream_benchmark.py` has occurred. |
| B2 | Full downstream post-fix benchmark — noisy variant (all 10 methods) | Same as B1 | `postfix_main_metrics.csv` should have 10 noisy rows with `source: measured` | No noisy rows present at all in `postfix_main_metrics.csv`. `02_downstream_postfix/` does not exist. | **NOT_RUN** | Same blocker as B1. |
| B3 | Full downstream post-fix benchmark — short variant (all 10 methods) | Same as B1 | `postfix_main_metrics.csv` should have 10 short rows with `source: measured` | No short rows present at all. | **NOT_RUN** | Same blocker as B1. |
| C1 | Pre-fix vs post-fix ablation (measured TypeMatch delta, orig) | `run_full_downstream_benchmark.py` — ablation loop (lines 333–384), patches `_is_type_match` in memory to simulate pre-fix | `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` rows with concrete measured post-fix values; `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md` | `prefix_vs_postfix_ablation.csv` post-fix column contains `~0.55–0.65`, `~0.70–0.80`, `slightly higher`, `TBD`. `source` column: `post_fix=structural-estimate-not-measured`. `14_reports/prefix_vs_postfix_ablation.md` **does not exist**. | **PLACEHOLDER_REPORT_ONLY** | Structural analysis of the fix is documented in `03_prefix_vs_postfix/prefix_vs_postfix_ablation.md`. The float fix is correct and its expected impact is explained. But no end-to-end run was performed. |
| C2 | Pre-fix vs post-fix ablation (noisy/short variants) | Same as C1 | Noisy/short rows in ablation CSV | No noisy/short rows in the ablation CSV. | **NOT_RUN** | Blocked by HF_TOKEN requirement. |
| D1 | ESWA table — `postfix_main_metrics.csv` (canonical downstream results) | `run_full_downstream_benchmark.py` (writes this file) | 30 rows (10 methods × 3 variants) with `source: measured` | 10 rows, orig only, all with `source: placeholder-pre-fix-manuscript-era (run NLP4LP benchmark workflow to replace)`. Numbers are pre-fix manuscript-era values. | **PLACEHOLDER_REPORT_ONLY** | The file explicitly labels itself as placeholder. The measured run is the ONLY way to replace it. |
| D2 | ESWA table — `prefix_vs_postfix_ablation.csv` | `run_full_downstream_benchmark.py` | Post-fix columns with real numbers | Post-fix column values: `~0.55–0.65`, `TBD`, `slightly higher`. Source: `structural-estimate-not-measured`. | **PLACEHOLDER_REPORT_ONLY** | See C1. |
| D3 | ESWA table — `retrieval_main.csv` | `training/run_baselines.py` | 9 rows (3 variants × 3 methods) with real metrics | 9 rows present with precise floats matching `retrieval_results.json`. E.g. `orig,tfidf,0.9094,0.9637,0.9334,0.9451,331`. | **DONE_AND_MEASURED** | See A1–A3 for evidence. |
| D4 | ESWA table — `deterministic_method_comparison_orig.csv` | Numbers sourced from `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` and related docs | 10 rows with pre-fix numbers | 10 rows present with pre-fix manuscript-era numbers. Source note in `04_method_comparison/deterministic_method_comparison.md`: "pre-fix for TypeMatch; post-fix TypeMatch will be higher". | **PARTIALLY_DONE** | Schema R@1, Coverage, Exact20 are valid. TypeMatch and InstReady values are pre-fix and will change after the float fix is measured. |
| D5 | ESWA table — `robustness_by_variant.csv` | Numbers for orig from manuscripts; noisy/short downstream from existing docs | 4 methods × 3 variants | 14 rows present. orig rows have real numbers. noisy/short rows for non-tfidf-typed methods are `N/A` (not measured). | **PARTIALLY_DONE** | Retrieval R@1 across variants is measured (DONE_AND_MEASURED). Downstream coverage/TypeMatch for noisy/short variants beyond tfidf_typed is not run. |
| D6 | ESWA table — `error_taxonomy_counts.csv` | Code audit and heuristic analysis | Error counts by category | 9 rows present with counts and `evidence_note` column labelling each as `est.`, `~70% of float slots mistyped pre-fix`, `est. from TypeMatch_hits`, etc. None are direct end-to-end counts from a measured run. | **PARTIALLY_DONE** | Float mismatch count (~230) derived from token-level structural analysis of `_is_type_match` (valid as code evidence). Schema miss count (31) is directly computable from retrieval results (DONE). Others are estimates. |
| D7 | ESWA table — `learning_summary.csv` | `batch/learning/train_nlp4lp_real_data_only_learning_check.sbatch` (HPC GPU job 854626) | Learned model vs rule baseline metrics | 2 rows: rule baseline (pairwise_acc=0.247) and learned model (pairwise_acc=0.197). `note` column records job 854626 and split (230/50/50). | **DONE_AND_MEASURED** | Authentic negative result from a real GPU job. No HF token needed (uses local NLP4LP eval data). |
| D8 | ESWA table — `runtime_summary.csv` | Retrieval: timed locally. Downstream: estimated from docs. | Per-method runtime | Retrieval rows have `"locally measured"` note and precise elapsed_s values matching `retrieval_results.json`. Downstream rows have `"estimate from docs"` note. | **PARTIALLY_DONE** | Retrieval runtimes are measured. Downstream runtimes (~0.01–0.10 s/query) are estimates only, not directly timed on a canonical run. |
| E1 | `.github/workflows/check-hf-access.yml` — "Check HF access" | Manual `workflow_dispatch` trigger; runs only `verify_hf_access.py` | HF token validation result (~60 s) | Workflow file exists and is correctly structured. `hf_access_check_runtime.md` status: `⏳ AWAITING GITHUB ACTIONS TRIGGER`. | **ONLY_SCAFFOLDING_EXISTS** | This workflow is a smoke test only. It does NOT run any experiment. A successful run of this workflow is NOT evidence that any benchmark ran. |
| E2 | `.github/workflows/nlp4lp.yml` — "NLP4LP benchmark" (3-phase) | Manual `workflow_dispatch`; runs Phase 0 (verify) + Phase 1 (build eval sets) + Phase 2 (full downstream benchmark) | All of B1–B3 and C1–C2 output artifacts | Workflow file exists, correctly structured (30-setting loop, `contents: write`, `git push`). Previous runs in GitHub Actions were all fast (~20 s) push-triggered stubs (now removed). No run of this workflow has ever completed Phase 2. | **ONLY_SCAFFOLDING_EXISTS** | The `BENCHMARK_STATUS.md` file explicitly records: "fast (<20 second) workflow completions" were registration-only stubs. No 2–3 hour run has occurred. `02_downstream_postfix/` absent is conclusive. |
| E3 | `.github/workflows/downstream_benchmark.yml` — "NLP4LP downstream benchmark (authenticated)" | Manual `workflow_dispatch`; runs Phase 0 (verify) + Phase 2 (full downstream benchmark) — skips eval-set build | Same as B1–B3 and C1–C2 | Workflow file exists and is correctly structured. Has never been triggered (requires `workflow_dispatch`; no prior runs recorded in sandbox context). | **ONLY_SCAFFOLDING_EXISTS** | Standalone alternative to `nlp4lp.yml` for cases where eval sets already exist. Not triggered. |

---

## Differentiating: CI checks vs retrieval vs full downstream

| Run type | Typical duration | Uses HF_TOKEN | Produces benchmark numbers | Which workflow |
|----------|-----------------|---------------|---------------------------|----------------|
| HF access check | ~60 s | Yes (verify only) | **NO** | `check-hf-access.yml` |
| Push-triggered registration stub (historical, now removed) | ~20 s | No | **NO** | `nlp4lp.yml` (old behavior) |
| Retrieval-only run | ~10–30 s | **NO** (uses local catalog) | Yes — retrieval R@1/MRR only | `training/run_baselines.py` inline |
| Full downstream benchmark (all 10 methods × 3 variants) | ~2–3 hours | **YES** (loads gold params from HF) | Yes — Coverage, TypeMatch, InstReady | `nlp4lp.yml` Phase 2 or `downstream_benchmark.yml` |

**Key distinction:** Retrieval does NOT need `HF_TOKEN` and has been run. The full downstream benchmark requires `HF_TOKEN` for gold parameters and has NOT been run.

---

## Evidence Checklist for DONE_AND_MEASURED Claims

### A1–A3: Retrieval (DONE_AND_MEASURED)

Concrete evidence:

1. **File:** `results/eswa_revision/01_retrieval/retrieval_results.json` — 9 entries (3 variants × 3 methods), each with `"n_queries": 331` and multi-decimal-place metrics. E.g.: `"tfidf" orig: {"P@1": 0.9093655589123867, "elapsed_s": 1.47, "n_queries": 331}`.
2. **File:** `results/eswa_revision/13_tables/retrieval_main.csv` — 9 data rows, values matching the JSON to 4 decimal places.
3. **File:** `results/eswa_revision/01_retrieval/retrieval_summary.md` — timestamps 2026-03-10, records git commit `e3fdaf4`, provides inline Python reproduction command.
4. **File:** `results/eswa_revision/manifests/experiment_manifest.json` — entry `"id": "01_retrieval"` with `"status": "SUCCESS"` and matching key results.
5. **Reproducibility note in summary:** elapsed_s values for retrieval (1.98 s, 1.47 s, 2.61 s) are consistent with the catalog size (335 entries) and BM25/TF-IDF/LSA complexity on CPU.

### D7: Learning Appendix (DONE_AND_MEASURED)

1. **File:** `results/eswa_revision/13_tables/learning_summary.csv` — 2 rows with specific numeric values (rule: pairwise_acc=0.247; learned: 0.197). Notes reference SLURM job 854626 and split counts (230/50/50).
2. **Supporting doc:** `docs/learning_runs/real_data_only_learning_check.md` — records same job ID, split details, and reproduces the same 4-metric table.

---

## Why NOT_RUN / PLACEHOLDER_REPORT_ONLY Claims Are Made

### B1–B3: Full downstream benchmark (NOT_RUN)

1. **Primary evidence:** `results/eswa_revision/02_downstream_postfix/` **directory does not exist** (`ls -la` returns `No such file or directory`). `run_full_downstream_benchmark.py` creates this directory as its first action. Its absence is conclusive.
2. **Secondary evidence:** `results/eswa_revision/13_tables/postfix_main_metrics.csv` — all 10 rows (orig only; should have 30 for all 3 variants) have `source: placeholder-pre-fix-manuscript-era (run NLP4LP benchmark workflow to replace)`. A measured run writes `source: measured`.
3. **Tertiary evidence:** `results/eswa_revision/14_reports/postfix_main_metrics.md` **does not exist**. The script creates this on success.
4. **Explicit confirmation:** `results/eswa_revision/manifests/experiment_manifest.json` records `02_downstream_postfix` as `"status": "BLOCKED — HF_TOKEN required"`.
5. **Explicit confirmation:** `results/eswa_revision/00_env/hf_access_check_runtime.md` states: `**Status:** ⏳ AWAITING GITHUB ACTIONS TRIGGER`, `huggingface.co DNS lookup: BLOCKED`.
6. **Explicit confirmation:** `results/eswa_revision/manifests/commands_run_runtime.md` states: `"huggingface.co: DNS lookup blocked by sandbox DNS monitoring proxy"`.
7. **Workflow history:** `results/eswa_revision/00_env/BENCHMARK_STATUS.md` records all past fast runs (~20 s) as registration-only stubs that exited immediately without running any experiment code.

### C1–C2: Pre-fix vs post-fix ablation (PLACEHOLDER_REPORT_ONLY)

1. **File:** `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` — post-fix column values are literal strings: `~0.55–0.65`, `~0.70–0.80`, `slightly higher`, `TBD`. Source column: `post_fix=structural-estimate-not-measured`.
2. **File:** `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md` **does not exist**. The script creates this when the ablation loop completes.
3. **Context:** The ablation is implemented inside `run_full_downstream_benchmark.py` (monkey-patches `_is_type_match`). Since the script has never run (see B1), the ablation has also never run.
4. **What exists:** `results/eswa_revision/03_prefix_vs_postfix/prefix_vs_postfix_ablation.md` — a thorough structural analysis of the fix's expected impact. This is analysis, not measurement.

### D1: `postfix_main_metrics.csv` (PLACEHOLDER_REPORT_ONLY)

Every cell in the `source` column reads: `placeholder-pre-fix-manuscript-era (run NLP4LP benchmark workflow to replace)`. The numbers are copied from manuscript-era documentation and are (a) pre-fix for TypeMatch and (b) not from the post-fix code path. They will be overwritten when the benchmark runs.

### D2: `prefix_vs_postfix_ablation.csv` (PLACEHOLDER_REPORT_ONLY)

Every post-fix value is a range estimate or `TBD`, not a number. Source: `structural-estimate-not-measured`.

---

## What Is Already Sufficient for the Manuscript

| Claim | Basis | Sufficient? |
|-------|-------|-------------|
| "Retrieval R@1 ≥ 0.77 on short, ≥ 0.90 on orig for TF-IDF" | DONE_AND_MEASURED — `retrieval_results.json` | ✅ Yes |
| "Downstream grounding is the bottleneck — oracle retrieval improves InstReady by ≤ 1pp" | Pre-fix manuscript numbers; Oracle–TF-IDF gap is structural (Coverage difference, not retrieval difference) | ⚠️ Caveat: based on pre-fix TypeMatch values only |
| "No single deterministic method dominates — Pareto frontier between coverage and precision" | Pre-fix comparison table | ⚠️ Caveat: TypeMatch ordering may shift post-fix |
| "Float type classification was the largest error source (~70% of float slots mistyped)" | Structural code analysis (`_is_type_match` logic + token counts) | ✅ Yes — this is a code fact, not a measurement |
| "Learning (distilroberta-base, 500 steps) scored below rule baseline on all metrics" | DONE_AND_MEASURED — job 854626 | ✅ Yes |
| "All methods run in under 1 second per query on CPU" | Retrieval measured; downstream estimated | ⚠️ Caveat: downstream runtime is estimated |
| "Post-fix TypeMatch is materially higher than pre-fix" | Structural estimate only (~+0.33pp overall) | ❌ No — requires measured run |

---

## What Still Must Be Run

**Priority 1 — The full post-fix downstream benchmark (blocks most manuscript claims):**

```bash
# Trigger from GitHub Actions UI:
# Actions → "NLP4LP benchmark" → Run workflow → branch: copilot/experiment-overview

# Or via CLI:
gh workflow run nlp4lp.yml \
  --repo SoroushVahidi/combinatorial-opt-agent \
  --ref copilot/experiment-overview
```

This single ~2–3 hour workflow run will produce:
- `results/eswa_revision/02_downstream_postfix/` — all 30-setting per-query results
- `results/eswa_revision/13_tables/postfix_main_metrics.csv` with `source: measured` (replaces all placeholder rows)
- `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` with real pre-fix vs post-fix TypeMatch deltas (replaces structural estimates)
- `results/eswa_revision/14_reports/postfix_main_metrics.md` and `prefix_vs_postfix_ablation.md`
- `results/paper/nlp4lp_downstream_summary.csv` and `nlp4lp_downstream_types_summary.csv`

**Prerequisite:** `HF_TOKEN` must be set in repository Secrets (Settings → Secrets and variables → Actions → `HF_TOKEN`). The `check-hf-access` workflow (~60 s) can verify this first.

---

## Do We Still Need to Run More Experiments?

**YES.**

The two most important missing experiments — the **real post-fix downstream benchmark** and the **real pre-fix vs post-fix ablation** — have not been run. Every number in `postfix_main_metrics.csv` is a placeholder from the manuscript era and every post-fix TypeMatch value in `prefix_vs_postfix_ablation.csv` is a structural estimate.

The manuscript cannot credibly claim any specific post-fix TypeMatch or InstantiationReady improvement without running the `NLP4LP benchmark` workflow (`.github/workflows/nlp4lp.yml`) on a GitHub Actions runner with `HF_TOKEN` configured. This is a single manual trigger of a workflow that already exists and is ready to run.

Until that run completes and commits `source: measured` results back to the branch, the manuscript's central empirical claim — that the `_is_type_match` float fix materially improves downstream grounding — rests on a structural estimate, not a measurement.
