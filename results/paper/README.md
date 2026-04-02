# results/paper — Artifact Provenance Guide

This directory holds all paper-facing result artifacts for the EAAI manuscript
*"Retrieval-Assisted Instantiation of Natural-Language Optimization Problems"*.

---

## ★ Canonical manuscript artifacts

The following subdirectories contain the authoritative camera-ready artifacts.
These are the files that should be cited in README, docs, and the manuscript.

### `eaai_camera_ready_tables/`

| File | Contents | Status |
|------|----------|--------|
| `table1_main_benchmark_summary.csv` | Main benchmark: Schema R@1, Coverage, TypeMatch, InstReady for TF-IDF / BM25 / Oracle on NLP4LP orig (331 queries) | ★ Canonical |
| `table2_engineering_structural_subset.csv` | Engineering structural subset (60 instances): schema hit, structural valid, inst. complete | ★ Canonical |
| `table3_executable_attempt_with_blockers.csv` | Executable-attempt subset (269 instances) with documented blockers | ★ Canonical |
| `table4_final_solver_backed_subset.csv` | Solver-backed subset (20 instances, SciPy HiGHS shim) | ★ Canonical |
| `table5_failure_taxonomy.csv` | Cross-experiment failure taxonomy | ★ Canonical |
| `camera_ready_tables.md` | Markdown bundle of Tables 1–5 with source notes | ★ Canonical |

**Table 1 key values** (TF-IDF typed-greedy, orig variant):
- Schema R@1 = **0.9094**, Coverage = **0.8639**, TypeMatch = **0.7513**, InstReady = **0.5257**

Provenance source: `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`

### `eaai_camera_ready_figures/`

| Files | Contents | Status |
|-------|----------|--------|
| `figure{1–5}.{png,pdf}` | Camera-ready figures for the manuscript | ★ Canonical |
| `figure{1–5}_source.csv` | Source data for each figure | ★ Canonical |

---

## ⚠ Legacy / pre-fix files

The following files in this directory are **not** the canonical manuscript results.
They are preserved for historical provenance only and should not be cited as the
paper's current main results.

| File(s) | Why legacy |
|---------|-----------|
| `nlp4lp_downstream_summary.csv` | Pre-fix run — TypeMatch and InstReady values are from before the float type-match fixes. These numbers are lower than the canonical Table 1 values. |
| `nlp4lp_downstream_{noisy,orig,short}_{bm25,lsa,tfidf}.json` | Same pre-fix run (9 files). Raw per-run outputs. |
| `nlp4lp_downstream_per_query_*.csv` | Per-query detail for the same pre-fix run (9 files). |
| `nlp4lp_downstream_types_summary.csv` | Type-level summary from the same pre-fix run. |
| `nlp4lp_llm_baseline_comparison.csv` | LLM comparison stub — LLM rows are blank/incomplete. |

---

## Summary

| Subdirectory / file | Status |
|--------------------|--------|
| `eaai_camera_ready_tables/` | ★ Canonical |
| `eaai_camera_ready_figures/` | ★ Canonical |
| `nlp4lp_downstream_summary.csv` and related files | ⚠ Legacy (pre-fix) |

For the master paper framing and full canonical artifact list, see
[`docs/EAAI_SOURCE_OF_TRUTH.md`](../../docs/EAAI_SOURCE_OF_TRUTH.md) and
[`docs/RESULTS_PROVENANCE.md`](../../docs/RESULTS_PROVENANCE.md).
