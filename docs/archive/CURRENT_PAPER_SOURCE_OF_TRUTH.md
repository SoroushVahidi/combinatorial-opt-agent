# Current Paper Source of Truth

**Status:** Authoritative for EAAI submission
**Extends:** `docs/EAAI_SOURCE_OF_TRUTH.md` (read that first for full background)
**Last updated:** 2026-03-30

---

## Overview

This document is the single authoritative reference for what the current EAAI manuscript
claims, what the repository demonstrates, and which artifacts are canonical. When in doubt
about any number, file path, or framing decision, consult this document and
`docs/EAAI_SOURCE_OF_TRUTH.md` first.

---

## Exact Intended Paper Framing

The paper title is:

> **"Retrieval-Assisted Instantiation of Natural-Language Optimization Problems"**

### Core Framing: Retrieval-Assisted Schema Grounding

The system is a **retrieval-assisted schema grounding** pipeline, not a full compiler. The
key distinction:

- **What it does:** Given a natural-language optimization problem, retrieve the most
  relevant schema from a catalog, then deterministically instantiate scalar parameters
  from numeric evidence in the problem text.
- **What it does not do:** Generate arbitrary new optimization formulations, produce
  universally solver-ready code, or serve as an end-to-end "AI compiler" for optimization.

The pipeline has six stages:
1. NL query → schema retrieval (TF-IDF / BM25 / Oracle baseline)
2. Retrieved schema → scalar parameter extraction (typed greedy grounding)
3. Optimization-role repair (constraint/objective alignment)
4. Structural validation (LP consistency without a live solver)
5. Executable attempt (full execution, documents blockers)
6. Solver-backed subset (20 instances, SciPy HiGHS shim)

### Primary Contribution

The primary contribution is a **rigorous, reproducible benchmark study** of this pipeline
on the NLP4LP dataset, with an honest bottleneck analysis showing that schema retrieval
is strong (R@1 = 0.9094) but downstream grounding remains the binding constraint
(InstReady = 0.5257).

---

## What the Repo Does and Does Not Claim

### The Repository DOES claim

- TF-IDF schema retrieval achieves R@1 = **0.9094** on the NLP4LP `orig` variant (331 test queries).
- Downstream typed-greedy grounding achieves Coverage = **0.8639**, TypeMatch = **0.7513**,
  InstantiationReady = **0.5257** (TF-IDF baseline).
- Structural validation (LP objective/variable consistency) is implemented and reproducible
  without a live solver.
- On a **restricted 20-instance subset**, real nonzero solver outcomes are achievable:
  TF-IDF feasibility = **0.80**, Oracle feasibility = **0.75** (via SciPy HiGHS shim).
- An **engineering-oriented structural subset** (60 instances) achieves TF-IDF
  structural-valid rate = **0.75**, instantiation-complete rate = **0.75**.
- The main bottleneck is downstream number-to-slot grounding, not schema retrieval.

### The Repository does NOT claim

- Solver-ready output for arbitrary NL optimization problems (only a restricted 20-instance subset).
- Generalisation beyond the NLP4LP benchmark or the engineering-oriented subset.
- That the learned retrieval model outperforms the TF-IDF rule baseline (it does not —
  documented in `KNOWN_ISSUES.md`).
- That Gurobi is available or required (paper uses SciPy HiGHS shim for the solver subset).
- That the executable-attempt rate on the 269-instance subset is nonzero (it is 0.0,
  due to `gurobipy` absence; this is transparently documented as a blocker study).
- That E5 or BGE dense retrieval is the primary result (these are supplementary baselines).

---

## Authoritative Reports, Tables, and Figures

### Tables (camera-ready, in `results/paper/eaai_camera_ready_tables/`)

| File | Contents | Key Numbers |
|------|----------|-------------|
| `table1_main_benchmark_summary.csv` | NLP4LP orig variant main metrics (4 data rows + header) | TF-IDF R@1=0.9094, InstReady=0.5257 |
| `table2_engineering_structural_subset.csv` | Engineering subset (60 instances) | TF-IDF struct valid=0.75, inst complete=0.75 |
| `table3_executable_attempt_with_blockers.csv` | Executable-attempt study (269 instances) | All exec rates=0.0 (gurobipy blocker) |
| `table4_final_solver_backed_subset.csv` | Solver-backed subset (20 instances) | TF-IDF feasible=0.80, Oracle feasible=0.75 |
| `table5_failure_taxonomy.csv` | Failure categories and counts | 7 failure rows |

### Figures (camera-ready, in `results/paper/eaai_camera_ready_figures/`)

| File (PDF + PNG) | Contents |
|------------------|----------|
| `figure1_pipeline_overview.*` | Six-step pipeline schematic |
| `figure2_main_benchmark_comparison.*` | Grouped bar chart, 3 baselines × 4 metrics |
| `figure3_engineering_validation_comparison.*` | Engineering subset bar chart, 3 baselines × 3 metrics |
| `figure4_final_solver_backed_subset.*` | Solver-backed subset bar chart |
| `figure5_failure_breakdown.*` | Failure taxonomy stacked bar chart |

### Experiment Reports (in `analysis/`)

| File | Contents |
|------|----------|
| `analysis/eaai_engineering_subset_report.md` | Engineering subset (60 instances) experiment |
| `analysis/eaai_executable_subset_report.md` | Executable-attempt study (269 instances) |
| `analysis/eaai_final_solver_attempt_report.md` | Final solver-backed subset (20 instances) |
| `analysis/eaai_tables_build_report.md` | Table provenance, conflict resolution |
| `analysis/eaai_figures_build_report.md` | Figure build log |
| `analysis/eaai_figures_reproduction_report.md` | Figure reproduction log (this pass) |
| `analysis/eaai_tables_reproduction_report.md` | Table verification log (this pass) |
| `analysis/eaai_repo_validation_report.md` | Full test-suite validation log (this pass) |

---

## Historical-Only Documents

The following documents contain earlier ESWA-era framing or intermediate experiment records.
They are **not authoritative** for the EAAI submission and should not be cited in the paper:

| Document | Why Historical |
|----------|---------------|
| `docs/JOURNAL_READINESS_AUDIT.md` | ESWA readiness audit; scope broader than current paper |
| `docs/Q1_JOURNAL_AUDIT.md` | Earlier quality audit; pre-EAAI framing |
| `docs/CURRENT_STATE_AUDIT.md` | Point-in-time snapshot; superseded |
| `docs/FULL_REPO_SUMMARY.md` | Broad description pre-dating EAAI scope narrowing |
| `docs/EAAI_REPO_ALIGNMENT_AUDIT.md` | Earlier alignment pass; superseded by this SoT |
| `docs/EAAI_COPILOT_HANDOFF_REPORT.md` | Prior copilot handoff; superseded |
| `docs/eswa_revision/` | ESWA revision materials |
| `current_repo_vs_manuscript_rerun.md` | Intermediate comparison; superseded |
| `literature_informed_rerun_report.md` | Pre-EAAI method exploration |
| `publish_now_decision_report.md` | Internal decision log |

**Exception:** `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`
is the *upstream source* for Table 1 and remains authoritative even though the rest of
`results/eswa_revision/` is historical.

---

## Current Benchmark Numbers Safe to Cite

All values below are sourced from the camera-ready tables and verified in
`analysis/eaai_tables_reproduction_report.md`.

### Table 1 — Main Benchmark (NLP4LP, `orig` variant, 331 test queries)

| Method | Schema R@1 | Coverage | TypeMatch | InstReady |
|--------|-----------|----------|-----------|-----------|
| **TF-IDF** | **0.9094** | **0.8639** | **0.7513** | **0.5257** |
| BM25 | 0.8822 | 0.8509 | 0.7386 | 0.5196 |
| Oracle | 1.0000 | 0.9151 | 0.8030 | 0.5650 |

### Table 2 — Engineering Structural Subset (60 instances)

| Baseline | Schema Hit | Structural Valid | Inst. Complete |
|----------|-----------|-----------------|----------------|
| TF-IDF | 0.9333 | 0.7500 | 0.7500 |
| BM25 | 0.9000 | 0.7333 | 0.7333 |
| Oracle | 1.0000 | 0.7667 | 0.7833 |

### Table 3 — Executable-Attempt Study (269 instances)

All executable/solver/feasible rates = **0.0** for all baselines.
Dominant blocker: `ModuleNotFoundError: gurobipy`.

### Table 4 — Final Solver-Backed Subset (20 instances, SciPy HiGHS shim)

| Baseline | Executable | Solver Success | Feasible | Objective |
|----------|-----------|----------------|----------|-----------|
| TF-IDF | 0.95 | 0.80 | **0.80** | 0.80 |
| Oracle | 0.95 | 0.75 | 0.75 | 0.75 |

### Key Subset Sizes

| Subset | Size | Note |
|--------|------|------|
| NLP4LP test set (primary benchmark) | 331 queries | `orig` variant |
| Engineering structural subset | **60 instances** | 2–8 scalar params |
| Executable-attempt subset | **269 instances** | Rows with non-empty `optimus_code` |
| Solver-backed subset | **20 instances** | Deterministically filtered, SciPy shim |

---

## Submission Notes

1. **Double-anonymization:** The EAAI submission is double-anonymized. The manuscript body
   must not contain the GitHub URL, author names, or institution identifiers.
2. **Solver:** The paper uses SciPy HiGHS via a MILP compatibility shim for the 20-instance
   solver-backed subset. Gurobi is NOT required to reproduce these results.
3. **Dataset access:** Full benchmark reproduction requires HuggingFace account with
   `udell-lab/NLP4LP` dataset access approved (gated dataset).
4. **Learned model:** The learned retrieval model checkpoint is not publicly available in
   this repository; training requires GPU and a HuggingFace token.
5. **GAMSPy:** GAMSPy examples and integration are outside manuscript scope and require a
   GAMS license. Do not cite GAMSPy results in the paper.

---

## Unresolved Conflicts Needing Human Review

As of 2026-03-30, no number conflicts are known between the camera-ready tables and the
manuscript text. The following items were investigated and resolved:

| Item | Resolution |
|------|-----------|
| "331 vs 335" instance count discrepancy | Only "331" appears in authoritative files; "335" not found |
| "Best method" label in Table 1 | `tfidf_typed_greedy` = same values as TF-IDF row; documented in `analysis/eaai_tables_build_report.md` |
| Oracle R@1 = 1.0000 vs upper-bound meaning | Oracle is the retrieval upper bound; clearly documented |

If the manuscript draft uses any metric value not in the tables above, flag it for manual
reconciliation against the camera-ready CSVs before submission.
