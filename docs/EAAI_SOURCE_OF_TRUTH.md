# EAAI Source of Truth

**Created:** 2026-03-30  
**Status:** Authoritative — supersedes older broad repo framing docs for manuscript purposes

---

## Intended Paper Framing

The paper is:

> **"Retrieval-Assisted Instantiation of Natural-Language Optimization Problems"**  
> Submitted to *Engineering Applications of Artificial Intelligence* (EAAI)

This repository is the companion codebase for that manuscript.

### What the paper IS

- A transparent, reproducible pipeline for **retrieval-assisted optimization schema grounding**
- A study of **deterministic scalar parameter instantiation** from numeric evidence in NL text
- A benchmark study on **NLP4LP**, the primary evaluation dataset
- A set of **restricted engineering-oriented validations**:
  1. **Structural engineering subset** (60 instances): end-to-end structural consistency
  2. **Executable-attempt subset** (269 instances): full execution with documented blockers
  3. **Final solver-backed subset** (20 instances): real solver execution via SciPy HiGHS shim
- An honest analysis of the **main bottleneck**: downstream number-to-slot grounding, not schema retrieval

### What the paper is NOT

- Not a full natural-language-to-optimization compiler (solver-ready output is restricted)
- Not mainly an "expert systems" paper
- Not a paper claiming benchmark-wide solver readiness
- Not claiming E5 or BGE dense retrieval as primary results (these are supplementary)
- Not claiming that LLM-based generation for unknown problems is part of the evaluated system

---

## What the Repository Does and Does Not Claim

### Repository DOES claim

- Schema retrieval is strong: TF-IDF Schema R@1 = 0.9094 on NLP4LP `orig` variant
- Downstream grounding (typed greedy, optimization-role repair) is implemented and benchmarked
- Structural validation (without a live solver) is reproducible
- A 20-instance restricted subset achieves real nonzero solver outcomes via SciPy HiGHS shim

### Repository does NOT claim

- Full solver-ready code generation for arbitrary NL optimization problems
- That all NLP4LP instances are executable
- That Gurobi is available or required for paper results (it is NOT; paper uses SciPy shim for the solver subset)
- That the learned retrieval model beats the rule baseline (it does not: documented in KNOWN_ISSUES.md)

---

## Authoritative Files for Current Manuscript Evidence

The following files are authoritative for the current EAAI submission:

| File | Role |
|------|------|
| `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` | Main benchmark results (NLP4LP, orig variant) |
| `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv` | Engineering subset (60 instances) |
| `results/paper/eaai_camera_ready_tables/table3_executable_attempt_with_blockers.csv` | Executable-attempt study (269 instances) |
| `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv` | Solver-backed subset (20 instances) |
| `results/paper/eaai_camera_ready_tables/table5_failure_taxonomy.csv` | Failure taxonomy |
| `results/paper/eaai_camera_ready_figures/figure*.png` and `*.pdf` | Camera-ready figures |
| `analysis/eaai_engineering_subset_report.md` | Engineering subset experiment report |
| `analysis/eaai_executable_subset_report.md` | Executable-attempt experiment report |
| `analysis/eaai_final_solver_attempt_report.md` | Final solver-backed experiment report |
| `analysis/eaai_tables_build_report.md` | Table provenance and conflict resolution notes |
| `analysis/eaai_figures_build_report.md` | Figure build report |
| `analysis/eaai_figures_reproduction_report.md` | Figure reproduction log (this pass) |
| `analysis/eaai_tables_reproduction_report.md` | Table verification log (this pass) |
| `tools/run_eaai_engineering_subset_experiment.py` | Engineering subset experiment script |
| `tools/run_eaai_executable_subset_experiment.py` | Executable-attempt experiment script |
| `tools/run_eaai_final_solver_attempt.py` | Final solver-backed experiment script |
| `tools/build_eaai_camera_ready_figures.py` | Figure build script |

---

## Historical-Only Docs (Not Authoritative for EAAI)

The following docs contain earlier experiment records or ESWA-era framing and should not be cited as authoritative for the current EAAI submission:

- `docs/JOURNAL_READINESS_AUDIT.md` — ESWA readiness audit (superseded)
- `docs/Q1_JOURNAL_AUDIT.md` — Earlier quality audit
- `docs/CURRENT_STATE_AUDIT.md` — Point-in-time snapshot
- `docs/FULL_REPO_SUMMARY.md` — Broad summary, not EAAI-specific
- `docs/eswa_revision/` — ESWA revision materials
- `current_repo_vs_manuscript_rerun.md` — Intermediate comparison
- `literature_informed_rerun_report.md` — Earlier literature rerun
- `publish_now_decision_report.md` — Internal decision document

Note: The ESWA revision tables in `results/eswa_revision/13_tables/` ARE still authoritative for **Table 1** (main benchmark metrics), because they contain the canonical NLP4LP orig-variant measured results.

---

## Benchmark and Evaluation Story

### Primary Benchmark: NLP4LP (orig variant)

- **Dataset:** `udell-lab/NLP4LP` (HuggingFace, gated)
- **Eval set size:** 331 test queries
- **Primary retrieval baseline:** TF-IDF
- **Comparison baselines:** Random, LSA, BM25, Oracle
- **Dense baselines (supplementary):** SBERT, E5, BGE
- **Primary metric:** Schema R@1 = 0.9094 (TF-IDF, orig)
- **Downstream grounding metric:** Coverage, TypeMatch, Exact20, InstantiationReady
- **Main bottleneck:** Downstream grounding (not retrieval)

### Three EAAI Validation Subsets

1. **Engineering structural subset (60 instances)**
   - Purpose: End-to-end structural consistency on engineering-flavored NLP4LP instances
   - Key result: TF-IDF structural valid rate = 0.75, inst. complete = 0.75
   - Blocker: No live solver; HF dataset gated

2. **Executable-attempt subset (269 instances)**
   - Purpose: Study full execution pipeline on all NLP4LP rows with gold optimus_code
   - Key result: All executable/solver rates = 0.0 due to gurobipy absence
   - This is intentionally presented as documenting blockers, not as a success rate

3. **Final solver-backed subset (20 instances)**
   - Purpose: Real nonzero solver outcomes on a deterministically-filtered subset
   - Solver: SciPy HiGHS MILP compatibility shim (no Gurobi license required)
   - Key result: TF-IDF feasible = 0.80, Oracle feasible = 0.75
   - Clearly scoped as a restricted pragmatic demonstration

---

## Key Metrics (Verified from Latest Committed Artifacts)

All values below are from `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` (sourced from `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`):

| Method | Schema R@1 | Coverage | TypeMatch | InstReady |
|--------|-----------|----------|-----------|-----------|
| TF-IDF | 0.9094 | 0.8639 | 0.7513 | 0.5257 |
| BM25 | 0.8822 | 0.8509 | 0.7386 | 0.5196 |
| Oracle | 1.0000 | 0.9151 | 0.8030 | 0.5650 |

Engineering subset (60 instances, from `table2_engineering_structural_subset.csv`):

| Baseline | Schema Hit | Structural Valid | Inst. Complete |
|----------|-----------|-----------------|----------------|
| TF-IDF | 0.9333 | 0.7500 | 0.7500 |
| BM25 | 0.9000 | 0.7333 | 0.7333 |
| Oracle | 1.0000 | 0.7667 | 0.7833 |

Final solver-backed subset (20 instances, from `table4_final_solver_backed_subset.csv`):

| Baseline | Executable | Solver Success | Feasible | Objective |
|----------|-----------|----------------|----------|-----------|
| TF-IDF | 0.95 | 0.80 | 0.80 | 0.80 |
| Oracle | 0.95 | 0.75 | 0.75 | 0.75 |

---

## Unresolved Number Conflicts

None found as of this audit. The following potential conflicts were investigated and resolved:

- **331 vs 335 instance counts**: Only "331" appears in EXPERIMENTS.md and all EAAI experiment reports. No "335" found in current authoritative files.
- **Best method label ambiguity**: "Best downstream grounding" in Table 1 maps to `tfidf_typed_greedy` with same values as TF-IDF row. This is documented in `analysis/eaai_tables_build_report.md` under Conflict 2.

---

## Submission-Related Warnings

1. **Double-anonymization**: The EAAI submission is double-anonymized. The manuscript body must NOT contain the GitHub repository URL, author names, or institution identifiers. Keep these in the cover letter or supplementary material only.

2. **Generative AI declaration**: If the journal requires a generative AI use statement, it belongs in the designated submission section (typically a separate field or cover letter), NOT in the manuscript body if doing so would violate double-blind rules or journal formatting.

3. **Solver availability in paper**: The paper should clearly state which solver is used for the 20-instance subset (SciPy HiGHS via compatibility shim) and that Gurobi is NOT required to reproduce these results.

4. **Dataset access**: Reproduction of full benchmark results requires a HuggingFace account with `udell-lab/NLP4LP` dataset access approved. This should be documented in the paper's supplementary materials or README.
