# Start Here for the Current Paper

**Audience:** New collaborators, reviewers, or returning contributors approaching this
repository for the first time in the context of the EAAI manuscript.

---

## What This Repository Is

This is the companion codebase for:

> **"Retrieval-Assisted Instantiation of Natural-Language Optimization Problems"**
> Submitted to *Engineering Applications of Artificial Intelligence* (EAAI)

The system retrieves the best-matching optimization problem schema from a catalog given a
natural-language description, then deterministically instantiates scalar parameters from
numeric evidence in the text. It is benchmarked on the NLP4LP dataset. It is **not** a
general-purpose NL-to-solver compiler.

If you want to understand what the paper claims and what the code demonstrates, read the
two source-of-truth documents first (listed below).

---

## Where to Look First

| Priority | File | Why |
|----------|------|-----|
| 1 | `docs/EAAI_SOURCE_OF_TRUTH.md` | Authoritative framing, scope, and benchmark numbers |
| 2 | `docs/CURRENT_PAPER_SOURCE_OF_TRUTH.md` | Extended SoT with all citable numbers and table provenance |
| 3 | `README.md` | Quick start, architecture diagram, capabilities table |
| 4 | `EXPERIMENTS.md` | Full experiment log with commands and results |
| 5 | `KNOWN_ISSUES.md` | Known failures and blockers (read before running anything) |

---

## Files That Matter for the Current Manuscript

### Paper Artifacts

```
results/paper/
├── eaai_camera_ready_tables/
│   ├── table1_main_benchmark_summary.csv        # Main results (TF-IDF R@1=0.9094)
│   ├── table2_engineering_structural_subset.csv  # 60-instance engineering subset
│   ├── table3_executable_attempt_with_blockers.csv # 269-instance blocker study
│   ├── table4_final_solver_backed_subset.csv     # 20-instance solver-backed subset
│   └── table5_failure_taxonomy.csv               # Failure categories
└── eaai_camera_ready_figures/
    ├── figure1_pipeline_overview.{pdf,png}
    ├── figure2_main_benchmark_comparison.{pdf,png}
    ├── figure3_engineering_validation_comparison.{pdf,png}
    ├── figure4_final_solver_backed_subset.{pdf,png}
    └── figure5_failure_breakdown.{pdf,png}
```

### Analysis Reports (Experiment Provenance)

```
analysis/
├── eaai_engineering_subset_report.md       # Source for Table 2
├── eaai_executable_subset_report.md        # Source for Table 3
├── eaai_final_solver_attempt_report.md     # Source for Table 4
├── eaai_tables_build_report.md             # Table construction and conflict notes
├── eaai_figures_build_report.md            # Figure build log
├── eaai_figures_reproduction_report.md     # Reproduction log (latest pass)
└── eaai_tables_reproduction_report.md      # Table verification (latest pass)
```

### Core Pipeline Code

```
retrieval/          # Schema retrieval (TF-IDF, BM25, LSA, dense baselines)
formulation/        # Schema catalog and structural validation
tools/              # Experiment and artifact-build scripts (see §Scripts below)
src/                # Supporting utilities
```

---

## Scripts That Reproduce Current Paper Evidence

All scripts live in `tools/`. Run them from the repository root.

| Script | What It Produces | Table / Figure |
|--------|-----------------|----------------|
| `tools/run_eaai_engineering_subset_experiment.py` | Engineering subset metrics | Table 2 |
| `tools/run_eaai_executable_subset_experiment.py` | Executable-attempt blocker study | Table 3 |
| `tools/run_eaai_final_solver_attempt.py` | Solver-backed subset outcomes | Table 4 |
| `tools/make_nlp4lp_paper_artifacts.py` | Assembles all camera-ready tables | Tables 1–5 |
| `tools/build_eaai_camera_ready_figures.py` | Renders all 5 camera-ready figures | Figures 1–5 |

> **Note:** Full benchmark experiments require HuggingFace access to the gated
> `udell-lab/NLP4LP` dataset. See `KNOWN_ISSUES.md` for all environment blockers before
> running.

### Quick Smoke Test (no HF access required)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q --timeout=30
```

Expected: ~1,469 passed, ~5 skipped, 0 failures (after fixes applied in this pass).

---

## Current Figures and Tables Location

| Artifact | Location |
|----------|----------|
| All 5 paper figures (PDF + PNG) | `results/paper/eaai_camera_ready_figures/` |
| All 5 paper tables (CSV) | `results/paper/eaai_camera_ready_tables/` |
| Human-readable table summary | `results/paper/eaai_camera_ready_tables/camera_ready_tables.md` |

Figure source CSVs (for regeneration) are co-located with the figures:
`results/paper/eaai_camera_ready_figures/*_source.csv`.

---

## Historical-Only Docs (skip these for the current paper)

The following documents are retained for historical context but are **not authoritative**
for the EAAI manuscript. Do not use them to look up metric values or paper claims.

| Document | Reason to Skip |
|----------|---------------|
| `docs/JOURNAL_READINESS_AUDIT.md` | ESWA-era readiness audit; scope broader than current paper |
| `docs/Q1_JOURNAL_AUDIT.md` | Earlier quality audit; pre-EAAI framing |
| `docs/CURRENT_STATE_AUDIT.md` | Point-in-time snapshot; superseded |
| `docs/FULL_REPO_SUMMARY.md` | Broad overview predating EAAI scope narrowing |
| `docs/EAAI_REPO_ALIGNMENT_AUDIT.md` | Earlier alignment pass; superseded |
| `docs/EAAI_COPILOT_HANDOFF_REPORT.md` | Prior handoff notes; superseded |
| `docs/eswa_revision/` | ESWA revision materials |
| `current_repo_vs_manuscript_rerun.md` | Intermediate comparison; superseded |
| `literature_informed_rerun_report.md` | Pre-EAAI method exploration |
| `publish_now_decision_report.md` | Internal decision log |
| `docs/NLP4LP_*.md` (most files) | Implementation notes for individual NLP4LP methods; not paper-facing |
| `docs/GAMSPY_*.md` | GAMSPy integration; outside manuscript scope |
| `docs/LEARNING_*.md` | Training experiments; learned model not paper-primary |

For a full current-vs-historical classification, see `docs/REPO_POLISH_AUDIT.md §5`.
