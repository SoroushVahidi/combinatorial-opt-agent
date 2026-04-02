# Repository Polish Audit

**Date:** 2026-03-30
**Status:** Current — supersedes earlier ad-hoc audit notes (e.g., `docs/CURRENT_STATE_AUDIT.md`, `docs/Q1_JOURNAL_AUDIT.md`)

---

## Overview

This document records the results of a systematic polish pass on the repository prior to
EAAI camera-ready submission. It covers scope claims, metric currency, doc-vs-manuscript
conflicts, and the status of all paper-facing artifacts.

---

## Overbroad Claims

A review of `README.md` and all files in `docs/` was conducted to identify language that
could be read as claiming:

1. **Full solver-ready code generation** for arbitrary NL optimization problems
2. **Full open-world generalization** beyond the NLP4LP benchmark
3. **"AI compiler" framing** (implying end-to-end autonomous compilation)

### Findings

| Location | Potential Claim | Verdict |
|----------|----------------|---------|
| `README.md` → Capabilities table | "Restricted solver execution … 20-instance subset via SciPy HiGHS shim" | ✅ Appropriately scoped |
| `README.md` → Architecture section | Structural validation described as "without requiring a live solver" | ✅ Accurate |
| `README.md` → Project Phases checklist | Phase 4 solver row marked 🔬 Experimental | ✅ Honest framing |
| `README.md` → footer disclaimer | "All claims are benchmark-scoped (NLP4LP `orig` variant, 331 test queries)" | ✅ Correct |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | "NOT a full natural-language-to-optimization compiler" | ✅ Explicit disclaimer |
| `docs/FULL_REPO_SUMMARY.md` | Older broad language describing the system as a "pipeline … generates solver-ready code" | ⚠️ Historical doc — not paper-facing; see §5 |
| `docs/JOURNAL_READINESS_AUDIT.md` | References "end-to-end code generation" for general instances | ⚠️ Historical doc — not paper-facing; see §5 |

**Summary:** `README.md` already contains appropriate scope-limiting language. The two
flagged items are in historical-only documents that are not cited in the manuscript.
No changes to README were required on this pass.

---

## Stale Metrics or Claims

### Authoritative Result Files

The following files contain the canonical, paper-facing metrics:

| File | Metric | Value | Status |
|------|--------|-------|--------|
| `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` | TF-IDF Schema R@1 | 0.9094 | ✅ Authoritative |
| `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` | TF-IDF Coverage | 0.8639 | ✅ Authoritative |
| `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` | TF-IDF TypeMatch | 0.7513 | ✅ Authoritative |
| `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` | TF-IDF InstReady | 0.5257 | ✅ Authoritative |
| `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv` | TF-IDF Feasible (20-inst) | 0.80 | ✅ Authoritative |
| `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv` | All Table 1 raw values | — | ✅ Authoritative source for Table 1 |

### Files from Older ESWA Revision History (Not Authoritative for EAAI Figures/Tables)

| File | Era | Reason Not Authoritative |
|------|-----|--------------------------|
| `results/eswa_revision/` (except `13_tables/`) | ESWA R1/R2 | Intermediate revision artifacts; superseded by camera-ready tables |
| `current_repo_vs_manuscript_rerun.csv` | Pre-EAAI | Intermediate comparison run; superseded |
| `current_repo_vs_manuscript_rerun.md` | Pre-EAAI | Narrative for the above; superseded |
| `literature_informed_rerun_report.md` | Pre-EAAI | Literature-informed method exploration; not paper-facing |
| `publish_now_decision_report.md` | Pre-EAAI | Internal go/no-go decision log; not paper-facing |

**Note:** `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv` is the
*upstream source* for Table 1 and remains authoritative. All other ESWA revision artifacts
are historical context only.

---

## Conflicts Between Docs and Manuscript

The following older docs may contain framing that diverges from the current EAAI manuscript:

| Document | Conflict Type | Severity |
|----------|--------------|----------|
| `docs/JOURNAL_READINESS_AUDIT.md` | ESWA readiness framing; references "full pipeline" language broader than current paper scope | Low — historical, not linked from README |
| `docs/CURRENT_STATE_AUDIT.md` | Point-in-time snapshot with metric values from an earlier evaluation run | Low — clearly a snapshot doc |
| `docs/FULL_REPO_SUMMARY.md` | Broad system description written before EAAI scope narrowing | Low — not paper-facing |
| `docs/Q1_JOURNAL_AUDIT.md` | Earlier journal quality audit; some claims pre-date EAAI framing | Low — historical |
| `docs/EAAI_REPO_ALIGNMENT_AUDIT.md` | Earlier alignment audit; superseded by `EAAI_SOURCE_OF_TRUTH.md` | Low — partially superseded |

None of these documents are linked from `README.md`'s main documentation table. They are
retained as historical audit trail but are not authoritative for the EAAI submission.

---

## Which Docs are Current vs Historical

| Document | Status | Purpose |
|----------|--------|---------|
| `docs/EAAI_SOURCE_OF_TRUTH.md` | ✅ **Current** | Authoritative EAAI framing |
| `docs/REPO_POLISH_AUDIT.md` (this file) | ✅ **Current** | This polish pass |
| `docs/CURRENT_PAPER_SOURCE_OF_TRUTH.md` | ✅ **Current** | Extended paper SoT |
| `docs/START_HERE_FOR_CURRENT_PAPER.md` | ✅ **Current** | New-collaborator orientation |
| `docs/COPILOT_REPO_POLISH_HANDOFF.md` | ✅ **Current** | Polish pass handoff notes |
| `README.md` | ✅ **Current** | Main project README (already aligned) |
| `EXPERIMENTS.md` | ✅ **Current** | Experiment log |
| `KNOWN_ISSUES.md` | ✅ **Current** | Known issues and blockers |
| `docs/CURRENT_STATE.md` | ⚠️ **Historical** | Point-in-time snapshot |
| `docs/CURRENT_STATE_AUDIT.md` | ⚠️ **Historical** | Point-in-time audit |
| `docs/FULL_REPO_SUMMARY.md` | ⚠️ **Historical** | Broad summary, pre-EAAI |
| `docs/JOURNAL_READINESS_AUDIT.md` | ⚠️ **Historical** | ESWA readiness; superseded |
| `docs/Q1_JOURNAL_AUDIT.md` | ⚠️ **Historical** | Earlier quality audit |
| `docs/EAAI_REPO_ALIGNMENT_AUDIT.md` | ⚠️ **Historical** | Superseded by SoT |
| `docs/EAAI_COPILOT_HANDOFF_REPORT.md` | ⚠️ **Historical** | Prior handoff; superseded |
| `docs/eswa_revision/` | ⚠️ **Historical** | ESWA revision materials |
| `literature_informed_rerun_report.md` | ⚠️ **Historical** | Pre-EAAI method exploration |
| `publish_now_decision_report.md` | ⚠️ **Historical** | Internal decision doc |
| `current_repo_vs_manuscript_rerun.md` | ⚠️ **Historical** | Intermediate comparison |

---

## Figure Status

All 5 EAAI camera-ready figures are present and verified at
`results/paper/eaai_camera_ready_figures/`:

| Figure | PDF | PNG | PDF Size | PNG Size | Status |
|--------|-----|-----|----------|----------|--------|
| `figure1_pipeline_overview` | ✅ | ✅ | 48 KB | 20 KB | ✅ Verified |
| `figure2_main_benchmark_comparison` | ✅ | ✅ | 80 KB | 40 KB | ✅ Verified |
| `figure3_engineering_validation_comparison` | ✅ | ✅ | 64 KB | 28 KB | ✅ Verified |
| `figure4_final_solver_backed_subset` | ✅ | ✅ | 68 KB | 32 KB | ✅ Verified |
| `figure5_failure_breakdown` | ✅ | ✅ | 76 KB | 36 KB | ✅ Verified |

Figures were regenerated from authoritative source CSVs using
`tools/build_eaai_camera_ready_figures.py`. See
`analysis/eaai_figures_reproduction_report.md` for full provenance.

---

## Authoritative Paper-Facing Files

The following files constitute the complete set of paper-facing artifacts for the current
EAAI submission:

### Tables (5 files)

- `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
- `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv`
- `results/paper/eaai_camera_ready_tables/table3_executable_attempt_with_blockers.csv`
- `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv`
- `results/paper/eaai_camera_ready_tables/table5_failure_taxonomy.csv`
- `results/paper/eaai_camera_ready_tables/camera_ready_tables.md` (human-readable summary)

### Figures (10 files, 5 pairs)

- `results/paper/eaai_camera_ready_figures/figure{1-5}_*.{pdf,png}`

### Analysis Reports

- `analysis/eaai_engineering_subset_report.md`
- `analysis/eaai_executable_subset_report.md`
- `analysis/eaai_final_solver_attempt_report.md`
- `analysis/eaai_tables_build_report.md`
- `analysis/eaai_figures_build_report.md`
- `analysis/eaai_figures_reproduction_report.md`
- `analysis/eaai_tables_reproduction_report.md`
- `analysis/eaai_repo_validation_report.md`

### Framing and Orientation Docs

- `docs/EAAI_SOURCE_OF_TRUTH.md`
- `docs/REPO_POLISH_AUDIT.md` (this file)
- `docs/CURRENT_PAPER_SOURCE_OF_TRUTH.md`
- `README.md`
- `EXPERIMENTS.md`
- `KNOWN_ISSUES.md`

### Reproduction Scripts

- `tools/build_eaai_camera_ready_figures.py`
- `tools/make_nlp4lp_paper_artifacts.py`
- `tools/run_eaai_engineering_subset_experiment.py`
- `tools/run_eaai_executable_subset_experiment.py`
- `tools/run_eaai_final_solver_attempt.py`
