# EAAI Repository Alignment Audit

**Generated:** 2026-03-30  
**Purpose:** Identify all places where the repository diverges from the current EAAI manuscript framing

---

## Scope

This audit covers: README.md, EXPERIMENTS.md, HOW_TO_RUN_BENCHMARK.md, KNOWN_ISSUES.md, docs/, analysis/, results/paper/, tools/, and other paper-facing files.

---

## Summary of Findings

### Critical Overclaims (need correction or softening)

| Location | Issue | Severity |
|----------|-------|----------|
| README.md line 1–7 | Framing as "translates plain-English optimization problem descriptions into structured ILP/LP formulations and **solver-ready code**" — too broad | High |
| README.md "Project vision" section | "returns both the ILP/LP formulation and solver code ready to run" — overclaims full solver-readiness | High |
| README.md tech stack | Lists Gurobi, Pyomo, PuLP, GAMSPy as output solvers without noting that live solver integration is blocked in practice | Medium |
| README.md architecture diagram | Output box says "solver code (Pyomo / Gurobi / PuLP / GAMSPy)" without qualification | Medium |
| README.md line 95–96 | "for **unknown problems** it is generated via LLM and structurally verified" — this is speculative/future; not an EAAI main result | Medium |
| EXPERIMENTS.md title | Document is comprehensive but does not distinguish between pre-EAAI experiments and the three current EAAI validation subsets | Low |

### Stale / Outdated Claims

| Location | Issue | Severity |
|----------|-------|----------|
| README.md "Current evidence-based status" table | `Solver-based output validation ⚠️ Partial — Structural consistency checks only; no LP solver` — outdated; the final solver-backed subset (Table 4) now shows real solver outcomes via SciPy shim | Medium |
| EXPERIMENTS.md Section 1 | Lists E5 and BGE as evaluated baselines; paper-facing EAAI results only use Random/LSA/BM25/TF-IDF/Oracle | Low |
| README.md retrieval table | Mentions SBERT, E5, BGE without noting these are not in the EAAI main table | Low |

### Inconsistent Candidate/Instance Counts

| Location | Claim | Authoritative Value |
|----------|-------|---------------------|
| EXPERIMENTS.md Section 1 | "331 NLP4LP queries" | 331 — consistent with EAAI |
| README.md problem recognition | "matches queries to 90+ known problem types" | Catalog size is separate from NLP4LP 331; not a conflict but could be clarified |

### Missing Paper Artifact Pointers

| What is missing | Recommended location |
|----------------|---------------------|
| No link to `results/paper/eaai_camera_ready_figures/` | README.md "Documentation" or new "Paper artifacts" section |
| No link to `results/paper/eaai_camera_ready_tables/` | Same |
| No link to `analysis/eaai_*` reports | README.md |
| No mention of three EAAI validation subsets | README.md "Current evidence-based status" |

### Docs That Are Historical Only (not authoritative for current EAAI manuscript)

The following docs contain older experiment records, pre-fix metrics, or ESWA-era framing that may conflict with or predate the current EAAI manuscript. They should be treated as historical:

- `docs/JOURNAL_READINESS_AUDIT.md` — ESWA-era audit
- `docs/Q1_JOURNAL_AUDIT.md` — Pre-EAAI journal audit
- `docs/CURRENT_STATE_AUDIT.md` — Snapshot from earlier in the project
- `docs/BRANCH_VS_MAIN_COMPARISON.md` — Branch merge history
- `docs/FULL_REPO_SUMMARY.md` — Broad summary, not EAAI-specific
- `docs/eswa_revision/` — ESWA revision materials (metrics are canonical for Table 1, but framing is for ESWA not EAAI)
- `current_repo_vs_manuscript_rerun.md` — Intermediate comparison artifact
- `literature_informed_rerun_report.md` — Earlier literature-informed rerun
- `publish_now_decision_report.md` — Internal decision document

### Manuscript-Critical Consistency Issues

1. **Setup text mentions E5/BGE** (`EXPERIMENTS.md`, README tech stack) but the EAAI main paper results only show Random/LSA/BM25/TF-IDF/Oracle. This mismatch could confuse a reader trying to reproduce paper results. EAAI manuscript should make clear that dense retrieval methods (E5/BGE) are supplementary experiments.

2. **"Solver-ready code" framing** — The repo title and top description claim "solver-ready code" generation. The EAAI paper story is that the bottleneck is downstream number-to-slot grounding, and full solver execution is only achieved on a restricted 20-instance subset via a compatibility shim. The framing should be softened to "retrieval-assisted optimization schema grounding and scalar parameter instantiation."

3. **"Unknown problems" / LLM generation** — README.md mentions LLM-based generation for unknown problems. This is not part of the EAAI manuscript story and should either be removed from the paper-facing description or clearly labeled as "demo-only / outside paper scope."

4. **InstantiationReady metric** — EXPERIMENTS.md section 2 shows InstantiationReady=0.076 for typed greedy. README shows 0.277 for Exact20. These are different metrics (InstReady vs Exact20) so there is no numerical conflict, but the README table conflates them in a way that could be misleading.

---

## What Was Correct

- All five EAAI camera-ready tables exist and have verified provenance.
- All three EAAI experiment reports (`analysis/eaai_*_report.md`) are present and consistent.
- The `tools/run_eaai_*` scripts exist and are functional.
- `tools/build_eaai_camera_ready_figures.py` exists and generates all figures correctly.
- `KNOWN_ISSUES.md` is accurate and up-to-date.
- `EXPERIMENTS.md` contains detailed and accurate experiment records.
- The NLP4LP benchmark instance count (331) is consistent across all relevant files.

---

## Actions Taken

1. ✅ `README.md` — Updated to use EAAI-aligned framing (see revised file)
2. ✅ `requirements.txt` — Added `Pillow>=9.0.0` and `scipy>=1.9.0`
3. ✅ All 5 EAAI figures (PNG + PDF) reproduced from verified table sources
4. ✅ Created `docs/EAAI_SOURCE_OF_TRUTH.md`
5. ✅ Created `docs/EAAI_COPILOT_HANDOFF_REPORT.md`
6. ✅ Created `analysis/eaai_figures_reproduction_report.md`
7. ✅ Created `analysis/eaai_tables_reproduction_report.md`
8. ✅ Created `analysis/eaai_repo_validation_report.md`
9. ✅ Added historical notes header to `EXPERIMENTS.md`

---

## Remaining Items (Recommended for Author)

1. Decide whether E5/BGE results should be in an appendix of the EAAI manuscript or footnoted as supplementary.
2. Double-check the "LLM generation for unknown problems" claim — remove from manuscript or clearly scope.
3. Add paper artifact map section to README.md pointing to camera-ready figures and tables (done in this pass).
4. Verify that the double-anonymized submission does not include the GitHub repo URL or author names in the manuscript body.
5. Consider adding a prominent `[HISTORICAL]` header to the oldest pre-EAAI docs listed above to prevent confusion.
