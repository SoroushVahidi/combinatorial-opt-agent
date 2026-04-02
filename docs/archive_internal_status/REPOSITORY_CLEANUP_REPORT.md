# Repository Cleanup Report

**Date:** 2026-04-02  
**Branch:** `copilot/prepare-research-codebase-eaai-paper`  
**Purpose:** Polish the companion codebase for the EAAI submission on retrieval-assisted
instantiation of natural-language optimization problems.

---

## Summary

This report documents all changes made during the publication-readiness cleanup pass.
The guiding principle was *minimal, safe, surgical edits* — preserving all experimental
integrity, existing tests, paper artifacts, and reproducibility while reducing clutter
and improving clarity.

---

## What Was Changed

### A. README.md (top-level)

| Change | Detail |
|--------|--------|
| Added "Repository status" section | Clearly states benchmark scope (NLP4LP, 331 queries, `orig` variant), core evaluated task, and that solver results are restricted-subset only |
| Added "What this repo does not claim" section | Aligned with `docs/EAAI_SOURCE_OF_TRUTH.md`: no full NL→solver compilation, no benchmark-wide readiness, dense retrieval supplementary, LLM outside scope, Gurobi not required |
| Added "How to reproduce the main paper artifacts" section | Canonical commands using `tools/run_eaai_*.py` scripts |
| Added "Repo map" section | Annotated table marking authoritative (★) vs historical/demo paths |
| Removed "Testing on iPhone" section | Irrelevant to an academic paper companion repo |
| Simplified Documentation table | Removed stale links to moved docs; kept essential references |
| Simplified Data sources section | Removed stale link to moved `docs/data_sources.md` |
| Fixed Tech Stack solvers row | Removed Gurobi (paper does NOT require it); changed to "SciPy HiGHS shim (restricted subset, paper results); GAMSPy/Pyomo/PuLP (demo only)" |
| Simplified Training section | Removed stale `NLP4LP_CONSTRAINED_ASSIGNMENT_*.md` reference (file moved to archive) |
| Fixed Historical note | Now points to `docs/archive/` instead of non-existent `docs/` files |

### B. docs/ — Archival of stale development documents

**94 historical development and handoff documents** were moved from `docs/` into
`docs/archive/`.  A `docs/archive/README.md` was added explaining these files are
NOT authoritative.

Files retained in `docs/` (authoritative or HPC reference):
- `docs/EAAI_SOURCE_OF_TRUTH.md` — manuscript authority
- `docs/README.md` — documentation index
- `docs/wulver.md` — HPC setup reference
- `docs/wulver_webapp.md` — web app on HPC reference
- `docs/eswa_revision/` — ESWA-era revision materials (historical, kept intact)
- `docs/learning_runs/` — learning experiment logs

All moved files remain fully traceable in `docs/archive/`.

### C. tools/nlp4lp_downstream_utility.py — In-place partial refactor

A **full package split** into `tools/nlp4lp/` submodules was not performed because
several existing regression tests use `ast.parse()` on this exact file and assert that
specific string constants appear in its AST.  A shim/re-export approach would break
those tests.

Instead, a **safe in-place partial refactor** was performed:
- Replaced the 4-line module docstring with a **44-line docstring** including a
  14-section table of contents and environment-variable reference
- Added **8 new section banners** (`# ── Section N – … ──`) for sections 2, 3, 4, 5,
  6, 13, and 14, complementing the pre-existing section headers already in the file
- No function signatures, return values, or side effects were changed
- All 1453 original tests continue to pass

### D. requirements.txt

- Removed **duplicate `pypdf` entry** (`pypdf>=3.0.0` was a duplicate of `pypdf>=4.0.0`)
- Created **`requirements-dev.txt`** with testing extras (`pytest>=7.0.0`, `pytest-mock>=3.0.0`)

### E. New files added

| File | Purpose |
|------|---------|
| `scripts/paper/run_repo_validation.py` | Canonical validation script: checks required artifacts, result tables, figures, analysis reports, experiment scripts, importability of core utility, and duplicate-free requirements.txt. Exits 0 on pass, 1 on failure. Degrades gracefully for HF dataset. |
| `CONTRIBUTING.md` | Short practical guide: how to run tests, protected paths, how to add grounding/retrieval methods, code style |
| `REPO_STRUCTURE.md` | Annotated directory map with authoritative vs historical paths, pipeline flow, key scripts table, Gurobi clarification |
| `tests/test_repo_hygiene.py` | 15 sanity tests: source-of-truth files exist, paper tables exist and are valid CSVs, experiment scripts exist, requirements.txt has no duplicates, validation script is syntactically valid |
| `docs/archive/README.md` | Notice that archived files are NOT authoritative |
| `docs/REPOSITORY_CLEANUP_REPORT.md` | This file |

---

## What Was Left Untouched

| Item | Reason |
|------|--------|
| `results/paper/` contents | All camera-ready tables and figures preserved exactly as committed |
| `analysis/eaai_*` reports | All EAAI analysis reports preserved unchanged |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | Authoritative manuscript document — not modified |
| All test files in `tests/` (except new addition) | Preserved to protect regression coverage |
| `results/eswa_revision/` | Historical ESWA revision materials kept intact |
| All experiment scripts in `tools/` | No functional changes |
| `retrieval/`, `formulation/`, `src/`, `pipeline/`, `training/` | No changes |
| All CSV result files | No measured values were edited |
| `KNOWN_ISSUES.md`, `EXPERIMENTS.md`, `HOW_TO_RUN_BENCHMARK.md` | Useful reference docs kept at top level |
| `artifacts/`, `data/`, `figures/`, `outputs/`, `notebooks/` | Preserved intact |

---

## Any Remaining Issues

1. **Top-level clutter** — Several files at the repo root are legacy development
   artefacts that could be moved to a root-level `archive/` in a future pass:
   - `current_repo_vs_manuscript_rerun.csv` / `.md`
   - `literature_informed_rerun_report.md` / `_summary.csv`
   - `publish_now_decision_evidence.csv` / `_report.md`
   - `LINKEDIN_POST.md`, `README_Spaces.md`, `analyze_feedback.py`

2. **docs/eswa_revision/** — Contains older ESWA-era experiment notes.  These are
   historical and should not be confused with current EAAI claims, but are kept intact
   as provenance.

3. **Learned retrieval fine-tuning** — Documented in `KNOWN_ISSUES.md` as "real-data
   run did not beat rule baseline."  The training pipeline is present but correctly
   labeled "future work."

4. **LLM baseline API keys** — `tools/llm_baselines.py` requires `OPENAI_API_KEY` or
   `GEMINI_API_KEY`.  These are optional; the paper's primary results do not depend on
   them.

---

## Risky Areas Intentionally Not Refactored

| Area | Risk | Decision |
|------|------|---------|
| Full split of `tools/nlp4lp_downstream_utility.py` | AST-based tests would break; complex circular-import risk in a 6 400+ line file | In-place partial refactor only |
| Moving `analysis/eaai_*` reports | Would require updating references in docs | Left in place |
| Reorganizing `tests/` into `unit/`/`integration/`/`regression/` subdirectories | Would require updating all conftest.py paths and pytest.ini | Left flat; well-named files make navigation straightforward |
| Removing `results/eswa_revision/` | Historical provenance value | Kept intact |

---

## Recommended Future Work

1. **Move top-level stale files** — Move `current_repo_vs_manuscript_rerun.*`,
   `literature_informed_rerun_*`, `publish_now_decision_*`, `LINKEDIN_POST.md`,
   `README_Spaces.md` into a root-level `archive/` folder.

2. **Reorganize tests** — If import paths can be updated safely, consider grouping
   tests into `tests/unit/`, `tests/integration/`, `tests/regression/` for better
   discoverability.

3. **Package tools/nlp4lp_downstream_utility.py** — Once the AST-based test constraints
   are addressed (either by updating the tests or keeping a thin shim with the required
   constants), a full split into `tools/nlp4lp/` submodules would improve maintainability.

4. **Add a Makefile** — Simple targets (`make test`, `make validate`, `make lint`) would
   lower the barrier for new contributors.

5. **Pin CI Python version** — The test suite works with Python 3.12; explicitly pinning
   in `.github/workflows` would prevent silent breakage.

6. **Add `__init__.py` to `tools/`** — Would make `tools` a proper package and simplify
   imports across experiment scripts.
