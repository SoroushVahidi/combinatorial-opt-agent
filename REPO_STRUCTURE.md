# Repository Structure

Annotated directory map for the EAAI companion codebase.
★ = authoritative for the paper  |  ⚠ = historical / non-authoritative

```
combinatorial-opt-agent/
│
├── ★ results/paper/                   Camera-ready EAAI tables and figures
│   ├── eaai_camera_ready_tables/      Table 1–5 CSV files (DO NOT EDIT)
│   └── eaai_camera_ready_figures/     Figure 1–5 PNG/PDF files
│
├── ★ analysis/                        EAAI experiment reports
│   ├── eaai_engineering_subset_report.md
│   ├── eaai_executable_subset_report.md
│   ├── eaai_final_solver_attempt_report.md
│   ├── eaai_tables_build_report.md
│   ├── eaai_figures_build_report.md
│   ├── eaai_figures_reproduction_report.md
│   └── eaai_tables_reproduction_report.md
│
├── ★ docs/EAAI_SOURCE_OF_TRUTH.md     Manuscript authority — paper framing + claims
├── ★ docs/CURRENT_STATUS.md           Reviewer-facing status + headline pointers
├──   docs/wulver.md                   HPC setup reference (Wulver @ NJIT)
├── ⚠ docs/archive_internal_status/   Internal audits / decision logs (provenance only)
├── ⚠ docs/archive/                    Historical dev notes (NOT authoritative)
├── ⚠ docs/eswa_revision/              Earlier ESWA-era revision materials
│
├── ★ tools/nlp4lp_downstream_utility.py   Core grounding pipeline (6 000+ lines)
│   │   Sections: extraction → slot records → basic scoring →
│   │             semantic IR → opt-role → GCG → evaluation → CLI
│   └── tools/run_eaai_*.py            Canonical EAAI experiment scripts
│
├──   retrieval/                       Schema retrieval (TF-IDF, BM25, LSA, SBERT, E5, BGE)
│   ├── search.py                      Main retrieval interface
│   ├── baselines.py                   TF-IDF, BM25, LSA baseline retrievers
│   └── utils.py                       Short-query expansion helpers
│
├──   formulation/                     LP structural validation (no live solver)
│   └── verify.py                      Objective-sense + variable-symbol checks
│
├──   src/                             Number-role repair subsystem
│   ├── features/number_role_features.py
│   └── analysis/consistency_benchmark.py
│
├──   tests/                           Pytest test suite (1 400+ tests, CPU-only)
├──   scripts/paper/                   Paper-support scripts
│   └── run_repo_validation.py         Canonical validation entrypoint
│
├──   training/                        Retrieval fine-tuning pipeline
├──   pipeline/                        Data collection pipeline
├──   data/                            Catalogs, manifests, benchmark splits
├──   artifacts/                       Copilot-vs-model comparison artifacts
├──   batch/                           Batch job definitions
├──   configs/                         Configuration files
├──   notebooks/                       Exploratory Jupyter notebooks
├──   figures/                         Raw figure sources
├──   outputs/                         Script output directory (gitignored or archival)
├──   parsers/                         Data parsers
├──   schema/                          Problem schema definitions
├──   static/                          Web UI static assets
├──   collectors/                      Data collection scripts
│
├── ⚠ results/eswa_revision/           ESWA-era experiment results
│
├──   requirements.txt                 Runtime dependencies
├──   requirements-dev.txt             Testing/dev extras (pytest, pytest-mock)
├──   pytest.ini                       Pytest configuration
├──   README.md                        ★ Main documentation
├──   REPO_STRUCTURE.md                This file
├──   CONTRIBUTING.md                  Contribution guidelines
└──   EXPERIMENTS.md                   Consolidated experiments overview
```

## Pipeline flow

```
NL query
  → retrieval/ (TF-IDF / BM25 / dense)  →  schema ID
  → tools/nlp4lp_downstream_utility.py  →  slot assignment
  → formulation/verify.py               →  structural check
  → [optional] SciPy HiGHS shim         →  solver result (restricted subset)
```

## Authoritative vs historical

| Path | Status | Notes |
|------|--------|-------|
| `results/paper/` | ★ Authoritative | DO NOT edit; regenerate via canonical scripts only |
| `analysis/eaai_*` | ★ Authoritative | Experiment reports; regenerate by re-running scripts |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | ★ Authoritative | Master reference for paper claims |
| `docs/CURRENT_STATUS.md` | ★ Summary | Reviewer-facing status; points to tables |
| `docs/archive_internal_status/` | ⚠ Provenance | Internal audits; not headline source |
| `docs/archive/` | ⚠ Historical | Dev notes; preserved for provenance, not citation |
| `docs/eswa_revision/` | ⚠ Historical | ESWA-era revision; not EAAI-authoritative |
| `results/eswa_revision/` | ⚠ Historical | Earlier experiment results |

## Key scripts

| Script | Purpose |
|--------|---------|
| `tools/run_eaai_engineering_subset_experiment.py` | Engineering subset (60 instances) |
| `tools/run_eaai_executable_subset_experiment.py` | Executable-attempt study (269 instances) |
| `tools/run_eaai_final_solver_attempt.py` | Solver-backed subset (20 instances) |
| `tools/build_eaai_camera_ready_figures.py` | Regenerate camera-ready figures |
| `scripts/paper/run_repo_validation.py` | Validate repo integrity for paper use |

## Note on Gurobi

The paper does **not** require Gurobi. The solver-backed subset (Table 4, 20 instances)
uses a SciPy HiGHS shim. GAMSPy/Pyomo/PuLP appear in demo code only and are outside
the paper scope.
