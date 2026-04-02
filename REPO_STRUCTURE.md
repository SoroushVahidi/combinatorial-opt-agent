# Repository Structure

Annotated directory map for the EAAI companion codebase.
★ = authoritative for the paper  |  ⚠ = historical / non-authoritative  |  ✦ = demo only

```
combinatorial-opt-agent/
│
├── ★ results/paper/                   Camera-ready EAAI tables and figures (DO NOT EDIT)
│   ├── eaai_camera_ready_tables/      Tables 1–5 CSV files
│   └── eaai_camera_ready_figures/     Figures 1–5 PNG/PDF files
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
├── ★ docs/EAAI_SOURCE_OF_TRUTH.md     Master manuscript authority
├── ★ docs/RESULTS_PROVENANCE.md       Canonical metrics + provenance chain
├──   docs/paper_vs_demo_scope.md      Paper scope vs demo scope explanation
├──   docs/wulver.md                   HPC setup (Wulver @ NJIT)
├──   docs/wulver_webapp.md            Web app on Wulver
├──   docs/learning_runs/              Benchmark-safe splits, real-data-only check
├── ⚠ docs/archive/                    Historical dev notes (NOT authoritative)
├── ⚠ docs/eswa_revision/              ESWA-era revision materials
├── ⚠ docs/audits/                     Legacy audit/decision reports
│   ├── current_repo_vs_manuscript_rerun.{md,csv}   Pre-EAAI comparison (superseded)
│   ├── literature_informed_rerun_{report.md,summary.csv}  Method exploration
│   └── publish_now_decision_{report.md,evidence.csv}      Internal go/no-go log
│
├── ★ tools/nlp4lp_downstream_utility.py   Core grounding pipeline (6 000+ lines)
│   │   Sections: extraction → slot records → basic scoring →
│   │             semantic IR → opt-role → GCG → evaluation → CLI
│   └── tools/run_eaai_*.py                Canonical EAAI experiment scripts
│
├──   retrieval/                       Schema retrieval
│   ├── search.py                      Main retrieval interface
│   ├── baselines.py                   TF-IDF, BM25, LSA, SBERT, E5, BGE
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
├──   scripts/                         Utility and paper-support scripts
│   └── paper/run_repo_validation.py   Canonical validation entrypoint
│
├──   training/                        Retrieval fine-tuning pipeline
├──   pipeline/                        Data collection pipeline
├──   data/                            Catalogs, manifests, benchmark splits
├──   artifacts/                       Copilot-vs-model comparison artifacts
├──   configs/                         Configuration files
├──   notebooks/                       Exploratory Jupyter notebooks
├──   figures/                         Raw figure sources
├──   outputs/                         Script output directory
├──   parsers/                         Data parsers
├──   schema/                          Problem schema definitions
├──   static/                          Web UI static assets
├──   collectors/                      Data collection scripts
├──   jobs/                            SLURM job definitions
│
├── ✦ app.py                           Gradio web UI (demo only)
├── ✦ feedback_server.py               Feedback collection server (demo only)
├── ✦ analyze_feedback.py              Feedback analysis script (demo only)
├── ✦ deploy_to_hf.py                  HuggingFace Spaces deployment (demo only)
├── ✦ launch_and_capture_url.py        URL capture helper (demo only)
├── ✦ run_app_wulver.sh                Wulver app launch wrapper (demo only)
├── ✦ telemetry.py                     Query telemetry module (demo only)
├── ✦ demo/                            Demo documentation + HF Spaces guide
│
├── ⚠ results/eswa_revision/           ESWA-era experiment results
│
├──   HOW_TO_REPRODUCE.md             ★ Canonical reproduction commands
├──   EXPERIMENTS.md                   Consolidated experiments overview
├──   KNOWN_ISSUES.md                  Active blockers, limitations, resolved issues
├──   CONTRIBUTING.md                  Contribution guidelines
├──   REPO_STRUCTURE.md                This file
├──   README.md                        ★ Main documentation
├──   LICENSE                          MIT License
├──   requirements.txt                 Runtime dependencies
├──   requirements-dev.txt             Testing/dev extras
├──   pytest.ini                       Pytest configuration
├──   run_search.py                    CLI entrypoint: NL query → schema search
├──   build_extended_catalog.py        Extend the problem catalog
└──   setup_catalog.sh                 One-step catalog setup script
```

---

## Zone summary

| Zone | Purpose | Authority |
|------|---------|-----------|
| `results/paper/` | Camera-ready artifacts | ★ DO NOT edit; regenerate via canonical scripts |
| `analysis/eaai_*` | Experiment reports | ★ Regenerate by re-running experiment scripts |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | Paper framing | ★ Edit only when manuscript changes |
| `docs/RESULTS_PROVENANCE.md` | Metrics provenance | ★ Update when canonical results change |
| `tools/run_eaai_*.py` | Experiment scripts | Core paper pipeline |
| `retrieval/`, `formulation/`, `src/` | Core source | Paper-scoped |
| `tests/` | Test suite | Core source |
| `app.py` and ✦-marked files | Demo application | Demo only; outside paper scope |
| `demo/` | Demo documentation | Demo only |
| `docs/archive/`, `docs/audits/`, `results/eswa_revision/` | Historical | ⚠ Not authoritative |

---

## Pipeline flow

```
NL query
  → retrieval/ (TF-IDF / BM25 / dense)   →  schema ID
  → tools/nlp4lp_downstream_utility.py   →  slot assignment
  → formulation/verify.py                →  structural check
  → [optional] SciPy HiGHS shim          →  solver result (restricted 20-instance subset)
```

---

## Key scripts

| Script | Purpose |
|--------|---------|
| `tools/run_eaai_engineering_subset_experiment.py` | Engineering subset (60 instances) |
| `tools/run_eaai_executable_subset_experiment.py` | Executable-attempt study (269 instances) |
| `tools/run_eaai_final_solver_attempt.py` | Solver-backed subset (20 instances) |
| `tools/build_eaai_camera_ready_figures.py` | Regenerate camera-ready figures |
| `scripts/paper/run_repo_validation.py` | Validate repo integrity for paper use |

For full reproduction commands, see `HOW_TO_REPRODUCE.md`.

---

## Note on Gurobi

The paper does **not** require Gurobi.  The solver-backed subset (Table 4, 20 instances)
uses a SciPy HiGHS shim.  GAMSPy/Pyomo/PuLP appear in demo code only and are outside
the paper scope.
