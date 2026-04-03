# Documentation index

Canonical documentation for the EAAI companion codebase. **Start with the reviewer row**, then drill into scope and reproduction.

---

## Reviewer-facing (canonical story)

| Doc | Role |
|-----|------|
| **[REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)** | **Read first** — orientation, official metrics pointer, limitations |
| **[CURRENT_STATUS.md](CURRENT_STATUS.md)** | One-page public truth — validated vs auxiliary |
| **[EAAI_SOURCE_OF_TRUTH.md](EAAI_SOURCE_OF_TRUTH.md)** | Manuscript authority — what IS / IS NOT claimed |
| **[RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md)** | Definitions + provenance chain (“which file is this number?”) |
| **[GEMINI_RERUN_REPORT.md](GEMINI_RERUN_REPORT.md)** | Gemini Slurm/preflight/cache — **infrastructure**; not Table 1–5 authority |
| **[DATASET_EXPANSION_PLAN.md](DATASET_EXPANSION_PLAN.md)** | Text2Zinc & CP-Bench — why / scope (NLP4LP stays canonical) |
| **[DATASET_EXPANSION_STATUS.md](DATASET_EXPANSION_STATUS.md)** | What is integrated vs still pending (no fabricated benchmark claims) |
| **[paper_vs_demo_scope.md](paper_vs_demo_scope.md)** | Fixed-catalog benchmark vs demo / open-domain |

**Same folder (`docs/`):** [HOW_TO_REPRODUCE.md](HOW_TO_REPRODUCE.md) · [HOW_TO_RUN_BENCHMARK.md](HOW_TO_RUN_BENCHMARK.md) · [REPO_STRUCTURE.md](REPO_STRUCTURE.md) · [KNOWN_ISSUES.md](KNOWN_ISSUES.md) · [EXPERIMENTS.md](EXPERIMENTS.md) · [README.md](../README.md) (repo root)

**Integrity:** `python scripts/check_docs_integrity.py` (from repo root)

---

## Terminology (consistent)

| Term | Meaning here |
|------|----------------|
| **Fixed-catalog NLP4LP benchmark** | 331 test queries, `orig` primary; retrieval over NLP4LP schema catalog |
| **Deterministic scalar grounding** | Rule-based / repair pipeline in `tools/nlp4lp_downstream_utility.py` (not LLM slot-fill for paper tables) |
| **Restricted solver-backed subset** | 20 instances, SciPy HiGHS shim (Table 4) |
| **Rerun infrastructure** | Slurm `batch/learning/`, `results/rerun/`, optional LLM APIs |
| **Canonical vs legacy** | Camera-ready `results/paper/eaai_camera_ready_tables/` + `docs/EAAI_SOURCE_OF_TRUTH.md` vs `docs/archive/` / `results/eswa_revision/` |

---

## HPC and cluster

| Doc | Role |
|-----|------|
| [wulver.md](wulver.md) | NJIT Wulver / SLURM |
| [wulver_webapp.md](wulver_webapp.md) | Web app on Wulver |

---

## Learning experiments (non-paper-core)

| Doc | Role |
|-----|------|
| [learning_runs/README.md](learning_runs/README.md) | Splits, records, benchmark-safe practice |

---

## Provenance and archives (not headline sources)

| Location | Role |
|----------|------|
| [provenance/](provenance/README.md) | Dated audits / cleanup CSVs — **not** `CURRENT_STATUS.md` |
| [archive_internal_status/](archive_internal_status/README.md) | Internal go/no-go and manuscript comparison trails |
| [archive/](archive/README.md) | Historical development notes |
| [eswa_revision/](eswa_revision/README.md) | ESWA-era materials (superseded by EAAI framing) |
| [audits/](audits/README.md) | Index into archive_internal_status |
| [../analysis/archive/](../analysis/archive/README.md) | Non-EAAI analysis artifacts |

---

## Camera-ready artifacts (paths only)

| Need | Location |
|------|----------|
| Tables 1–5 | `results/paper/eaai_camera_ready_tables/` |
| Figures 1–5 | `results/paper/eaai_camera_ready_figures/` |
| EAAI experiment reports | `analysis/eaai_*_report.md` |
