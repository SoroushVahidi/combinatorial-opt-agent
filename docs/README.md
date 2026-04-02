# Documentation index

This folder contains the canonical documentation for the EAAI companion codebase.
Start with the reviewer-facing docs below; the archive folders hold historical material.

---

## Reviewer-facing (start here)

| Doc | Purpose |
|-----|---------|
| **[REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)** | Practical guide for reviewers — what to look at, where metrics come from, limitations |
| **[CURRENT_STATUS.md](CURRENT_STATUS.md)** | Single concise status page — headline metrics, validated vs auxiliary, limitations |
| **[EAAI_SOURCE_OF_TRUTH.md](EAAI_SOURCE_OF_TRUTH.md)** | Manuscript authority — paper framing, authoritative file list, benchmark story |
| **[RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md)** | Canonical metrics + full provenance chain (which number came from where) |
| **[paper_vs_demo_scope.md](paper_vs_demo_scope.md)** | Paper scope vs demo scope — what is benchmarked vs demo-only |
| **[../REPO_STRUCTURE.md](../REPO_STRUCTURE.md)** | Annotated directory map (canonical vs historical) |
| **[../KNOWN_ISSUES.md](../KNOWN_ISSUES.md)** | Active blockers, limitations, and resolved historical issues |
| **[../HOW_TO_REPRODUCE.md](../HOW_TO_REPRODUCE.md)** | Step-by-step reproduction commands |
| **[../EXPERIMENTS.md](../EXPERIMENTS.md)** | Consolidated experiments overview (retrieval, grounding, learning) |

---

## HPC and infrastructure

| Doc | Purpose |
|-----|---------|
| [wulver.md](wulver.md) | NJIT Wulver cluster setup and batch job submission |
| [wulver_webapp.md](wulver_webapp.md) | Running the web app on Wulver |

---

## Learning experiments

| Doc | Purpose |
|-----|---------|
| [learning_runs/README.md](learning_runs/README.md) | Benchmark-safe splits and learning experiment records |
| [learning_runs/real_data_only_learning_check.md](learning_runs/real_data_only_learning_check.md) | Real-data-only learning check (no synthetic aux); conclusion: learning did not beat rule baseline |

---

## Archive folders (provenance, not authoritative)

| Folder | Contents |
|--------|---------|
| [archive/](archive/README.md) | Historical dev notes, method iteration logs, handoff documents. **Not** authoritative for the EAAI manuscript. |
| [archive_internal_status/](archive_internal_status/README.md) | Internal audits, manuscript-vs-repo comparisons, go/no-go decision logs. **Not** the public headline source. |
| [eswa_revision/](eswa_revision/README.md) | ESWA-era revision materials (superseded by EAAI framing). |
| [audits/](audits/README.md) | Index pointing to `archive_internal_status/` (files consolidated there). |

---

## Quick links

| Need | Location |
|------|----------|
| Camera-ready tables (Tables 1–5) | `results/paper/eaai_camera_ready_tables/` |
| Camera-ready figures (Figures 1–5) | `results/paper/eaai_camera_ready_figures/` |
| EAAI experiment reports | `analysis/eaai_*_report.md` |
| Non-canonical analysis archive | `analysis/archive/` |
