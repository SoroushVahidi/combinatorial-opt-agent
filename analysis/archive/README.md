# analysis/archive — Non-canonical analysis reports

> **This folder is retained for provenance and is not the canonical public summary.** See **`docs/CURRENT_STATUS.md`** and **`docs/REVIEWER_GUIDE.md`** first.

This folder holds analysis and audit reports that are **not** authoritative for the
EAAI manuscript. They are retained for provenance and research history. For the
annotated repository map and canonical code paths, use **`docs/REPO_STRUCTURE.md`**.

**Do not cite these files as the paper's main results.**

For authoritative sources, see:

- `analysis/eaai_*_report.md` — EAAI experiment reports (canonical)
- `results/paper/eaai_camera_ready_tables/` — Camera-ready tables (Tables 1–5)
- `docs/EAAI_SOURCE_OF_TRUTH.md` — Manuscript authority and authoritative file list
- `docs/RESULTS_PROVENANCE.md` — Canonical metrics + provenance chain

## Files in this folder

| File | Description |
|------|-------------|
| `classic_problem_family_performance.md` | Per-family TF-IDF performance breakdown (exploratory; pre-EAAI camera-ready) |
| `classic_problem_family_performance.csv` | Data source for the above |
| `grounding_failure_examples.md` | Per-category grounding failure case study (TF-IDF typed-greedy on 331 NLP4LP test instances) |
| `FIGURE_REPRODUCTION_REPORT.md` | Internal figure reproduction validation log |
| `REPO_VALIDATION_REPORT.md` | Internal repo integrity validation log |
| `TABLE_REPRODUCTION_REPORT.md` | Internal table reproduction validation log |
| `binary_cleanup_report.md` | Binary file cleanup audit |
| `dataset_parallel_work_audit.md` | Parallel dataset integration audit |
| `missing_normalized_sources_audit.md` | Audit of missing normalized data sources |
| `new_dataset_integration_audit.md` | New dataset integration audit |
