# results/eswa_revision — Historical Source Data

This directory contains the experiment outputs produced during the ESWA journal
revision cycle.  It is **not** the primary paper-facing artifact folder for the
current EAAI submission.

---

## Purpose

These results support internal reproducibility audits and provide the provenance
chain for the canonical EAAI manuscript values.  They are preserved as-is and
should be treated as read-only historical data.

---

## ★ Provenance source for EAAI Table 1

One file in this directory is directly cited as the provenance source for the
canonical manuscript Table 1 values:

| File | Role |
|------|------|
| `13_tables/deterministic_method_comparison_orig.csv` | Provenance source for `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` |

**Key values from that file (TF-IDF typed-greedy, orig variant):**

| Schema R@1 | Coverage | TypeMatch | Exact20 | InstReady |
|-----------|----------|-----------|---------|-----------|
| 0.9094 | 0.8639 | 0.7513 | 0.1991 | 0.5257 |

Additional canonical retrieval metrics are in `13_tables/retrieval_main.csv` and
`01_retrieval/retrieval_results.json`.

---

## ⚠ Status of other subdirectories

All other subdirectories (`00_env/`, `02_downstream_postfix/`, `03_prefix_vs_postfix/`,
`04_method_comparison/`, `05_retrieval_vs_grounding/`, `06_robustness/`, `07_sae/`,
`08_error_taxonomy/`, `09_case_studies/`, `10_learning_appendix/`, `11_runtime/`,
`14_reports/`, `manifests/`) are **historical ESWA revision outputs**.

They are useful for internal reproducibility checks but are:
- Not directly cited in the EAAI manuscript main tables
- Not the primary canonical source for any number reported in the paper
- Preserved to document the full experiment history

---

## Canonical paper artifacts

For the authoritative EAAI manuscript results, see:
- [`results/paper/eaai_camera_ready_tables/`](../paper/eaai_camera_ready_tables/)
- [`results/paper/eaai_camera_ready_figures/`](../paper/eaai_camera_ready_figures/)
- [`docs/EAAI_SOURCE_OF_TRUTH.md`](../../docs/EAAI_SOURCE_OF_TRUTH.md)
- [`docs/RESULTS_PROVENANCE.md`](../../docs/RESULTS_PROVENANCE.md)
