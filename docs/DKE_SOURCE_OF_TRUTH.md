# DKE / SN Computer Science — Source of Truth

**This is the current entry point for the manuscript submission.** It exists because
`PROJECT_STATUS.md`, root `README.md`, `docs/CURRENT_STATUS.md`, `docs/SCIENTIFIC_STATE.md`,
`docs/KNOWN_ISSUES.md`, `docs/RESULTS_PROVENANCE.md`, `results/CANONICAL_RESULTS.md`, and
`docs/KAIS_SOURCE_OF_TRUTH.md` / `docs/EAAI_SOURCE_OF_TRUTH.md` all predate the 2026-08-13→08-15
DKE migration and still describe an earlier, retracted headline result (InstantiationReady =
0.5287) and earlier target venues (EAAI, then KAIS). They remain valuable as a historical record
of how the numbers evolved but are **not** current — see the banner added to the top of each.

## Current manuscript

- **Authoritative submission source:** [`manuscript/sncs/main.tex`](../manuscript/sncs/main.tex)
  (Springer Nature `sn-jnl` class, `sn-basic` numbered references; migrated 2026-08-27). Submission
  package: `manuscript/sncs/submission_package/` (clean-room build verified, 39 pages, 0 undefined
  citations/references). Metadata: `manuscript/sncs/SUBMISSION_METADATA.md`,
  `manuscript/sncs/SUBMISSION_CHECKLIST.md`.
- **Migration source (also corrected, kept for historical/DKE-track reference):**
  [`manuscript/dke/main.tex`](../manuscript/dke/main.tex) (elsarticle class; scientifically
  corrected 2026-08-27, same content as the SNCS version modulo template).
- **Target journal:** SN Computer Science (Springer Nature).
- **Superseded manuscript versions:** `manuscript/main.tex` / `manuscript/submission_package/main.tex`
  (byte-identical, 2026-08-11, KAIS-targeted, contains the retracted 0.5287-era numbers) and the
  original EAAI/Elsevier draft referenced in `manuscript/MANUSCRIPT_README.md`.

## Current authoritative result artifacts

| Result family | Authoritative artifact |
|---|---|
| TF-IDF downstream (Coverage/TypeMatch/InstReady/Strict) | `results/final_resubmission_method/metrics.json` (git SHA `72f7e29`) |
| Oracle control | `results/oracle_recomputation_2026-08-15/oracle_frozen_verification.json` |
| BGE-M3 retrieval + downstream | `results/dense_retrieval_bge_m3/{retrieval_metrics,downstream_metrics,significance_tests}.json` |
| Structural / executable / solver-backed subsets (60/269/20) | `results/paper/eaai_camera_ready_tables/{table2,table3,table4,table5}*.csv` |
| External baseline comparison (PaMOP/ORLM/OptMATH/Generic-LLM/DeepOR/OR-R1) | `results/external_baseline_comparison/comparison.json`/`.md` |
| External validation (OptMATH-Train numeric extraction) | `results/external_validation/optmath/final_verification/verified_metrics.json` |

Full provenance and per-family authority/supersession detail: see
[`docs/DKE_STAGE1_RESULT_MIGRATION_2026-08-15.md`](DKE_STAGE1_RESULT_MIGRATION_2026-08-15.md)
and Section 4 of the audit below.

## Audit trail (read before editing the manuscript)

- [`docs/SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md`](SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md)
  — full manuscript-vs-repository consistency audit: authoritative-manuscript determination,
  complete numerical-claim inventory, the Exact20 denominator root-cause, error-taxonomy
  staleness, statistical-reproduction gaps, structural/solver-subset verification, external-baseline
  provenance problems (DeepOR/OR-R1, OptMATH "manual audit" mischaracterization), and ranked
  MUST-FIX / HIGH-VALUE items.
- [`docs/SNCS_STAGE2_LOCAL_WULVER_GITHUB_AUDIT_2026-08-27.md`](SNCS_STAGE2_LOCAL_WULVER_GITHUB_AUDIT_2026-08-27.md)
  — local/GitHub/Wulver reconciliation, MUST-FIX resolution mapping, and the deterministic
  reproduction/verification artifacts preserved in this stage.
- [`docs/SNCS_STAGE3_MANUSCRIPT_SPRINGER_REPOSITORY_FINALIZATION_2026-08-27.md`](SNCS_STAGE3_MANUSCRIPT_SPRINGER_REPOSITORY_FINALIZATION_2026-08-27.md)
  — all Stage-1 MUST-FIX items resolved and applied to the manuscript text; SN Computer Science
  Springer migration.
- [`docs/SNCS_STAGE4_FINAL_SUBMISSION_AUDIT.md`](SNCS_STAGE4_FINAL_SUBMISSION_AUDIT.md)
  — final fresh read-through, official-requirement re-verification, a genuine table layout defect
  found and fixed, clean-room build verification, and submission package preparation.

## Issues found in Stages 1-2 — all now fixed in the manuscript text (Stage 3)

1. **Exact20 (on hits)** — corrected to a uniform, definition-consistent rule across TF-IDF/BGE-M3/
   Oracle. Evidence: `results/final_resubmission_method/exact20_uniform_2026-08-27.json`
   (`tools/compute_uniform_exact20.py`).
2. **Error-taxonomy table** — replaced with an exact, reproducible, mutually exclusive 5-category
   residual-error decomposition. Evidence: `results/final_resubmission_method/residual_error_analysis_2026-08-27.json`
   (`tools/recompute_residual_error_analysis.py`).
3. **DeepOR/OR-R1 provenance sentence** — corrected in both manuscripts (4 occurrences each) to
   accurately state DeepOR has neither code nor a checkpoint, and OR-R1 has code but no released
   checkpoint.
4. **OptMATH-Train "single-reviewer manual audit"** — corrected to accurately describe an automated
   second-pass rule-based classifier cross-check; the unsupported "148/150 generated-code omission"
   claim was removed (no corroborating evidence exists).
5. **Statistical reproduction** — all 5 significance-table rows (including the 3 previously
   unreproduced ones) now reproduce exactly, including p-values, from
   `tools/recompute_dke_significance.py`.

All of the above are applied in both `manuscript/dke/main.tex` and `manuscript/sncs/main.tex`. The
only remaining open item is an **author-confirmation** field (not evidence-dependent): the exact
Azure OpenAI funding/credit disclosure — see `manuscript/sncs/SUBMISSION_METADATA.md`.
