# DKE / SN Computer Science — Source of Truth

**This is the current entry point for the manuscript submission.** It exists because
`PROJECT_STATUS.md`, root `README.md`, `docs/CURRENT_STATUS.md`, `docs/SCIENTIFIC_STATE.md`,
`docs/KNOWN_ISSUES.md`, `docs/RESULTS_PROVENANCE.md`, `results/CANONICAL_RESULTS.md`, and
`docs/KAIS_SOURCE_OF_TRUTH.md` / `docs/EAAI_SOURCE_OF_TRUTH.md` all predate the 2026-08-13→08-15
DKE migration and still describe an earlier, retracted headline result (InstantiationReady =
0.5287) and earlier target venues (EAAI, then KAIS). They remain valuable as a historical record
of how the numbers evolved but are **not** current — see the banner added to the top of each.

## Current manuscript

- **Authoritative source:** [`manuscript/dke/main.tex`](../manuscript/dke/main.tex) (elsarticle
  class; last finalized 2026-08-15). Compiled PDF: `manuscript/dke/main.pdf`.
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

## Known open issues (not yet fixed in the manuscript text — see Stage-1 report Section 13)

1. Exact20 (on hits) uses an aggregation rule for TF-IDF/BGE-M3 inconsistent with its own stated
   definition and with the Oracle row in the same table. Evidence:
   `results/final_resubmission_method/exact20_denominator_audit.json` (`tools/audit_exact20_denominator.py`).
2. The error-taxonomy table (`tab:nlp4lp-error-taxonomy`) is 8/9 rows a stale, unattributed
   pre-bugfix carry-over. Partial current recount:
   `results/final_resubmission_method/type_mismatch_recount_2026-08-27.json`
   (`tools/recount_type_mismatch_frozen.py`).
3. The DeepOR/OR-R1 provenance sentence is factually inconsistent with `docs/DEEPOR_PROVENANCE.md`
   / `docs/ORR1_PROVENANCE.md`.
4. The OptMATH-Train "single-reviewer manual audit" characterization does not match what
   `scripts/verify_optmath_external.py` actually does (a second rule-based classifier).
5. Three of five significance-table rows previously had no reproducible committed script; fixed by
   `tools/recompute_dke_significance.py` → `results/final_resubmission_method/significance_recomputation_2026-08-27.json`.

None of the above have been edited into the manuscript yet — that is deliberately deferred to a
later manuscript-editing stage (see the Stage-2 report's Section 23).
