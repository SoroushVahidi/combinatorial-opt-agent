# Attribution correction v1 — author review

Status: bounded scientific attribution revision of the public pre-submission manuscript; no new manuscript release, journal submission or acceptance claim. Author review remains required.

## Authoritative versions

- Baseline manuscript: [`jdsa-manuscript-v1.0.0`](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0), commit `00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc`, `manuscript/jdsa-stage3/`.
- Immutable evidence: [`jdsa-evidence-v2.0.0`](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0), commit `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`.
- Original top-k experiment: `2d386ccce98398bb0aa5540a9c1ccc5f5eb0c6cd`.
- Correction branch: `manuscript/jdsa-attribution-corrections-20261029`.

The older `manuscript/jdsa/source/` is not canonical for this revision. All existing release PDFs, bibliography builds and submission ZIPs remain unchanged. Corrected sources remain in `manuscript/jdsa-stage3/`; newly built PDFs are separate in [`manuscript/jdsa-attribution-correction-v1/`](../../manuscript/jdsa-attribution-correction-v1/). The notices distinguish them from the baseline artifacts and release.

## C01 — Oracle identity and the two-case difference

The previous sentence, “A gold-schema top-ten oracle yields 270 ordinary-ready queries”, compressed two distinct selection rules. This is an attribution correction, not a new result or historical result revision.

| Original configuration | Selection | Ready /331 | Correct schemas /331 | Wrong-schema ready |
|---|---|---:|---:|---:|
| `oracle_A_gold_schema_k10` | Gold schema whenever available in top ten; baseline otherwise | 268 | 325 | 0 |
| `oracle_C_gold_ready_k10` | Gold schema only if its grounding is ready; baseline otherwise | 270 | 322 | 2 |
| `oracle_B_readiness_k10` | Highest-ranked ready candidate regardless of schema correctness; baseline if none | 286 | 293 | 32 |

The top-five heuristic `all_k5_R3_ready_cov_type_tfidf` remains 282 ready with 20 schema regressions. A different top-five heuristic (`all_k5_R5_small_consistency_score`) also has 270 ready; this coincident count is not the origin of oracle C's identity.

For `nlp4lp_test_286`, the baseline selects wrong schema `nlp4lp_test_285`, with coverage/type both 1. Its gold candidate at rank 4 has coverage 2/3 and type 1. For `nlp4lp_test_299`, the baseline selects wrong schema `nlp4lp_test_321`, also with coverage/type 1; the gold candidate at rank 6 has coverage 1/2 and type 1. A switches both to non-ready gold candidates; C preserves both wrong but ready baseline choices. From baseline 257, both gain 13 ready cases; A loses two and C loses none. Both therefore have 268 schema-correct ready cases. No interpretation of 270 as universally correct-schema output is justified.

Primary original files (all at the original top-k commit):

- [`candidate_groundings.csv`](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/2d386ccce98398bb0aa5540a9c1ccc5f5eb0c6cd/results/topk_schema_rerank_stage_a/candidate_groundings.csv): 3,310 rows, ten per query; SHA-256 `e7fd444b059b52247a5cf1a4927eb29c7fe408af9e004f569868643caa3ce632`.
- [`oracle_topk.csv`](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/2d386ccce98398bb0aa5540a9c1ccc5f5eb0c6cd/results/topk_schema_rerank_stage_a/oracle_topk.csv) and [`summary.json`](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/2d386ccce98398bb0aa5540a9c1ccc5f5eb0c6cd/results/topk_schema_rerank_stage_a/summary.json).
- [`tools/topk_schema_rerank_stage_a.py`](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/2d386ccce98398bb0aa5540a9c1ccc5f5eb0c6cd/tools/topk_schema_rerank_stage_a.py), selection definitions at lines 373–385 and readiness at line 163.

The candidate-level accounting is a read of saved observations, not a new grounding run. The original inputs are reference-assisted; readiness checks coverage/type thresholds and not scalar-value correctness.

## C02 — Historical P@5 naming

The ranked-retrieval display heading is now **Hit@5**. The numerical cells are identical. The historical `P@5` JSON name and evaluator are unchanged. `training/metrics.py:14–19,77–84` computes an indicator of whether the single target occurs in the first five results, then averages across queries. This is Hit@5 (equivalently Recall@5 for one relevant target), not precision among five retrieved documents.

The implementation is present at the retrieval report's documented revision [`e3fdaf49bf507e7eed5f36621e7e4615a3342c5d`](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/e3fdaf49bf507e7eed5f36621e7e4615a3342c5d/training/metrics.py). Historical higher-rank cells retain summary-supported status because their complete original ranked outputs have not been authenticated. No new rankings were generated.

## C03 — Scientific positioning and literature

The contributions, Related Work and Discussion now specify reference-assisted supplied-slot grounding, complete stored-reference agreement, ordered gate accounting and fixed-score decoder attribution. They acknowledge prior diagnostic work and do not claim priority for component-level evaluation, a new matching algorithm, autonomous formulation, independent semantic accuracy or unseen-template generalization.

Added references:

- Refai, Dania; Ahmed, Moataz (2026). *A component-level evaluation framework for diagnosing LLM errors in optimization modelling*. Neurocomputing 695, article 133981. [DOI](https://doi.org/10.1016/j.neucom.2026.133981). Metadata and supported component-metric description verified against the [authors' institutional publication record](https://pure.kfupm.edu.sa/en/publications/a-component-level-evaluation-framework-for-diagnosing-llm-errors-/). Publisher full text returned HTTP 403. Detailed methods, datasets and validation safeguards are **UNVERIFIED**; the manuscript comparison is limited to the primary abstract.
- Chen, Yitian; Cheng, Cheng; Sun, Yinan; Ling, Zi; Ge, Dongdong (2026). *OPT-Engine: Benchmarking the Limits of LLMs in Optimization Modeling via Complexity Scaling*. ICML, PMLR 306, pp. 13807–13834. [Official proceedings](https://proceedings.mlr.press/v306/chen26o.html). No DOI was invented. The comparison concerns documented complexity-based formulation/solving diagnosis; no baseline performance is attributed to this project.

Existing [OptMATH](https://proceedings.mlr.press/v267/lu25o.html) and [Autoformulation](https://proceedings.mlr.press/v267/astorga25a.html) metadata were verified and left unchanged. Their data-synthesis and formulation-search tasks are distinguished from supplied-slot assignment. No empirical cross-paper comparison is introduced.

## C04 — Evidence navigation and provenance

The absent local register path is replaced by the [immutable Phase 1 register](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650/docs/jdsa_revision_phase1_20261029/EXPERIMENT_EVIDENCE_REGISTER.md), [frozen released claims](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/claims.json), and [baseline manuscript crosswalk](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc/docs/jdsa_manuscript_v1_20261029/CLAIM_LOCATION_CROSSWALK_V1.md).

Those records retain historical wording. This note and the new location overlay qualify E35 and the ranked-retrieval heading without overwriting `claims.json`, historical reports or releases. The new overlay records current source locations and compiled destinations without promoting any evidence classification.

Local uncommitted G1/G2/G3, D1 review and relationship-protocol reports retain their original status. Worktree base `7324dfdacef5e102d57a2e64288e824fb267a44e` does not authenticate their uncommitted content. This correction relies on the identified committed artifacts for oracle/metric claims and does not publish private reports or raw provider material.

## Review and limitations

The exact sequential before/after replacements are in [EXACT_SOURCE_CHANGES.json](EXACT_SOURCE_CHANGES.json), with readable excerpts in [BEFORE_AFTER.md](BEFORE_AFTER.md). Validation is documented in [VALIDATION_REPORT.md](VALIDATION_REPORT.md) and its machine-readable companion. Existing scientific limitations remain: missing original headline scalar maps, reference assistance, exposed populations, unavailable historical outcomes and absent independent semantic references.

Scientifically consistent within those stated boundaries. Further author review is required before submission or any new manuscript release. No new experiment, provider call, training, solver execution, Slurm submission, evaluator change, data regeneration, merge or release replacement was performed.
