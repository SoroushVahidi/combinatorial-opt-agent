# External Baseline Comparison

**Status: `PRELIMINARY_EXTERNAL_BASELINE_STATUS`** — not a final paper comparison. Generated 2026-08-15T05:58:06.373364+00:00, repository HEAD `69df7c030fabb3bd785684b68a844929487603b9`.

## Evaluation protocol

See `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` for the frozen protocol this report follows (metric definitions, run-selection rules, proxy semantics, no-cherry-picking policy).

## Common benchmark manifest

- pilot_ids (n=6): `[14, 23, 34, 59, 69, 72]`
- future_evaluation_ids (n=18): `[14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262]`
- source_subset: `pamop_possible_269`

**Known divergence:** PaMOP's original gpt-5.4 diagnostic executed 6 IDs ([14, 23, 34, 72, 84, 88]), which were NOT the shared pilot_ids convention [14, 23, 34, 59, 69, 72]. That divergence is RESOLVED at the 18-instance level: the 2026-08-15 scaled extension executed the remaining 12 IDs, so PaMOP's empirical evidence now covers the full `future_evaluation_ids` set [14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262]. Any 6-instance pilot-vs-pilot comparison must still state which 6 it means.

## System availability

| System | Status | Has empirical NLP4LP rows | Classification |
|---|---|---|---|
| ours | VALIDATED (whole-benchmark); common-18-subset status tracked separately | True | `PAPER_CORE_VALIDATED` |
| pamop | COMMON-18 COMPLETE (gpt-5.4, AMPL/HiGHS execution) | True | `PAMOP_COMMON18_COMPLETE` |
| orlm | COMMON-18 COMPLETE (official checkpoint; execution blocked on coptpy) | True | `ORLM_COMMON18_COMPLETE_EXECUTION_BLOCKED` |
| optmath | COMMON-18 COMPLETE (official checkpoint; solver execution pending) | True | `OPTMATH_COMMON18_INFERENCE_COMPLETE` |
| generic | COMMON-18 COMPLETE (gpt-5.4, zero-shot gurobipy; no solver execution) | True | `GENERIC_LLM_COMMON18_COMPLETE_EXECUTION_NOT_ATTEMPTED` |
| deepor | PAPER RECONSTRUCTION, OFFICIAL CHECKPOINT UNAVAILABLE | False | `DEEPOR_PAPER_RECONSTRUCTION_READY` |
| orr1 | OFFICIAL CODE INTEGRATED, CHECKPOINT UNAVAILABLE | False | `ORR1_CODE_INTEGRATED_CHECKPOINT_BLOCKED` |

## Implementation fidelity

| System | Rows | Fidelity levels seen |
|---|---|---|
| ours | 18 | NATIVE_METHOD |
| pamop | 18 | INDEPENDENT_RECONSTRUCTION |
| orlm | 18 | ADAPTED_OFFICIAL |
| optmath | 18 | ADAPTED_OFFICIAL |
| generic | 18 | INDEPENDENT_RECONSTRUCTION |
| deepor | 0 | PENDING |
| orr1 | 0 | PENDING |

## Native metrics

Native metrics are NOT comparable numerically across systems -- see `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` §metric taxonomy. Full values are in `native_metrics.csv`.

## Shared end-to-end metrics

| Metric | System | n | Rate/state |
|---|---|---|---|
| parse_success_rate | pamop | 18 | 0.7222 |
| parse_success_rate | orlm | 18 | 1.0000 |
| parse_success_rate | optmath | 18 | 1.0000 |
| parse_success_rate | generic | 18 | 1.0000 |
| parse_success_rate | deepor | 0 | PENDING |
| parse_success_rate | orr1 | 0 | PENDING |
| executable_rate | pamop | 18 | 0.7222 |
| executable_rate | orlm | 0 | NOT_APPLICABLE |
| executable_rate | optmath | 0 | NOT_APPLICABLE |
| executable_rate | generic | 0 | NOT_APPLICABLE |
| executable_rate | deepor | 0 | PENDING |
| executable_rate | orr1 | 0 | PENDING |
| feasible_rate | pamop | 18 | 0.7222 |
| feasible_rate | orlm | 0 | NOT_APPLICABLE |
| feasible_rate | optmath | 0 | NOT_APPLICABLE |
| feasible_rate | generic | 0 | NOT_APPLICABLE |
| feasible_rate | deepor | 0 | PENDING |
| feasible_rate | orr1 | 0 | PENDING |
| objective_agreement_rate | pamop | 11 | 0.7273 |
| objective_agreement_rate | orlm | 0 | NOT_APPLICABLE |
| objective_agreement_rate | optmath | 0 | NOT_APPLICABLE |
| objective_agreement_rate | generic | 0 | NOT_APPLICABLE |
| objective_agreement_rate | deepor | 0 | PENDING |
| objective_agreement_rate | orr1 | 0 | PENDING |

## Paired comparison results

| Metric | A | B | n paired | both | A only | B only | neither | McNemar p |
|---|---|---|---|---|---|---|---|---|
| objective_agreement | pamop | orlm | 0 | 0 | 0 | 0 | 0 | None |
| objective_agreement | pamop | optmath | 0 | 0 | 0 | 0 | 0 | None |
| objective_agreement | pamop | generic | 0 | 0 | 0 | 0 | 0 | None |
| objective_agreement | orlm | optmath | 0 | 0 | 0 | 0 | 0 | None |
| objective_agreement | orlm | generic | 0 | 0 | 0 | 0 | 0 | None |
| objective_agreement | optmath | generic | 0 | 0 | 0 | 0 | 0 | None |

## Resource requirements

| System | Compute | Solver | Test-time learning | Training required for faithful result |
|---|---|---|---|---|
| ours | CPU-only | N/A (no solver call) | False | False |
| pamop | CPU-only (API calls) + local AMPL/HiGHS | AMPL + HiGHS (this repo's config) | False | False |
| orlm | 1x GPU, >=24GB VRAM | coptpy (COPT) | False | False |
| optmath | 1x GPU, >=24GB VRAM | gurobipy (Gurobi) | False | False |
| generic | None local (external Azure OpenAI API) | gurobipy (Gurobi) | False | False |
| deepor | Unknown (official code unavailable; paper implies multi-GPU training) | Pyomo (paper case study) | True | True |
| orr1 | 1x GPU >=24GB for inference; multi-GPU DeepSpeed ZeRO-3 for any training | coptpy (COPT) | True | True |

## Failure analysis

See `failure_summary.csv` for the full per-row native-category -> top-level-bucket mapping.

## Important fairness caveats

- `ours` performs fixed-catalog scalar grounding, not full NL-to-model generation; it is excluded from every SharedMetric above (see `END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY` in `metrics.py`).
- PaMOP's objective-value agreement is an exact-match proxy on execution-successful rows only, never a structural/semantic correctness judgment.
- ORLM has 18 real common-18 rows but **no solver execution** (coptpy not installed); its executable/feasible/objective-agreement cells are NOT_APPLICABLE, never zero. DeepOR/OR-R1 currently have **zero** empirical rows; any non-zero number for them in this report would be fabricated and must be treated as a bug.
- `generic` (gpt-5.4, zero-shot gurobipy) has 18 real common-18 rows with valid generation/parse/static-validation but **no solver execution**; it is a general-purpose LLM floor, not an optimization-trained baseline, and its executable/objective cells are NOT_APPLICABLE.

## OR-R1 transductive-protocol note

The official OR-R1 TGRPO training set is the union of all official evaluation test sets, including all 242 official NLP4LP rows (verified by direct file inspection, `docs/ORR1_PROVENANCE.md`). Any future OR-R1 empirical row in this report MUST carry `transductive_training=True` and must not be compared to an inductively-evaluated system without this caveat restated in the same table.

## Missing experiments / blockers

- **deepor**: No official code, checkpoint, or requirements file located anywhere. Lightweight reconstruction is mock-tested only; zero empirical result rows exist and none can exist without a released artifact.
- **orr1**: Official code verified (cited directly by the arXiv paper). No SFT/TGRPO/merged checkpoint released anywhere. Faithful reproduction additionally requires training TGRPO transductively over the evaluation set itself. Zero empirical result rows exist.

## Provenance

- Repository HEAD at generation time: `69df7c030fabb3bd785684b68a844929487603b9`
- Generator: `baselines/comparison/report.py` via `python -m baselines.comparison.cli`
- Protocol: `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md`

