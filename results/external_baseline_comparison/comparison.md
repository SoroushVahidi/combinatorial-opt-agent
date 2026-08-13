# External Baseline Comparison

**Status: `PRELIMINARY_EXTERNAL_BASELINE_STATUS`** — not a final paper comparison. Generated 2026-08-13T21:31:43.933110+00:00, repository HEAD `b2bb4f35b053e24300b07492ffa8de3ee1e4a392`.

## Evaluation protocol

See `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` for the frozen protocol this report follows (metric definitions, run-selection rules, proxy semantics, no-cherry-picking policy).

## Common benchmark manifest

- pilot_ids (n=6): `[14, 23, 34, 59, 69, 72]`
- future_evaluation_ids (n=18): `[14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262]`
- source_subset: `pamop_possible_269`

**Known divergence:** The shared pilot_ids convention [14, 23, 34, 59, 69, 72] (used by the ORLM/OptMATH/DeepOR/OR-R1 lightweight manifests) is NOT identical to the 6 problem IDs PaMOP's gpt-5.4 fidelity diagnostic actually executed ([14, 23, 34, 72, 84, 88]). Overlap: [14, 23, 34, 72] (4 of 6). Both ID sets are subsets of the same 18-instance future_evaluation_ids superset [14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262], so they remain comparable at the 18-instance level, but any 6-instance pilot-vs-pilot comparison must state which 6 it means.

## System availability

| System | Status | Has empirical NLP4LP rows | Classification |
|---|---|---|---|
| ours | VALIDATED (whole-benchmark); common-18-subset status tracked separately | True | `PAPER_CORE_VALIDATED` |
| pamop | PILOT VALIDATED (6-instance gpt-5.4 diagnostic) | True | `PAMOP_PILOT_VALIDATED` |
| orlm | IMPLEMENTED, NOT YET RUN | False | `ORLM_IMPLEMENTED_READY_FOR_INFERENCE` |
| optmath | IMPLEMENTED, NOT YET RUN | False | `OPTMATH_IMPLEMENTED_READY_FOR_INFERENCE` |
| deepor | PAPER RECONSTRUCTION, OFFICIAL CHECKPOINT UNAVAILABLE | False | `DEEPOR_PAPER_RECONSTRUCTION_READY` |
| orr1 | OFFICIAL CODE INTEGRATED, CHECKPOINT UNAVAILABLE | False | `ORR1_CODE_INTEGRATED_CHECKPOINT_BLOCKED` |

## Implementation fidelity

| System | Rows | Fidelity levels seen |
|---|---|---|
| ours | 18 | NATIVE_METHOD |
| pamop | 6 | INDEPENDENT_RECONSTRUCTION |
| orlm | 0 | PENDING |
| optmath | 0 | PENDING |
| deepor | 0 | PENDING |
| orr1 | 0 | PENDING |

## Native metrics

Native metrics are NOT comparable numerically across systems -- see `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` §metric taxonomy. Full values are in `native_metrics.csv`.

## Shared end-to-end metrics

| Metric | System | n | Rate/state |
|---|---|---|---|
| parse_success_rate | pamop | 6 | 0.8333 |
| parse_success_rate | orlm | 0 | PENDING |
| parse_success_rate | optmath | 0 | PENDING |
| parse_success_rate | deepor | 0 | PENDING |
| parse_success_rate | orr1 | 0 | PENDING |
| executable_rate | pamop | 6 | 0.8333 |
| executable_rate | orlm | 0 | PENDING |
| executable_rate | optmath | 0 | PENDING |
| executable_rate | deepor | 0 | PENDING |
| executable_rate | orr1 | 0 | PENDING |
| feasible_rate | pamop | 6 | 0.8333 |
| feasible_rate | orlm | 0 | PENDING |
| feasible_rate | optmath | 0 | PENDING |
| feasible_rate | deepor | 0 | PENDING |
| feasible_rate | orr1 | 0 | PENDING |
| objective_agreement_rate | pamop | 5 | 0.8000 |
| objective_agreement_rate | orlm | 0 | PENDING |
| objective_agreement_rate | optmath | 0 | PENDING |
| objective_agreement_rate | deepor | 0 | PENDING |
| objective_agreement_rate | orr1 | 0 | PENDING |

## Paired comparison results

NOT_APPLICABLE — no two systems currently share empirical rows on the same problem_ids.


## Resource requirements

| System | Compute | Solver | Test-time learning | Training required for faithful result |
|---|---|---|---|---|
| ours | CPU-only | N/A (no solver call) | False | False |
| pamop | CPU-only (API calls) + local AMPL/HiGHS | AMPL + HiGHS (this repo's config) | False | False |
| orlm | 1x GPU, >=24GB VRAM | coptpy (COPT) | False | False |
| optmath | 1x GPU, >=24GB VRAM | gurobipy (Gurobi) | False | False |
| deepor | Unknown (official code unavailable; paper implies multi-GPU training) | Pyomo (paper case study) | True | True |
| orr1 | 1x GPU >=24GB for inference; multi-GPU DeepSpeed ZeRO-3 for any training | coptpy (COPT) | True | True |

## Failure analysis

See `failure_summary.csv` for the full per-row native-category -> top-level-bucket mapping.

## Important fairness caveats

- `ours` performs fixed-catalog scalar grounding, not full NL-to-model generation; it is excluded from every SharedMetric above (see `END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY` in `metrics.py`).
- PaMOP's objective-value agreement is an exact-match proxy on execution-successful rows only, never a structural/semantic correctness judgment.
- ORLM/OptMATH/DeepOR/OR-R1 currently have **zero** empirical NLP4LP rows; any non-zero number for them in this report would be fabricated and must be treated as a bug.

## OR-R1 transductive-protocol note

The official OR-R1 TGRPO training set is the union of all official evaluation test sets, including all 242 official NLP4LP rows (verified by direct file inspection, `docs/ORR1_PROVENANCE.md`). Any future OR-R1 empirical row in this report MUST carry `transductive_training=True` and must not be compared to an inductively-evaluated system without this caveat restated in the same table.

## Missing experiments / blockers

- **orlm**: Official code (Apache-2.0) and one public checkpoint confirmed. No GPU inference has been run against NLP4LP in this repository; zero result rows exist.
- **optmath**: Official code and checkpoint confirmed public. No GPU inference has been run; zero result rows exist.
- **deepor**: No official code, checkpoint, or requirements file located anywhere. Lightweight reconstruction is mock-tested only; zero empirical result rows exist and none can exist without a released artifact.
- **orr1**: Official code verified (cited directly by the arXiv paper). No SFT/TGRPO/merged checkpoint released anywhere. Faithful reproduction additionally requires training TGRPO transductively over the evaluation set itself. Zero empirical result rows exist.

## Provenance

- Repository HEAD at generation time: `b2bb4f35b053e24300b07492ffa8de3ee1e4a392`
- Generator: `baselines/comparison/report.py` via `python -m baselines.comparison.cli`
- Protocol: `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md`

