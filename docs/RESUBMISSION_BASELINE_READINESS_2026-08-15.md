# External-Baseline Readiness for Manuscript Resubmission — 2026-08-15

**Status snapshot:** this is the empirical-evidence readiness matrix for the
external-baseline section of the resubmission. It is generated from real
result files on disk (fixed locations, see `baselines/comparison/ingest.py`),
not from promises. Every cell below is either a measured value or an explicit
blocker; no cell is fabricated.

Frozen method (unchanged): TF-IDF top-1 retrieval + typed greedy grounding +
deterministic multiplicative-expression extraction. Whole-benchmark native
numbers: Schema R@1 = 301/331, InstantiationReady = 265/331,
StrictInstantiationReady = 255/331.

## Common-18 benchmark manifest

All external systems are evaluated on the same 18 NLP4LP instances:

`[14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262]`

Selection is outcome-independent (deterministic stratified bucket pass over
the pamop_possible_269 subset, stable SHA-256 tie-breaking). See
`baselines/comparison/manifests/nlp4lp_common_18.json`.

## Readiness matrix

| System | Empirical common-18 rows | Generation / parse / static | Solver execution | Objective-value proxy | Status | Blocker |
|---|---|---|---|---|---|---|
| **ours** (frozen) | 18/18 | Schema R@1 17/18; InstantiationReady 16/18; Strict 16/18 | N/A (scalar grounding, no solver) | N/A (ineligible by protocol) | `PAPER_CORE_VALIDATED` | none |
| **PaMOP** (reconstruction, gpt-5.4) | 18/18 | 13/18 execution success; 5 AMPL parse failures | AMPL + HiGHS (local) | 8/11 evaluable success (proxy) | `PAMOP_COMMON18_COMPLETE` | none |
| **ORLM** (official checkpoint) | 18/18 | 18/18 generation; 18/18 parse; 18/18 static | BLOCKED (coptpy missing) | NOT_EVALUABLE | `ORLM_COMMON18_COMPLETE_EXECUTION_BLOCKED` | coptpy not installed |
| **OptMATH** (official checkpoint) | 18/18 | verified complete | PENDING (gurobipy available in dedicated venv) | PENDING | `OPTMATH_COMMON18_INFERENCE_COMPLETE` | solver execution not yet run |
| **generic** (gpt-5.4, zero-shot gurobipy) | 18/18 | 18/18 generation; 18/18 parse; 18/18 static | NOT_ATTEMPTED | NOT_APPLICABLE | `GENERIC_LLM_COMMON18_COMPLETE_EXECUTION_NOT_ATTEMPTED` | execution intentionally deferred (floor bound) |
| **DeepOR** | 0 | — | — | — | `DEEPOR_PAPER_RECONSTRUCTION_READY` | no official code/checkpoint (rechecked 2026-08-15) |
| **OR-R1** | 0 | — | — | — | `ORR1_CODE_INTEGRATED_CHECKPOINT_BLOCKED` | no SFT/TGRPO/merged checkpoint (rechecked 2026-08-15) |

## What this means for the resubmission

1. **Three full-LLM baselines now have real common-18 evidence**: ORLM
   (official checkpoint, execution blocked on coptpy), OptMATH (official
   checkpoint, inference complete, execution pending), and PaMOP
   (reconstruction, fully executed with AMPL + HiGHS). This is the empirical
   backbone the manuscript revision needs.
2. **A generic-purpose LLM floor** (gpt-5.4, zero-shot gurobipy, 18/18
   parse/static valid) bounds how much of the task is attributable to a plain
   API model — it is deliberately NOT optimization-trained and is never
   compared as if it were.
3. **Shared end-to-end metrics** (parse/executable/feasible/objective
   agreement) are computed only where solver execution exists. `ours` is
   excluded by protocol (scalar grounding, not NL-to-model generation); ORLM
   and `generic` report NOT_APPLICABLE objective cells rather than zero.
4. **DeepOR and OR-R1 cannot contribute empirical rows** without released
   artifacts. OR-R1 additionally has the mandatory transductive-protocol
   caveat (its official training data is the union of the evaluation test
   sets, including NLP4LP). Both remain documented blockers.

## Evidence locations

- `results/orlm/common18_official_checkpoint/results.jsonl` (+ run_metadata.json, inference.log)
- `results/optmath/common18_official_checkpoint/results.jsonl` (+ run_metadata.json, inference.log); pilot at `results/optmath/pilot_official_checkpoint/`
- `results/generic_llm/common18_official/results.jsonl` (+ run_metadata.json, inference.log); pilot at `results/generic_llm/pilot_official/`
- `results/pamop/fidelity_diagnostic_gpt5/per_problem.csv` (18 rows, + summary.json, run_metadata.json, selected_ids.json, incremental/)
- `results/external_baseline_comparison/` — regenerated comparison report
- Provenance: `docs/DEEPOR_PROVENANCE.md`, `docs/ORR1_PROVENANCE.md`, `docs/OPTMATH_PROVENANCE.md`, `docs/ORLM_PROVENANCE.md`

## Remaining work before the report is manuscript-final

- Regenerate `results/external_baseline_comparison/` with all evidence above
  (done once OptMATH inference validated).
- Optional: run gurobipy solver execution for OptMATH and generic rows using
  `/home/soroush/.venvs/gurobi/bin/python` to promote objective cells from
  NOT_APPLICABLE to measured; ORLM would need coptpy installed for the same.
- Final resource/fairness table is in `baselines/comparison/resource_profile.py`
  and rendered into the comparison report.