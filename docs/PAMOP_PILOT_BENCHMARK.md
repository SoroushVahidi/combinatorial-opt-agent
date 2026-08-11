# PaMOP Pilot Benchmark

**Status:** BLOCKED BY ENVIRONMENT, not executed.

**Label:** PILOT / SMALL-SLICE RESULTS.

## Slice Construction

The pilot slice was selected from `pamop_possible_269`, excluding ids `28`,
`51`, `57`, `123`, `126`, and `135` because their structured NLP4LP data are
missing.

Selected ids:

```text
14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262
```

The selection is deterministic (`pamop-pilot-v1`) and uses only non-gated
metadata in the committed artifact: LP/MILP, objective sense, variable count,
parameter count, constraint count, numeric-mention count, partition node
count/depth, gold-code availability, and stable SHA-256 tie-breaking. No
execution outcomes were used for selection.

The slice includes LP and MILP, maximize and minimize problems, simple and
multi-constraint cases, low and high numeric-mention buckets, and both
single-node and multi-node partition trees. The exact metadata are in
`results/pamop/pilot/selected_ids.json`.

## Failure Categories

The primary categories were fixed before execution:

`A. SUCCESS_NO_CORRECTION`, `B. SUCCESS_AFTER_CORRECTION`,
`C. MODEL_PARSE_FAILURE`, `D. AMPL_RENDER_FAILURE`,
`E. AMPL_PARSE_FAILURE`, `F. SOLVER_INFEASIBLE`,
`G. SOLVER_UNBOUNDED`, `H. SOLVER_RUNTIME_ERROR`,
`I. CORRECTION_EXHAUSTED`, `J. DATA_FAILURE`,
`K. ENVIRONMENT_FAILURE`, `L. OTHER_MODEL_FAILURE`.

Infrastructure failures are not counted as model failures.

## Execution Status

The benchmark was not run because this environment does not provide Slurm:

```text
sbatch batch/pamop/run_pamop_pilot.sbatch
sbatch: command not found
```

The runner also refused local execution by default, as required by the pilot
protocol. Therefore:

- selected problems: 18
- actually evaluated: 0
- Slurm job id: unavailable
- run failure category: `K. ENVIRONMENT_FAILURE`

## Current Results

No PaMOP problem reached Azure OpenAI, AMPL, Gurobi, correction, or gold-code
comparison in this blocked run. All execution, correction, token, latency,
objective, feasibility, and semantic-correctness counts are therefore zero
or not evaluable.

Cost is unavailable because no Azure calls were made. Even for a completed
run, exact Azure cost should remain unavailable unless the deployment's
billing region/account pricing can be established safely.

## Semantic Correctness

No semantic correctness assessment was performed because no generated AMPL
model exists for this pilot attempt.

The runner is set up to record separate fields for generation validity, AMPL
executability, solver success, feasibility, objective production, and partial
gold comparison. It does not convert solver feasibility into PaMOP accuracy.

For this slice, available NLP4LP gold artifacts include `optimus-code.py`,
`parameters.json`, and `solution.json` where accessible. A completed Slurm
run should execute gold code separately and compare solver success,
feasibility, and objective value where meaningful. Objective equality alone
must still be treated as partial evidence, not full semantic equivalence.

## Correction-Loop Effect

Not measured in this environment. The committed runner records:

- initial execution success rate
- final execution success rate
- correction rescue count
- harmed-by-correction count
- correction iterations
- correction token overhead
- correction latency overhead

These fields will populate incrementally once the sbatch job can run.

## Our-Method Comparison

`results/pamop/pilot/comparison_with_ours.csv` contains operational
comparison rows for the same selected ids. Our method is recorded as:

- external LLM calls: 0
- LLM tokens: 0
- API cost: 0
- deterministic: true

No common PaMOP accuracy metric is claimed. The comparison is operational
only unless a genuinely common structural/grounding metric is established.

## Recommendation

Decision gate: **B. FIX SYSTEMATIC ISSUE FIRST**.

Exact next step: run `sbatch batch/pamop/run_pamop_pilot.sbatch` from a
Slurm login node where `sbatch` is available and the required Azure, HF,
AMPL, and Gurobi environment variables/licenses are configured.
