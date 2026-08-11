# PaMOP Pilot Benchmark

**Status:** COMPLETED LOCALLY.

**Label:** PILOT / SMALL-SLICE RESULTS.

The earlier Slurm-only launch attempt evaluated 0 problems because the pilot
runner was incorrectly required to use `sbatch` in an environment where Slurm
is not installed. That 0/0 attempt is superseded and is not a scientific
result. The actual pilot was subsequently run locally on `al-khwarizmi` in a
detached `tmux` checkpoint run.

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

## Execution Setup

- Execution mode: local `tmux` checkpoint run.
- Provider: Azure OpenAI.
- Deployment: `gpt-4.1-mini`.
- Underlying model observed: `gpt-4.1-mini-2025-04-14`.
- Temperature: `0.2`.
- Maximum correction iterations: `5`.
- Execution: AMPL -> Gurobi through `/home/soroush/.venvs/gurobi/bin/python`.
- Cost: unavailable; Azure account/region billing price was not established.

No raw gated NLP4LP text, API keys, HF tokens, or AMPL/Gurobi license material
are committed.

## Execution Results

| Metric | Count / value |
|---|---:|
| Selected problems | 18 |
| Evaluated problems | 18 |
| Initial execution successes | 0 / 18 (0.0%) |
| Final execution successes | 6 / 18 (33.3%) |
| Success without correction | 0 |
| Success after correction | 6 |
| Correction rescue count | 6 |
| Correction harm count | 0 |
| Mean correction iterations, all correction-invoked cases | 4.39 |
| Mean correction iterations, final-success cases | 3.17 |
| Solver-feasible models | 6 / 18 (33.3%) |
| Objective produced | 6 / 18 (33.3%) |

Primary terminal categories:

| Category | Count |
|---|---:|
| `B. SUCCESS_AFTER_CORRECTION` | 6 |
| `D. AMPL_RENDER_FAILURE` | 7 |
| `E. AMPL_PARSE_FAILURE` | 4 |
| `H. SOLVER_RUNTIME_ERROR` | 1 |

No case succeeded initially. Correction improved execution from 0/18 to 6/18,
but most cases still exhausted the correction loop.

## Semantic Correctness

Executability is not counted as correctness. The pilot records separate
generation, AMPL parse, solver, feasibility, objective, and semantic/gold
comparison fields.

Gold executable code was available for the slice, but full model-structure
equivalence is not established. The reliable semantic denominator in this
pilot is the subset where both the generated model produced an objective and
the gold run produced a comparable objective:

| Metric | Count / value |
|---|---:|
| Semantically evaluable by objective comparison | 5 |
| Objective matches gold | 0 / 5 (0.0%) |
| Feasible but gold-objective mismatched | 5 |

One additional solver-feasible generated model did not have a comparable gold
objective in the available gold output, so it is recorded as feasible but not
semantically evaluated.

No PaMOP "accuracy" metric is reported or inferred.

## Token And Latency Cost

| Metric | Value |
|---|---:|
| Total LLM calls | 304 |
| Prompt tokens | 202,377 |
| Completion tokens | 58,397 |
| Total tokens | 260,774 |
| Mean tokens/problem | 14,487.44 |
| Total wall-clock latency recorded across problems | 974.48 s |
| Mean latency/problem | 54.14 s |
| Min latency/problem | 19.09 s |
| Max latency/problem | 87.27 s |
| Correction tokens | 212,697 |
| Correction latency | 748.18 s |

## Our-Method Comparison

`results/pamop/pilot/comparison_with_ours.csv` contains operational
comparison rows for the same selected ids. Our deterministic method is
recorded as:

- external LLM calls: 0
- LLM tokens: 0
- API cost: 0
- deterministic inference: true

PaMOP reproduction consumed 304 LLM calls and 260,774 tokens for this 18-case
pilot. No common PaMOP accuracy metric is claimed. The comparison is
operational only unless a genuinely common structural/grounding metric is
established.

## Failure Analysis

The dominant failure mode is not missing infrastructure. It is model/output
instability after correction:

- malformed or statically invalid AMPL after remodeling (`D`);
- AMPL parse failures after correction exhaustion (`E`);
- one solver/runtime failure (`H`);
- feasible generated models whose objectives disagree with gold where
  objective comparison is meaningful.

The correction loop provides real executability value, rescuing 6 cases, but
it is expensive and does not establish semantic correctness.

## Recommendation

Decision gate: **B. FIX SYSTEMATIC ISSUE FIRST**.

Exact next step: inspect the failed and feasible-but-wrong generated models
using safe, non-gated paraphrases/metadata and improve the systematic AMPL
rendering/modeling/correction failure modes before any larger run.
