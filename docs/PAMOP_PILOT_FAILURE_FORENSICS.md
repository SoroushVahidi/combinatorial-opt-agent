# PaMOP Pilot Failure Forensics

**Status:** completed on 2026-08-11.

This pass investigates why the first 18-problem PaMOP pilot performed poorly.
It does not modify the manuscript, does not run a larger benchmark, and does
not claim reproduction of PaMOP's published 67-problem numbers.

## Source Pilot

The 18 selected ids remain:

```text
14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262
```

The local pilot produced 0/18 initial execution successes and 6/18 final
execution successes, all after correction. Five solver-feasible models had
comparable gold objectives, and all five disagreed with gold. This made the
pilot a failure-forensics target, not a larger-run trigger.

## Root-Cause Summary

Safe trace inspection found a dominant deterministic reproduction error before
attributing the result to the LLM:

| First bad stage | Count | Finding |
|---|---:|---|
| `G_extr_input_source` | 18 | The runner fed `problem_info.json:parametrized_description` to `G_extr` instead of the original NLP4LP `description.txt`. This removed concrete numeric grounding before extraction. |
| Deterministic AMPL preflight | 7 affected render-failure cases | The lightweight validator did not understand valid AMPL indexed declarations/constraint headers, causing indexed models to be over-flagged. |
| Merge stage | 0 | No deterministic bottom-up merge corruption was isolated from the safe traces. |
| AMPL renderer serialization | 0 | No case showed a valid merged IR that the renderer alone serialized into invalid AMPL. |

Per-case safe root-cause metadata is in
`results/pamop/forensics/root_cause_table.csv`. No gated NLP4LP text is
included.

## Renderer Findings

The 7 terminal `D. AMPL_RENDER_FAILURE` cases were not true renderer
serialization bugs. They were a mixture of:

- under-grounded input reaching `G_extr`;
- free index symbols such as unbound loop variables;
- undeclared index sets or indexed dimensions;
- generated prose/pseudocode-like AMPL fragments;
- validator limitations around valid indexed AMPL syntax.

Fix made: the validator now recognizes `set` declarations, indexed parameter
and variable declarations, indexed `subject to ... {i in SET}:` headers, and
statement-local index scope. It still flags free indices outside their AMPL
statement scope.

## IR Findings

The main IR/data-flow bug was not loss inside the partition tree or merge. It
was the input chosen for `G_extr`: the benchmark runner used the structured
record's parametrized description rather than the original natural-language
description. This systematically deprived extraction/modeling of concrete
numeric values and explains the repeated missing-parameter and gold-objective
mismatch behavior.

Fix made: `baselines.pamop.data.load_problem_text()` now loads
`description.txt` through the same bare-id/suffixed-id Hugging Face path logic
used by structured records, and the pilot runner uses that raw text in memory
for `G_extr`. Raw gated text is not written to artifacts.

## Prompt Findings

The reconstructed prompts are a reproduction choice because PaMOP's exact
prompts are unavailable. Trace inspection found avoidable prompt gaps:

- leaf modeling did not explicitly require statement-local index bindings;
- root modeling did not explicitly allow `set` declarations for indexed AMPL;
- scalar numeric parameter values were not strongly requested as AMPL `:=` or
  `default` values;
- correction remodeling did not strongly forbid Markdown/prose/pseudocode or
  duplicate declarations.

Fix made: the reconstructed prompts were tightened to require valid AMPL-only
output, declared sets for indexed notation, bound local indices, no duplicate
declarations, and preservation of numeric parameter values when present. These
are documented reproduction choices consistent with PaMOP's AMPL-generation
stages, not a new algorithm.

## Merge Findings

No deterministic merge bug was isolated. The visible failures were already
present before or within generated AMPL fragments, and the corrected targeted
run did not require merge-stage changes.

## Correction-Loop Findings

The original pilot correction loop mostly repeated invalid candidates:

| Transition class | Count |
|---|---:|
| bad -> same | 58 |
| bad -> different bad/new errors | 11 |
| bad -> better but still non-executable | 4 |
| bad -> executable | 6 |
| executable -> worse | 0 |

Main pathology: correction often received enough diagnostics to identify a
surface AMPL error but not enough grounded context to repair the underlying
modeling mistake. It also allowed repeated invalid candidate forms such as
free indices and duplicate declarations.

The fix pass did not raise the 5-iteration cap.

## Semantic Error Taxonomy

The scientifically important failure mode is feasible-but-wrong output. In the
original pilot, 5 comparable feasible models disagreed with gold objectives.
In the targeted rerun after deterministic fixes, all 6 cases executed and all
6 were gold-objective evaluable, but only 1 matched gold.

Observed semantic error classes from safe metadata:

- objective value mismatch against trusted gold code;
- likely coefficient/value placement errors;
- missing or invented parameter values;
- wrong resource/capacity relationships;
- variable/domain or indexed-structure drift.

No PaMOP "accuracy" metric is inferred from these partial gold checks.

## Fixes Made

- Load raw NLP4LP `description.txt` for `G_extr` instead of
  `parametrized_description`.
- Add deterministic `description.txt` loader with bare-id/suffixed-id lookup.
- Update AMPL validator for `set` declarations and indexed AMPL scope.
- Tighten reconstructed modeling and correction prompts around valid AMPL,
  index binding, set declaration, duplicate declarations, and numeric values.
- Add synthetic regression tests for raw text loading, indexed AMPL acceptance,
  free-index rejection, and prompt guardrails.

## Targeted Rerun

Diagnostic subset, all from the original 18:

```text
14, 23, 34, 72, 84, 88
```

Selection rationale:

- prior render failures: `23`, `84`;
- prior parse failures: `72`, `88`;
- prior feasible-but-semantically-wrong cases: `14`, `34`.

Run mode: local `tmux` checkpoint run on `al-khwarizmi`, Azure OpenAI
`gpt-4.1-mini`, temperature `0.2`, max correction iterations `5`, AMPL ->
Gurobi.

| Metric | Before | After |
|---|---:|---:|
| Initial execution success | 0 / 6 | 2 / 6 |
| Final execution success | 2 / 6 | 6 / 6 |
| Semantic correctness by comparable gold objective | 0 / 2 | 1 / 6 |
| Mean correction iterations | 4.50 | 0.67 |
| Total LLM tokens | 81,968 | 24,194 |
| Mean LLM tokens/problem | 13,661.33 | 4,032.33 |

After the fixes, executability improved substantially and correction burden
dropped. Semantic correctness remains poor: 5 of 6 executable models were
feasible but gold-objective mismatched.

## Model-Fidelity Risk

Model-fidelity risk for `gpt-4.1-mini-2025-04-14`: **HIGH**.

This is not the first explanation for the original pilot failure; deterministic
implementation bugs were real and have now been fixed. After those fixes,
however, the remaining error signal is mostly semantic modeling mismatch under
a smaller modern GPT-4-family deployment, while PaMOP reports only "GPT-4" and
does not publish the exact model snapshot or prompts.

## Decision

Decision: **C. IMPLEMENTATION SOUND, MODEL/PROMPT IS PRIMARY LIMITATION**.

Do not automatically rerun all 18 yet. The exact next step is a tiny
model-fidelity/prompt-fidelity diagnostic before any larger benchmark:
compare the same six ids under a closer GPT-4-class Azure deployment if one is
available, or otherwise manually audit `G_extr`/`G_mod` structured outputs
against gold structure using safe metadata only.
