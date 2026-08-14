# External baseline comparison protocol

Frozen 2026-08-13. This document is the source of truth for
`baselines/comparison/`; change it deliberately, not by editing the
generator code first and letting the doc drift. It governs comparisons
across `ours` (`tfidf_typed_greedy`), PaMOP, ORLM, OptMATH, DeepOR, and OR-R1.

## Common manifest

Authoritative file: `baselines/comparison/manifests/nlp4lp_common_18.json`.

- `pilot_ids` (6): `[14, 23, 34, 59, 69, 72]` — used by the ORLM, OptMATH,
  DeepOR, and OR-R1 lightweight baseline manifests (verified byte-identical
  across all four, 2026-08-13).
- `future_evaluation_ids` (18): `[14, 23, 34, 59, 69, 72, 84, 88, 96, 117,
  190, 202, 208, 219, 232, 237, 254, 262]`.
- Selection is the deterministic `pamop-pilot-v1` stratified bucket pass
  over the `pamop_possible_269` subset, fixed before any external-baseline
  outcome existed. Do not cherry-pick or silently change these IDs.

**Known, documented divergence:** PaMOP's actually-executed empirical pilot
(`results/pamop/fidelity_diagnostic_gpt5/`) covers IDs `[14, 23, 34, 72, 84,
88]`, not `pilot_ids`. Overlap is 4 of 6 (`14, 23, 34, 72`). Both sets are
subsets of `future_evaluation_ids`, so an 18-instance comparison remains
valid; any 6-instance pilot-vs-pilot comparison must state which 6 IDs it
means and must never silently treat the two 6-ID sets as the same manifest.
See `baselines/comparison/manifests.py::pamop_empirical_manifest_note`.

## Metric definitions

### Native metrics
System-specific; never compared numerically across systems. Full list and
sources: `baselines/comparison/metrics.py::NATIVE_METRICS`.

### Shared end-to-end metrics
Computed identically from a `UnifiedRow` across the systems listed in each
metric's `applicable_systems`:

- `parse_success_rate` — generated output yields a parseable code/model
  artifact.
- `executable_rate` — the artifact executes against its target solver
  without error.
- `feasible_rate` — execution reports a feasible solution.
- `objective_agreement_rate` — predicted objective within a predeclared
  tolerance of gold. **This is a PROXY, never "semantic correctness".**

Applicable to: `pamop`, `orlm`, `optmath`, `deepor`, `orr1`. **`ours` is
excluded from every shared metric** — it performs fixed-catalog scalar
grounding, not full NL-to-model generation, so "generated code", "execution",
"feasible", and "objective" do not apply to it in the same sense. Its closest
analogue, `InstantiationReady`, is a distinct claim (full, correctly-typed
scalar instantiation of the one correct catalog entry) and must never share
a column with objective-value agreement.

### END_TO_END_OBJECTIVE_SUCCESS eligibility
A system is eligible for this composite claim (executable + solved +
objective agrees within tolerance) only per
`baselines/comparison/metrics.py::END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY`.
Do not add a system to a shared metric's `applicable_systems` without
updating that eligibility table and its justification in the same commit.

## Cell-state semantics

Every metric-shaped field is either a real measurement or exactly one of:
`PENDING` (implemented, not yet run), `NOT_APPLICABLE` (concept doesn't
apply), `UNAVAILABLE` (blocked by a missing artifact), `MOCK_ONLY` (never a
comparison row), `PROXY` (a value exists but is explicitly a proxy),
`UNSUPPORTED` (input excluded), `UNKNOWN` (genuinely undetermined). A report
must never render one of these as a blank cell or as `0`/`0%`.

## Run-selection rules (no cherry-picking)

- One `(system, problem_id)` pair must resolve to exactly one accepted row.
- If more than one distinct run identity (`method_variant` +
  `checkpoint_revision`) exists for the same `(system, problem_id)`, the
  entire group is **rejected**, not resolved by picking the best outcome
  (`validation.py::detect_ambiguous_runs`, `select_rows`). Explicit run
  selection (by run ID, config hash, model revision, or timestamp) is
  required before such a group can be re-included.
- Mock-only evidence (`validation.py::is_mock_evidence`) is excluded from
  every report by default. `--include-mock` does not exist in the CLI as of
  this writing — mock rows must never reach a canonical comparison report.

## Paired-testing rules

- Pairing (`pairing.py::pair_systems`) only pairs `problem_id`s where both
  systems have a MEASURED boolean value for the chosen metric; everything
  else is reported as unpaired/unmeasured, never guessed.
- McNemar's exact test (`statistics.py::mcnemar_exact`) returns
  `p_value=None` with an explicit note when there are zero discordant
  pairs — this is "undefined", never reported as "not significant".
- Treat any result with fewer than 10 discordant pairs as a small-sample
  indication, not a definitive finding (`mcnemar_exact`'s own note enforces
  this in the generated report).
- Do not run significance tests across systems using different metrics,
  different problem sets, or unpaired outcomes.

## Proxy semantics

- `objective_agreement_rate` / `objective_value_proxy` style fields are
  always tolerance-based numeric agreement, never a claim about whether the
  generated model is structurally/semantically the correct formulation.
- PaMOP's own `semantic_correctness_status` field is mapped to
  `UnifiedRow.semantic_correct = CellState.PROXY` (not `True`/`False`) in
  `adapters.py::adapt_pamop`, specifically because it is an exact-objective
  -match proxy, not a structural judgment.
- No system in this repository currently has a genuine, solver-verified
  structural semantic-equivalence metric (`semantic_metric_available=False`
  everywhere as of 2026-08-13).

## OR-R1 transductive-protocol caveat (mandatory)

The official OR-R1 TGRPO training set is the union of all nine official
evaluation test sets, including all 242 official NLP4LP rows (verified by
direct file inspection — see `docs/ORR1_PROVENANCE.md`). Any OR-R1 row in
this comparison harness **must** carry `transductive_training=True`
(`adapters.py::adapt_orr1` sets this unconditionally) and any report
including such a row must restate this caveat in the same table
(`report.py::_render_markdown` does so under "OR-R1 transductive-protocol
note"). Never compare an OR-R1 row silently against an inductively-evaluated
system without this caveat visible in the same output.

## What this protocol does not yet cover

- A completed empirical result for ORLM, OptMATH, DeepOR, or OR-R1. ORLM's
  official-checkpoint six-instance pilot is currently running, but its output
  remains excluded until rows complete and pass provenance validation (see
  `baselines/comparison/availability.py`).
- Any statistical correction for multiple comparisons across metrics/systems.
- A canonical single "leaderboard" ranking — deliberately out of scope; see
  the critical scientific principle in this document's introduction.
