# Cross-baseline comparison harness

**Status: `CROSS_BASELINE_HARNESS_COMPLETE` (infrastructure), report status
`PRELIMINARY_EXTERNAL_BASELINE_STATUS` (data), 2026-08-13.** This package is
an ANALYSIS VIEW that sits above each baseline's own native
`result_schema.py`/`evaluator.py` (`baselines/pamop/`, `baselines/orlm/`,
`baselines/optmath/`, `baselines/deepor/`, `baselines/orr1/`) and this
repository's own NLP4LP downstream benchmark. It never replaces any of
those, and it never invents a number: a system with zero real result rows
appears only in the availability/resource tables, never with a fabricated
metric value.

## Why this exists

The six systems compared here (`ours`, PaMOP, ORLM, OptMATH, DeepOR, OR-R1)
do not naturally expose the same metrics — one is deterministic fixed-catalog
scalar grounding, the others are five different flavors of full NL-to-model
generation with different solvers, training regimes, and rollout counts. A
single blended "score" across them would misrepresent the science. This
package instead distinguishes four kinds of information (frozen in
`docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md`):

- **NATIVE metrics** (`metrics.py::NATIVE_METRICS`) — meaningful only within
  one system/family (e.g. `InstantiationReady` and
  `StrictInstantiationReady` for `ours`, `pass@8` for OR-R1). For `ours`,
  `InstantiationReady` is the predicted-schema proxy
  `Coverage >= 0.8 AND TypeMatch >= 0.8`; `StrictInstantiationReady` adds
  `schema_hit` and is therefore the native schema-gated readiness diagnostic.
- **SHARED metrics** (`metrics.py::SHARED_METRICS`) — verified genuinely
  computable identically across multiple systems (`parse_success_rate`,
  `executable_rate`, `feasible_rate`, `objective_agreement_rate`). `ours` is
  deliberately excluded from all of these — see
  `metrics.py::END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY`.
- **RESOURCE/COST metrics** (`resource_profile.py`) — runtime, model size,
  solver/GPU requirements, rollout count, test-time-learning/training
  requirements. Never used to rank systems.
- **AVAILABILITY/FIDELITY metadata** (`availability.py`,
  `schema.py::UnifiedRow.implementation_fidelity`) — whether a result comes
  from an official implementation/checkpoint, an adapted official
  implementation, an independent reconstruction, a paper-level
  reconstruction, or mock-only evidence (which is always excluded from
  reports).

## Layout

- `schema.py` — `UnifiedRow`, the shared analysis-view record, and
  `CellState` (`PENDING`/`NOT_APPLICABLE`/`UNAVAILABLE`/`MOCK_ONLY`/`PROXY`/
  `UNSUPPORTED`/`UNKNOWN`) so no cell is ever an ambiguous blank.
- `adapters.py` — `adapt_ours`/`adapt_pamop`/`adapt_orlm`/`adapt_optmath`/
  `adapt_deepor`/`adapt_orr1`: native record -> `UnifiedRow`, discarding
  nothing (native fields land in `native_record`/`native_metrics`).
- `metrics.py` — the native/shared metric taxonomy and the
  `END_TO_END_OBJECTIVE_SUCCESS` eligibility notes per system.
- `pairing.py` — pairs two systems' rows on `problem_id` for one boolean
  metric, producing a both/A-only/B-only/neither transition table.
- `statistics.py` — Wilson confidence intervals and an exact (not
  chi-square-approximated) McNemar test; returns `p_value=None` (never a
  fabricated number) when there are zero discordant pairs.
- `resource_profile.py` — static per-system compute/solver/determinism
  facts, for context, never for ranking.
- `failure_taxonomy.py` — maps each system's native failure-category
  strings to one shared top-level bucket without deleting native detail.
- `validation.py` — provenance validation, mock-evidence exclusion
  (`is_mock_evidence`), and duplicate/ambiguous-run rejection (never
  best-of-runs cherry-picking — ambiguous groups are rejected entirely).
- `manifests.py` + `manifests/nlp4lp_common_18.json` — the authoritative
  common manifest, drift-checked against the four baseline packages' own
  manifests, and the documented (not silently resolved) divergence between
  PaMOP's actually-executed 6-ID pilot and the other four baselines'
  6-ID `pilot_ids` convention.
- `availability.py` — per-system availability status, kept structurally
  separate from performance metrics.
- `ingest.py` — explicit, fixed-location ingestion for `ours` (a fresh,
  CPU-only rerun of `tools.nlp4lp_downstream_utility`, filtered to the
  common manifest) and `pamop` (`results/pamop/fidelity_diagnostic_gpt5/`).
  No filesystem crawling. ORLM's six-row pilot is complete, but this harness
  is fixed to the common-18 manifest and has no explicit common-6 ingestion
  mode; ORLM remains empty here until the common-18 output finishes and its
  fixed location is explicitly registered.
- `report.py` + `cli.py` — Markdown/CSV/JSON report generation, run via
  `python -m baselines.comparison.cli`.

## Running it

```bash
python -m baselines.comparison.cli --output-dir results/external_baseline_comparison
python -m baselines.comparison.cli --validate-only --systems pamop orr1
```

Output: `results/external_baseline_comparison/{availability,native_metrics,
shared_metrics,resource_profile,paired_results,failure_summary,
confidence_intervals}.csv`, `comparison.json`, `comparison.md`, `README.md`.
Regenerate rather than hand-edit.

## Frozen protocol

`docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` freezes the common manifest,
metric definitions, run-selection rules, paired-testing rules, proxy
semantics, and no-cherry-picking policy this package implements. Change the
protocol document deliberately, not by editing code first.
