# `max_weight_matching` mechanism + error analysis

Generator: `scripts/analysis/mwm_full_analysis.py` (reuses
`tools/nlp4lp_downstream_utility.py`'s own public functions; no changes to
that file). Full interpretation:
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` §5.

- `per_query_transition.csv` — per-query `schema_hit`, `n_expected_scalar`,
  typed-greedy-ready flag, `max_weight_matching`-ready flag, and
  `max_weight_matching`'s own coverage/type-match, for all 331 `orig`
  queries, both recomputed fresh from canonical inputs.
- `mechanism_and_error_analysis_summary.json` — the typed-greedy vs.
  `max_weight_matching` transition matrix (both ready / only one ready /
  neither) and a slot-level residual-failure taxonomy for
  `max_weight_matching` (same-type ambiguity, total/per-unit confusion,
  objective/constraint confusion, min/max polarity, percent ambiguity,
  missing mentions, schema-retrieval miss, zero-expected-scalar).

**Headline:** `max_weight_matching` does not "solve" typed greedy's
bottleneck taxonomy — its own residual failures are dominated by the same
same-type/total-vs-coefficient confusions the local pairwise score already
had, and typed greedy rescues more than twice as many queries in the
transition matrix (18) as `max_weight_matching` rescues in the other
direction (7).
