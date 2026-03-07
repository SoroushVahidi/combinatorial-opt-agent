# NLP4LP Semantic IR + Repair Assignment — Implementation

## Where the new method plugs in

- **File:** `tools/nlp4lp_downstream_utility.py`
- **Retrieval:** Unchanged. `run_setting(...)` still uses `rank_fn` from `retrieval.baselines.get_baseline(baseline).rank`; assignment mode only affects how extracted numbers are mapped to schema slots after the top-1 schema is chosen.
- **Entry point:** In `run_one(label, mode)` inside `run_setting`, the branch `if assignment_mode == "semantic_ir_repair" and expected_scalar` calls `_run_semantic_ir_repair(query, variant, expected_scalar)` and then fills `filled`, `n_filled`, `type_matches`, and `comparable_errs` from the returned values and mention IRs.
- **CLI:** `--assignment-mode semantic_ir_repair` with `--baseline tfidf` or `--baseline oracle` produces baseline names `tfidf_semantic_ir_repair` and `oracle_semantic_ir_repair`; results are written to the same summary and per-query paths with these baseline names.

## Files and functions changed/added

| Location | Change |
|----------|--------|
| `tools/nlp4lp_downstream_utility.py` | New: `SEMANTIC_ROLE_WORDS`, `OPERATOR_MIN/MAX_PHRASES`, `PERCENT/CURRENCY_MARKER_TOKENS`, `SEMANTIC_IR_WEIGHTS`, `REPAIR_WEIGHTS`, `MentionIR`, `SlotIR`, `_context_to_semantic_tags`, `_detect_operator_tags`, `_detect_unit_tags`, `_extract_enriched_mentions`, `_slot_semantic_expansion`, `_build_slot_irs`, `_score_mention_slot_ir`, `_semantic_ir_global_assignment`, `_validate_slot_assignment`, `_validation_and_repair`, `_run_semantic_ir_repair`. |
| Same file | Extended: `run_one` — new branch for `assignment_mode == "semantic_ir_repair"`; `main()` — `--assignment-mode` choices include `semantic_ir_repair`, `effective_baseline` gets suffix `_semantic_ir_repair` when that mode is selected. |

No other files are modified. Existing methods (typed greedy, untyped, constrained) are unchanged.

## Mention representation (Stage A)

- **Enriched extraction:** `_extract_enriched_mentions(query, variant)` returns a list of `MentionIR`.
- **MentionIR fields:** `mention_id`, `value`, `type_bucket` (percent|int|currency|float|unknown), `raw_surface`, `context_tokens` (±12 tokens), `sentence_tokens`, `semantic_role_tags`, `operator_tags` (min/max), `unit_tags` (percent_marker, currency_marker, integer_like, decimal_like), `polarity_or_bound`, `target_entity_tokens`, `tok` (original `NumTok`).
- **Semantic tags:** Rule-based map `SEMANTIC_ROLE_WORDS` from context words to tags: budget, cost, expense, spend, profit, revenue, return, demand, requirement, need, capacity, limit, available, supply, minimum, lower_bound, maximum, upper_bound, at_least, at_most, ratio, fraction, percentage, rate, share, total, fixed_cost, variable_cost, penalty, resource, time, quantity, item_count.
- **Operator detection:** `_detect_operator_tags` uses `OPERATOR_MIN_PHRASES` / `OPERATOR_MAX_PHRASES` (e.g. at least, minimum, at most, maximum, up to).
- **Unit tags:** `_detect_unit_tags` sets percent_marker, currency_marker, integer_like, decimal_like from token kind and context.

## Slot representation (Stage B)

- **SlotIR fields:** `name`, `norm_tokens`, `expected_type`, `alias_tokens`, `semantic_target_tags`, `operator_preference` (min/max), `unit_preference`.
- **Slot semantic expansion:** `_slot_semantic_expansion(param_name)` returns a frozenset of tags (e.g. budget → budget, total, available, spending_limit, resource_limit; capacity → capacity, limit, maximum, available, upper_bound; min/max/percent slots get corresponding tags). Built in `_build_slot_irs(expected_scalar)`.

## Compatibility scoring (Stage C)

- **Function:** `_score_mention_slot_ir(m: MentionIR, s: SlotIR)` returns `(score, features)`.
- **Components (additive, weights in `SEMANTIC_IR_WEIGHTS`):**
  1. **Type:** Strong bonus for exact type match, soft bonus for compatible numeric types; large penalty (forbidden) for percent/currency mismatch.
  2. **Semantic tag overlap:** Overlap of `m.semantic_role_tags` and `s.semantic_target_tags` — one of the strongest signals.
  3. **Lexical:** Context-token and sentence-token overlap with slot name/aliases.
  4. **Operator:** Min-like context with min-like slot and max-like with max-like.
  5. **Unit:** percent_marker with percent slot, currency_marker with currency slot.
  6. **Entity/target:** Overlap of `target_entity_tokens` with slot words.
  7. **Weak-match penalty** when score would otherwise be ≤ 0.
- **Global assignment:** `_semantic_ir_global_assignment(mentions_ir, slots_ir)` uses `scipy.optimize.linear_sum_assignment` on cost = −score when scipy is available; otherwise the same DP as constrained assignment. Returns `(assignments, scores_out, debug)`.

## Validation and repair (Stage D)

- **Validator:** `_validate_slot_assignment(slot_name, m, s, score)` checks type compatibility, semantic/operator/unit support, and applies a small bonus/penalty; returns `(valid, adjustment)`.
- **Repair:** `_validation_and_repair(...)` (1) optionally unassigns slots with low score and no semantic support; (2) sorts unfilled slots by priority (richer semantic/alias info first); (3) fills unfilled slots from debug candidate list with type-compatible, moderate-support mentions; (4) records whether each slot was filled in the initial pass or in the repair pass (`filled_in_repair`).
- **Objective:** Implicitly improves a validated objective (initial_pair_score + validation_bonus − inconsistency_penalty); coverage repair adds slots that pass the relaxed validity check.

## Method names used in outputs

- **Baseline names:** `tfidf_semantic_ir_repair`, `oracle_semantic_ir_repair` (and e.g. `bm25_semantic_ir_repair`, `lsa_semantic_ir_repair` if `--baseline bm25/lsa` with `--assignment-mode semantic_ir_repair`).
- **Summary CSV:** Same file `nlp4lp_downstream_summary.csv`; rows keyed by `(variant, baseline)`.
- **Per-query CSV/JSON:** `nlp4lp_downstream_per_query_{variant}_{baseline}.csv`, `nlp4lp_downstream_{variant}_{baseline}.json` with the same columns/aggregates as other modes.
