# NLP4LP Optimization-Role Method — Implementation

## Where the method plugs in

- **File:** `tools/nlp4lp_downstream_utility.py`
- **Retrieval:** Unchanged. Assignment mode only affects downstream slot filling after top-1 schema selection.
- **Entry point:** In `run_one()` inside `run_setting()`, the branch `if assignment_mode == "optimization_role_repair" and expected_scalar` calls `_run_optimization_role_repair(query, variant, expected_scalar)` and then fills metrics from the returned values and mention IRs.
- **CLI:** `--assignment-mode optimization_role_repair` with `--baseline tfidf` or `--baseline oracle` yields baseline names `tfidf_optimization_role_repair` and `oracle_optimization_role_repair`. Results are written to the same summary and per-query paths with these baseline names.

## Files and functions changed/added

| Location | Change |
|----------|--------|
| `tools/nlp4lp_downstream_utility.py` | New: `OPT_ROLE_WORDS`, `PER_UNIT_PATTERNS`, `OBJECTIVE_PATTERNS`, `BOUND_PATTERNS`, `TOTAL_RESOURCE_PATTERNS`, `RATIO_PATTERNS`, `REQUIREMENT_PATTERNS`, `OPT_UNIT_*`, `OPT_ROLE_WEIGHTS`, `OPT_REPAIR_WEIGHTS`, `MentionOptIR`, `SlotOptIR`, `_context_to_opt_role_tags`, `_classify_fragment_type`, `_detect_opt_unit_tags`, `_extract_opt_role_mentions`, `_slot_opt_role_expansion`, `_build_slot_opt_irs`, `_score_mention_slot_opt`, `_opt_role_global_assignment`, `_opt_role_validate_one`, `_opt_role_validate_and_repair`, `_run_optimization_role_repair`. |
| Same file | Extended: `run_one` — new branch for `assignment_mode == "optimization_role_repair"`; `main()` — `--assignment-mode` choices include `optimization_role_repair`, `effective_baseline` gets suffix `_optimization_role_repair` when that mode is selected. |

Existing methods (typed, untyped, constrained, semantic_ir_repair) are unchanged.

## Optimization-role tags

**Mention-side role tags** (inferred from context via `OPT_ROLE_WORDS` and patterns):  
`objective_coeff`, `unit_profit`, `unit_revenue`, `unit_return`, `unit_cost`, `resource_consumption`, `capacity_limit`, `demand_requirement`, `total_budget`, `total_available`, `lower_bound`, `upper_bound`, `ratio_constraint`, `percentage_constraint`, `share_constraint`, `fixed_cost`, `penalty`, `setup_cost`, `time_requirement`, `quantity_limit`, `cardinality_limit`, `minimum_requirement`, `maximum_allowance`.

**Unit markers:** `percent_marker`, `currency_marker`, `count_marker`, `time_marker`, `decimal_marker`.

**Fragment types** (from `_classify_fragment_type`): `objective`, `constraint`, `resource`, `ratio`, `bound`, or `""`.

## MentionOptIR / SlotOptIR

- **MentionOptIR:** `mention_id`, `value`, `type_bucket`, `raw_surface`, `role_tags`, `operator_tags`, `unit_tags`, `fragment_type`, `is_per_unit`, `is_total_like`, `nearby_entity_tokens`, `nearby_resource_tokens`, `nearby_product_tokens`, `context_tokens`, `sentence_tokens`, `tok` (NumTok).
- **SlotOptIR:** `name`, `norm_tokens`, `expected_type`, `alias_tokens`, `slot_role_tags`, `operator_preference`, `unit_preference`, `is_objective_like`, `is_bound_like`, `is_total_like`, `is_coefficient_like`.

## Scoring features (Stage 5)

Additive score components (weights in `OPT_ROLE_WEIGHTS`):

1. **Type compatibility** — exact/loose bonus; incompatibility penalty.
2. **Optimization-role tag overlap** — `opt_role_overlap` (strong signal).
3. **Objective-vs-constraint compatibility** — fragment_type vs slot flags (`fragment_objective`, `fragment_bound`, `fragment_resource`, `fragment_ratio`).
4. **Operator compatibility** — min/max match.
5. **Lexical overlap** — context and sentence vs slot name/aliases.
6. **Unit compatibility** — unit_tags vs unit_preference.
7. **Entity/resource token overlap** — nearby_entity_tokens, nearby_resource_tokens, nearby_product_tokens vs slot words.
8. **Coefficient-vs-total distinction** — total_match (slot total_like + mention total_like), coefficient_match (slot coefficient_like + mention is_per_unit).
9. **Weak-match penalty** — when score would otherwise be ≤ 0.
10. **Schema-prior bonus** — small constant for every pair.

## Validation and repair logic (Stage 6)

- **Validator** `_opt_role_validate_one`: type compatibility; role overlap bonus; bound plausibility (min/max alignment); total-vs-coefficient penalty (e.g. total slot with per-unit mention, coefficient slot with total-like mention).
- **Repair** `_opt_role_validate_and_repair`: (1) drop assignments with low score and weak role support; (2) sort unfilled slots by richness of slot_role_tags/alias_tokens; (3) fill unfilled slots from debug candidate list with type-compatible, role-plausible mentions; (4) record `filled_in_repair` (initial vs repair).
- **Swap repair:** Not implemented as a separate pass; the global assignment and repair fill already prefer role-consistent pairs. Optional future: try swapping two slot assignments if both improve validated score.

## Method names used in outputs

- **Baseline names:** `tfidf_optimization_role_repair`, `oracle_optimization_role_repair` (and e.g. `bm25_optimization_role_repair`, `lsa_optimization_role_repair` if used with those baselines).
- **Summary CSV:** Same file `nlp4lp_downstream_summary.csv`; rows keyed by `(variant, baseline)`.
- **Per-query CSV/JSON:** `nlp4lp_downstream_per_query_{variant}_{baseline}.csv`, `nlp4lp_downstream_{variant}_{baseline}.json`.
