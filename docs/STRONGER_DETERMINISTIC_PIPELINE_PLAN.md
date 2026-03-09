# Stronger Deterministic Pipeline — Design Plan

**Date:** 2026-03-09  
**Based on:** actual code in `tools/nlp4lp_downstream_utility.py`, existing results in `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md`, and bottleneck evidence in `docs/BOTTLENECK_ANALYSIS.md`.

---

## 1. What the current downstream pipeline really does

The pipeline has three assignment methods, all in `tools/nlp4lp_downstream_utility.py`:

### 1a. Typed greedy baseline (`assignment_mode="typed"`)

Entry: `_choose_token()`.

For each scalar slot in order:
1. Classify the slot name into `percent | int | currency | float` via `_expected_type()`.
2. Pick the best remaining numeric token from the flat list using a hand-tuned score tuple (kind match, absolute value, raw string).
3. Remove the used token and proceed to the next slot.

No global view. No semantic context. No type-incompatibility block.

### 1b. Optimization-role repair (`assignment_mode="optimization_role_repair"`)

Entry: `_run_optimization_role_repair()`.

**Stage 1 — mention extraction:** `_extract_opt_role_mentions()` builds `MentionOptIR` for each numeric token in the query. Each mention carries:
- `type_bucket` (percent/int/currency/float)
- `role_tags` (inferred from context words, e.g. `objective_coeff`, `capacity_limit`)
- `operator_tags` (min/max inferred from context cues)
- `unit_tags` (percent_marker, currency_marker, count_marker, time_marker, decimal_marker)
- `fragment_type` (objective/constraint/resource/ratio/bound)
- `is_per_unit`, `is_total_like`
- `nearby_entity_tokens`, `nearby_resource_tokens`, `nearby_product_tokens`

**Stage 2 — slot expansion:** `_build_slot_opt_irs()` builds `SlotOptIR` for each expected scalar parameter. Each slot carries:
- `expected_type`, `slot_role_tags`, `operator_preference`, `unit_preference`
- `is_objective_like`, `is_bound_like`, `is_total_like`, `is_coefficient_like`

**Stage 3 — scoring:** `_score_mention_slot_opt()` computes an additive compatibility score:
- Type exact bonus (+4.0), type loose bonus (+1.5), type incompatibility veto (−∞)
- Optimization-role overlap (+3.0 × overlap count)
- Fragment-type compatibility bonus (+1.5)
- Operator match bonus (+1.2)
- Lexical context overlap (+0.5), sentence overlap (+0.2)
- Unit match bonus (+2.0)
- Entity/resource overlap (+0.8)
- Total/coefficient consistency (+1.2)
- Schema prior bonus (+0.5), weak-match penalty (−1.0)

**Stage 4 — global assignment:** `_opt_role_global_assignment()` runs maximum-weight bipartite matching using `scipy.optimize.linear_sum_assignment` when available, or a DP fallback.

**Stage 5 — validate and repair:** `_opt_role_validate_and_repair()` removes implausible assignments (type-incompatible or score below threshold) and tries to fill unfilled slots from remaining candidates.

---

## 2. Why the current pipeline likely fails

### 2a. Multiple confusable numeric values (87% of queries)

The current scoring relies primarily on context-word overlap and role tag overlap. When a query has 5+ numeric values with similar contexts (e.g., multiple costs, demands, capacities), the role tag overlap signal is too diffuse — many mentions share the same tags (`objective_coeff`, `capacity_limit`), so the matching devolves to weak lexical overlap. Result: effectively random assignment among confusable values.

### 2b. Lower vs upper confusion

The `operator_match_bonus = 1.2` is a **positive bonus** for a match, but there is **no penalty** for a mismatch. A mention appearing in a "minimum" context can still score acceptably for a "maximum" slot if it has strong role overlap. The polarity signal is therefore asymmetric and too weak.

### 2c. Total vs per-unit confusion

`coefficient_vs_total_bonus = 1.2` applies only when the assignment is correct. For the **reverse** direction (total mention assigned to coefficient slot, or per-unit mention assigned to total slot), there is **no explicit cross-penalty** in the current code. The result is that total-style numbers (large absolute values, "budget", "available") can be freely assigned to coefficient slots.

### 2d. Percent vs absolute mismatch

`type_incompatible` blocks currency→percent and percent→currency. But the type incompatibility check is **not applied when the slot is percent-type and there exist percent-kind mentions that were not used**. The current scoring has `type_exact_bonus = 4.0` for percent-to-percent, but `type_loose_bonus = 1.5` still allows non-percent mentions to compete. If the percent mention is used for a different slot, a non-percent mention (with strong role overlap) can win a percent slot.

### 2e. Variable/entity association

`entity_resource_overlap = 0.8` checks if `nearby_entity_tokens | nearby_resource_tokens | nearby_product_tokens` overlap with `slot_words`. But the entity token extraction is: `frozenset(t for t in ctx_tokens if len(t) > 2 and t in OPT_ROLE_WORDS)`. This only captures words that happen to be in `OPT_ROLE_WORDS` — **not the actual entity names** that appear in the query ("product A", "machine 1", "supplier X"). Real entity associations are therefore largely missed.

---

## 3. The new stronger deterministic design: `global_consistency_grounding`

### Design overview

The new method (`assignment_mode="global_consistency_grounding"`, implementation: `_run_global_consistency_grounding()`) reuses the mention extraction and slot building from `optimization_role_repair` but adds six new scoring signals and a post-assignment conflict repair pass.

**Reused from optimization_role_repair:**
- `_extract_opt_role_mentions()` → `MentionOptIR`
- `_build_slot_opt_irs()` → `SlotOptIR`
- Bipartite matching infrastructure

**New:**
1. **Polarity mismatch penalty** (−4.0): if slot's `operator_preference` contains `"min"` but the mention's `operator_tags` contains `"max"` (or vice versa), apply a hard penalty. Currently optimization_role_repair only gives a +1.2 bonus for a match — no penalty for a mismatch.

2. **Percent firewall** (−6.0): if the slot `expected_type == "percent"` AND there exist at least one `percent`-kind mention in the query, then assigning a non-percent mention to this slot receives a −6.0 penalty. This enforces that percent slots go to percent mentions when percent mentions are available, preventing the `type_loose_bonus` from undermining type consistency.

3. **Total/coefficient cross-penalty** (−3.0): if `s.is_coefficient_like AND m.is_total_like`, penalize by −3.0 (currently no such penalty). Similarly if `s.is_total_like AND m.is_per_unit`, penalize by −3.0. The current method has only a +1.2 bonus for a match, with no cross-penalty.

4. **Entity anchor bonus** (+2.0): extract "anchor tokens" from the slot name — tokens longer than 3 characters that are not function words. If any anchor token appears in the mention's context window, apply a +2.0 bonus. This is a direct entity-name match that is independent of `OPT_ROLE_WORDS`.

5. **Magnitude plausibility penalties** (−1.5 each):
   - If `expected_type == "percent"` and `m.value > 100.0`: implausible percentage → −1.5.
   - If `expected_type == "int"` and `m.value is not None` and `float(int(m.value)) != m.value`: decimal assigned to integer slot → −1.5.

6. **Post-assignment min/max conflict repair**: after the bipartite matching, detect sibling slot pairs (min_x / max_x, or lower_x / upper_x). If both are assigned and `assigned_min_value > assigned_max_value`, swap the two assignments. This directly addresses the lower vs upper confusion.

### Sibling detection

`_detect_gcg_sibling_slots()` strips `min_`, `max_`, `lower_`, `upper_`, `_min`, `_max`, `_lower`, `_upper`, `minimum`, `maximum` prefixes/suffixes from slot names, groups by base, and links complementary min/max slots.

### Why this is meaningfully different

| Signal | optimization_role_repair | global_consistency_grounding |
|--------|--------------------------|------------------------------|
| Polarity match | +1.2 bonus only | +2.0 bonus + −4.0 mismatch penalty |
| Percent type | +4.0 exact bonus | +4.0 exact + −6.0 firewall if percent available |
| Total vs coeff | +1.2 match bonus only | +1.8 match + −3.0 cross-penalty |
| Entity anchor | OPT_ROLE_WORDS lookup only (0.8) | Direct slot-name token lookup (+2.0) |
| Magnitude plausibility | None | −1.5 for percent>100; −1.5 for int+decimal |
| Min/max conflict | None | Post-assignment swap pass |

The key insight: the current method has **no penalties for the most common errors**, only bonuses for correct matches. Adding explicit mismatch penalties makes the scoring surface more discriminative.

---

## 4. Metrics and slices to evaluate

### Main metrics (same denominators as existing)

- **Coverage** (`param_coverage`): mean over 331 queries of n_filled / n_expected_scalar
- **TypeMatch**: mean over 331 queries of type_matches / n_filled
- **Exact5 / Exact20**: mean over schema-hit queries with comparable errors
- **InstantiationReady**: fraction of 331 queries with coverage ≥ 0.8 AND type_match ≥ 0.8

### Bottleneck slices

The following slices target the five identified bottlenecks. They can be computed from the per-query CSV output:

1. **Multi-numeric slice**: queries where `n_expected_scalar >= 3` (proxy for confusable values). Target: higher TypeMatch on this slice.
2. **Lower/upper polarity slice**: queries where at least one slot name contains `min`/`minimum`/`lower` AND at least one contains `max`/`maximum`/`upper`. Target: lower swap-error rate.
3. **Percent slice**: queries where at least one slot has `expected_type == "percent"`. Target: higher TypeMatch on this slice.
4. **Per-unit vs total slice**: queries where slot names include both coefficient-like and total-like names. Target: lower total/coeff confusion.
5. **Entity-association slice**: queries where slot names include entity-specific tokens (names longer than 4 characters that are not generic optimization vocabulary). Target: higher entity-anchor hit rate.

---

## 5. Implementation location

All new code lives in `tools/nlp4lp_downstream_utility.py`, inserted after `_run_optimization_role_repair` (line 1620) and before `_constrained_assignment` (line 1621).

New functions:
- `GCG_WEIGHTS` — weight constants
- `GCG_REPAIR_WEIGHTS` — repair weight constants
- `_detect_gcg_sibling_slots(expected_scalar)` → dict
- `_score_mention_slot_gcg(m, s, has_percent_mention)` → (score, features)
- `_gcg_global_assignment(mentions, slots, has_percent_mention)` → (assignments, scores, debug)
- `_gcg_conflict_repair(assignments, scores_out, siblings)` → assignments
- `_gcg_validate_and_repair(mentions, slots, initial_assignments, initial_scores, debug, siblings, has_percent_mention)` → (filled, filled_in_repair)
- `_run_global_consistency_grounding(query, variant, expected_scalar)` → (filled_values, filled_mentions, filled_in_repair)

New `assignment_mode` value: `"global_consistency_grounding"`.

Wired into `run_setting()` dispatch (same pattern as `optimization_role_repair`) and `main()` argparse choices.

---

## 6. Expected behaviour

- **Coverage**: should match or exceed `optimization_role_repair` (0.822 TF-IDF) because we reuse the same extraction and add repair, not restrict it.
- **TypeMatch**: should improve over `optimization_role_repair` (0.243) due to percent firewall, polarity enforcement, magnitude plausibility.
- **Exact20**: should improve over `typed` (0.205) and approximately match `optimization_role_repair` (0.277) or better.
- **InstantiationReady**: should improve over `optimization_role_repair` (0.060) because TypeMatch and Coverage both target the 0.8-threshold simultaneously.
- **Polarity errors**: should decrease relative to typed and optimization_role_repair on the lower/upper slice.
