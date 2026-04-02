# NLP4LP Relation-Aware Method Implementation (RAT-SQL + PICARD Inspired)

## What was already present (from Phase A audit)

- **Mention-to-slot linking:** `_score_mention_slot_opt` with type, role overlap, fragment_type vs slot flags, operator/unit, entity_resource, total/coefficient.
- **Post-hoc validation and repair:** `_opt_role_validate_one`, `_opt_role_validate_and_repair` (used after global matching).
- **One-to-one global matching:** `_opt_role_global_assignment` (bipartite / DP over pair scores only; no partial-state checks).

**Missing before:** Slot-slot and mention-mention relation features; incremental assignment with admissibility; relation-aware scoring in context of partial assignment.

---

## What was added (RAT-SQL–inspired)

1. **Slot–slot relation tags**  
   `_slot_slot_relation_tags(s1, s2)` returns: `both_coeff`, `min_max_pair`, `total_and_coeff`, `same_role_family` when applicable.

2. **Mention–mention relation tags**  
   `_mention_mention_relation_tags(m1, m2)` returns: `same_sentence`, `same_fragment_type`, `both_per_unit`, `both_total`, `same_role_family` when applicable.

3. **Relation bonus in context of partial assignment**  
   `_relation_bonus(m, s, partial, slots, mentions)` adds:
   - Bonus when two coefficient slots get two per-unit mentions that are same_sentence or same_fragment_type.
   - Bonus when total/coefficient slot pair gets total-like and per-unit mentions consistently.
   - Bonus when min/max slot pair gets min and max operator mentions consistently.
   - Penalty when total slot gets per-unit mention or coefficient slot gets total mention and conflicts with current partial.

4. **Use in assignment**  
   Relation bonus is applied when choosing the next (mention, slot) in `_opt_role_incremental_admissible_assignment`: candidate score = base_score(m,s) + relation_bonus(m, s, current_partial, slots, mentions).

Weights: `RELATION_WEIGHTS["consistent_pair_bonus"] = 1.2`, `RELATION_WEIGHTS["inconsistent_pair_penalty"] = -2.0`.

---

## What was added (PICARD–inspired)

1. **Partial-state admissibility**  
   `_is_partial_admissible(partial, slots)` enforces:
   - One-to-one (no duplicate mention).
   - Type compatibility for every assigned pair.
   - Total slot not assigned a per-unit-only mention; coefficient slot not assigned a total-only mention.
   - Percent mentions only to ratio/percentage slots.
   - Min slots not assigned max-only mentions; max slots not assigned min-only mentions.

2. **Incremental admissible assignment**  
   `_opt_role_incremental_admissible_assignment(mentions, slots, base_score_matrix, debug)`:
   - Orders slots by priority (role/alias richness, then name).
   - For each slot in order, picks best unused mention such that tentative partial = current ∪ {slot: mention} is admissible (`_is_partial_admissible`).
   - Candidate score = base_score(i,j) + relation_bonus(mention_i, slot_j, current_partial, slots, mentions).
   - Only admissible extensions are accepted; invalid partials are never extended.

3. **Pipeline**  
   `_run_optimization_role_relation_repair`: extract mentions → build slot opt IRs → base pair scores (same as optimization_role_repair) → **incremental admissible assignment with relation bonus** → existing `_opt_role_validate_and_repair` for unfilled slots.

---

## Method name and CLI

- **Assignment mode:** `optimization_role_relation_repair`
- **Effective baseline names:** `tfidf_optimization_role_relation_repair`, `oracle_optimization_role_relation_repair`, etc. (baseline + `_optimization_role_relation_repair`).

CLI:

```bash
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode optimization_role_relation_repair
python tools/nlp4lp_downstream_utility.py --variant orig --baseline oracle --assignment-mode optimization_role_relation_repair
```

---

## Files changed

- **tools/nlp4lp_downstream_utility.py**
  - Added: `_slot_slot_relation_tags`, `_mention_mention_relation_tags`, `_is_partial_admissible`, `RELATION_WEIGHTS`, `_relation_bonus`, `_opt_role_incremental_admissible_assignment`, `_run_optimization_role_relation_repair`.
  - Extended: `run_one` branch for `assignment_mode == "optimization_role_relation_repair"`; `--assignment-mode` choices; `effective_baseline` for the new mode.
- **docs/NLP4LP_RATSQL_PICARD_AUDIT.md** (Phase A audit).
- **docs/NLP4LP_RELATION_AWARE_METHOD_IMPLEMENTATION.md** (this file).

---

## Design choices

- **Deterministic:** No learned parameters; all scoring and admissibility are rule-based.
- **Backward compatible:** Existing modes (typed, constrained, semantic_ir_repair, optimization_role_repair) unchanged.
- **Repair preserved:** Unfilled slots after incremental assignment are still filled via existing `_opt_role_validate_and_repair` so coverage is preserved when possible.
