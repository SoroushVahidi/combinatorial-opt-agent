# Implementation plan: anchor-linking and bottom-up beam (NLP4LP downstream)

## Audit summary (repo-specific)

### Where mention-slot compatibility is computed
- **File**: `tools/nlp4lp_downstream_utility.py`
- **Function**: `_score_mention_slot_opt(m: MentionOptIR, s: SlotOptIR)` (lines ~1290–1370)
- Returns `(float score, dict features)`. Uses: type compatibility, role overlap, fragment_type vs slot flags, operator match, context/sentence overlap with slot words, unit match, entity/resource/product overlap, total/coefficient match. Weights in `OPT_ROLE_WEIGHTS`.

### Current assignment strategies
| Method | Flow | Strategy |
|--------|------|----------|
| **optimization_role_repair** | `_run_optimization_role_repair` → `_opt_role_global_assignment` → `_opt_role_validate_and_repair` | **Max-weight bipartite matching** (scipy `linear_sum_assignment` or fallback DP over slot mask). One-to-one; then validate and fill unfilled from debug. |
| **optimization_role_relation_repair** | `_run_optimization_role_relation_repair` → `_opt_role_incremental_admissible_assignment` → `_opt_role_validate_and_repair` | **Greedy incremental**: slot order by role/alias size; for each slot pick best mention that keeps partial admissible; add relation bonus. No-reuse via `used_mention_ids`. |

### No-reuse and admissibility
- **No-reuse**: Bipartite path: scipy/DP enforces one-to-one. Incremental path: `used_mention_ids` in `_opt_role_incremental_admissible_assignment`. Validate-and-repair: `used_mention_ids` when filling unfilled slots from debug.
- **Admissibility**: `_is_partial_admissible(partial, slots)` used only in relation_repair path. Checks: distinct mention ids, type compat, total vs per-unit, bound min/max consistency.

### Where to register new baselines
- **main()**: `--assignment-mode` choices; add `optimization_role_anchor_linking`, `optimization_role_bottomup_beam_repair`. Map to effective_baseline `tfidf_optimization_role_anchor_linking`, `tfidf_optimization_role_bottomup_beam_repair`.
- **run_one()**: Add two `elif assignment_mode == "..."` blocks that call the new runners and fill `filled`, `type_matches`, `comparable_errs` like the existing repair blocks.

### Existing structures reused
- **MentionOptIR**: Already has raw_surface, value, type_bucket, context_tokens, sentence_tokens, role_tags, operator_tags, unit_tags, fragment_type, is_per_unit, is_total_like, nearby_entity/resource/product_tokens. No schema text at mention level.
- **SlotOptIR**: name, norm_tokens, expected_type, alias_tokens, slot_role_tags, operator_preference, unit_preference, is_objective_like, is_bound_like, is_total_like, is_coefficient_like. Slot “profile” = name + aliases + role tags (no separate schema description in code).
- **Focused eval**: `tools/run_nlp4lp_focused_eval.py` uses `BASELINE_ASSIGNMENT` list; `tools/build_nlp4lp_per_instance_comparison.py` calls `_run_optimization_role_repair` and `_run_optimization_role_relation_repair` and writes CSV with pred_opt_repair, pred_relation_repair, exact_*, inst_ready_*.

---

## Implementation plan

### Method A: optimization_role_anchor_linking
1. **Richer mention/slot use**  
   Reuse MentionOptIR and SlotOptIR as-is. Add a small **slot profile** (set of tokens) = norm_tokens + alias_tokens + tokenized slot_role_tags for alignment.

2. **Anchor scoring**  
   Add `_score_mention_slot_anchor(m, s, use_entity_alignment=True, use_edge_pruning=True)`:
   - Base: same type/role/fragment/operator/unit as `_score_mention_slot_opt` (call it or reuse logic).
   - Add: operator compatibility (min↔min, max↔max, total↔total, per↔per-unit, percent↔percent, objective cue↔objective slot).
   - Add: entity alignment = overlap(mention context + nearby_entity/resource, slot name + aliases).
   - Add: lexical alignment = overlap(mention context, slot profile set).
   - Edge pruning (if use_edge_pruning): large penalty (e.g. -1e8) for: percent mention → non-percent slot; at-least/min mention → upper-bound-only slot; per-unit mention → total-only slot; total mention → per-unit-only slot.

3. **Assignment**  
   Add `_opt_role_global_assignment_with_score_matrix(mentions, slots, score_matrix)` (or add optional `score_fn` to existing global assignment). Anchor runner builds `score_matrix[i][j] = _score_mention_slot_anchor(mentions[i], slots[j], ...)[0]` and calls global assignment with it. Then `_opt_role_validate_and_repair` as today.

4. **Runner**  
   `_run_optimization_role_anchor_linking(query, variant, expected_scalar, use_entity_alignment=True, use_edge_pruning=True)` → same pipeline as repair (extract mentions, build slots, score matrix, global assignment, validate_and_repair). Return (filled_values, filled_mentions, filled_in_repair).

### Method B: optimization_role_bottomup_beam_repair
1. **Atoms**  
   All (mention_id, slot_name) pairs with score from `_score_mention_slot_opt` or `_score_mention_slot_anchor` (configurable via `use_anchor_scores`).

2. **Bundles**  
   A bundle = (assignments: dict slot_name -> MentionOptIR, score: float). Or (frozenset of (mention_id, slot_name), score); then decode to dict for admissibility.

3. **Beam**  
   Beam width K (default 5). Start with top-K single-pair bundles. Iterate: extend each bundle with one (mention_id, slot_name) not in bundle; no mention/slot reuse; build partial dict and run `_is_partial_admissible`; score = sum of pair scores (+ optional relation bonus). Keep top-K by score (tie-break by n_filled then score).

4. **Stop**  
   When no bundle can be extended (all slots or all mentions used, or no admissible extension). Return best bundle (max score; tie-break by coverage).

5. **Decode**  
   Best bundle → filled dict. Run `_opt_role_validate_and_repair` on it. Return (filled_values, filled_mentions, filled_in_repair).

6. **Runner**  
   `_run_optimization_role_bottomup_beam_repair(query, variant, expected_scalar, beam_width=5, use_anchor_scores=True)`.

### Global assignment refactor
- Add a variant that takes a precomputed score matrix: e.g. `_opt_role_global_assignment(..., score_matrix=None)`. If `score_matrix` is provided, use it (score_matrix[i][j]); else compute from `_score_mention_slot_opt`. This avoids duplicating the scipy/DP logic.

### Focused eval and artifacts
- **run_nlp4lp_focused_eval.py**: Add to `FOCUSED_BASELINES` and `BASELINE_ASSIGNMENT`: `tfidf_optimization_role_anchor_linking`, `tfidf_optimization_role_bottomup_beam_repair` (with baseline `tfidf`, modes `optimization_role_anchor_linking`, `optimization_role_bottomup_beam_repair`).
- **build_nlp4lp_per_instance_comparison.py**: Add columns for anchor and beam: pred_anchor_linking, pred_bottomup_beam, n_filled_anchor, n_filled_beam, exact_anchor, exact_beam, inst_ready_anchor, inst_ready_beam. Call `_run_optimization_role_anchor_linking` and `_run_optimization_role_bottomup_beam_repair` per instance.
- **analyze_nlp4lp_downstream_disagreements.py**: Keep as-is (compares opt_repair vs relation_repair). Optional: later add pairwise disagreement across anchor/beam if needed.
- **build_nlp4lp_failure_audit.py**: Reads comparison CSV; if new columns exist, include anchor/beam in hard-cases and failure patterns where applicable. Prefer minimal change: comparison CSV already has multiple pred_* columns; audit can use any subset.

### Ablations (optional, if easy)
- Anchor: `use_entity_alignment=False`, `use_edge_pruning=False` (internal params; can expose via extra baseline names later).
- Beam: `use_anchor_scores=False` (use _score_mention_slot_opt for beam). Expose via param; no extra assignment mode unless we add e.g. `optimization_role_bottomup_beam_repair_old_scores`.

---

## Classification of current codebase

- **optimization_role_repair**: **Bipartite matching** (maximum-weight assignment via scipy or DP).
- **optimization_role_relation_repair**: **Greedy** (incremental slot-by-slot with admissibility).
- **Hybrid**: Both use the same validate-and-repair post-pass. So the repo is a **hybrid**: one method is bipartite, one is greedy + admissibility; both share mention/slot extraction and repair.

A **richer constrained maximum-weight assignment** (e.g. min/max ordering constraints, total vs per-unit constraints in the objective) would be a clean future extension: keep the same cost matrix but add linear or constraint-based terms to the assignment formulation (e.g. in scipy or a small MIP). The current admissibility is applied incrementally in the relation path and in validate_repair; moving some of that into the assignment itself would be the next step.
