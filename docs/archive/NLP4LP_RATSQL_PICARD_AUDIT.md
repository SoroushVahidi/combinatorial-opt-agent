# Phase A: Audit — RAT-SQL and PICARD Ideas in Current Repo

**Purpose:** Identify what is already implemented vs missing before adding relation-aware and incremental-admissibility features.

---

## A. RAT-SQL–like ideas: what exists

### A.1 Explicit mention-to-slot linking features
**Present.**  
- `_score_mention_slot` (typed/constrained), `_score_mention_slot_ir` (semantic_ir_repair), `_score_mention_slot_opt` (optimization_role_repair) all compute a compatibility score and feature breakdown per (mention, slot).  
- Features include: type match/incompatible, lexical context/sentence overlap, cue overlap, operator semantics, and in opt-role: role_overlap, fragment_type vs is_objective_like/is_bound_like/is_total_like/is_coefficient_like, operator_match, unit_match, entity_resource_overlap, total_match, coefficient_match.  
- **Evidence:** `tools/nlp4lp_downstream_utility.py` lines 391–436 (_score_mention_slot), 1289–1366 (_score_mention_slot_opt).

### A.2 Explicit slot-to-slot / schema-element relation features
**Missing.**  
- There is no representation of relations between slots (e.g. “both coefficient”, “min vs max”, “total and coefficient”) or between schema elements.  
- Slot–slot structure is only implicit in the list of SlotOptIRs; no `slot_slot_relation(s1, s2)` or relation matrix.

### A.3 Relation-aware scoring between mention and slot beyond lexical overlap
**Partially present.**  
- Opt-role scoring is already “relation-aware” in the sense of role overlap, fragment_type vs slot flags, operator/unit, total-vs-coefficient.  
- **Missing:** any use of relations between *pairs* of (mention, slot) or between mentions or between slots. Scoring is per-pair only; no “if slot A and slot B are both coefficient, prefer assigning two per-unit mentions from the same sentence”.

### A.4 Structured relations
**Partially present.**  
- **Same sentence:** Implicit via `sentence_tokens` and `sent_overlap` in pair score.  
- **Same clause:** Not represented.  
- **Same entity/resource context:** Partially via `nearby_entity_tokens`, `nearby_resource_tokens`, `nearby_product_tokens` and `entity_resource_overlap` with slot words.  
- **Same optimization-role family:** Via `role_tags` and `slot_role_tags` overlap.  
- **Lower-bound vs upper-bound:** Via `operator_tags`, `operator_preference` (min/max) and `is_bound_like`.  
- **Total-vs-per-unit:** Via `is_total_like`, `is_per_unit`, `coefficient_match`, `total_match`.  
- **Sibling-slot competition:** Not represented.  
- **Coefficient-set consistency:** Not represented (e.g. “two coefficient slots should get two per-unit mentions in parallel structure”).  

### A.5 Graph-style or relation-table representation
**Missing.**  
- Only the cost matrix for bipartite matching exists. No explicit relation graph, relation table, or mention–mention / slot–slot matrices used in scoring.

### A.6 Better schema-linking style scoring
**Present (per-pair).**  
- Opt-role provides strong schema-linking style scoring (type, role, fragment, operator, unit, entity/resource, total/coefficient).  
- **Missing:** use of slot–slot and mention–mention relations to improve which slot each number goes to.

---

## B. PICARD-like ideas: what exists

### B.1 Constraint-aware incremental assignment
**Partially present.**  
- `_constrained_assignment` and `_opt_role_global_assignment` use DP over (mention index × slot mask) or linear_sum_assignment, so assignment is built in a globally optimal way subject to one-to-one.  
- **Missing:** any check that a *partial* assignment is semantically valid (type/role/bound/total-vs-coeff) before extending it. The DP only maximizes sum of pair scores; it does not reject states that are already inconsistent.

### B.2 Partial-state validator while filling slots
**Missing.**  
- `_opt_role_validate_one` and `_validate_slot_assignment` exist but are used **after** the full assignment in `_opt_role_validate_and_repair` and `_validation_and_repair`.  
- There is no `_is_partial_assignment_admissible(partial, slots)` called during the assignment build.

### B.3 Early rejection of invalid assignments
**Partially present.**  
- Pair-level: type-incompatible pairs get a large penalty (-1e9 or 1e9) so they are not chosen.  
- **Missing:** rejection of a *partial* assignment that is already invalid (e.g. total slot with per-unit mention while a coefficient slot has total mention), so we never extend that state.

### B.4 Hard validity checks during assignment
**Missing.**  
- Validity is enforced only in the repair phase (drop weak, fill unfilled). No hard “this partial assignment is inadmissible, do not use it” during the main assignment step.

### B.5 Stateful repair guided by validity
**Partially present.**  
- Repair is stateful (tracks `used_mention_ids`, fills unfilled slots from debug list).  
- Validity is used when *choosing* a repair candidate (`_opt_role_validate_one`), but the *order* of assignment in the main step is not guided by admissibility.

### B.6 Maintaining feasible partial assignments
**Partially present.**  
- One-to-one is enforced by the DP/matching.  
- Role plausibility, lower/upper consistency, total-vs-coefficient, ratio compatibility are only checked in `_opt_role_validate_one` after the fact, not as conditions for “this partial state is still feasible”.

---

## Summary

1. **RAT-SQL–like already in repo:**  
   Explicit mention–slot linking (A.1), per-pair relation-aware scoring beyond plain lexical (A.3 partial), several structured relations at the pair level (A.4 partial: same sentence, role family, bound direction, total-vs-per-unit), and strong schema-linking style scoring per pair (A.6).  

2. **PICARD–like already in repo:**  
   One-to-one constraint in assignment (B.6 partial), stateful repair that uses validity when filling unfilled slots (B.5 partial).  

3. **Highest-value missing pieces:**  
   - **RAT-SQL:** Slot–slot and mention–mention relation features, and using them in scoring (e.g. coefficient-pair consistency, total-vs-coeff consistency across pairs).  
   - **PICARD:** A partial-state admissibility check and incremental assignment that only extends admissible partial assignments (reject or avoid invalid states during the build, not only in repair).

---

*Evidence: `tools/nlp4lp_downstream_utility.py` — _score_mention_slot*, _opt_role_global_assignment, _opt_role_validate_one, _opt_role_validate_and_repair, _constrained_assignment, _validation_and_repair.*
