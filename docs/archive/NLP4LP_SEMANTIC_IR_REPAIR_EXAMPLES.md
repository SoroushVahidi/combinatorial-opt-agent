# NLP4LP Semantic IR + Repair — Examples

Below are illustrative examples of how the pipeline behaves. The exact queries and slots depend on the eval set and gold schema; the structure (mentions → semantic tags → candidate slots → initial scores → repair → final assignment) is what the implementation provides.

---

## Example 1: Budget and demand (semantic tags help)

- **Query snippet:** "... with a budget of 5000 and demand of 100 units ..."
- **Extracted mentions:** e.g. (5000, currency), (100, int) with context including "budget", "demand".
- **Semantic tags:** First mention gets tags like budget, total; second gets demand, quantity/item_count.
- **Candidate slots:** e.g. `total_budget`, `demand`.
- **Initial scores:** High semantic_tag_overlap for 5000→total_budget and 100→demand; type and unit alignment.
- **Repair:** Often none needed; assignment is already valid.
- **Final assignment:** total_budget ← 5000, demand ← 100.
- **Outcome:** Method improves over a baseline that might map 100 to budget by using semantic role tags.

---

## Example 2: Min/max operator alignment

- **Query snippet:** "... at least 3 facilities and at most 10 ..."
- **Extracted mentions:** 3 and 10 with operator_tags min and max.
- **Candidate slots:** e.g. `min_facilities`, `max_facilities`.
- **Initial scores:** Operator compatibility gives bonus for 3→min_facilities and 10→max_facilities.
- **Repair:** If initial assignment swapped them, validation would penalize; repair can try the other pairing if scores are close.
- **Final assignment:** min_facilities ← 3, max_facilities ← 10.
- **Outcome:** Operator tags reduce min/max confusions.

---

## Example 3: Repair fills an unfilled slot

- **Query snippet:** "... budget 10000, cost 3 per unit, profit 20% ..."
- **Slots:** e.g. budget, unit_cost, profit_share.
- **Initial assignment:** Strong pairs get budget and profit_share; unit_cost may stay unfilled if the best mention was assigned elsewhere.
- **Repair:** Unfilled slot unit_cost is prioritized; candidate "3" (or "20") is type-compatible; repair assigns the best remaining mention (e.g. 3 → unit_cost) if semantic support is moderate.
- **Final assignment:** budget ← 10000, unit_cost ← 3, profit_share ← 0.20 (or as normalized).
- **Outcome:** Coverage and InstantiationReady improve vs leaving unit_cost empty.

---

## Example 4: Weak assignment unassigned by validation

- **Query snippet:** "... 5 items and 200 dollars ..." with slots requiring e.g. min_items, budget.
- **Initial assignment:** Sometimes the global matching assigns 200 to min_items (weak lexical overlap).
- **Validation:** Type is compatible (int-like), but semantic support for min_items is weak (no min/at_least near 200); score is low → validation flags.
- **Repair:** Assignment to min_items is dropped; 200 goes back to pool; repair may assign 5 to min_items and 200 to budget if they fit.
- **Final assignment:** min_items ← 5, budget ← 200.
- **Outcome:** Validation-and-repair corrects a plausible but wrong pairing, improving TypeMatch/readiness.

---

## Example 5: Still failing (no matching mention)

- **Query snippet:** "... we need capacity of 50 ..." with only one numeric mention (50).
- **Slots:** e.g. budget, capacity, min_order (three slots).
- **Initial assignment:** One slot gets 50; the other two stay empty.
- **Repair:** Tries to fill the other slots from the same mention only if we allowed multiple use (we do not); so remaining slots stay empty.
- **Final assignment:** e.g. capacity ← 50; budget and min_order unfilled.
- **Outcome:** Method still fails to be instantiation-ready when there are fewer mentions than slots; improvement is in better choice of which slot gets 50 (semantic tags prefer capacity) and in not over-penalizing coverage on other queries.

---

These examples are representative of the design: entity/semantic tagging, IR-based scoring, and validation-and-repair together improve TypeMatch and InstantiationReady versus the constrained-only method, while keeping determinism and no LLM/solver at inference.
