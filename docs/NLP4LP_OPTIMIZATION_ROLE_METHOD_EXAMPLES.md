# NLP4LP Optimization-Role Method — Examples

Illustrative examples of how optimization-specific assumptions affect the pipeline.

---

## Example 1: Total budget vs per-unit cost

- **Query snippet:** "With a total budget of $5000, each item costs $3."
- **Mentions:** 5000 (currency, context: "total", "budget"); 3 (currency, context: "each", "item", "costs").
- **Optimization-role tags:** 5000 → total_budget, total_available, upper_bound; 3 → unit_cost, objective_coeff.
- **Fragment types:** 5000 → resource; 3 → objective (or coefficient-like).
- **Slots:** e.g. `total_budget`, `unit_cost`.
- **Scores:** 5000–total_budget gets total_match + fragment_resource; 3–unit_cost gets coefficient_match + is_per_unit. Assignment: total_budget ← 5000, unit_cost ← 3.
- **Where it helped:** Total-vs-coefficient distinction prevents swapping budget and unit cost.

---

## Example 2: Min/max bounds

- **Query snippet:** "At least 2 facilities, at most 10."
- **Mentions:** 2 and 10 with operator_tags min and max.
- **Slots:** min_facilities, max_facilities.
- **Fragment type:** bound/constraint. Operator match gives bonus for 2→min, 10→max.
- **Repair:** If initial assignment were swapped, validation would penalize bound plausibility; repair prefers role-consistent pairs.
- **Final assignment:** min_facilities ← 2, max_facilities ← 10.
- **Where it helped:** Lower_bound/upper_bound and operator tags align mentions to the correct bound slots.

---

## Example 3: Demand and capacity

- **Query snippet:** "Demand is 100 units; capacity is 500."
- **Mentions:** 100 (demand_requirement, lower_bound); 500 (capacity_limit, upper_bound).
- **Slots:** demand, capacity.
- **Role overlap:** demand_requirement + lower_bound for 100; capacity_limit + upper_bound for 500. Strong opt_role_overlap for correct pairs.
- **Final assignment:** demand ← 100, capacity ← 500.
- **Where it helped:** Optimization-role tags separate demand (requirement/lower-bound) from capacity (limit/upper-bound).

---

## Example 4: Ratio/percentage slot

- **Query snippet:** "At least 20% must be from category A."
- **Mentions:** 20 (or 0.20) with percent_marker, percentage_constraint.
- **Slots:** e.g. min_percentage_A.
- **Fragment type:** ratio. Slot has ratio_constraint / percentage_constraint. Unit and fragment_ratio bonuses apply.
- **Final assignment:** min_percentage_A ← 0.20 (or 20 depending on normalization).
- **Where it helped:** Percentage/ratio role and unit markers align the mention with the ratio/percentage slot.

---

## Example 5: Still failing — ambiguous coefficient

- **Query snippet:** "Profit is 5 and cost is 3." (two numbers, no "per unit" or "total" disambiguation.)
- **Mentions:** 5 and 3; both may get objective_coeff / unit_profit / unit_cost if context is shared.
- **Slots:** profit_per_unit, cost_per_unit.
- **Issue:** Without strong per-unit or total cues, both mentions can get similar role tags; assignment may depend on order or lexical overlap and can still swap.
- **Where it failed:** Optimization-role method improves role-based scoring but cannot resolve ambiguity when cues are missing; repair does not run a full swap search.

---

These examples show that optimization-role tags, fragment classification, and total-vs-coefficient signals help when the text carries clear cues (budget vs per-unit, min vs max, demand vs capacity, ratio/percentage). When cues are absent or shared across mentions, the method still benefits from global assignment and type/role scoring but may not fully resolve the ambiguity.
