# NLP4LP Relation-Aware Method — Examples

## 1. Coefficient-pair consistency (RAT-SQL–style)

**Setup:** Schema has two coefficient slots (e.g. ProfitPerSandwichA, ProfitPerSandwichB) and two per-unit mentions in the same sentence (“$3 per regular sandwich and $4 per special sandwich”).

**Behavior:**  
- Slot–slot: `both_coeff`.  
- Mention–mention: `both_per_unit`, `same_sentence` (or same_fragment_type).  
- When the first (coefficient slot, per-unit mention) is assigned, the second candidate gets a relation bonus for the same_sentence + both_per_unit consistency, so the two coefficient slots tend to get the two per-unit numbers without swapping.

## 2. Total vs per-unit (PICARD–style admissibility)

**Setup:** Schema has TotalBudget and CostPerUnit. Query has “total budget $100” and “cost $5 per unit”.

**Behavior:**  
- Partial admissibility: total slot must not get a per-unit-only mention; coefficient slot must not get a total-only mention.  
- If we tentatively assign the $100 (total-like) to CostPerUnit and $5 (per-unit) to TotalBudget, `_is_partial_admissible` rejects (total slot with per-unit mention, coefficient with total mention).  
- Incremental assignment only extends with admissible partials, so this swap is never accepted.

## 3. Min/max bound pair

**Setup:** Schema has MinRadioAds and MaxRadioAds. Query has “at least 15 but no more than 40 radio ads”.

**Behavior:**  
- Slot–slot: `min_max_pair`.  
- Relation bonus: assigning the 15 to MinRadioAds and 40 to MaxRadioAds gives consistent operator alignment (min mention → min slot, max → max); bonus applied when choosing the second assignment given the first.  
- Admissibility: min slot is not assigned a mention that has only “max” operator; max slot not assigned only “min” mention.

## 4. Percent only to ratio slots

**Setup:** Schema has MinimumPercentageCondos (ratio) and TotalBudget (currency). Query has “20%” and “$760000”.

**Behavior:**  
- `_is_partial_admissible`: if 20% is assigned to TotalBudget, partial is invalid (percent mention to non-ratio slot). So that assignment is never extended.  
- 20% is only allowed for ratio/percentage slots; $760000 for total/currency.

## 5. Same-sentence coefficient pair

**Setup:** “Each regular sandwich requires 2 eggs and 3 slices of bacon. Each special requires 3 eggs and 5 slices.” Schema has RequiredRegularEggs, RequiredSpecialEggs (both coefficient-like).

**Behavior:**  
- The two “2” and “3” (or 3 and 5) in the first sentence can get `same_sentence` and `same_fragment_type`.  
- When the second coefficient slot is filled, relation_bonus rewards picking a mention that is same_sentence + both_per_unit with the already-assigned mention for the first coefficient slot, encouraging the correct pairing.

---

## How to inspect per-query behavior

After running with `--assignment-mode optimization_role_relation_repair`, per-query outputs are under `results/paper/`, e.g.:

- `nlp4lp_downstream_per_query_orig_tfidf_optimization_role_relation_repair.csv`

Compare with:

- `nlp4lp_downstream_per_query_orig_tfidf_optimization_role_repair.csv` (no relation/incremental)

to see where the new method changes assignments or fills more slots correctly.
