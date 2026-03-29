# Case Studies

**Date:** 2026-03-10  
**Note:** These are illustrative cases constructed from the documented failure patterns.
Full per-instance case extraction requires HF_TOKEN for gold slot values.

---

## Case 1: Easy full success (schema-hit + high coverage)

**Profile:** Short, clear query with numeric values matching schema types exactly.  
**Example query type:** "A company has 3 products with profit rates 10%, 15%, 20% and
budgets of $500, $800, $300 respectively."  
**Prediction quality:** High — TF-IDF retrieves correct LP schema; numeric mentions
include clear percent and currency values that map uniquely to slots.  
**Lesson:** Well-formed queries with explicit type indicators (%, $) succeed reliably.

---

## Case 2: Retrieval correct but grounding wrong (float confusion — PRE-FIX)

**Profile:** Query with many integer-valued parameters in a float-typed schema.  
**Example query type:** "There are 5 types of feed with protein contents 2.3, 1.5, 3.1, 0.9, 2.7."  
**Pre-fix behavior:** All float tokens had kind="int" (digit-only tokens);
_expected_type returned "float" for all slots; TypeMatch was 0 for every slot.  
**Post-fix behavior:** `_is_type_match("float","int") = True`; TypeMatch should now count.  
**Lesson:** The _is_type_match fix is expected to convert this from a total failure to a success.

---

## Case 3: Short-query degradation

**Profile:** Only first sentence of a multi-sentence problem description.  
**Example:** "A diet problem requires food items with specific calorie constraints."  
**Prediction quality:** Schema retrieval: good (0.78 R@1 on short). 
Coverage: near zero (no numeric values in one sentence).  
**Lesson:** Short queries fundamentally lack numeric information; grounding is impossible.

---

## Case 4: Percent confusion

**Profile:** Query with both percent and integer values.  
**Example:** "20% of production goes to market A, remaining 80 units to market B."  
**Error:** Percent token (20%) assigned to an integer slot; integer token (80) mistyped.  
**Post-fix:** Partially addressed; percent TypeMatch ≈ 0.484 pre-fix, higher post-fix.  
**Lesson:** Mixed-type queries remain challenging; percent slots need strong contextual signals.

---

## Case 5: Noisy query (structural block)

**Profile:** Noisy variant of any query; all numbers replaced with `<num>`.  
**Query:** "A factory produces `<num>` types of products each requiring `<num>` units..."  
**Prediction:** Schema retrieval: 0.90+ R@1 (schema text doesn't depend on values).
Grounding: 0 TypeMatch, 0 InstReady (no parseable values exist).  
**Lesson:** Noisy variant is a retrieval success / grounding no-op by design.

---

## How to generate more cases

When HF_TOKEN is available:
```bash
export HF_TOKEN=hf_...
python tools/build_nlp4lp_per_instance_comparison.py --variant orig
# Outputs: results/paper/nlp4lp_focused_per_instance_comparison.csv
# Then select interesting cases from that CSV
```
