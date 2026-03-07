## NLP4LP Constrained Assignment Examples

This document provides a few concrete examples illustrating how the new constrained assignment behaves relative to the original typed greedy method. All examples are drawn from the **orig** variant.

Per-query metrics come from:

- Typed TF-IDF: `results/paper/nlp4lp_downstream_per_query_orig_tfidf.csv`
- Constrained TF-IDF: `results/paper/nlp4lp_downstream_per_query_orig_tfidf_constrained.csv`
- Typed Oracle: `results/paper/nlp4lp_downstream_per_query_orig_oracle.csv`
- Constrained Oracle: `results/paper/nlp4lp_downstream_per_query_orig_oracle_constrained.csv`
- Queries: `data/processed/nlp4lp_eval_orig.jsonl`

The examples focus on:

- Query snippet
- Extracted numeric context (conceptually)
- Candidate slots (conceptually)
- Aggregate pair behavior (via per-query metrics)
- Whether constrained assignment helped or hurt

---

### Example 1: Mrs. Watson’s investment (query_id = nlp4lp_test_0)

**Query snippet (orig):**

> Mrs. Watson wants to invest in the real-estate market and has a total budget of at most \$760000. … Each dollar invested in condos yields a \$0.50 profit and each dollar invested in detached houses yields a \$1 profit. A minimum of 20% of all money invested must be in condos, and at least \$20000 must be in detached houses. …

**Conceptual slots (from schema):**

- `total_budget` (currency)
- `profit_per_dollar_condos` (currency/float)
- `profit_per_dollar_detached` (currency/float)
- `min_condo_fraction` (percent)
- `min_detached_amount` (currency)

**Per-query metrics (TF-IDF):**

- Typed TF-IDF (`nlp4lp_downstream_per_query_orig_tfidf.csv`):
  - `n_expected_scalar = 5`, `n_filled = 4`, coverage = 0.8
  - type_match = 0.75
  - exact5 = 0.25, exact20 = 0.25
- Constrained TF-IDF (`nlp4lp_downstream_per_query_orig_tfidf_constrained.csv`):
  - `n_expected_scalar = 5`, `n_filled = 4`, coverage = 0.8
  - type_match = 1.0
  - exact5 = 1.0, exact20 = 1.0

**Interpretation:**

- Both methods fill 4 out of 5 scalar slots.
- Constrained assignment selects mention–slot pairs that:
  - Are fully type-consistent (all four filled slots have the expected coarse type).
  - Achieve perfect relative error (all comparable errors ≤ 5% and ≤ 20%).
- This is a clean case where constrained assignment **strictly improves** numeric correctness without sacrificing coverage.

---

### Example 2: Cleaning company advertising (query_id = nlp4lp_test_2)

**Query snippet (orig):**

> A cleaning company… has a \$250,000 advertising budget. Each radio ad costs \$5,000; each social media ad costs \$9,150. … At least 15 but no more than 40 radio ads… at least 35 social media ads…

**Conceptual slots:**

- `total_budget` (currency)
- `cost_per_radio_ad` (currency)
- `cost_per_social_media_ad` (currency)
- `min_radio_ads` (integer)
- `max_radio_ads` (integer)
- `min_social_media_ads` (integer)
- Exposure-related slots (viewers per ad)

**Per-query metrics (TF-IDF):**

- Typed TF-IDF:
  - `n_expected_scalar = 8`, `n_filled = 6`, coverage = 0.75
  - type_match ≈ 0.1667
  - exact5 = 0.0, exact20 = 0.0
- Constrained TF-IDF:
  - `n_expected_scalar = 8`, `n_filled = 6`, coverage = 0.75
  - type_match = 0.5
  - exact5 ≈ 0.3333, exact20 ≈ 0.3333

**Interpretation:**

- Constrained assignment **keeps the same coverage** but:
  - Increases the proportion of correctly typed slots from ~1/6 to 1/2.
  - Improves relative error on the scalar slots it fills (instead of assigning loosely related numbers).
- This is a typical case where the new method prioritizes **semantically and numerically plausible pairs** over just “using every number”.

---

### Example 3: Breakfast sandwiches (query_id = nlp4lp_test_1)

**Query snippet (orig):**

> A breakfast joint makes two different sandwiches: a regular and a special. Both need eggs and bacon. … The joint has a total of 40 eggs and 70 slices of bacon. It makes a profit of \$3 per regular sandwich and a profit of \$4 per special sandwich. …

**Conceptual slots:**

- `eggs_per_regular`, `bacon_per_regular`
- `eggs_per_special`, `bacon_per_special`
- `total_eggs`, `total_bacon`
- `profit_per_regular`, `profit_per_special`

**Per-query metrics (TF-IDF):**

- Typed TF-IDF:
  - `n_expected_scalar = 2` (for this per-query summary) , `n_filled = 2`, coverage = 1.0
  - type_match = 1.0
  - exact5 = 0.0, exact20 = 0.0
- Constrained TF-IDF:
  - `n_expected_scalar = 2`, `n_filled = 2`, coverage = 1.0
  - type_match = 0.0
  - exact5 = 0.0, exact20 = 0.0

**Interpretation:**

- This is a **failure case** for constrained assignment:
  - The greedy typed method fills both slots with correct coarse types.
  - The constrained method chooses assignments that do not satisfy the type expectation (type_match = 0), even though coverage is unchanged.
- This shows that constrained assignment is not universally better; it is more conservative and can make suboptimal choices when several numbers have similar contextual support.

---

### Example 4: Oracle vs Oracle constrained (same queries)

Using the same first few queries but with oracle schemas:

- For `nlp4lp_test_0` and `nlp4lp_test_2`:
  - Oracle and Oracle-constrained both have schema_hit = 1 and identical coverage (0.8 and 0.75 respectively).
  - Oracle-constrained improves type_match and Exact20, similar to TF-IDF.
- For `nlp4lp_test_1`:
  - Oracle behaves similarly to TF-IDF (perfect type_match).
  - Oracle-constrained can again degrade type_match on some slots.

This reinforces that constrained assignment is mostly **about the downstream numeric instantiation**, independent of whether schemas came from retrieval or oracle, and that it should be presented as a **precision-oriented alternative** rather than a strict improvement.

---

### 5. Summary of example behavior

- The constrained assignment:
  - Clearly improves some multi-constraint problems (budget + percentages + min/max constraints) by better aligning mentions to semantically appropriate slots.
  - Leaves schema retrieval untouched (same schema_hit and key_overlap).
  - Can **over-prune** or mis-prioritize mentions in simple cases, harming type_match without changing coverage.
- For the manuscript:
  - Use examples like Mrs. Watson’s investment and the cleaning company to illustrate **what constrained assignment is trying to do** (align numbers to slots under global constraints).
  - Also show a failure case (e.g., breakfast sandwiches) to avoid over-claiming and to make clear that this is a diagnostic improvement, not a solved downstream instantiation problem.

