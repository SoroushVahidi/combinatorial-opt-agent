# Grounding Failure Examples

> **Source:** NLP4LP test set (`orig` variant), live grounding pipeline.
> **Methods:** `tfidf + optimization_role_repair` vs `tfidf + max_weight_matching`.
> **Gold values:** HF `udell-lab/NLP4LP` test split (cached at `results/eswa_revision/00_env/nlp4lp_gold_cache.json`).
> **All examples have `schema_hit = 1`:** TF-IDF retrieved the correct schema,
> so every failure here is a pure **grounding failure** (not a retrieval failure).
> **How produced:** `_run_optimization_role_repair` and `_run_max_weight_matching_grounding`
> were run on all 331 test instances; predictions were compared to gold (relative error threshold: 5%).

## Summary

- Total test instances: **331**
- Instances with ≥1 slot error (> 5% relative error): **318** of 330 valid
- Examples in this file: **36** (selected for type diversity)

### Error-type breakdown

| Error Type | All failing | Selected |
|---|---|---|
| swapped quantities | 159 | 7 |
| template query (no values) | 50 | 3 |
| implicit count | 38 | 5 |
| percent vs count / incompatible type | 25 | 6 |
| min vs max swap | 24 | 5 |
| wrong assignment | 16 | 4 |
| total vs unit coefficient | 4 | 4 |
| missing value | 2 | 2 |

### Key observations

1. **Implicit count is the most pervasive failure pattern.** Many schemas abstract the number of product/resource types as a parameter (e.g., `NumProducts = 2`). Queries never state this count explicitly — they just mention the objects by name. The grounder then assigns the nearest integer-valued token (a per-unit coefficient) to the count slot, causing a cascade of further misassignments.

2. **Swapped quantities dominate among value-extraction failures.** When a query contains two or more values with similar magnitude and syntactic structure (e.g., two costs, two resource limits), the one-to-one matching assigns them to the wrong slots. Both `opt_repair` and `max_weight_matching` fail identically on these cases, indicating the score matrix lacks sufficient discriminating signal.

3. **Min/max and bound swaps are systematically present.** Parallel phrasing ("at least X ... at most Y") causes the grounder to swap lower and upper bounds. The global assignment optimum favours whichever slot-score is marginally higher, and the difference is often close to zero.

4. **Percent vs. count incompatibility blocks some correct assignments.** The strengthened `_is_type_incompatible` rule now hard-blocks integer tokens from filling percent slots. This is correct behaviour for most cases but leaves the percent slot unfilled when the query only provides a bare integer (e.g., '20' instead of '20%').

5. **`max_weight_matching` and `opt_repair` almost always fail on the same instances.** On the catastrophic failures (wrong count assignment, large swaps), both modes produce identical or near-identical predictions, confirming the bottleneck is the score matrix, not the assignment algorithm.

---

## Example 1 — implicit count

- **ID:** `nlp4lp_test_8`
- **Error type:** implicit count
- **Schema hit:** ✓
- **Slots:** 3 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> An artisan makes two types of terracotta jars: a thin jar and a stubby jar. Each thin jar requires 50 minutes of shaping time and 90 minutes of baking time. Each stubby jar requires 30 minutes of shaping time and 150 minutes of baking time. Per week, there are 3000 minutes available for shaping and 4000 minutes available for baking. The profit per thin jar is $5 and the profit per stubby jar is $9. How many jars of each type should the artisan make to maximize profit?

### Schema (preview)

> An artisan produces NumJarTypes different types of terracotta jars. Each jar type requires ShapingTimePerType shaping time and BakingTimePerType baking time. Each week, there is a total shaping time available of ShapingTimeAvailable and a total baking time available of BakingTimeAvailable. The profi

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumJarTypes` | int | 2 | ❌ 50 | ❌ 50 | 2400% | 2400% |
| `ShapingTimeAvailable` | float | 3000 | ❌ 150 | ❌ 150 | 95% | 95% |
| `BakingTimeAvailable` | float | 4000 | ❌ 3000 | ❌ 3000 | 25% | 25% |

### Mismatch

- **`NumJarTypes`** (gold = 2) → predicted **50** (2400% error)
- **`ShapingTimeAvailable`** (gold = 3000) → predicted **150** (95% error)
- **`BakingTimeAvailable`** (gold = 4000) → predicted **3000** (25% error)

### Why this is hard

The schema abstracts the number of product types, resource types, etc. as a scalar slot (e.g., `NumProducts = 2`). The query never states this count explicitly — instead it mentions the two products by name. The pipeline extracts no matching token and falls back to the nearest integer-valued mention, which is typically one of the per-unit coefficients.

---

## Example 2 — implicit count

- **ID:** `nlp4lp_test_12`
- **Error type:** implicit count
- **Schema hit:** ✓
- **Slots:** 2 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> A souvenir shop makes wooden elephants and tigers with plastic ornaments. Each elephant requires 50 grams of wood and 20 grams of plastic. Each tiger requires 40 grams of wood and 30 grams of plastic. In a week, 5000 grams of wood and 4000 grams of plastic are available. The profit per elephant sold is $5 and the profit per tiger sold is $4. How many of each should be made in order to maximize profit?

### Schema (preview)

> A souvenir shop produces NumProducts different products using NumResources different resources. Each product has a profit defined by Profit and requires specific amounts of resources as specified by ResourceRequirement. The ResourceAvailability defines the total amount of each resource available. Th

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumProducts` | int | 2 | ❌ 50 | ❌ 50 | 2400% | 2400% |
| `NumResources` | int | 2 | ❌ 20 | ❌ 20 | 900% | 900% |

### Mismatch

- **`NumProducts`** (gold = 2) → predicted **50** (2400% error)
- **`NumResources`** (gold = 2) → predicted **20** (900% error)

### Why this is hard

The schema abstracts the number of product types, resource types, etc. as a scalar slot (e.g., `NumProducts = 2`). The query never states this count explicitly — instead it mentions the two products by name. The pipeline extracts no matching token and falls back to the nearest integer-valued mention, which is typically one of the per-unit coefficients.

---

## Example 3 — implicit count

- **ID:** `nlp4lp_test_18`
- **Error type:** implicit count
- **Schema hit:** ✓
- **Slots:** 2 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> A candy store mixes regular candy and sour candy to prepare two products, regular mix and sour surprise mix. Each kilogram of the regular mix contains 0.8 kg of regular candy and 0.2 kg of sour candy. The profit per kilogram of the regular mix is $3. Each kilogram of the sour surprise mix contains 0.1 kg of regular candy and 0.9 kg of sour candy. The profit per kilogram of the sour surprise mix is $5. The candy store has 80 kg of regular candy and 60 kg of sour candy available. How many kilograms of each type of candy mix should be created to maximize profits?

### Schema (preview)

> A candy store prepares NumMixes different candy mixes using NumCandyTypes different types of candy. Each kilogram of each mix requires specific amounts of each candy type as defined by CompositionRequired. The profit per kilogram of each mix is given by ProfitPerMix. The store has AvailableCandy kil

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumMixes` | int | 2 | ❌ 60 | ❌ 60 | 2900% | 2900% |
| `NumCandyTypes` | int | 2 | ❌ 0.2 | ❌ 0.2 | 90% | 90% |

### Mismatch

- **`NumMixes`** (gold = 2) → predicted **60** (2900% error)
- **`NumCandyTypes`** (gold = 2) → predicted **0.2** (90% error)

### Why this is hard

The schema abstracts the number of product types, resource types, etc. as a scalar slot (e.g., `NumProducts = 2`). The query never states this count explicitly — instead it mentions the two products by name. The pipeline extracts no matching token and falls back to the nearest integer-valued mention, which is typically one of the per-unit coefficients.

---

## Example 4 — implicit count

- **ID:** `nlp4lp_test_19`
- **Error type:** implicit count
- **Schema hit:** ✓
- **Slots:** 4 expected; opt\_repair: 4 filled, 0 exact@5%; max\_weight\_matching: 4 filled, 0 exact@5%

### Query

> A suspicious factory has 100 sq. feet of space. It makes bootleg phones and laptops. Phones require 2 hours of labor and cost $12 for each sq. foot of space allocated for phone production (cost of electricity and equipment). Laptops require 3 hours of labor and cost $15 for each sq. foot of space allocated for laptop production. Phones produce a net revenue of $50 per sq. foot while laptops produce a net revenue of $70 per sq. foot. The factory wants to spend at most $5000 and 2000 hours of labor. What is the optimal factory layout to maximize revenue?

### Schema (preview)

> A factory has TotalSpace square feet of available space. It produces NumberOfProducts different products. Each product requires LaborRequiredPerSqFt labor hours per square foot and costs CostPerSqFt dollars per square foot to produce. Each product generates RevenuePerSqFt dollars of net revenue per 

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `TotalSpace` | float | 100 | ❌ 3 | ❌ 3 | 97% | 97% |
| `Budget` | currency | 5000 | ❌ 2 | ❌ 2 | 100% | 100% |
| `LaborHoursAvailable` | float | 2000 | ❌ 12 | ❌ 12 | 99% | 99% |
| `NumberOfProducts` | int | 2 | ❌ 100 | ❌ 100 | 4900% | 4900% |

### Mismatch

- **`TotalSpace`** (gold = 100) → predicted **3** (97% error)
- **`Budget`** (gold = 5000) → predicted **2** (100% error)
- **`LaborHoursAvailable`** (gold = 2000) → predicted **12** (99% error)
- **`NumberOfProducts`** (gold = 2) → predicted **100** (4900% error)

### Why this is hard

The schema abstracts the number of product types, resource types, etc. as a scalar slot (e.g., `NumProducts = 2`). The query never states this count explicitly — instead it mentions the two products by name. The pipeline extracts no matching token and falls back to the nearest integer-valued mention, which is typically one of the per-unit coefficients.

---

## Example 5 — implicit count

- **ID:** `nlp4lp_test_22`
- **Error type:** implicit count
- **Schema hit:** ✓
- **Slots:** 3 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> You are designing an office space with two types of desks: long desks and short desks. You can spend at most $2000. Long desks cost $300, take up 10 square feet of space, and seat 6 employees. Short desks cost $100, take up 4 square feet of space, and seat 2 employees. The office can have at most 200 square feet of desks. How many of each desk should you buy in order to maximize the seating availability?

### Schema (preview)

> You are designing an office space with NumDeskTypes different desk types. Each desk type has a Price, Space, and Seats. The total cost of the desks must not exceed MaxBudget, and the total space occupied by the desks must not exceed MaxSpace. The goal is to determine the number of each desk type to 

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumDeskTypes` | int | 2 | ❌ 200 | ❌ 200 | 9900% | 9900% |
| `MaxBudget` | currency | 2000 | ❌ 300 | ❌ 300 | 85% | 85% |
| `MaxSpace` | float | 200 | ❌ 10 | ❌ 10 | 95% | 95% |

### Mismatch

- **`NumDeskTypes`** (gold = 2) → predicted **200** (9900% error)
- **`MaxBudget`** (gold = 2000) → predicted **300** (85% error)
- **`MaxSpace`** (gold = 200) → predicted **10** (95% error)

### Why this is hard

The schema abstracts the number of product types, resource types, etc. as a scalar slot (e.g., `NumProducts = 2`). The query never states this count explicitly — instead it mentions the two products by name. The pipeline extracts no matching token and falls back to the nearest integer-valued mention, which is typically one of the per-unit coefficients.

---

## Example 6 — total vs unit coefficient

- **ID:** `nlp4lp_test_53`
- **Error type:** total vs unit coefficient
- **Schema hit:** ✓
- **Slots:** 3 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> There are two specialized containers, a small and large one, that are used to make a pharmaceutical paste. The small container requires 10 units of water and 15 units of the powdered pill to make 20 units of the paste. The large container requires 20 units of water and 20 units of the powdered pill to make 30 units of the paste. The pharmacy has available 500 units of water and 700 units of the powdered pill. How many of each container should be used to maximize the amount of paste that can be made?

### Schema (preview)

> There are NumContainers types of specialized containers used to produce paste. Each container type requires WaterRequiredPerContainer units of water and PowderedPillRequiredPerContainer units of powdered pill to produce PasteProducedPerContainer units of paste. The pharmacy has WaterAvailability uni

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumContainers` | int | 2 | ❌ 10 | ❌ 10 | 400% | 400% |
| `WaterAvailability` | float | 500 | ❌ 15 | ❌ 15 | 97% | 97% |
| `PowderedPillAvailability` | float | 700 | ❌ 20 | ❌ 20 | 97% | 97% |

### Mismatch

- **`NumContainers`** (gold = 2) → predicted **10** (400% error)
- **`WaterAvailability`** (gold = 500) → predicted **15** (97% error)
- **`PowderedPillAvailability`** (gold = 700) → predicted **20** (97% error)

### Why this is hard

One slot expects a per-unit rate or coefficient and another expects a total/aggregate quantity. Both appear as plain numbers in similar syntactic contexts, and the pipeline assigns the total to the coefficient slot or vice versa, producing a >5× magnitude error.

---

## Example 7 — total vs unit coefficient

- **ID:** `nlp4lp_test_203`
- **Error type:** total vs unit coefficient
- **Schema hit:** ✓
- **Slots:** 3 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> A woman has $100000 to gamble on two sports bets: a basketball tournament, a horse race, and a soccer game. Based on simple analysis, the woman determines her chance of losing her money would be 50% for basketball tournament, 25% for horse race, and 10% for the soccer game. The payout for each dollar put on each bet will be $1.2 for basketball tournament, $0.5 for horse race, and $0.1 for the soccer game. Knowing herself, she limits her average chance of losing her money should be at most 30%. Could you help her determine how much to money to put on each sport bet to maximize her average payout?

### Schema (preview)

> A woman has TotalMoney to gamble on NumBets different sports bets. Each sport bet has a loss probability given by LossProbabilities and a payout per dollar given by Payouts. She limits her average chance of losing her money to at most MaxAverageLossProbability. Determine the allocation of money to e

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `TotalMoney` | float | 100000 | ❌ 0.5 | ❌ 0.5 | 100% | 100% |
| `NumBets` | int | 3 | ❌ 0.1 | ❌ 0.1 | 97% | 97% |
| `MaxAverageLossProbability` | float | 0.3 | ❌ 1.2 | ❌ 1.2 | 90% | 90% |

### Mismatch

- **`TotalMoney`** (gold = 100000) → predicted **0.5** (100% error)
- **`NumBets`** (gold = 3) → predicted **0.1** (97% error)
- **`MaxAverageLossProbability`** (gold = 0.3) → predicted **1.2** (90% error)

### Why this is hard

One slot expects a per-unit rate or coefficient and another expects a total/aggregate quantity. Both appear as plain numbers in similar syntactic contexts, and the pipeline assigns the total to the coefficient slot or vice versa, producing a >5× magnitude error.

---

## Example 8 — total vs unit coefficient

- **ID:** `nlp4lp_test_132`
- **Error type:** total vs unit coefficient
- **Schema hit:** ✓
- **Slots:** 2 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> A lab has 1000 units of medicinal ingredients to make two pills, a large pill and a small pill. A large pill requires 3 units of medicinal ingredients and 2 units of filler. A small pill requires 2 units of medicinal ingredients and 1 unit of filler. The lab has to make at least 100 large pills. However, since small pills are more popular at least 60% of the total number of pills must be small. How many of each should be made to minimize the total number of filler material needed?

### Schema (preview)

> A lab has TotalMedicinalIngredients units of medicinal ingredients available to produce NumPillTypes different types of pills. Each pill type i requires RequiredMedicinal[i] units of medicinal ingredients and RequiredFiller[i] units of filler. The lab must produce at least MinimumPills[i] units of p

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `TotalMedicinalIngredients` | int | 1000 | ❌ 1 | ❌ 1 | 100% | 100% |
| `NumPillTypes` | int | 2 | ❌ 3 | ❌ 3 | 50% | 50% |

### Mismatch

- **`TotalMedicinalIngredients`** (gold = 1000) → predicted **1** (100% error)
- **`NumPillTypes`** (gold = 2) → predicted **3** (50% error)

### Why this is hard

One slot expects a per-unit rate or coefficient and another expects a total/aggregate quantity. Both appear as plain numbers in similar syntactic contexts, and the pipeline assigns the total to the coefficient slot or vice versa, producing a >5× magnitude error.

---

## Example 9 — total vs unit coefficient

- **ID:** `nlp4lp_test_247`
- **Error type:** total vs unit coefficient
- **Schema hit:** ✓
- **Slots:** 2 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> There are two ways to extract a metal from mined ores. The first way is to use process J and the second is process P. Process J can extract 5 units of metal using 8 units of water and produces 3 units of pollution. Process P can extract 9 units of metal using 6 units of water and produces 5 units of pollution. There can be at most 1500 units of water 1350 units of pollution. How many of each type of processes should be performed to maximize the amount of metal extracted?

### Schema (preview)

> Determine the number of each process to perform in order to maximize the total MetalExtraction, subject to the constraints that the total WaterUsage does not exceed MaxWater and the total PollutionProduction does not exceed MaxPollution.

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `MaxWater` | float | 1500 | ❌ 6 | ❌ 6 | 100% | 100% |
| `MaxPollution` | float | 1350 | ❌ 5 | ❌ 5 | 100% | 100% |

### Mismatch

- **`MaxWater`** (gold = 1500) → predicted **6** (100% error)
- **`MaxPollution`** (gold = 1350) → predicted **5** (100% error)

### Why this is hard

One slot expects a per-unit rate or coefficient and another expects a total/aggregate quantity. Both appear as plain numbers in similar syntactic contexts, and the pipeline assigns the total to the coefficient slot or vice versa, producing a >5× magnitude error.

---

## Example 10 — swapped quantities

- **ID:** `nlp4lp_test_61`
- **Error type:** swapped quantities
- **Schema hit:** ✓
- **Slots:** 6 expected; opt\_repair: 6 filled, 0 exact@5%; max\_weight\_matching: 6 filled, 0 exact@5%

### Query

> A bank can build small and large branches to serve their customers. A small branch can serve 50 customers per day and requires 10 bank tellers. A large branch can serve 100 customers per day and requires 15 bank tellers. The bank has available 200 bank tellers and needs to be able to serve at least 1200 customers per day. How many of each branch size should they build to minimize the total number of branches needed?

### Schema (preview)

> A bank can build small and large branches to serve their customers. A small branch can serve CustomersSmall customers per day and requires TellersSmall bank tellers. A large branch can serve CustomersLarge customers per day and requires TellersLarge bank tellers. The bank has available TotalTellers 

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `CustomersSmall` | float | 50 | ❌ 1200 | ❌ 1200 | 2300% | 2300% |
| `TellersSmall` | float | 10 | ❌ 15 | ❌ 15 | 50% | 50% |
| `CustomersLarge` | float | 100 | ❌ 50 | ❌ 50 | 50% | 50% |
| `TellersLarge` | float | 15 | ❌ 10 | ❌ 10 | 33% | 33% |
| `TotalTellers` | float | 200 | ❌ 100 | ❌ 100 | 50% | 50% |
| `MinCustomers` | float | 1200 | ❌ 200 | ❌ 200 | 83% | 83% |

### Mismatch

- **`CustomersSmall`** (gold = 50) → predicted **1200** (2300% error)
- **`TellersSmall`** (gold = 10) → predicted **15** (50% error)
- **`CustomersLarge`** (gold = 100) → predicted **50** (50% error)
- **`TellersLarge`** (gold = 15) → predicted **10** (33% error)
- **`TotalTellers`** (gold = 200) → predicted **100** (50% error)
- **`MinCustomers`** (gold = 1200) → predicted **200** (83% error)

### Why this is hard

Correct values were extracted but assigned to the wrong slots. Multiple mentions share similar magnitude, token kind, or syntactic context, making it hard to determine which value belongs to which parameter (e.g., cost vs. exposure, two types of machines, two available resources).

---

## Example 11 — swapped quantities

- **ID:** `nlp4lp_test_69`
- **Error type:** swapped quantities
- **Schema hit:** ✓
- **Slots:** 3 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> A fire department employs regular and emergency fire fighters. A regular fire fighter works 10 hours per shift and earns $300. An emergency fire fighter works 6 hours per shift and earns $100. Due to wildfires in the region, the fire department needs at least 300 hours of fire fighter time. If the fire department has a budget of $7000, how many of each should the fire department hire to minimize the total number of fire fighters?

### Schema (preview)

> A fire department employs NumFireFighterTypes different fire fighter types. Each fire fighter type works HoursPerShift hours per shift and incurs CostPerShift cost per shift. The fire department needs at least TotalHoursRequired fire fighter hours and has a budget of Budget. The objective is to mini

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumFireFighterTypes` | int | 2 | ❌ 10 | ❌ 10 | 400% | 400% |
| `TotalHoursRequired` | float | 300 | ❌ 7000 | ❌ 7000 | 2233% | 2233% |
| `Budget` | currency | 7000 | ❌ 300 | ❌ 300 | 96% | 96% |

### Mismatch

- **`NumFireFighterTypes`** (gold = 2) → predicted **10** (400% error)
- **`TotalHoursRequired`** (gold = 300) → predicted **7000** (2233% error)
- **`Budget`** (gold = 7000) → predicted **300** (96% error)

### Why this is hard

Correct values were extracted but assigned to the wrong slots. Multiple mentions share similar magnitude, token kind, or syntactic context, making it hard to determine which value belongs to which parameter (e.g., cost vs. exposure, two types of machines, two available resources).

---

## Example 12 — swapped quantities

- **ID:** `nlp4lp_test_77`
- **Error type:** swapped quantities
- **Schema hit:** ✓
- **Slots:** 5 expected; opt\_repair: 5 filled, 0 exact@5%; max\_weight\_matching: 5 filled, 0 exact@5%

### Query

> A water company sells water in glass and plastic bottles. A glass bottle can hole 500 ml of water while a plastic bottle can hold 750 ml of water. Because most customer prefer plastic bottles, the number of plastic bottles must be at least 3 times the number of glass bottles. However, there must be at least 20 glass bottles. If the company has available 250000 ml of water, how many of each bottle should be made to maximize the total number of bottles?

### Schema (preview)

> A water company sells glass and plastic bottles. Each glass bottle can hold GlassBottleCapacity milliliters of water while each plastic bottle can hold PlasticBottleCapacity milliliters of water. The number of plastic bottles must be at least MinPlasticRatio times the number of glass bottles. There 

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `GlassBottleCapacity` | currency | 500 | ❌ 750 | ❌ 750 | 50% | 50% |
| `PlasticBottleCapacity` | currency | 750 | ❌ 250000 | ❌ 250000 | 33233% | 33233% |
| `MinPlasticRatio` | float | 3 | ❌ 20 | ❌ 20 | 567% | 567% |
| `MinGlassBottles` | float | 20 | ❌ 3 | ❌ 3 | 85% | 85% |
| `TotalWater` | float | 250000 | ❌ 500 | ❌ 500 | 100% | 100% |

### Mismatch

- **`GlassBottleCapacity`** (gold = 500) → predicted **750** (50% error)
- **`PlasticBottleCapacity`** (gold = 750) → predicted **250000** (33233% error)
- **`MinPlasticRatio`** (gold = 3) → predicted **20** (567% error)
- **`MinGlassBottles`** (gold = 20) → predicted **3** (85% error)
- **`TotalWater`** (gold = 250000) → predicted **500** (100% error)

### Why this is hard

Correct values were extracted but assigned to the wrong slots. Multiple mentions share similar magnitude, token kind, or syntactic context, making it hard to determine which value belongs to which parameter (e.g., cost vs. exposure, two types of machines, two available resources).

---

## Example 13 — swapped quantities

- **ID:** `nlp4lp_test_145`
- **Error type:** swapped quantities
- **Schema hit:** ✓
- **Slots:** 3 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> A shoe company supplies shoes to stores via vans and trucks. A van can transport 50 pairs of shoes while a truck can transport 100 pairs of shoes. The company must supply a minimum of 2000 pairs of shoes around the city. Since most stores are small, the number of trucks used cannot exceed the number of vans used.  Find the minimum number of vans that can be used?

### Schema (preview)

> A company supplies at least MinPairsToSupply items using vehicles of two types: vans and trucks. Each van can transport VanCapacity items and each truck can transport TruckCapacity items. The number of trucks used cannot exceed the number of vans used. Determine the minimum number of vans required.

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `VanCapacity` | currency | 50 | ❌ 2000 | ❌ 2000 | 3900% | 3900% |
| `TruckCapacity` | currency | 100 | ❌ 50 | ❌ 50 | 50% | 50% |
| `MinPairsToSupply` | float | 2000 | ❌ 100 | ❌ 100 | 95% | 95% |

### Mismatch

- **`VanCapacity`** (gold = 50) → predicted **2000** (3900% error)
- **`TruckCapacity`** (gold = 100) → predicted **50** (50% error)
- **`MinPairsToSupply`** (gold = 2000) → predicted **100** (95% error)

### Why this is hard

Correct values were extracted but assigned to the wrong slots. Multiple mentions share similar magnitude, token kind, or syntactic context, making it hard to determine which value belongs to which parameter (e.g., cost vs. exposure, two types of machines, two available resources).

---

## Example 14 — swapped quantities

- **ID:** `nlp4lp_test_153`
- **Error type:** swapped quantities
- **Schema hit:** ✓
- **Slots:** 8 expected; opt\_repair: 7 filled, 0 exact@5%; max\_weight\_matching: 7 filled, 0 exact@5%

### Query

> A sand company delivers sand for playgrounds in small and large containers. A small container requires 1 person to unload and can hold 20 units of sand. A large container requires 3 people to unload and can hold 50 units of sand. Since most playgrounds are small, the number of small containers used must be thrice the number of large containers used. In addition, there must be at least 5 small containers and 3 large containers used. If the company has 100 people available, maximize the amount of sand that they can deliver.

### Schema (preview)

> A sand company delivers sand using small containers and large containers. Each small container requires UnloadPersonsSmall persons to unload and can hold CapacitySmall units of sand. Each large container requires UnloadPersonsLarge persons to unload and can hold CapacityLarge units of sand. The numb

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `UnloadPersonsSmall` | int | 1 | ❌ 3 | ❌ 3 | 200% | 200% |
| `CapacitySmall` | currency | 20 | ❌ 100 | ❌ 100 | 400% | 400% |
| `UnloadPersonsLarge` | int | 3 | ❌ 20 | ❌ 20 | 567% | 567% |
| `CapacityLarge` | currency | 50 | ❌ *missing* | ❌ *missing* | — | — |
| `RatioSmallToLargeContainers` | float | 3 | ❌ 50 | ❌ 50 | 1567% | 1567% |
| `MinSmallContainers` | float | 5 | ❌ 3 | ❌ 3 | 40% | 40% |
| `MinLargeContainers` | float | 3 | ❌ 5 | ❌ 5 | 67% | 67% |
| `TotalPersonsAvailable` | int | 100 | ❌ 1 | ❌ 1 | 99% | 99% |

### Mismatch

- **`UnloadPersonsSmall`** (gold = 1) → predicted **3** (200% error)
- **`CapacitySmall`** (gold = 20) → predicted **100** (400% error)
- **`UnloadPersonsLarge`** (gold = 3) → predicted **20** (567% error)
- **`CapacityLarge`** (gold = 50) → **not filled**
- **`RatioSmallToLargeContainers`** (gold = 3) → predicted **50** (1567% error)
- **`MinSmallContainers`** (gold = 5) → predicted **3** (40% error)
- **`MinLargeContainers`** (gold = 3) → predicted **5** (67% error)
- **`TotalPersonsAvailable`** (gold = 100) → predicted **1** (99% error)

### Why this is hard

Correct values were extracted but assigned to the wrong slots. Multiple mentions share similar magnitude, token kind, or syntactic context, making it hard to determine which value belongs to which parameter (e.g., cost vs. exposure, two types of machines, two available resources).

---

## Example 15 — swapped quantities

- **ID:** `nlp4lp_test_175`
- **Error type:** swapped quantities
- **Schema hit:** ✓
- **Slots:** 5 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> A construction company in the tropics uses cows and elephants to carry bricks. A cow can carry 20 bricks on its back while an elephant can carry 50 bricks on its back. To avoid having elephants create too much traffic, the number of elephant cannot exceed the number of cows. In addition, there can be at most twice the number of cows as elephants. If the company needs to transport at least 1000 bricks, find the minimum number of animals, cows and elephants, that can be used..

### Schema (preview)

> A construction company uses cows and elephants to carry bricks. Each cow can carry BrickCapacityCow bricks and each elephant can carry BrickCapacityElephant bricks. The number of elephants cannot exceed MaxElephantsToCowsRatio times the number of cows, and the number of cows cannot exceed MaxCowsToE

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `BrickCapacityCow` | currency | 20 | ❌ *missing* | ❌ *missing* | — | — |
| `BrickCapacityElephant` | currency | 50 | ❌ 1000 | ❌ 1000 | 1900% | 1900% |
| `MaxElephantsToCowsRatio` | float | 1 | ❌ 20 | ❌ 20 | 1900% | 1900% |
| `MaxCowsToElephantsRatio` | float | 2 | ❌ 50 | ❌ 50 | 2400% | 2400% |
| `RequiredBricks` | float | 1000 | ❌ *missing* | ❌ *missing* | — | — |

### Mismatch

- **`BrickCapacityCow`** (gold = 20) → **not filled**
- **`BrickCapacityElephant`** (gold = 50) → predicted **1000** (1900% error)
- **`MaxElephantsToCowsRatio`** (gold = 1) → predicted **20** (1900% error)
- **`MaxCowsToElephantsRatio`** (gold = 2) → predicted **50** (2400% error)
- **`RequiredBricks`** (gold = 1000) → **not filled**

### Why this is hard

Correct values were extracted but assigned to the wrong slots. Multiple mentions share similar magnitude, token kind, or syntactic context, making it hard to determine which value belongs to which parameter (e.g., cost vs. exposure, two types of machines, two available resources).

---

## Example 16 — swapped quantities

- **ID:** `nlp4lp_test_185`
- **Error type:** swapped quantities
- **Schema hit:** ✓
- **Slots:** 3 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> An international shipping company uses large and small ships to transport containers around the world. A large ship can carry 500 containers while a small ship can carry 200 containers. Because most ports are small, the number of large ships cannot exceed the number of small ships. If the company is under contract needs to transport at least 3000 containers, find the minimum number of ships that can be used.

### Schema (preview)

> An international shipping company uses large ships and small ships to transport containers. A large ship can carry LargeShipCapacity containers while a small ship can carry SmallShipCapacity containers. The number of large ships cannot exceed the number of small ships. The company needs to transport

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `LargeShipCapacity` | currency | 500 | ❌ 200 | ❌ 200 | 60% | 60% |
| `SmallShipCapacity` | currency | 200 | ❌ 3000 | ❌ 3000 | 1400% | 1400% |
| `RequiredContainers` | float | 3000 | ❌ 500 | ❌ 500 | 83% | 83% |

### Mismatch

- **`LargeShipCapacity`** (gold = 500) → predicted **200** (60% error)
- **`SmallShipCapacity`** (gold = 200) → predicted **3000** (1400% error)
- **`RequiredContainers`** (gold = 3000) → predicted **500** (83% error)

### Why this is hard

Correct values were extracted but assigned to the wrong slots. Multiple mentions share similar magnitude, token kind, or syntactic context, making it hard to determine which value belongs to which parameter (e.g., cost vs. exposure, two types of machines, two available resources).

---

## Example 17 — min vs max swap

- **ID:** `nlp4lp_test_113`
- **Error type:** min vs max swap
- **Schema hit:** ✓
- **Slots:** 7 expected; opt\_repair: 7 filled, 0 exact@5%; max\_weight\_matching: 7 filled, 0 exact@5%

### Query

> A boy needs to get enough magnesium and zinc in his diet by eating chewable gummies and taking pills. Each gummy contains 3 units of magnesium and 4 units of zinc. Each pill contains 2 units of magnesium and 5 units of zinc. The boy must take at least 10 pills. Since he prefers gummies more, he must eat at least 3 times the amount of gummies as pills. If the boy can consume at most 200 units of magnesium, how many of each should he eat to maximize his zinc intake?

### Schema (preview)

> A subject must determine the number of gummies and the number of pills to consume in order to maximize the total zinc intake, which is calculated as UnitsZincPerGummy multiplied by the number of gummies plus UnitsZincPerPill multiplied by the number of pills. The consumption must satisfy the followi

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `UnitsMagnesiumPerGummy` | float | 3 | ❌ 200 | ❌ 200 | 6567% | 6567% |
| `UnitsZincPerGummy` | float | 4 | ❌ 3 | ❌ 3 | 25% | 25% |
| `UnitsMagnesiumPerPill` | float | 2 | ❌ 4 | ❌ 4 | 100% | 100% |
| `UnitsZincPerPill` | float | 5 | ❌ 2 | ❌ 2 | 60% | 60% |
| `MinimumNumberOfPills` | int | 10 | ❌ 5 | ❌ 5 | 50% | 50% |
| `MinimumGummiesToPillsRatio` | currency | 3 | ❌ 10 | ❌ 10 | 233% | 233% |
| `MaximumUnitsOfMagnesium` | currency | 200 | ❌ 3 | ❌ 3 | 98% | 98% |

### Mismatch

- **`UnitsMagnesiumPerGummy`** (gold = 3) → predicted **200** (6567% error)
- **`UnitsZincPerGummy`** (gold = 4) → predicted **3** (25% error)
- **`UnitsMagnesiumPerPill`** (gold = 2) → predicted **4** (100% error)
- **`UnitsZincPerPill`** (gold = 5) → predicted **2** (60% error)
- **`MinimumNumberOfPills`** (gold = 10) → predicted **5** (50% error)
- **`MinimumGummiesToPillsRatio`** (gold = 3) → predicted **10** (233% error)
- **`MaximumUnitsOfMagnesium`** (gold = 200) → predicted **3** (98% error)

### Why this is hard

Both a lower bound (`min` / `lower`) and an upper bound (`max` / `upper`) appear in the query. The pipeline swaps them, filling the minimum slot with the maximum value or vice versa, because both occur in structurally parallel phrases with similar syntactic contexts.

---

## Example 18 — min vs max swap

- **ID:** `nlp4lp_test_124`
- **Error type:** min vs max swap
- **Schema hit:** ✓
- **Slots:** 5 expected; opt\_repair: 4 filled, 0 exact@5%; max\_weight\_matching: 4 filled, 0 exact@5%

### Query

> Both sulfate and ginger need to be added to a shampoo. One unit of sulfate takes 0.5 minutes to be effective while one unit of ginger takes 0.75 minutes to be effective. The shampoo must contain at least 100 units of sulfates and a total of 400 units of both ingredient. Since too much sulfate can damage the hair, there can be at most twice the amount of sulfate as ginger in the shampoo. How many units of each should be added to the shampoo to minimize the total amount of time it takes for the mixture to be effective? (Note: one must be added before the other)

### Schema (preview)

> Determine the quantities of sulfate and ginger to add to the shampoo such that the number of sulfate units is at least MinSulfateUnits, the combined number of sulfate and ginger units equals TotalUnits, the number of sulfate units does not exceed MaxSulfateToGingerRatio multiplied by the number of g

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `TimePerUnitSulfate` | float | 0.5 | ❌ *missing* | ❌ *missing* | — | — |
| `TimePerUnitGinger` | float | 0.75 | ❌ 0.5 | ❌ 0.5 | 25% | 25% |
| `MinSulfateUnits` | float | 100 | ❌ 0.75 | ❌ 0.75 | 99% | 99% |
| `TotalUnits` | float | 400 | ❌ 100 | ❌ 100 | 75% | 75% |
| `MaxSulfateToGingerRatio` | float | 2 | ❌ 400 | ❌ 400 | 19900% | 19900% |

### Mismatch

- **`TimePerUnitSulfate`** (gold = 0.5) → **not filled**
- **`TimePerUnitGinger`** (gold = 0.75) → predicted **0.5** (25% error)
- **`MinSulfateUnits`** (gold = 100) → predicted **0.75** (99% error)
- **`TotalUnits`** (gold = 400) → predicted **100** (75% error)
- **`MaxSulfateToGingerRatio`** (gold = 2) → predicted **400** (19900% error)

### Why this is hard

Both a lower bound (`min` / `lower`) and an upper bound (`max` / `upper`) appear in the query. The pipeline swaps them, filling the minimum slot with the maximum value or vice versa, because both occur in structurally parallel phrases with similar syntactic contexts.

---

## Example 19 — min vs max swap

- **ID:** `nlp4lp_test_142`
- **Error type:** min vs max swap
- **Schema hit:** ✓
- **Slots:** 6 expected; opt\_repair: 6 filled, 0 exact@5%; max\_weight\_matching: 6 filled, 0 exact@5%

### Query

> A soda company sends bottles of their soda to stores around the city in old and new vans. An old van can take 100 soda bottles while a new van can take 80 soda bottles. An old van produces 50 units of pollution while a new van only produces 30 units of pollution. The company needs to send at least 5000 bottles. In addition, at most 30 new vans can be used. How many of each van should be used to minimize the total amount of pollution produced?

### Schema (preview)

> A soda company uses old and new vans to transport soda bottles to stores. Each old van has a capacity of OldVanCapacity bottles and produces OldVanPollution units of pollution. Each new van has a capacity of NewVanCapacity bottles and produces NewVanPollution units of pollution. The company needs to

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `OldVanCapacity` | currency | 100 | ❌ 30 | ❌ 30 | 70% | 70% |
| `NewVanCapacity` | currency | 80 | ❌ 100 | ❌ 100 | 25% | 25% |
| `OldVanPollution` | float | 50 | ❌ 80 | ❌ 80 | 60% | 60% |
| `NewVanPollution` | float | 30 | ❌ 50 | ❌ 50 | 67% | 67% |
| `MinimumBottles` | currency | 5000 | ❌ 30 | ❌ 30 | 99% | 99% |
| `MaximumNewVans` | currency | 30 | ❌ 5000 | ❌ 5000 | 16567% | 16567% |

### Mismatch

- **`OldVanCapacity`** (gold = 100) → predicted **30** (70% error)
- **`NewVanCapacity`** (gold = 80) → predicted **100** (25% error)
- **`OldVanPollution`** (gold = 50) → predicted **80** (60% error)
- **`NewVanPollution`** (gold = 30) → predicted **50** (67% error)
- **`MinimumBottles`** (gold = 5000) → predicted **30** (99% error)
- **`MaximumNewVans`** (gold = 30) → predicted **5000** (16567% error)

### Why this is hard

Both a lower bound (`min` / `lower`) and an upper bound (`max` / `upper`) appear in the query. The pipeline swaps them, filling the minimum slot with the maximum value or vice versa, because both occur in structurally parallel phrases with similar syntactic contexts.

---

## Example 20 — min vs max swap

- **ID:** `nlp4lp_test_149`
- **Error type:** min vs max swap
- **Schema hit:** ✓
- **Slots:** 7 expected; opt\_repair: 6 filled, 0 exact@5%; max\_weight\_matching: 6 filled, 0 exact@5%

### Query

> A florist transports his flowers to stores in small bouquets and large bouquets. A small bouquet has 5 flowers while a large bouquet has 10 flowers. The florist can transport at most 80 small bouquets and 50 large bouquets. In total, he can transport at most 70 bouquets and he must transport at least 20 large bouquets. Since small bouquets are more popular, he must transport at least twice as many small bouquets as large bouquets. How many of each bouquet should he transport to maximize the total number of flowers that reach the stores?

### Schema (preview)

> A florist transports small and large bouquets, where each small bouquet contains FlowersPerSmallBouquet flowers and each large bouquet contains FlowersPerLargeBouquet flowers. The florist can transport at most MaxSmallBouquets small bouquets and MaxLargeBouquets large bouquets. In total, the florist

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `FlowersPerSmallBouquet` | float | 5 | ❌ 80 | ❌ 80 | 1500% | 1500% |
| `FlowersPerLargeBouquet` | float | 10 | ❌ *missing* | ❌ *missing* | — | — |
| `MaxSmallBouquets` | float | 80 | ❌ 5 | ❌ 5 | 94% | 94% |
| `MaxLargeBouquets` | float | 50 | ❌ 10 | ❌ 10 | 80% | 80% |
| `MaxTotalBouquets` | float | 70 | ❌ 20 | ❌ 20 | 71% | 71% |
| `MinLargeBouquets` | float | 20 | ❌ 50 | ❌ 50 | 150% | 150% |
| `MinSmallToLargeRatio` | float | 2 | ❌ 70 | ❌ 70 | 3400% | 3400% |

### Mismatch

- **`FlowersPerSmallBouquet`** (gold = 5) → predicted **80** (1500% error)
- **`FlowersPerLargeBouquet`** (gold = 10) → **not filled**
- **`MaxSmallBouquets`** (gold = 80) → predicted **5** (94% error)
- **`MaxLargeBouquets`** (gold = 50) → predicted **10** (80% error)
- **`MaxTotalBouquets`** (gold = 70) → predicted **20** (71% error)
- **`MinLargeBouquets`** (gold = 20) → predicted **50** (150% error)
- **`MinSmallToLargeRatio`** (gold = 2) → predicted **70** (3400% error)

### Why this is hard

Both a lower bound (`min` / `lower`) and an upper bound (`max` / `upper`) appear in the query. The pipeline swaps them, filling the minimum slot with the maximum value or vice versa, because both occur in structurally parallel phrases with similar syntactic contexts.

---

## Example 21 — min vs max swap

- **ID:** `nlp4lp_test_165`
- **Error type:** min vs max swap
- **Schema hit:** ✓
- **Slots:** 4 expected; opt\_repair: 3 filled, 0 exact@5%; max\_weight\_matching: 3 filled, 0 exact@5%

### Query

> A jam company sends its product out in small and large jars. A small jar can hold 50 ml of jam while a large jar can hold 200 ml of jam. Most store prefer the smaller size and so the number of large jars cannot exceed the number of small jars. If the company wants to ship at least 100000 ml of jam, find the minimum number of jars that can be used.

### Schema (preview)

> A jam company sends its product out in small and large jars. A small jar can hold SmallJarCapacity milliliters of jam while a large jar can hold LargeJarCapacity milliliters of jam. Most stores prefer the smaller size and so the number of large jars cannot exceed MaxLargeJarsRatio times the number o

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `SmallJarCapacity` | currency | 50 | ❌ 100000 | ❌ 100000 | 199900% | 199900% |
| `LargeJarCapacity` | currency | 200 | ❌ *missing* | ❌ *missing* | — | — |
| `MinJamVolume` | float | 100000 | ❌ 200 | ❌ 200 | 100% | 100% |
| `MaxLargeJarsRatio` | float | 1 | ❌ 50 | ❌ 50 | 4900% | 4900% |

### Mismatch

- **`SmallJarCapacity`** (gold = 50) → predicted **100000** (199900% error)
- **`LargeJarCapacity`** (gold = 200) → **not filled**
- **`MinJamVolume`** (gold = 100000) → predicted **200** (100% error)
- **`MaxLargeJarsRatio`** (gold = 1) → predicted **50** (4900% error)

### Why this is hard

Both a lower bound (`min` / `lower`) and an upper bound (`max` / `upper`) appear in the query. The pipeline swaps them, filling the minimum slot with the maximum value or vice versa, because both occur in structurally parallel phrases with similar syntactic contexts.

---

## Example 22 — percent vs count / incompatible type

- **ID:** `nlp4lp_test_67`
- **Error type:** percent vs count / incompatible type
- **Schema hit:** ✓
- **Slots:** 5 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> A banana company sells their bananas in small and large crates. A small crate can hold 20 bananas while a large crate can hole 50 bananas. Since large crates are more manageable, the number of large crates must be at least twice the number of small crates. However, at least 5 small crates should be used. If the company has available 500 bananas, how many of each crate should the company use to maximize the total number of crates produced?

### Schema (preview)

> A banana company sells their bananas in small and large crates. A small crate can hold CapacitySmallCrate bananas while a large crate can hold CapacityLargeCrate bananas. The number of large crates must be at least LargeToSmallRatio times the number of small crates. At least MinSmallCrates should be

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `CapacitySmallCrate` | percent | 20 | ❌ *missing* | ❌ *missing* | — | — |
| `CapacityLargeCrate` | percent | 50 | ❌ *missing* | ❌ *missing* | — | — |
| `TotalBananas` | float | 500 | ❌ 5 | ❌ 5 | 99% | 99% |
| `MinSmallCrates` | percent | 5 | ❌ *missing* | ❌ *missing* | — | — |
| `LargeToSmallRatio` | float | 2 | ❌ 50 | ❌ 50 | 2400% | 2400% |

### Mismatch

- **`CapacitySmallCrate`** (gold = 20) → **not filled**
- **`CapacityLargeCrate`** (gold = 50) → **not filled**
- **`TotalBananas`** (gold = 500) → predicted **5** (99% error)
- **`MinSmallCrates`** (gold = 5) → **not filled**
- **`LargeToSmallRatio`** (gold = 2) → predicted **50** (2400% error)

### Why this is hard

The query contains both percentage values and integer counts. A slot typed as `percent` either (a) receives a bare integer token that triggers the hard type-incompatibility rule, leaving the slot unfilled, or (b) a percent token is incorrectly routed to an integer count slot.

---

## Example 23 — percent vs count / incompatible type

- **ID:** `nlp4lp_test_100`
- **Error type:** percent vs count / incompatible type
- **Schema hit:** ✓
- **Slots:** 9 expected; opt\_repair: 7 filled, 1 exact@5%; max\_weight\_matching: 7 filled, 1 exact@5%

### Query

> A sailor can eat either a crab cakes or a lobster roll for his meals. He needs to ensure he gets at least 80 units of vitamin A and 100 units of vitamin C. Each crab cake contains 5 units of vitamin A and 7 units of vitamin C. Each lobster roll contains 8 units of vitamin A and 4 units of vitamin C. In addition, since lobster is more expensive, at most 40% of his meals should be lobster rolls. If each crab cake contains 4 units of unsaturated fat and each lobster roll contains 6 units of unsaturated fat, how many of each should he eat to minimize his unsaturated fat intake?

### Schema (preview)

> Determine the quantities of crab cakes and lobster rolls that minimize the total unsaturated fat, where total unsaturated fat is calculated as UnsaturatedFatPerCrabCake multiplied by the number of crab cakes plus UnsaturatedFatPerLobsterRoll multiplied by the number of lobster rolls. Ensure that the

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `MinimumVitaminA` | currency | 80 | ❌ 4 | ❌ 4 | 95% | 95% |
| `MinimumVitaminC` | currency | 100 | ❌ 7 | ❌ 7 | 93% | 93% |
| `VitaminAPerCrabCake` | float | 5 | ❌ 80 | ❌ 80 | 1500% | 1500% |
| `VitaminCPerCrabCake` | float | 7 | ❌ 100 | ❌ 100 | 1329% | 1329% |
| `VitaminAPerLobsterRoll` | float | 8 | ❌ 5 | ❌ 5 | 38% | 38% |
| `VitaminCPerLobsterRoll` | float | 4 | ❌ 6 | ❌ 6 | 50% | 50% |
| `UnsaturatedFatPerCrabCake` | percent | 4 | ❌ *missing* | ❌ *missing* | — | — |
| `UnsaturatedFatPerLobsterRoll` | percent | 6 | ❌ *missing* | ❌ *missing* | — | — |
| `MaximumLobsterFraction` | percent | 0.4 | ✓ 0.4 | ✓ 0.4 | 0% | 0% |

### Mismatch

- **`MinimumVitaminA`** (gold = 80) → predicted **4** (95% error)
- **`MinimumVitaminC`** (gold = 100) → predicted **7** (93% error)
- **`VitaminAPerCrabCake`** (gold = 5) → predicted **80** (1500% error)
- **`VitaminCPerCrabCake`** (gold = 7) → predicted **100** (1329% error)
- **`VitaminAPerLobsterRoll`** (gold = 8) → predicted **5** (38% error)
- **`VitaminCPerLobsterRoll`** (gold = 4) → predicted **6** (50% error)
- **`UnsaturatedFatPerCrabCake`** (gold = 4) → **not filled**
- **`UnsaturatedFatPerLobsterRoll`** (gold = 6) → **not filled**

### Why this is hard

The query contains both percentage values and integer counts. A slot typed as `percent` either (a) receives a bare integer token that triggers the hard type-incompatibility rule, leaving the slot unfilled, or (b) a percent token is incorrectly routed to an integer count slot.

---

## Example 24 — percent vs count / incompatible type

- **ID:** `nlp4lp_test_260`
- **Error type:** percent vs count / incompatible type
- **Schema hit:** ✓
- **Slots:** 9 expected; opt\_repair: 5 filled, 1 exact@5%; max\_weight\_matching: 5 filled, 1 exact@5%

### Query

> A clinical firm has two factories, a northern factory and a western factory, where they make expensive anti-itch injections and topical cream. Every hour, the northern factory makes 800 g of anti-itch injections and 700 g of topical cream. Every hour, the western factory makes 650 g of anti-itch injections and 750 g of topical cream. The northern factory requires 40 units of plastic per hour while the western factory requires 35 units of plastic to manufacture the packaging. The clinical firm has available 60,000 units of plastic. Further, they must make at least 800,000 g of anti-itch injections and 700,000 g of topical cream. How many hours should each factory be run to minimize the total time needed?

### Schema (preview)

> A clinical firm operates two factories, northern and western. The firm decides the number of hours to run each factory. The northern factory produces NorthernFactoryAntiItchRate grams of anti-itch injections and NorthernFactoryTopicalCreamRate grams of topical cream per hour. The western factory pro

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NorthernFactoryAntiItchRate` | percent | 800 | ❌ *missing* | ❌ *missing* | — | — |
| `NorthernFactoryTopicalCreamRate` | percent | 700 | ❌ *missing* | ❌ *missing* | — | — |
| `WesternFactoryAntiItchRate` | percent | 650 | ❌ *missing* | ❌ *missing* | — | — |
| `WesternFactoryTopicalCreamRate` | percent | 750 | ❌ *missing* | ❌ *missing* | — | — |
| `NorthernFactoryPlasticUsage` | float | 40 | ❌ 800 | ❌ 800 | 1900% | 1900% |
| `WesternFactoryPlasticUsage` | float | 35 | ❌ 700 | ❌ 700 | 1900% | 1900% |
| `TotalPlasticAvailable` | float | 60000 | ❌ 35 | ❌ 35 | 100% | 100% |
| `MinimumAntiItchProduction` | currency | 800000 | ✓ 800000 | ✓ 800000 | 0% | 0% |
| `MinimumTopicalCreamProduction` | currency | 700000 | ❌ 60000 | ❌ 60000 | 91% | 91% |

### Mismatch

- **`NorthernFactoryAntiItchRate`** (gold = 800) → **not filled**
- **`NorthernFactoryTopicalCreamRate`** (gold = 700) → **not filled**
- **`WesternFactoryAntiItchRate`** (gold = 650) → **not filled**
- **`WesternFactoryTopicalCreamRate`** (gold = 750) → **not filled**
- **`NorthernFactoryPlasticUsage`** (gold = 40) → predicted **800** (1900% error)
- **`WesternFactoryPlasticUsage`** (gold = 35) → predicted **700** (1900% error)
- **`TotalPlasticAvailable`** (gold = 60000) → predicted **35** (100% error)
- **`MinimumTopicalCreamProduction`** (gold = 700000) → predicted **60000** (91% error)

### Why this is hard

The query contains both percentage values and integer counts. A slot typed as `percent` either (a) receives a bare integer token that triggers the hard type-incompatibility rule, leaving the slot unfilled, or (b) a percent token is incorrectly routed to an integer count slot.

---

## Example 25 — percent vs count / incompatible type

- **ID:** `nlp4lp_test_180`
- **Error type:** percent vs count / incompatible type
- **Schema hit:** ✓
- **Slots:** 7 expected; opt\_repair: 7 filled, 1 exact@5%; max\_weight\_matching: 7 filled, 1 exact@5%

### Query

> A shipping company need to transport packages by either truck or car. A truck can transport 50 packages per trip while a car can transport 30 packages per trip. In addition, a truck uses 20 liters of gas per trip while a car uses 15 liters of gas per trip. There can be at most 5 truck trips made and at least 30% of all the trips must be made by car. The company needs to transport at least 500 packages. How many of each transportation should they use to minimize the total amount of gas consumed?

### Schema (preview)

> A shipping company transports packages using trucks and cars. Each truck transports TruckCapacity packages per trip and uses TruckGas liters of gas per trip. Each car transports CarCapacity packages per trip and uses CarGas liters of gas per trip. The number of truck trips is limited by MaxTruckTrip

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `TruckCapacity` | currency | 50 | ❌ 30 | ❌ 30 | 40% | 40% |
| `CarCapacity` | currency | 30 | ❌ 50 | ❌ 50 | 67% | 67% |
| `TruckGas` | float | 20 | ✓ 20 | ✓ 20 | 0% | 0% |
| `CarGas` | float | 15 | ❌ 500 | ❌ 500 | 3233% | 3233% |
| `MaxTruckTrips` | int | 5 | ❌ 15 | ❌ 15 | 200% | 200% |
| `MinCarTripPercentage` | percent | 30 | ❌ 0.3 | ❌ 0.3 | 99% | 99% |
| `MinTotalPackages` | float | 500 | ❌ 5 | ❌ 5 | 99% | 99% |

### Mismatch

- **`TruckCapacity`** (gold = 50) → predicted **30** (40% error)
- **`CarCapacity`** (gold = 30) → predicted **50** (67% error)
- **`CarGas`** (gold = 15) → predicted **500** (3233% error)
- **`MaxTruckTrips`** (gold = 5) → predicted **15** (200% error)
- **`MinCarTripPercentage`** (gold = 30) → predicted **0.3** (99% error)
- **`MinTotalPackages`** (gold = 500) → predicted **5** (99% error)

### Why this is hard

The query contains both percentage values and integer counts. A slot typed as `percent` either (a) receives a bare integer token that triggers the hard type-incompatibility rule, leaving the slot unfilled, or (b) a percent token is incorrectly routed to an integer count slot.

---

## Example 26 — percent vs count / incompatible type

- **ID:** `nlp4lp_test_5`
- **Error type:** percent vs count / incompatible type
- **Schema hit:** ✓
- **Slots:** 10 expected; opt\_repair: 7 filled, 2 exact@5%; max\_weight\_matching: 7 filled, 2 exact@5%

### Query

> A company is deciding where to promote their product. Some options include z-tube, soorchle engine, and wassa advertisements. The cost for each option and the number of viewers they each attract is given. On z-tube, each ad costs $1000 and attracts 400,000 viewers. On soorchle, each ad costs $200 and attracts 5,000 viewers. On wassa, each ad costs $100 and attracts 3,000 viewers. Soorchle limits the number of advertisements from a single company to fifteen. Moreover, in order to balance the advertising among the three types of media, at most a third of the total number of advertisements should occur on wassa. And at least 5% should occur on z-tube. The weekly advertising budget is $10000. How many advertisements should be run in each of the three types of media to maximize the total audien
> 
> *(query truncated — structured MIP template format)*

### Schema (preview)

> Let x_Z, x_S, and x_W represent the number of advertisements on z-tube, soorchle, and wassa respectively. The objective is to maximize the total audience, which is calculated as ViewersZTube multiplied by x_Z plus ViewersSoorchle multiplied by x_S plus ViewersWassa multiplied by x_W. The constraints

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `CostZTube` | currency | 1000 | ✓ 1000 | ✓ 1000 | 0% | 0% |
| `ViewersZTube` | float | 400000 | ❌ *missing* | ❌ *missing* | — | — |
| `CostSoorchle` | currency | 200 | ❌ 400000 | ❌ 400000 | 199900% | 199900% |
| `ViewersSoorchle` | float | 5000 | ❌ *missing* | ❌ *missing* | — | — |
| `CostWassa` | currency | 100 | ❌ 200 | ❌ 200 | 100% | 100% |
| `ViewersWassa` | float | 3000 | ❌ 100 | ❌ 100 | 97% | 97% |
| `MaxAdsSoorchle` | float | 15 | ❌ 3000 | ❌ 3000 | 19900% | 19900% |
| `MaxFractionWassaAds` | percent | 0.3333 | ❌ *missing* | ❌ *missing* | — | — |
| `MinFractionZTubeAds` | percent | 0.05 | ✓ 0.05 | ✓ 0.05 | 0% | 0% |
| `WeeklyAdvertisingBudget` | currency | 10000 | ❌ 5000 | ❌ 5000 | 50% | 50% |

### Mismatch

- **`ViewersZTube`** (gold = 400000) → **not filled**
- **`CostSoorchle`** (gold = 200) → predicted **400000** (199900% error)
- **`ViewersSoorchle`** (gold = 5000) → **not filled**
- **`CostWassa`** (gold = 100) → predicted **200** (100% error)
- **`ViewersWassa`** (gold = 3000) → predicted **100** (97% error)
- **`MaxAdsSoorchle`** (gold = 15) → predicted **3000** (19900% error)
- **`MaxFractionWassaAds`** (gold = 0.3333) → **not filled**
- **`WeeklyAdvertisingBudget`** (gold = 10000) → predicted **5000** (50% error)

### Why this is hard

The query contains both percentage values and integer counts. A slot typed as `percent` either (a) receives a bare integer token that triggers the hard type-incompatibility rule, leaving the slot unfilled, or (b) a percent token is incorrectly routed to an integer count slot.

---

## Example 27 — percent vs count / incompatible type

- **ID:** `nlp4lp_test_177`
- **Error type:** percent vs count / incompatible type
- **Schema hit:** ✓
- **Slots:** 5 expected; opt\_repair: 5 filled, 1 exact@5%; max\_weight\_matching: 5 filled, 1 exact@5%

### Query

> A factory provides rides for its employees in either taxis or company cars. Each taxi ride can take 2 employees while each company car ride can take 3 employees. Since buying and maintaining cars is expensive, at most 60% of the rides can be company car rides. However, there has to be at least 30 company car rides. If the company needs to transport at least 500 employees, how many rides of each should be done to minimize the total number of taxi rides.

### Schema (preview)

> A factory provides rides for its employees using either taxis or company cars. Each taxi ride can transport EmployeesPerTaxiRide employees, and each company car ride can transport EmployeesPerCompanyCarRide employees. At most MaxCompanyCarRidePercentage of the total rides can be company car rides, a

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `EmployeesPerTaxiRide` | int | 2 | ✓ 2 | ✓ 2 | 0% | 0% |
| `EmployeesPerCompanyCarRide` | int | 3 | ❌ 500 | ❌ 500 | 16567% | 16567% |
| `MaxCompanyCarRidePercentage` | percent | 60 | ❌ 0.6 | ❌ 0.6 | 99% | 99% |
| `MinCompanyCarRides` | float | 30 | ❌ 3 | ❌ 3 | 90% | 90% |
| `MinEmployees` | int | 500 | ❌ 30 | ❌ 30 | 94% | 94% |

### Mismatch

- **`EmployeesPerCompanyCarRide`** (gold = 3) → predicted **500** (16567% error)
- **`MaxCompanyCarRidePercentage`** (gold = 60) → predicted **0.6** (99% error)
- **`MinCompanyCarRides`** (gold = 30) → predicted **3** (90% error)
- **`MinEmployees`** (gold = 500) → predicted **30** (94% error)

### Why this is hard

The query contains both percentage values and integer counts. A slot typed as `percent` either (a) receives a bare integer token that triggers the hard type-incompatibility rule, leaving the slot unfilled, or (b) a percent token is incorrectly routed to an integer count slot.

---

## Example 28 — wrong assignment

- **ID:** `nlp4lp_test_201`
- **Error type:** wrong assignment
- **Schema hit:** ✓
- **Slots:** 2 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> A man takes two supplements to get his daily iron and calcium requirements. A pill of supplement A has 5 units of iron and 10 units of calcium. A pill of supplement B contains 4 units of iron and 15 units of calcium.  The man needs a minimum of 40 units of iron and 50 units of calcium per day. If the cost per pill of supplement A is $2 and the cost per pill of supplement B is  $3, how many of each should he buy to minimize costs?

### Schema (preview)

> A person purchases a number of each of NumSupplements supplement types to meet the minimum requirements for NumNutrients nutrients. Each supplement type i provides NutrientContent[i][j] units of nutrient j and has a cost of CostPerPill[i] per pill. The objective is to minimize the total cost while e

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumSupplements` | int | 2 | ❌ 5 | ❌ 5 | 150% | 150% |
| `NumNutrients` | int | 2 | ❌ 10 | ❌ 10 | 400% | 400% |

### Mismatch

- **`NumSupplements`** (gold = 2) → predicted **5** (150% error)
- **`NumNutrients`** (gold = 2) → predicted **10** (400% error)

### Why this is hard

A wrong numeric token is chosen for a slot. The query does not provide a strong enough contextual signal to uniquely associate each value with its correct optimization parameter.

---

## Example 29 — wrong assignment

- **ID:** `nlp4lp_test_83`
- **Error type:** wrong assignment
- **Schema hit:** ✓
- **Slots:** 2 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> There are two chemical reactions, chemical reaction A and chemical reaction B. Chemical reaction A requires 5 units of rare inert gas and 6 units of treated water to produce 10 units of a rare compound. Chemical reaction B requires 7 units of rare inert gas and 3 units of treater water to produce 8 units of a rare compound. There are 1000 units of the rare inert gas and 800 units of treated water available in the lab. How many reactions of each type should be done to maximize the amount of rare compound produced?

### Schema (preview)

> There are NumReactions different chemical reactions. Each reaction requires ResourceRequirement units of each of the NumResources types of resources and produces ProductionPerReaction units of a rare compound. There are ResourceAvailable units of each resource available in the lab. Determine the num

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumReactions` | int | 2 | ❌ 5 | ❌ 5 | 150% | 150% |
| `NumResources` | int | 2 | ❌ 6 | ❌ 6 | 200% | 200% |

### Mismatch

- **`NumReactions`** (gold = 2) → predicted **5** (150% error)
- **`NumResources`** (gold = 2) → predicted **6** (200% error)

### Why this is hard

A wrong numeric token is chosen for a slot. The query does not provide a strong enough contextual signal to uniquely associate each value with its correct optimization parameter.

---

## Example 30 — wrong assignment

- **ID:** `nlp4lp_test_24`
- **Error type:** wrong assignment
- **Schema hit:** ✓
- **Slots:** 2 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> Sleep inducing medicine and anti-inflammatory medicine is found in two pills, pill A and pill B. One pill A contains 3 units of sleep inducing medicine and 5 units of anti-inflammatory medicine. One pill B contains 6 units of sleep-inducing medicine and 1 unit of anti-inflammatory medicine. The cost per pill for pill A is $4 and the cost per pill for pill B is $5. A patient must consume these two pills to get at least 40 units of sleep-inducing medicine and 50 units of anti-inflammatory medicine. Formulate a LP to minimize the cost for the patient.

### Schema (preview)

> A patient selects non-negative quantities of each of the NumPillTypes pill types. Each pill type provides specific amounts of each of the NumMedicineTypes medicine types as defined by AmountPerPill. The total amount of each medicine must meet or exceed the RequiredAmount. The objective is to minimiz

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumPillTypes` | int | 2 | ❌ 3 | ❌ 3 | 50% | 50% |
| `NumMedicineTypes` | int | 2 | ❌ 5 | ❌ 5 | 150% | 150% |

### Mismatch

- **`NumPillTypes`** (gold = 2) → predicted **3** (50% error)
- **`NumMedicineTypes`** (gold = 2) → predicted **5** (150% error)

### Why this is hard

A wrong numeric token is chosen for a slot. The query does not provide a strong enough contextual signal to uniquely associate each value with its correct optimization parameter.

---

## Example 31 — wrong assignment

- **ID:** `nlp4lp_test_73`
- **Error type:** wrong assignment
- **Schema hit:** ✓
- **Slots:** 2 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> A scientist is conducting two experiments to produce electricity, experiment alpha and experiment beta. In experiment alpha, 3 units of metal and 5 units of acid are required to produce 8 units of electricity. In experiment beta, 5 units of metal and 4 units of acid are required to produced 10 units of electricity. The lab has 800 units of metal and 750 units of acid available. How many of each experiment should the scientist conduct to maximize the total amount of electricity produced?

### Schema (preview)

> A scientist is conducting NumExperiments different experiments to produce electricity. Each experiment i produces ElectricityProduced[i] units of electricity and requires specific amounts of NumResources types of resources as defined by ResourceRequired[j][i]. The laboratory has ResourceAvailable[j]

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumExperiments` | int | 2 | ❌ 3 | ❌ 3 | 50% | 50% |
| `NumResources` | int | 2 | ❌ 5 | ❌ 5 | 150% | 150% |

### Mismatch

- **`NumExperiments`** (gold = 2) → predicted **3** (50% error)
- **`NumResources`** (gold = 2) → predicted **5** (150% error)

### Why this is hard

A wrong numeric token is chosen for a slot. The query does not provide a strong enough contextual signal to uniquely associate each value with its correct optimization parameter.

---

## Example 32 — missing value

- **ID:** `nlp4lp_test_266`
- **Error type:** missing value
- **Schema hit:** ✓
- **Slots:** 9 expected; opt\_repair: 3 filled, 1 exact@5%; max\_weight\_matching: 3 filled, 1 exact@5%

### Query

> A keyboard manufacturer makes mechanical and standard keyboards. Mechanical keyboards are becoming more popular and thus the manufacturer aims to have five times as many mechanical than standard keyboards. A mechanical keyboard costs five units of plastic and two units of solder whereas a standard keyboard costs two units of plastic and one unit of solder. There are still customers that prefer a less noisy alternative. Therefore, there must be at least 30 standard keyboards. If the company has available 1000 units of plastic and 250 units of solder, how many of each type should be manufactured to maximize the total number of keyboards?

### Schema (preview)

> A manufacturer produces two types of keyboards: mechanical and standard. The number of mechanical keyboards should be MechanicalToStandardRatio times the number of standard keyboards. Each mechanical keyboard requires PlasticCostMechanical units of plastic and SolderCostMechanical units of solder, w

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `NumProductTypes` | int | 2 | ❌ *missing* | ❌ *missing* | — | — |
| `PlasticCostMechanical` | currency | 5 | ❌ *missing* | ❌ *missing* | — | — |
| `PlasticCostStandard` | currency | 2 | ❌ *missing* | ❌ *missing* | — | — |
| `SolderCostMechanical` | currency | 2 | ❌ *missing* | ❌ *missing* | — | — |
| `SolderCostStandard` | currency | 1 | ❌ *missing* | ❌ *missing* | — | — |
| `MechanicalToStandardRatio` | float | 5 | ❌ *missing* | ❌ *missing* | — | — |
| `MinimumStandardKeyboards` | currency | 30 | ❌ 1000 | ❌ 1000 | 3233% | 3233% |
| `TotalPlasticAvailable` | float | 1000 | ❌ 30 | ❌ 30 | 97% | 97% |
| `TotalSolderAvailable` | float | 250 | ✓ 250 | ✓ 250 | 0% | 0% |

### Mismatch

- **`NumProductTypes`** (gold = 2) → **not filled**
- **`PlasticCostMechanical`** (gold = 5) → **not filled**
- **`PlasticCostStandard`** (gold = 2) → **not filled**
- **`SolderCostMechanical`** (gold = 2) → **not filled**
- **`SolderCostStandard`** (gold = 1) → **not filled**
- **`MechanicalToStandardRatio`** (gold = 5) → **not filled**
- **`MinimumStandardKeyboards`** (gold = 30) → predicted **1000** (3233% error)
- **`TotalPlasticAvailable`** (gold = 1000) → predicted **30** (97% error)

### Why this is hard

Fewer extractable numeric tokens are present than there are slots to fill. A value may be implicit (expressed as a word like 'half' or 'twice'), subsumed by another slot's assignment, or absent because the value is not stated numerically.

---

## Example 33 — missing value

- **ID:** `nlp4lp_test_168`
- **Error type:** missing value
- **Schema hit:** ✓
- **Slots:** 6 expected; opt\_repair: 3 filled, 1 exact@5%; max\_weight\_matching: 3 filled, 1 exact@5%

### Query

> An industrial tire company delivers large tires for equipment to remote engineering sites either by cargo planes or ultrawide trucks. Each cargo plane can transport 10 tires per trip and costs $1000. Each ultrawide truck can transport 6 tires per trip and costs $700. The company needs to transport at least 200 tires and has available $22000. Because most remote sites don't have proper airports, the number of plane trips cannot exceed the number of ultrawide truck trips. How many trips of each should be done to minimize the total number of trips?

### Schema (preview)

> An industrial tire company transports tires using cargo planes and ultrawide trucks. Each cargo plane trip transports TiresPerPlaneTrip tires and costs CostPerPlaneTrip dollars. Each ultrawide truck trip transports TiresPerTruckTrip tires and costs CostPerTruckTrip dollars. The company needs to tran

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `TiresPerPlaneTrip` | float | 10 | ✓ 10 | ✓ 10 | 0% | 0% |
| `CostPerPlaneTrip` | currency | 1000 | ❌ *missing* | ❌ *missing* | — | — |
| `TiresPerTruckTrip` | float | 6 | ❌ 200 | ❌ 200 | 3233% | 3233% |
| `CostPerTruckTrip` | currency | 700 | ❌ *missing* | ❌ *missing* | — | — |
| `MinTires` | float | 200 | ❌ 6 | ❌ 6 | 97% | 97% |
| `AvailableBudget` | currency | 22000 | ❌ *missing* | ❌ *missing* | — | — |

### Mismatch

- **`CostPerPlaneTrip`** (gold = 1000) → **not filled**
- **`TiresPerTruckTrip`** (gold = 6) → predicted **200** (3233% error)
- **`CostPerTruckTrip`** (gold = 700) → **not filled**
- **`MinTires`** (gold = 200) → predicted **6** (97% error)
- **`AvailableBudget`** (gold = 22000) → **not filled**

### Why this is hard

Fewer extractable numeric tokens are present than there are slots to fill. A value may be implicit (expressed as a word like 'half' or 'twice'), subsumed by another slot's assignment, or absent because the value is not stated numerically.

---

## Example 34 — template query (no values)

- **ID:** `nlp4lp_test_286`
- **Error type:** template query (no values)
- **Schema hit:** ✓
- **Slots:** 6 expected; opt\_repair: 2 filled, 0 exact@5%; max\_weight\_matching: 2 filled, 0 exact@5%

### Query

> PROBLEM TYPE: MILP
> PROBLEM INFO:
> 
> - An engineering factory makes several products on the machines, and the number of machine \var{m} the factory has is \var{num_{m}}.
> - Each product \var{k} yields \var{profit_{k}} to profit (defined as £/unit selling price minus cost of raw materials).
> - The unit production times (hours) product \var{k} requires on machine \var{m} is \var{time_{k, m}}
> - In the present month (January) and several subsequent months, certain machines will be down for maintenance.
> - Each machine \var{m} has to be down for \var{down_{m}} months for maintenance.
> - There are marketing limitations on each product in each month. 
> - The limitation of product \var{k} in month \var{i} is \var{limit_{k, i}}.
> - It is possible to store up to 100 of each product at a time at a cost of \va
> 
> *(query truncated — structured MIP template format)*

### Schema (preview)

> What maintaining, selling, storing, and manufacturing policy should the company pursue in order to maximize profit? The company has M machines, each with a specific downtime Downtime. There are K products, each with an associated profit Profit and a production time on each machine given by Time. Pro

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `M` | float | 5 | ❌ *missing* | ❌ *missing* | — | — |
| `P` | float | 7 | ❌ *missing* | ❌ *missing* | — | — |
| `T` | float | 7 | ❌ *missing* | ❌ *missing* | — | — |
| `StorePrice` | currency | 0.5 | ❌ 100 | ❌ 100 | 9950% | 9950% |
| `KeepQuantity` | float | 100 | ❌ *missing* | ❌ *missing* | — | — |
| `WorkHours` | float | 8 | ❌ 24 | ❌ 24 | 200% | 200% |

### Mismatch

- **`M`** (gold = 5) → **not filled**
- **`P`** (gold = 7) → **not filled**
- **`T`** (gold = 7) → **not filled**
- **`StorePrice`** (gold = 0.5) → predicted **100** (9950% error)
- **`KeepQuantity`** (gold = 100) → **not filled**
- **`WorkHours`** (gold = 8) → predicted **24** (200% error)

### Why this is hard

The query is a structured MIP/LP template that uses `\var{...}` placeholders instead of concrete numbers. No numeric tokens can be extracted, so all slots remain unfilled. This is expected behaviour for this query format.

---

## Example 35 — template query (no values)

- **ID:** `nlp4lp_test_322`
- **Error type:** template query (no values)
- **Schema hit:** ✓
- **Slots:** 6 expected; opt\_repair: 6 filled, 0 exact@5%; max\_weight\_matching: 6 filled, 0 exact@5%

### Query

> PROBLEM TYPE: LP
> 
> PROBLEM INFO:
> 
> - A division of an auto parts manufacturer produces \var{P} different parts using \var{M} different machines.
> - Batch of 100 part \var{p} requires \var{time_{m,p}} hours on machine \var{m}.
> - The division incurs a charge of \var{cost_{m}} per hour for using machine \var{m}.
> - Machine \var{m} has an availability of up to \var{available_{m}} hours per month.
> - The division sells part \var{p} in batches of 100 at price of \var{price_{p}} per batch.
> - The division must produce at least \var{min_batches_{p}} batches of part \var{p} each month to fulfill a contract.
> - Machine \var{1} is being outsourced so that the manufacturer must pay for the labor.
> - The labor costs $\var{standard_cost}/h up to \var{overtime_hour} hours, after which it costs $\var{overtime_cos
> 
> *(query truncated — structured MIP template format)*

### Schema (preview)

> Determine the quantity of batches for each part the manufacturer should produce every month, ensuring all constraints are met, where the manufacturer has P different parts, M machines, TimeRequired matrix indicating the time each machine takes to produce a part, MachineCosts indicating the cost of e

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `M` | float | 3 | ❌ 1 | ❌ 1 | 67% | 67% |
| `P` | float | 4 | ❌ 1 | ❌ 1 | 75% | 75% |
| `StandardCost` | currency | 20 | ❌ 1 | ❌ 1 | 95% | 95% |
| `OvertimeCost` | currency | 30 | ❌ 100 | ❌ 100 | 233% | 233% |
| `OvertimeHour` | float | 400 | ❌ 100 | ❌ 100 | 75% | 75% |
| `MinProfit` | currency | 5000 | ❌ 1 | ❌ 1 | 100% | 100% |

### Mismatch

- **`M`** (gold = 3) → predicted **1** (67% error)
- **`P`** (gold = 4) → predicted **1** (75% error)
- **`StandardCost`** (gold = 20) → predicted **1** (95% error)
- **`OvertimeCost`** (gold = 30) → predicted **100** (233% error)
- **`OvertimeHour`** (gold = 400) → predicted **100** (75% error)
- **`MinProfit`** (gold = 5000) → predicted **1** (100% error)

### Why this is hard

The query is a structured MIP/LP template that uses `\var{...}` placeholders instead of concrete numbers. No numeric tokens can be extracted, so all slots remain unfilled. This is expected behaviour for this query format.

---

## Example 36 — template query (no values)

- **ID:** `nlp4lp_test_279`
- **Error type:** template query (no values)
- **Schema hit:** ✓
- **Slots:** 5 expected; opt\_repair: 5 filled, 0 exact@5%; max\_weight\_matching: 5 filled, 0 exact@5%

### Query

> PROBLEM TYPE: LP
> 
> PROBLEM INFO: 
> 
> - A company produces and sells \var{P} different products. 
> - The demand for each product is unlimited, but the company is constrained by cash availability and machine capacity.
> - Each unit of the \var{i}-th product requires \var{hour_i} machine hours.
> - There are \var{availableHours} machine hours available in the current production period.
> - The production costs are \var{cost_i} per unit of the \var{i}-th product.
> - The selling prices of the \var{i}-th product is \var{price_i} per unit.
> - The available cash is \var{cash}.
> - Furthermore, \var{investRate_i} of the sales revenues from the \var{i}-th product will be made available to finance operations during the current period.
> - \var{investPercentage_i} is a number between 0 and 1
> - The company could incre
> 
> *(query truncated — structured MIP template format)*

### Schema (preview)

> We are aiming at maximizing total net income subject to the Cash availability and machine capacity limitations. The problem parameters include: the initial Cash available, the Hour(s) required to produce each of the P products, the Cost to produce each of the products, the Price at which each produc

### Predicted vs Gold

| Slot | Type | Gold | opt\_repair | max\_weight | Repair err | Match err |
|---|---|---|---|---|---|---|
| `P` | float | 2 | ❌ 1 | ❌ 1 | 50% | 50% |
| `Cash` | float | 3000 | ❌ 1 | ❌ 1 | 100% | 100% |
| `UpgradeHours` | float | 2000 | ❌ 1 | ❌ 1 | 100% | 100% |
| `UpgradeCost` | currency | 400 | ❌ 1 | ❌ 1 | 100% | 100% |
| `AvailableHours` | float | 2000 | ❌ 0 | ❌ 0 | 100% | 100% |

### Mismatch

- **`P`** (gold = 2) → predicted **1** (50% error)
- **`Cash`** (gold = 3000) → predicted **1** (100% error)
- **`UpgradeHours`** (gold = 2000) → predicted **1** (100% error)
- **`UpgradeCost`** (gold = 400) → predicted **1** (100% error)
- **`AvailableHours`** (gold = 2000) → predicted **0** (100% error)

### Why this is hard

The query is a structured MIP/LP template that uses `\var{...}` placeholders instead of concrete numbers. No numeric tokens can be extracted, so all slots remain unfilled. This is expected behaviour for this query format.

---

