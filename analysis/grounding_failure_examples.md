# Grounding Failure Examples

> **Program:** `tfidf + typed_greedy` (TF-IDF schema retrieval + typed greedy slot assignment)  
> — the primary baseline reported in the paper.
> **Variant:** NLP4LP `orig` test split (331 instances).
> **Gold values:** local cache at `results/eswa_revision/00_env/nlp4lp_gold_cache.json`  
> (no HF token required; set `NLP4LP_GOLD_CACHE` to regenerate).
> **Selection:** one or more representative examples drawn from each failure category;
> balanced for diversity, not ranked by worst score.

---

## Summary

- Total test instances: **331**
- All-slots-exact@5% (passing): **2** / 331 (0.6%)
- Schema misses (wrong template retrieved): **30** (9.1%)
- Schema hit but grounding failed: **299** (90.3%)
- Examples in this file: **31** (selected for category diversity)

### Failure category counts

| Category | All failing instances | Selected for file |
|---|---|---|
| wrong schema / retrieval failure | 30 | 3 |
| implicit count — schema slot never stated in query | 55 | 4 |
| total vs per-unit coefficient confusion | 69 | 3 |
| swapped quantities | 102 | 5 |
| min / max (lower / upper bound) swap | 10 | 4 |
| percent vs integer type incompatibility | 5 | 3 |
| missing value — slot left unfilled | 42 | 3 |
| wrong assignment / distractor number | 67 | 4 |
| template or under-specified query (no numeric values) | 9 | 2 |

---

## Category: wrong schema / retrieval failure

## Example 1 — wrong schema / retrieval failure

- **Query ID:** `nlp4lp_test_4`
- **Schema hit:** ✗
- **Retrieved (wrong) schema:** `nlp4lp_test_41`
- **Gold schema:** `nlp4lp_test_4`
- **Slots:** 0 expected, 0 filled, 0/0 exact@5%

**Input problem statement:**

> A store employs senior citizens who earn $500 per week and young adults who earn $750 per week. The store must keep the weekly wage bill below $30000. On any day, the store requires at least 50 workers, of whom at least 10 must be young adults. To ensure the store runs smoothly, the number of young adults should be at least a third the number of senior citizens. Formulate a LP to minimize the wage bill.

**Gold schema template (first 200 chars):**

> Minimize (SeniorWage × NumberOfSeniorCitizens + YoungAdultWage × NumberOfYoungAdults) subject to (SeniorWage × NumberOfSeniorCitizens + YoungAdultWage × NumberOfYoungAdults ≤ MaxWeeklyWageBill), (Numb…

**Wrongly retrieved schema template (first 200 chars):**

> A sandwich company can open NumStoreTypes types of stores. Each store type produces SandwichesPerStoreType sandwiches per day and requires EmployeesPerStoreType employees to operate. The company must …

**Error type / why it failed:** TF-IDF retrieval returned the wrong schema template. All downstream slot-filling is then irrelevant — values may be filled but they belong to the wrong problem structure.

---

## Example 2 — wrong schema / retrieval failure

- **Query ID:** `nlp4lp_test_12`
- **Schema hit:** ✗
- **Retrieved (wrong) schema:** `nlp4lp_test_260`
- **Gold schema:** `nlp4lp_test_12`
- **Slots:** 0 expected, 0 filled, 0/0 exact@5%

**Input problem statement:**

> A souvenir shop makes wooden elephants and tigers with plastic ornaments. Each elephant requires 50 grams of wood and 20 grams of plastic. Each tiger requires 40 grams of wood and 30 grams of plastic. In a week, 5000 grams of wood and 4000 grams of plastic are available. The profit per elephant sold is $5 and the profit per tiger sold is $4. How many of each should be made in order to maximize profit?

**Gold schema template (first 200 chars):**

> A souvenir shop produces NumProducts different products using NumResources different resources. Each product has a profit defined by Profit and requires specific amounts of resources as specified by R…

**Wrongly retrieved schema template (first 200 chars):**

> A clinical firm operates two factories, northern and western. The firm decides the number of hours to run each factory. The northern factory produces NorthernFactoryAntiItchRate grams of anti-itch inj…

**Error type / why it failed:** TF-IDF retrieval returned the wrong schema template. All downstream slot-filling is then irrelevant — values may be filled but they belong to the wrong problem structure.

---

## Example 3 — wrong schema / retrieval failure

- **Query ID:** `nlp4lp_test_24`
- **Schema hit:** ✗
- **Retrieved (wrong) schema:** `nlp4lp_test_122`
- **Gold schema:** `nlp4lp_test_24`
- **Slots:** 0 expected, 0 filled, 0/0 exact@5%

**Input problem statement:**

> Sleep inducing medicine and anti-inflammatory medicine is found in two pills, pill A and pill B. One pill A contains 3 units of sleep inducing medicine and 5 units of anti-inflammatory medicine. One pill B contains 6 units of sleep-inducing medicine and 1 unit of anti-inflammatory medicine. The cost per pill for pill A is $4 and the cost per pill for pill B is $5. A patient must consume these two pills to get at least 40 units of sleep-inducing medicine and 50 units of anti-inflammatory medicine. Formulate a LP to minimize the cost for the patient.

**Gold schema template (first 200 chars):**

> A patient selects non-negative quantities of each of the NumPillTypes pill types. Each pill type provides specific amounts of each of the NumMedicineTypes medicine types as defined by AmountPerPill. T…

**Wrongly retrieved schema template (first 200 chars):**

> 

**Error type / why it failed:** TF-IDF retrieval returned the wrong schema template. All downstream slot-filling is then irrelevant — values may be filled but they belong to the wrong problem structure.

---

## Category: implicit count — schema slot never stated in query

## Example 4 — implicit count — schema slot never stated in query

- **Query ID:** `nlp4lp_test_1`
- **Schema hit:** ✓
- **Slots:** 2 expected, 2 filled, 0/2 exact@5%

**Input problem statement:**

> A breakfast joint makes two different sandwiches: a regular and a special. Both need eggs and bacon. Each regular sandwich requires 2 eggs and 3 slices of bacon. Each special sandwich requires 3 eggs and 5 slices of bacon. The joint has a total of 40 eggs and 70 slices of bacon. It makes a profit of $3 per regular sandwich and a profit of $4 per special sandwich. How many of each sandwich should be made to maximize profit?

**Schema template (first 200 chars):**

> A breakfast joint produces NumSandwichTypes different types of sandwiches using NumIngredients different ingredients. The amount of each ingredient required for each sandwich type is specified by Requ…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `NumSandwichTypes` | int | 2 | ❌ 70 | 3400% |
| `NumIngredients` | int | 2 | ❌ 40 | 1900% |

**Error type / why it failed:** The schema contains a `Num*` parameter that counts how many distinct objects appear (e.g., 2 jar types, 3 product categories). The query never states this count explicitly — it just names the objects by name. The typed-greedy grounder has no numeric token of the correct small-integer value, so it fills the slot with the nearest integer-compatible token (usually a per-unit coefficient), causing a cascade of further mis-assignments down the slot list.

---

## Example 5 — implicit count — schema slot never stated in query

- **Query ID:** `nlp4lp_test_8`
- **Schema hit:** ✓
- **Slots:** 3 expected, 3 filled, 0/3 exact@5%

**Input problem statement:**

> An artisan makes two types of terracotta jars: a thin jar and a stubby jar. Each thin jar requires 50 minutes of shaping time and 90 minutes of baking time. Each stubby jar requires 30 minutes of shaping time and 150 minutes of baking time. Per week, there are 3000 minutes available for shaping and 4000 minutes available for baking. The profit per thin jar is $5 and the profit per stubby jar is $9. How many jars of each type should the artisan make to maximize profit?

**Schema template (first 200 chars):**

> An artisan produces NumJarTypes different types of terracotta jars. Each jar type requires ShapingTimePerType shaping time and BakingTimePerType baking time. Each week, there is a total shaping time a…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `NumJarTypes` | int | 2 | ❌ 150 | 7400% |
| `ShapingTimeAvailable` | float | 3000 | ❌ 90 | 97% |
| `BakingTimeAvailable` | float | 4000 | ❌ 50 | 99% |

**Error type / why it failed:** The schema contains a `Num*` parameter that counts how many distinct objects appear (e.g., 2 jar types, 3 product categories). The query never states this count explicitly — it just names the objects by name. The typed-greedy grounder has no numeric token of the correct small-integer value, so it fills the slot with the nearest integer-compatible token (usually a per-unit coefficient), causing a cascade of further mis-assignments down the slot list.

---

## Example 6 — implicit count — schema slot never stated in query

- **Query ID:** `nlp4lp_test_9`
- **Schema hit:** ✓
- **Slots:** 2 expected, 2 filled, 0/2 exact@5%

**Input problem statement:**

> A grocery store wants to liquidate its stock of 10 apples, 20 bananas, and 80 grapes. Given past experience, the store knows that they can propose a banana-haters package with 6 apples and 30 grapes and that this package will bring a profit of six euros. Similarly, they can prepare a combo package with 5 apples, 6 bananas, and 20 grapes, yielding a profit of seven euros. They know they can sell any quantity of these two packages within the availability of its stock. What quantity of each package, banana-haters packages and combo packages, should the store prepare to maximize net profit?

**Schema template (first 200 chars):**

> A grocery store aims to liquidate its stock of NumItems different items. It has an Available quantity for each item. The store can prepare NumPackages different package types, where each package requi…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `NumItems` | int | 3 | ❌ 80 | 2567% |
| `NumPackages` | int | 2 | ❌ 30 | 1400% |

**Error type / why it failed:** The schema contains a `Num*` parameter that counts how many distinct objects appear (e.g., 2 jar types, 3 product categories). The query never states this count explicitly — it just names the objects by name. The typed-greedy grounder has no numeric token of the correct small-integer value, so it fills the slot with the nearest integer-compatible token (usually a per-unit coefficient), causing a cascade of further mis-assignments down the slot list.

---

## Example 7 — implicit count — schema slot never stated in query

- **Query ID:** `nlp4lp_test_18`
- **Schema hit:** ✓
- **Slots:** 2 expected, 2 filled, 0/2 exact@5%

**Input problem statement:**

> A candy store mixes regular candy and sour candy to prepare two products, regular mix and sour surprise mix. Each kilogram of the regular mix contains 0.8 kg of regular candy and 0.2 kg of sour candy. The profit per kilogram of the regular mix is $3. Each kilogram of the sour surprise mix contains 0.1 kg of regular candy and 0.9 kg of sour candy. The profit per kilogram of the sour surprise mix is $5. The candy store has 80 kg of regular candy and 60 kg of sour candy available. How many kilograms of each type of candy mix should be created to maximize profits?

**Schema template (first 200 chars):**

> A candy store prepares NumMixes different candy mixes using NumCandyTypes different types of candy. Each kilogram of each mix requires specific amounts of each candy type as defined by CompositionRequ…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `NumMixes` | int | 2 | ❌ 80 | 3900% |
| `NumCandyTypes` | int | 2 | ❌ 60 | 2900% |

**Error type / why it failed:** The schema contains a `Num*` parameter that counts how many distinct objects appear (e.g., 2 jar types, 3 product categories). The query never states this count explicitly — it just names the objects by name. The typed-greedy grounder has no numeric token of the correct small-integer value, so it fills the slot with the nearest integer-compatible token (usually a per-unit coefficient), causing a cascade of further mis-assignments down the slot list.

---

## Category: total vs per-unit coefficient confusion

## Example 8 — total vs per-unit coefficient confusion

- **Query ID:** `nlp4lp_test_0`
- **Schema hit:** ✓
- **Slots:** 5 expected, 5 filled, 3/5 exact@5%

**Input problem statement:**

> Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested in condos yields a $0.50 profit and each dollar invested in detached houses yields a $1 profit. A minimum of 20% of all money invested must be in condos, and at least $20000 must be in detached houses. Formulate an LP that can be used to maximize total profit earned from Mrs. Watson's investment.

**Schema template (first 200 chars):**

> Maximize the sum of ProfitPerDollarCondos multiplied by the investment in condos and ProfitPerDollarDetachedHouses multiplied by the investment in detached houses. The total investment must not exceed…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `TotalBudget` | currency | 760000 | ✅ 760000 | 0% |
| `ProfitPerDollarCondos` | currency | 0.5 | ❌ 20000 | 1999950% |
| `ProfitPerDollarDetachedHouses` | currency | 1 | ✅ 1 | 0% |
| `MinimumPercentageCondos` | percent | 0.2 | ✅ 0.2 | 0% |
| `MinimumInvestmentDetachedHouses` | currency | 20000 | ❌ 0.5 | 100% |

**Error type / why it failed:** A per-unit coefficient (profit per item, cost per unit, rate per hour) is swapped with a global resource total (total budget, total available capacity). Both values are numeric and often of compatible types; the greedy schema-order determines which token is consumed first with no semantic guidance.

---

## Example 9 — total vs per-unit coefficient confusion

- **Query ID:** `nlp4lp_test_2`
- **Schema hit:** ✓
- **Slots:** 8 expected, 8 filled, 2/8 exact@5%

**Input problem statement:**

> A cleaning company located in Edmonton wants to get the best exposure possible for promoting their new dishwashing detergent without exceeding their $250,000 advertising budget. To do so, the company decides to spend their money on two forms of advertising: (1) radio ads and (2) social media ads. Each radio ad costs $5,000; each social media ad costs $9,150. The expected exposure, based on industry ratings, is 60,500 viewers for each radio ad. Additionally, the expected exposure for each social media ad is 50,000 viewers. The company decides that at least 15 but no more than 40 radio ads should be ordered, and that at least 35 social media ads should be contracted. How many ads of each type should be run to obtain maximum exposure while staying within the budget?

**Schema template (first 200 chars):**

> A cleaning company aims to maximize exposure for promoting a new product without exceeding AdvertisingBudget. They allocate funds to two advertising methods: radio ads and social media ads. Each radio…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `AdvertisingBudget` | currency | 250000 | ✅ 250000 | 0% |
| `CostRadioAd` | currency | 5000 | ❌ 60500 | 1110% |
| `CostSocialMediaAd` | currency | 9150 | ❌ 50000 | 446% |
| `ExposureRadioAd` | float | 60500 | ❌ 40 | 100% |
| `ExposureSocialMediaAd` | float | 50000 | ❌ 35 | 100% |
| `MinRadioAds` | float | 15 | ✅ 15 | 0% |
| `MaxRadioAds` | float | 40 | ❌ 2 | 95% |
| `MinSocialMediaAds` | float | 35 | ❌ 9150 | 26043% |

**Error type / why it failed:** A per-unit coefficient (profit per item, cost per unit, rate per hour) is swapped with a global resource total (total budget, total available capacity). Both values are numeric and often of compatible types; the greedy schema-order determines which token is consumed first with no semantic guidance.

---

## Example 10 — total vs per-unit coefficient confusion

- **Query ID:** `nlp4lp_test_5`
- **Schema hit:** ✓
- **Slots:** 10 expected, 10 filled, 0/10 exact@5%

**Input problem statement:**

> A company is deciding where to promote their product. Some options include z-tube, soorchle engine, and wassa advertisements. The cost for each option and the number of viewers they each attract is given. On z-tube, each ad costs $1000 and attracts 400,000 viewers. On soorchle, each ad costs $200 and attracts 5,000 viewers. On wassa, each ad costs $100 and attracts 3,000 viewers. Soorchle limits the number of advertisements from a single company to fifteen. Moreover, in order to balance the advertising among the three types of media, at most a third of the total number of advertisements should occur on wassa. And at least 5% should occur on z-tube. The weekly advertising budget is $10000. How many advertisements should be run in each of the three types of media to maximize the total audience?

**Schema template (first 200 chars):**

> Let x_Z, x_S, and x_W represent the number of advertisements on z-tube, soorchle, and wassa respectively. The objective is to maximize the total audience, which is calculated as ViewersZTube multiplie…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `CostZTube` | currency | 1000 | ❌ 400000 | 39900% |
| `ViewersZTube` | float | 400000 | ❌ 15 | 100% |
| `CostSoorchle` | currency | 200 | ❌ 10000 | 4900% |
| `ViewersSoorchle` | float | 5000 | ❌ 3 | 100% |
| `CostWassa` | currency | 100 | ❌ 5000 | 4900% |
| `ViewersWassa` | float | 3000 | ❌ 3 | 100% |
| `MaxAdsSoorchle` | float | 15 | ❌ 3000 | 19900% |
| `MaxFractionWassaAds` | percent | 0.3333 | ❌ 0.05 | 28% |
| `MinFractionZTubeAds` | percent | 0.05 | ❌ 1000 | 99995% |
| `WeeklyAdvertisingBudget` | currency | 10000 | ❌ 200 | 98% |

**Error type / why it failed:** A per-unit coefficient (profit per item, cost per unit, rate per hour) is swapped with a global resource total (total budget, total available capacity). Both values are numeric and often of compatible types; the greedy schema-order determines which token is consumed first with no semantic guidance.

---

## Category: swapped quantities

## Example 11 — swapped quantities

- **Query ID:** `nlp4lp_test_3`
- **Schema hit:** ✓
- **Slots:** 7 expected, 7 filled, 5/7 exact@5%

**Input problem statement:**

> There is 1000 mg of gold available that is needed to make long and short cables. Long cables require 10 mg of gold while short cables require 7 mg of gold. Because of their compact size, at least 5 times the number of short cables are needed than the long cables. In addition, there needs to be at least 10 long cables made. If each long cable sold results in a $12 profit and each short cable sold results in a $5 profit, how many of each type of cable should be made to maximize profit?

**Schema template (first 200 chars):**

> There is TotalGold available to produce long and short cables. Each long cable requires GoldPerLong amount of gold, while each short cable requires GoldPerShort amount of gold. At least MinShortToLong…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `TotalGold` | float | 1000 | ❌ 10 | 99% |
| `GoldPerLong` | float | 10 | ✅ 10 | 0% |
| `GoldPerShort` | float | 7 | ✅ 7 | 0% |
| `MinShortToLongRatio` | float | 5 | ✅ 5 | 0% |
| `MinLongCables` | float | 10 | ❌ 1000 | 9900% |
| `ProfitPerLong` | currency | 12 | ✅ 12 | 0% |
| `ProfitPerShort` | currency | 5 | ✅ 5 | 0% |

**Error type / why it failed:** Two values of similar magnitude and the same numeric type are assigned to the wrong slots. Typed-greedy processes slots in schema order and consumes tokens greedily — when two values share the same token type the first schema slot always wins, regardless of semantic alignment with the query text.

---

## Example 12 — swapped quantities

- **Query ID:** `nlp4lp_test_6`
- **Schema hit:** ✓
- **Slots:** 8 expected, 8 filled, 0/8 exact@5%

**Input problem statement:**

> A chair produced by Elm Furniture yields a profit of $43, while every dresser yields a $52 profit. Each week, 17 gallons of stain and 11 lengths of oak wood are available. Each chair requires 1.4 gallons of stain and 2 lengths of oak wood, while each dresser requires 1.1 gallons of stain and 3 lengths of oak wood. Determine the maximum profit.

**Schema template (first 200 chars):**

> A company produces chairs and dressers. Each chair yields a profit of ProfitPerChair, while each dresser yields a profit of ProfitPerDresser. Each week, AvailableStain gallons of stain and AvailableOa…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `ProfitPerChair` | currency | 43 | ❌ 52 | 21% |
| `ProfitPerDresser` | currency | 52 | ❌ 43 | 17% |
| `AvailableStain` | float | 17 | ❌ 11 | 35% |
| `AvailableOak` | float | 11 | ❌ 3 | 73% |
| `StainPerChair` | float | 1.4 | ❌ 2 | 43% |
| `StainPerDresser` | float | 1.1 | ❌ 1.4 | 27% |
| `OakPerChair` | float | 2 | ❌ 1.1 | 45% |
| `OakPerDresser` | float | 3 | ❌ 17 | 467% |

**Error type / why it failed:** Two values of similar magnitude and the same numeric type are assigned to the wrong slots. Typed-greedy processes slots in schema order and consumes tokens greedily — when two values share the same token type the first schema slot always wins, regardless of semantic alignment with the query text.

---

## Example 13 — swapped quantities

- **Query ID:** `nlp4lp_test_10`
- **Schema hit:** ✓
- **Slots:** 8 expected, 7 filled, 1/7 exact@5%

**Input problem statement:**

> A bakery uses a stand-mixer and a slow bake oven to make bread and cookies. Each machine can run for at most 3000 hours per year. To bake a loaf of bread takes 1 hour in the stand mixer and 3 hours in the oven. A batch of cookies requires 0.5 hours in the mixer and 1 hour in the oven. The profit per loaf of bread is $5 and the profit per batch of cookies is $3. How should the bakery operate to maximize total profit?

**Schema template (first 200 chars):**

> A bakery uses a stand mixer with MixerMaximumHours available operating hours per year and an oven with OvenMaximumHours available operating hours per year. Producing one loaf of bread requires BreadMi…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `MixerMaximumHours` | currency | 3000 | ✅ 3000 | 0% |
| `OvenMaximumHours` | currency | 3000 | ❌ 5 | 100% |
| `BreadMixerTime` | float | 1 | ❌ 3 | 200% |
| `BreadOvenTime` | float | 3 | ❌ 1 | 67% |
| `CookiesMixerTime` | float | 0.5 | ❌ 1 | 50% |
| `CookiesOvenTime` | float | 1 | ❌ 0.5 | 50% |
| `BreadProfit` | currency | 5 | ❌ 3 | 40% |
| `CookiesProfit` | currency | 3 | ❌ (missing) | — |

**Error type / why it failed:** Two values of similar magnitude and the same numeric type are assigned to the wrong slots. Typed-greedy processes slots in schema order and consumes tokens greedily — when two values share the same token type the first schema slot always wins, regardless of semantic alignment with the query text.

---

## Example 14 — swapped quantities

- **Query ID:** `nlp4lp_test_11`
- **Schema hit:** ✓
- **Slots:** 8 expected, 8 filled, 2/8 exact@5%

**Input problem statement:**

> A glass factory makes two types of glass panes: a regular glass pane and a tempered glass pane. Both require time on a heating and cooling machine. Both machines are available for a maximum of 300 minutes per day. It takes 3 minutes in the heating machine and 5 minutes in the cooling machine to make one regular glass pane. It takes 5 minutes in the heating machine and 8 minutes in the cooling machine to make one tempered glass pane. The profit per pane of regular glass is $8 and the profit per pane of tempered glass is $10. How many panes of each glass type should the factory make to maximize profit? What is the maximum profit?

**Schema template (first 200 chars):**

> A glass factory produces Regular and Tempered glass panes. Producing one Regular pane requires HeatingRegular time on the heating machine and CoolingRegular time on the cooling machine. Producing one …

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `MaxHeatingTime` | float | 300 | ✅ 300 | 0% |
| `MaxCoolingTime` | float | 300 | ❌ 8 | 97% |
| `HeatingRegular` | float | 3 | ❌ 5 | 67% |
| `CoolingRegular` | float | 5 | ✅ 5 | 0% |
| `HeatingTempered` | float | 5 | ❌ 3 | 40% |
| `CoolingTempered` | float | 8 | ❌ 2 | 75% |
| `ProfitRegular` | currency | 8 | ❌ 10 | 25% |
| `ProfitTempered` | currency | 10 | ❌ 8 | 20% |

**Error type / why it failed:** Two values of similar magnitude and the same numeric type are assigned to the wrong slots. Typed-greedy processes slots in schema order and consumes tokens greedily — when two values share the same token type the first schema slot always wins, regardless of semantic alignment with the query text.

---

## Example 15 — swapped quantities

- **Query ID:** `nlp4lp_test_13`
- **Schema hit:** ✓
- **Slots:** 13 expected, 12 filled, 4/12 exact@5%

**Input problem statement:**

> An art store makes large and small art pieces. The store has available 100 units of paint, 50 units of glitter, and 70 units of glue. To make a large art piece requires 4 units of paint, 3 units of glitter, and 5 units of glue. To make a small art piece requires 2 units of paint, 1 unit of glitter, and 2 units of glue. The store must make at least 5 units of each large and small art pieces. If the profit per large art piece is $30 and the profit per small art piece is $15, how many of each should be made to maximize profit?

**Schema template (first 200 chars):**

> An art store produces large and small art pieces. Each large art piece requires PaintPerLarge units of paint, GlitterPerLarge units of glitter, and GluePerLarge units of glue. Each small art piece req…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `AvailablePaint` | float | 100 | ✅ 100 | 0% |
| `AvailableGlitter` | float | 50 | ❌ 70 | 40% |
| `AvailableGlue` | float | 70 | ❌ 50 | 29% |
| `PaintPerLarge` | float | 4 | ❌ 5 | 25% |
| `GlitterPerLarge` | float | 3 | ❌ 5 | 67% |
| `GluePerLarge` | float | 5 | ❌ 4 | 20% |
| `PaintPerSmall` | float | 2 | ❌ 3 | 50% |
| `GlitterPerSmall` | float | 1 | ❌ 2 | 100% |
| `GluePerSmall` | float | 2 | ✅ 2 | 0% |
| `ProfitLarge` | currency | 30 | ✅ 30 | 0% |
| `ProfitSmall` | currency | 15 | ✅ 15 | 0% |
| `MinLarge` | float | 5 | ❌ 1 | 80% |
| `MinSmall` | float | 5 | ❌ (missing) | — |

**Error type / why it failed:** Two values of similar magnitude and the same numeric type are assigned to the wrong slots. Typed-greedy processes slots in schema order and consumes tokens greedily — when two values share the same token type the first schema slot always wins, regardless of semantic alignment with the query text.

---

## Category: min / max (lower / upper bound) swap

## Example 16 — min / max (lower / upper bound) swap

- **Query ID:** `nlp4lp_test_26`
- **Schema hit:** ✓
- **Slots:** 8 expected, 7 filled, 3/7 exact@5%

**Input problem statement:**

> A food truck owner can spend at most $20000 on mangos and guavas. A mango costs the food truck owner $5 and a guava costs him $3. Spices are added and each mango is sold for a profit of $3 while each guava is sold for a profit of $4. The owner estimates that at least 100 mangos but at the most 150 are sold each month. He also estimates that the number of guavas sold is at most a third of the mangos sold. How many mangos and guavas should be sold in order to maximize the profit?

**Schema template (first 200 chars):**

> The food truck owner allocates funds up to MaxSpendingBudget for purchasing mangos and guavas, with each mango costing CostMango and each guava costing CostGuava. Each mango sold generates a profit of…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `MaxSpendingBudget` | currency | 20000 | ✅ 20000 | 0% |
| `CostMango` | currency | 5 | ✅ 5 | 0% |
| `CostGuava` | currency | 3 | ❌ 4 | 33% |
| `ProfitMango` | currency | 3 | ✅ 3 | 0% |
| `ProfitGuava` | currency | 4 | ❌ 3 | 25% |
| `MinMangosSold` | float | 100 | ❌ 150 | 50% |
| `MaxMangosSold` | float | 150 | ❌ 100 | 33% |
| `MaxGuavaToMangoRatio` | float | 0.3333 | ❌ (missing) | — |

**Error type / why it failed:** A lower-bound (Min*) and an upper-bound (Max*) slot receive each other's values. Queries often use parallel phrasing ('at least X … at most Y') and both values share the same numeric type. The greedy order of the schema, not the semantic role of each value, determines which token fills which slot.

---

## Example 17 — min / max (lower / upper bound) swap

- **Query ID:** `nlp4lp_test_30`
- **Schema hit:** ✓
- **Slots:** 7 expected, 7 filled, 0/7 exact@5%

**Input problem statement:**

> A flooring company produces engineered hardwood and vinyl planks. Their sales forecasts show an expected demand of at least 20,000 square foot of hardwood and 10,000 square feet of vinyl planks each week. To satisfy a shipping contract, a total of at least 60,000 square feet of flooring much be shipped each week. Due to a labor shortage issue, no more than 50,000 square feet of hardwood and 30,000  square feet of vinyl  can be produced weekly. If a square foot of hardwood flooring yields a profit of $2.5 and a square foot of vinyl planks produces a $3 profit, how many of each type of flooring should be made weekly to maximize the company's profit?

**Schema template (first 200 chars):**

> A flooring company produces two types of products: hardwood and vinyl planks. The weekly production of hardwood must be at least MinimumDemandHardwood and no more than MaxProductionHardwood. Similarly…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `MinimumDemandHardwood` | currency | 20000 | ❌ 60000 | 200% |
| `MinimumDemandVinyl` | currency | 10000 | ❌ 50000 | 400% |
| `MinimumTotalShipping` | currency | 60000 | ❌ 30000 | 50% |
| `MaxProductionHardwood` | float | 50000 | ❌ 20000 | 60% |
| `MaxProductionVinyl` | float | 30000 | ❌ 10000 | 67% |
| `ProfitHardwood` | currency | 2.5 | ❌ 3 | 20% |
| `ProfitVinyl` | currency | 3 | ❌ 2.5 | 17% |

**Error type / why it failed:** A lower-bound (Min*) and an upper-bound (Max*) slot receive each other's values. Queries often use parallel phrasing ('at least X … at most Y') and both values share the same numeric type. The greedy order of the schema, not the semantic role of each value, determines which token fills which slot.

---

## Example 18 — min / max (lower / upper bound) swap

- **Query ID:** `nlp4lp_test_62`
- **Schema hit:** ✓
- **Slots:** 6 expected, 6 filled, 4/6 exact@5%

**Input problem statement:**

> A shipping company can purchase regular and hybrid vans to make deliveries. A regular van can deliver 500 packages per day and produces 200 units of pollutants. A hybrid van can deliver 300 packages per day and produces 100 units of pollutants. Due to a new environmental law, they can produce at most 7000 units of pollutants per day. However, the company needs to be able to deliver at least 20000 packages per day. How many of each type of van should they buy to minimize the total number of vans needed?

**Schema template (first 200 chars):**

> A shipping company can purchase RegularVans and HybridVans to make deliveries. Each RegularVan delivers PackagesDeliveredRegular packages per day and produces PollutantsRegular units of pollutants, wh…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `PackagesDeliveredRegular` | float | 500 | ✅ 500 | 0% |
| `PackagesDeliveredHybrid` | float | 300 | ✅ 300 | 0% |
| `PollutantsRegular` | float | 200 | ✅ 200 | 0% |
| `PollutantsHybrid` | float | 100 | ✅ 100 | 0% |
| `MaxPollutants` | float | 7000 | ❌ 20000 | 186% |
| `MinPackages` | float | 20000 | ❌ 7000 | 65% |

**Error type / why it failed:** A lower-bound (Min*) and an upper-bound (Max*) slot receive each other's values. Queries often use parallel phrasing ('at least X … at most Y') and both values share the same numeric type. The greedy order of the schema, not the semantic role of each value, determines which token fills which slot.

---

## Example 19 — min / max (lower / upper bound) swap

- **Query ID:** `nlp4lp_test_170`
- **Schema hit:** ✓
- **Slots:** 7 expected, 7 filled, 1/7 exact@5%

**Input problem statement:**

> A tropical city full of islands sends mail either by submarine or by boat. A submarine can carry 100 pieces of mail per trip and uses 30 liters of gas. A boat can carry 80 pieces of mail per trip and uses 25 liters of gas. There can be at most 6 submarine trips and a minimum of 50% of the trips must be by boat. If the city needs to transport at least 1000 pieces of mail, how many of each transportation should they use to minimize the total amount of gas used?

**Schema template (first 200 chars):**

> A tropical city sends mail using submarines and boats. Each submarine can carry SubmarineCapacity pieces of mail per trip and consumes SubmarineGasUsage liters of gas per trip. Each boat can carry Boa…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `SubmarineCapacity` | currency | 100 | ❌ 1000 | 900% |
| `SubmarineGasUsage` | float | 30 | ❌ 100 | 233% |
| `BoatCapacity` | currency | 80 | ✅ 80 | 0% |
| `BoatGasUsage` | float | 25 | ❌ 30 | 20% |
| `MaxSubmarineTrips` | int | 6 | ❌ 25 | 317% |
| `MinBoatTripPercentage` | percent | 50 | ❌ 0.5 | 99% |
| `MailRequired` | float | 1000 | ❌ 6 | 99% |

**Error type / why it failed:** A lower-bound (Min*) and an upper-bound (Max*) slot receive each other's values. Queries often use parallel phrasing ('at least X … at most Y') and both values share the same numeric type. The greedy order of the schema, not the semantic role of each value, determines which token fills which slot.

---

## Category: percent vs integer type incompatibility

## Example 20 — percent vs integer type incompatibility

- **Query ID:** `nlp4lp_test_60`
- **Schema hit:** ✓
- **Slots:** 8 expected, 8 filled, 1/8 exact@5%

**Input problem statement:**

> A laundromat can buy two types of washing machines, a top-loading model and a front-loading model. The top-loading model can wash 50 items per day while the front-loading model can wash 75 items per day. The top-loading model consumes 85 kWh per day while the front-loading model consumes 100 kWh per day. The laundromat must be able to wash at least 5000 items per day and has available 7000 kWh per day. Since the top-loading machine are harder to use, at most 40% of the machines can be top-loading. Further, at least 10 machines should be front-loading. How many of each machine should the laundromat buy to minimize the total number of washing machines?

**Schema template (first 200 chars):**

> A laundromat can buy two types of washing machines, a top-loading model and a front-loading model. The top-loading model can wash WashRateTopLoading items per day while the front-loading model can was…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `WashRateTopLoading` | percent | 50 | ❌ 0.4 | 99% |
| `WashRateFrontLoading` | percent | 75 | ❌ 7000 | 9233% |
| `EnergyConsumptionTopLoading` | float | 85 | ❌ 100 | 18% |
| `EnergyConsumptionFrontLoading` | float | 100 | ❌ 85 | 15% |
| `MinItemsPerDay` | int | 5000 | ❌ 75 | 98% |
| `MaxEnergyPerDay` | float | 7000 | ❌ 50 | 99% |
| `MaxFractionTopLoading` | percent | 0.4 | ❌ 5000 | 499960% |
| `MinNumFrontLoading` | int | 10 | ✅ 10 | 0% |

**Error type / why it failed:** A slot typed as `percent` (expected value in [0, 1]) is filled with a bare integer (e.g., 20 instead of 0.20). No percent-typed token (like '20%') appears in the query text; the grounder accepts the integer token because it is the only candidate with an otherwise compatible type.

---

## Example 21 — percent vs integer type incompatibility

- **Query ID:** `nlp4lp_test_100`
- **Schema hit:** ✓
- **Slots:** 9 expected, 9 filled, 1/9 exact@5%

**Input problem statement:**

> A sailor can eat either a crab cakes or a lobster roll for his meals. He needs to ensure he gets at least 80 units of vitamin A and 100 units of vitamin C. Each crab cake contains 5 units of vitamin A and 7 units of vitamin C. Each lobster roll contains 8 units of vitamin A and 4 units of vitamin C. In addition, since lobster is more expensive, at most 40% of his meals should be lobster rolls. If each crab cake contains 4 units of unsaturated fat and each lobster roll contains 6 units of unsaturated fat, how many of each should he eat to minimize his unsaturated fat intake?

**Schema template (first 200 chars):**

> Determine the quantities of crab cakes and lobster rolls that minimize the total unsaturated fat, where total unsaturated fat is calculated as UnsaturatedFatPerCrabCake multiplied by the number of cra…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `MinimumVitaminA` | currency | 80 | ❌ 100 | 25% |
| `MinimumVitaminC` | currency | 100 | ❌ 80 | 20% |
| `VitaminAPerCrabCake` | float | 5 | ❌ 8 | 60% |
| `VitaminCPerCrabCake` | float | 7 | ✅ 7 | 0% |
| `VitaminAPerLobsterRoll` | float | 8 | ❌ 6 | 25% |
| `VitaminCPerLobsterRoll` | float | 4 | ❌ 5 | 25% |
| `UnsaturatedFatPerCrabCake` | percent | 4 | ❌ 0.4 | 90% |
| `UnsaturatedFatPerLobsterRoll` | percent | 6 | ❌ 4 | 33% |
| `MaximumLobsterFraction` | percent | 0.4 | ❌ 4 | 360% |

**Error type / why it failed:** A slot typed as `percent` (expected value in [0, 1]) is filled with a bare integer (e.g., 20 instead of 0.20). No percent-typed token (like '20%') appears in the query text; the grounder accepts the integer token because it is the only candidate with an otherwise compatible type.

---

## Example 22 — percent vs integer type incompatibility

- **Query ID:** `nlp4lp_test_253`
- **Schema hit:** ✓
- **Slots:** 6 expected, 6 filled, 0/6 exact@5%

**Input problem statement:**

> A researcher is outsourcing annotations and has two options: a specialized third-party or a common third-party annotation company. The specialized company can annotate at a rate of 60 images per hour whereas the common company can annotate at a rate of 40 images per hour. However, the specialized company charges $100 per hour and the common company charges $72 per hour. The researcher has deadlines to meet and must complete a dataset of at least 10,000 images. They also have some special images that only the specialized company can annotate. Therefore, at least a third of work must be allocated to the specialized company. How should the researcher distribute the annotations to the two companies to minimize the cost of annotating the whole dataset?

**Schema template (first 200 chars):**

> A researcher must annotate at least MinTotalImages images by distributing the work between a specialized company and a common company. The specialized company annotates at a rate of SpecializedAnnotRa…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `SpecializedAnnotRate` | percent | 60 | ❌ 10000 | 16567% |
| `CommonAnnotRate` | percent | 40 | ❌ 100 | 150% |
| `SpecializedCostPerHour` | currency | 100 | ❌ 72 | 28% |
| `CommonCostPerHour` | currency | 72 | ❌ 60 | 17% |
| `MinTotalImages` | float | 10000 | ❌ 40 | 100% |
| `MinSpecializedFraction` | percent | 0.3333 | ❌ 2 | 167% |

**Error type / why it failed:** A slot typed as `percent` (expected value in [0, 1]) is filled with a bare integer (e.g., 20 instead of 0.20). No percent-typed token (like '20%') appears in the query text; the grounder accepts the integer token because it is the only candidate with an otherwise compatible type.

---

## Category: missing value — slot left unfilled

## Example 23 — missing value — slot left unfilled

- **Query ID:** `nlp4lp_test_14`
- **Schema hit:** ✓
- **Slots:** 5 expected, 4 filled, 3/4 exact@5%

**Input problem statement:**

> My family has decided to invest in real state for the first time. Currently, they have $600,000 to invest, some in apartments and the rest in townhouses. The money invested in apartments must not be greater than $200,000. They have decided that the money invested in apartments must be at least a half as much as that in townhouses.  If the apartments earn 10%, and the townhouses earn 15%, how much money should they invest in each to maximize profit?

**Schema template (first 200 chars):**

> A family has a TotalInvestment to allocate between apartments and townhouses. The investment in apartments must not exceed MaxInvestmentApartments and must be at least MinInvestmentRatio times the inv…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `TotalInvestment` | currency | 600000 | ✅ 600000 | 0% |
| `MaxInvestmentApartments` | currency | 200000 | ✅ 200000 | 0% |
| `MinInvestmentRatio` | currency | 0.5 | ❌ 0.15 | 35% |
| `ReturnRateApartments` | percent | 0.1 | ✅ 0.1 | 0% |
| `ReturnRateTownhouses` | percent | 0.15 | ❌ (missing) | — |

**Error type / why it failed:** One or more slots remain unfilled because no compatible numeric token is available after earlier slots have consumed the candidate pool. This can arise from implicit values, number words ('half', 'double'), or tokens that were eliminated by the type-incompatibility filter.

---

## Example 24 — missing value — slot left unfilled

- **Query ID:** `nlp4lp_test_28`
- **Schema hit:** ✓
- **Slots:** 12 expected, 11 filled, 3/11 exact@5%

**Input problem statement:**

> An ice cream store makes chocolate and vanilla ice cream by the gallon. In a week, they must make at least 5 gallons of each type but at most 10 gallons of chocolate ice cream and at most 8 gallons of vanilla ice cream. It takes 1 hour to produce a gallon of chocolate ice cream and 2 hours to produce a gallon of vanilla ice cream. In a week, 30 hours are available to make ice cream. In addition at least 6 workers are needed with 1 working on the chocolate ice cream and 2 on the vanilla ice cream at any time. If the profit per gallon of chocolate ice cream is $200 and the profit per gallon of vanilla ice cream is $300, how many gallons of each should be made to maximize profit?

**Schema template (first 200 chars):**

> Maximize the total profit, which is ProfitChocolate multiplied by the number of gallons of chocolate ice cream produced plus ProfitVanilla multiplied by the number of gallons of vanilla ice cream prod…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `MinGallonsChocolate` | float | 5 | ❌ 30 | 500% |
| `MinGallonsVanilla` | float | 5 | ❌ 10 | 100% |
| `MaxGallonsChocolate` | float | 10 | ❌ 8 | 20% |
| `MaxGallonsVanilla` | float | 8 | ❌ 6 | 25% |
| `ProductionTimeChocolate` | float | 1 | ❌ 5 | 400% |
| `ProductionTimeVanilla` | float | 2 | ✅ 2 | 0% |
| `TotalProductionHours` | float | 30 | ❌ 2 | 93% |
| `WorkersNeededChocolate` | int | 1 | ✅ 1 | 0% |
| `WorkersNeededVanilla` | int | 2 | ❌ 1 | 50% |
| `MinTotalWorkers` | int | 6 | ❌ 300 | 4900% |
| `ProfitChocolate` | currency | 200 | ✅ 200 | 0% |
| `ProfitVanilla` | currency | 300 | ❌ (missing) | — |

**Error type / why it failed:** One or more slots remain unfilled because no compatible numeric token is available after earlier slots have consumed the candidate pool. This can arise from implicit values, number words ('half', 'double'), or tokens that were eliminated by the type-incompatibility filter.

---

## Example 25 — missing value — slot left unfilled

- **Query ID:** `nlp4lp_test_29`
- **Schema hit:** ✓
- **Slots:** 6 expected, 5 filled, 2/5 exact@5%

**Input problem statement:**

> Mark has 50 acres of land available to grow potatoes and cucumbers that he sells at a farmers' market. He must grow at least 12 acres of potatoes and 15 acres of cucumbers to meet his contract. Mark prefers to grow more cucumbers than potatoes, but he only has enough resources to grow at most twice the amount of cucumbers as potatoes. If the profit per acre of potatoes is $500 and the profit per acre of cucumbers is $650, how many acres of each should he grow to maximize his profit? What is that profit?

**Schema template (first 200 chars):**

> Mark has TotalLandAvailable acres of land to grow potatoes and cucumbers. He must grow at least MinAcresPotatoes acres of potatoes and at least MinAcresCucumbers acres of cucumbers to meet his contrac…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `TotalLandAvailable` | float | 50 | ✅ 50 | 0% |
| `MinAcresPotatoes` | float | 12 | ❌ 15 | 25% |
| `MinAcresCucumbers` | float | 15 | ❌ 12 | 20% |
| `MaxCucumbersPerPotatoesRatio` | float | 2 | ❌ 650 | 32400% |
| `ProfitPerAcrePotatoes` | currency | 500 | ✅ 500 | 0% |
| `ProfitPerAcreCucumbers` | currency | 650 | ❌ (missing) | — |

**Error type / why it failed:** One or more slots remain unfilled because no compatible numeric token is available after earlier slots have consumed the candidate pool. This can arise from implicit values, number words ('half', 'double'), or tokens that were eliminated by the type-incompatibility filter.

---

## Category: wrong assignment / distractor number

## Example 26 — wrong assignment / distractor number

- **Query ID:** `nlp4lp_test_7`
- **Schema hit:** ✓
- **Slots:** 8 expected, 8 filled, 2/8 exact@5%

**Input problem statement:**

> A farmer wants to mix his animal feeds, Feed A and Feed B, in such a way that the mixture will contain a minimum of 30 units of protein and 50 units of fat. Feed A costs $100 per kilogram and contains 10 units of protein and 8 units of fat. Feed B costs $80 per kilogram and contains 7 units of protein and 15 units of fat. Determine the minimum cost of the mixture.

**Schema template (first 200 chars):**

> A farmer wants to mix FeedA and FeedB such that the mixture contains at least MinProtein units of protein and at least MinFat units of fat. FeedA costs CostFeedA per kilogram and provides ProteinFeedA…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `CostFeedA` | currency | 100 | ✅ 100 | 0% |
| `CostFeedB` | currency | 80 | ✅ 80 | 0% |
| `ProteinFeedA` | float | 10 | ❌ 50 | 400% |
| `ProteinFeedB` | float | 7 | ❌ 30 | 329% |
| `FatFeedA` | float | 8 | ❌ 15 | 88% |
| `FatFeedB` | float | 15 | ❌ 10 | 33% |
| `MinProtein` | float | 30 | ❌ 8 | 73% |
| `MinFat` | float | 50 | ❌ 7 | 86% |

**Error type / why it failed:** A distractor number from the query (an index, a year, a unit multiplier, or an irrelevant quantity) is assigned to a slot that should hold a different value. Typed-greedy uses only the token's numeric type as a signal — it has no semantic understanding of which value belongs to which parameter.

---

## Example 27 — wrong assignment / distractor number

- **Query ID:** `nlp4lp_test_19`
- **Schema hit:** ✓
- **Slots:** 4 expected, 4 filled, 3/4 exact@5%

**Input problem statement:**

> A suspicious factory has 100 sq. feet of space. It makes bootleg phones and laptops. Phones require 2 hours of labor and cost $12 for each sq. foot of space allocated for phone production (cost of electricity and equipment). Laptops require 3 hours of labor and cost $15 for each sq. foot of space allocated for laptop production. Phones produce a net revenue of $50 per sq. foot while laptops produce a net revenue of $70 per sq. foot. The factory wants to spend at most $5000 and 2000 hours of labor. What is the optimal factory layout to maximize revenue?

**Schema template (first 200 chars):**

> A factory has TotalSpace square feet of available space. It produces NumberOfProducts different products. Each product requires LaborRequiredPerSqFt labor hours per square foot and costs CostPerSqFt d…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `TotalSpace` | float | 100 | ✅ 100 | 0% |
| `Budget` | currency | 5000 | ✅ 5000 | 0% |
| `LaborHoursAvailable` | float | 2000 | ❌ 3 | 100% |
| `NumberOfProducts` | int | 2 | ✅ 2 | 0% |

**Error type / why it failed:** A distractor number from the query (an index, a year, a unit multiplier, or an irrelevant quantity) is assigned to a slot that should hold a different value. Typed-greedy uses only the token's numeric type as a signal — it has no semantic understanding of which value belongs to which parameter.

---

## Example 28 — wrong assignment / distractor number

- **Query ID:** `nlp4lp_test_23`
- **Schema hit:** ✓
- **Slots:** 3 expected, 3 filled, 1/3 exact@5%

**Input problem statement:**

> Ayse produces a plant growth compound by mixing two types of fertilizer: C and Y. This growth compound must contain at least 5 units of nitrous oxide and 8 units of vitamin mix. Fertilizer C and Y cost $2 and $3 per kg respectively. Fertilizer C contains 1.5 units of nitrous oxide per kg and 3 units of vitamin mix per kg. Fertilizer Y contains 5 units of nitrous oxide per kg and 1 unit of vitamin mix per kg. Determine the minimum cost of Ayse's compound.

**Schema template (first 200 chars):**

> Ayse produces a plant growth compound by mixing NumFertilizers different types of fertilizer. The compound must contain at least RequiredNitrousOxide units of nitrous oxide and at least RequiredVitami…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `NumFertilizers` | int | 2 | ❌ 8 | 300% |
| `RequiredNitrousOxide` | float | 5 | ✅ 5 | 0% |
| `RequiredVitaminMix` | float | 8 | ❌ 5 | 38% |

**Error type / why it failed:** A distractor number from the query (an index, a year, a unit multiplier, or an irrelevant quantity) is assigned to a slot that should hold a different value. Typed-greedy uses only the token's numeric type as a signal — it has no semantic understanding of which value belongs to which parameter.

---

## Example 29 — wrong assignment / distractor number

- **Query ID:** `nlp4lp_test_35`
- **Schema hit:** ✓
- **Slots:** 8 expected, 8 filled, 1/8 exact@5%

**Input problem statement:**

> A man  only eats vegetable and fruits. A serving of vegetables contains 2 units of vitamins and 3 units of minerals. A serving of fruit contains 4 units of vitamins and 1 unit of minerals. He wants to eat at least 20 units of vitamins and 30 units of minerals. If vegetables cost $3 per serving and fruits cost $5 per serving, how many servings of each should he eat to minimize his cost?

**Schema template (first 200 chars):**

> An individual consumes vegetables and fruits. A serving of vegetables contains VegetableVitamins units of vitamins and VegetableMinerals units of minerals. A serving of fruits contains FruitVitamins u…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `VegetableVitamins` | float | 2 | ❌ 30 | 1400% |
| `VegetableMinerals` | float | 3 | ❌ 20 | 567% |
| `FruitVitamins` | float | 4 | ✅ 4 | 0% |
| `FruitMinerals` | float | 1 | ❌ 3 | 200% |
| `MinimumVitamins` | currency | 20 | ❌ 5 | 75% |
| `MinimumMinerals` | currency | 30 | ❌ 3 | 90% |
| `VegetableCost` | currency | 3 | ❌ 2 | 33% |
| `FruitCost` | currency | 5 | ❌ 1 | 80% |

**Error type / why it failed:** A distractor number from the query (an index, a year, a unit multiplier, or an irrelevant quantity) is assigned to a slot that should hold a different value. Typed-greedy uses only the token's numeric type as a signal — it has no semantic understanding of which value belongs to which parameter.

---

## Category: template or under-specified query (no numeric values)

## Example 30 — template or under-specified query (no numeric values)

- **Query ID:** `nlp4lp_test_295`
- **Schema hit:** ✓
- **Slots:** 1 expected, 0 filled, 0/0 exact@5%

**Input problem statement:**

> PROBLEM TYPE: LP
PROBLEM INFO:

- A quantity y is known to depend on another quantity x. A set of corresponding values has been collected for x and y and is presented.
- The \var{k}-th y value takes \var{y_{k}} and the \var{k}-th x value takes \var{x_{k}}.

INPUT FORMAT:

{
    "y": [y_{k} for k = 1,...,K],
    "x": [x_{k} for k = 1,...,K]
}


OBJECTIVE: Fit the ‘best’ straight line y = bx + a to this set of data points. The objective is to minimise the sum of absolute deviations of each observed value of y from the value predicted by the linear relationship.

OUTPUT INFO:

- \var{intercept} represents the intercept of the fitted line
- \var{slope} represents the slope of the fitted line

OUTPUT FORMAT:

{
    "intercept": intercept,
    "slope": slope
}

**Schema template (first 200 chars):**

> Fit the 'best' straight line Y = bX + a to this set of data points. The objective is to minimise the sum of absolute deviations of each observed value of Y from the value predicted by the linear relat…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `K` | float | 19 | ❌ (missing) | — |

**Error type / why it failed:** The query is written as a template or specification with no concrete numeric values (e.g., 'each unit requires X minutes of shaping time'). No tokens are available to fill any slot, so all slots remain empty and the instance is never instantiation-ready.

---

## Example 31 — template or under-specified query (no numeric values)

- **Query ID:** `nlp4lp_test_296`
- **Schema hit:** ✓
- **Slots:** 1 expected, 0 filled, 0/0 exact@5%

**Input problem statement:**

> PROBLEM TYPE: LP
PROBLEM INFO:

- A quantity y is known to depend on another quantity x. A set of corresponding values has been collected for x and y and is presented.
- The \var{k}-th y value takes \var{y_{k}} and the \var{k}-th x value takes \var{x_{k}}.

INPUT FORMAT:

{
    "y": [y_{k} for k = 1,...,K],
    "x": [x_{k} for k = 1,...,K]
}


OBJECTIVE: Fit the ‘best’ straight line y = bx + a where the objective is to minimize the maximum deviation of all the observed values of y from the value predicted by the linear relationship.

OUTPUT INFO:

- \var{intercept} represents the intercept of the fitted line
- \var{slope} represents the slope of the fitted line

OUTPUT FORMAT:

{
    "intercept": intercept,
    "slope": slope
}

**Schema template (first 200 chars):**

> Fit the ‘best’ straight line y = bx + a where the objective is to minimize the maximum deviation of all the K observed values of Y from the value predicted by the linear relationship. The observed val…

**Correct answer vs our program's answer:**

| Slot | Type | Correct answer | Our program | Error |
|---|---|---|---|---|
| `NumObs` | int | 19 | ❌ (missing) | — |

**Error type / why it failed:** The query is written as a template or specification with no concrete numeric values (e.g., 'each unit requires X minutes of shaping time'). No tokens are available to fill any slot, so all slots remain empty and the instance is never instantiation-ready.

---
