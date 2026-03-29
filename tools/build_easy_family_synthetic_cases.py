"""
Synthetic stress-test case generator for the 5 easier error families.

Generates deterministic, reproducible synthetic test cases for:
  1. percent vs integer / float incompatibility
  2. implicit count / number-word / enumeration-derived count
  3. min/max / lower-vs-upper bound confusion
  4. total vs per-unit coefficient confusion
  5. wrong schema / retrieval failure

Usage (from project root):
    python tools/build_easy_family_synthetic_cases.py
    python tools/build_easy_family_synthetic_cases.py --output results/synthetic_easy_families.json

Each output case has:
  - id:               unique identifier
  - family:           one of the 5 family names
  - sub_type:         specific sub-variant
  - query:            natural-language problem statement
  - expected_slots:   dict of slot_name -> expected_value (what a correct grounder should produce)
  - expected_schema:  expected schema family tag (for retrieval check)
  - notes:            explanation of the intended stress

All cases are deterministic (no randomness); re-running produces the same output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Family 1: percent vs integer / float incompatibility
# ---------------------------------------------------------------------------

PERCENT_CASES: list[dict] = [
    {
        "id": "pct_01_digit_percent",
        "family": "percent_vs_integer",
        "sub_type": "digit_with_percent_sign",
        "query": (
            "A factory can produce two products. At least 40% of all production "
            "must be product A. The total production capacity is 1000 units. "
            "Each unit of product A earns $5 profit and each unit of product B "
            "earns $8 profit. Maximize total profit."
        ),
        "expected_slots": {
            "MinPercentProductA": 0.40,
            "TotalCapacity": 1000,
            "ProfitA": 5,
            "ProfitB": 8,
        },
        "expected_schema": "production",
        "notes": (
            "40% should be stored as 0.40 (float fraction), NOT as 40 (integer). "
            "A grounder that treats percent signs naively will emit 40."
        ),
    },
    {
        "id": "pct_02_word_percent",
        "family": "percent_vs_integer",
        "sub_type": "written_word_percent",
        "query": (
            "An investor allocates her budget between stocks and bonds. "
            "At least 30 percent of her $500,000 budget must be in stocks. "
            "Stocks yield 8% annual return and bonds yield 4% annual return. "
            "Maximize total annual return."
        ),
        "expected_slots": {
            "MinPercentStocks": 0.30,
            "TotalBudget": 500000,
            "StockReturn": 0.08,
            "BondReturn": 0.04,
        },
        "expected_schema": "blending",
        "notes": (
            "'30 percent' and '8%' / '4%' should all become fractions in [0,1]. "
            "A naive grounder may store 30 instead of 0.30."
        ),
    },
    {
        "id": "pct_03_fraction_word_half",
        "family": "percent_vs_integer",
        "sub_type": "fraction_word_half",
        "query": (
            "A manufacturer produces widgets in two sizes. "
            "At least half of all widgets produced must be large. "
            "The plant can produce at most 2000 widgets per week. "
            "Large widgets earn $12 each and small widgets earn $7 each. "
            "Maximize weekly profit."
        ),
        "expected_slots": {
            "MinFractionLarge": 0.50,
            "MaxProduction": 2000,
            "ProfitLarge": 12,
            "ProfitSmall": 7,
        },
        "expected_schema": "production",
        "notes": (
            "'half' should be resolved to 0.5 (fraction/percent kind), not 1 or 2. "
        ),
    },
    {
        "id": "pct_04_fraction_word_one_third",
        "family": "percent_vs_integer",
        "sub_type": "fraction_word_one_third",
        "query": (
            "A diet plan mixes two foods: oats and bran. "
            "At least one-third of the mix must be bran by weight. "
            "The total daily intake is 300 grams. "
            "Oats cost $0.02 per gram and bran costs $0.05 per gram. "
            "Minimize total cost."
        ),
        "expected_slots": {
            "MinFractionBran": 1.0 / 3.0,
            "TotalGrams": 300,
            "CostOats": 0.02,
            "CostBran": 0.05,
        },
        "expected_schema": "blending",
        "notes": (
            "'one-third' should resolve to ~0.333 (fraction kind), not 1 or 3."
        ),
    },
    {
        "id": "pct_05_decimal_rate_vs_amount",
        "family": "percent_vs_integer",
        "sub_type": "decimal_rate_vs_amount",
        "query": (
            "A company invests in project X and project Y. "
            "Each dollar in X yields 0.12 return and each dollar in Y yields 0.08 return. "
            "At least 0.25 of the total $1,000,000 must go to project Y. "
            "Maximize total return."
        ),
        "expected_slots": {
            "ReturnX": 0.12,
            "ReturnY": 0.08,
            "MinFractionY": 0.25,
            "TotalBudget": 1_000_000,
        },
        "expected_schema": "blending",
        "notes": (
            "0.12 and 0.08 are per-unit rates (percent kind); "
            "0.25 is also a fraction. All three are NOT integers or amounts."
        ),
    },
]

# ---------------------------------------------------------------------------
# Family 2: implicit count / number-word / enumeration-derived count
# ---------------------------------------------------------------------------

COUNT_CASES: list[dict] = [
    {
        "id": "cnt_01_number_word_two",
        "family": "implicit_count",
        "sub_type": "number_word_in_text",
        "query": (
            "A bakery makes two types of bread: white and whole wheat. "
            "Each loaf of white bread requires 0.5 kg of flour and earns $2 profit. "
            "Each loaf of whole wheat requires 0.7 kg of flour and earns $3 profit. "
            "The bakery has 100 kg of flour available daily. "
            "How many loaves of each type should be baked to maximize profit?"
        ),
        "expected_slots": {
            "NumBreadTypes": 2,
            "FlourPerWhite": 0.5,
            "FlourPerWholeWheat": 0.7,
            "ProfitWhite": 2,
            "ProfitWholeWheat": 3,
            "TotalFlour": 100,
        },
        "expected_schema": "production",
        "notes": (
            "'two types' should produce NumBreadTypes=2. "
            "A grounder without number-word detection won't fill NumBreadTypes."
        ),
    },
    {
        "id": "cnt_02_enumeration_three_items",
        "family": "implicit_count",
        "sub_type": "enumeration_list",
        "query": (
            "A store sells apples, bananas, and grapes. "
            "Each kg of apples sells for $3, each kg of bananas for $2, "
            "and each kg of grapes for $5. "
            "The store has 50 kg of apples, 80 kg of bananas, and 30 kg of grapes. "
            "How many kg of each fruit should be sold to maximize revenue?"
        ),
        "expected_slots": {
            "NumFruits": 3,
            "PriceApples": 3,
            "PriceBananas": 2,
            "PriceGrapes": 5,
            "SupplyApples": 50,
            "SupplyBananas": 80,
            "SupplyGrapes": 30,
        },
        "expected_schema": "production",
        "notes": (
            "The enumeration 'apples, bananas, and grapes' implies NumFruits=3. "
            "A grounder must count items in the list."
        ),
    },
    {
        "id": "cnt_03_three_resources",
        "family": "implicit_count",
        "sub_type": "implicit_resource_count",
        "query": (
            "A furniture factory produces chairs and tables. "
            "Each chair requires 2 hours of carpentry, 1 hour of painting, and 0.5 kg of wood. "
            "Each table requires 4 hours of carpentry, 2 hours of painting, and 1 kg of wood. "
            "Available: 240 hours of carpentry, 100 hours of painting, 60 kg of wood. "
            "Each chair sells for $80 and each table for $180. Maximize profit."
        ),
        "expected_slots": {
            "NumProducts": 2,
            "NumResources": 3,
            "ProfitChair": 80,
            "ProfitTable": 180,
        },
        "expected_schema": "production",
        "notes": (
            "NumResources=3 (carpentry, painting, wood) is derived from the enumeration. "
            "This count is implicit — never directly stated as a number."
        ),
    },
    {
        "id": "cnt_04_explicit_number_word_five",
        "family": "implicit_count",
        "sub_type": "explicit_number_word",
        "query": (
            "A logistics company has five trucks available for deliveries. "
            "Each truck can carry at most 10 tons. "
            "There are three delivery locations requiring 4, 6, and 8 tons. "
            "Each delivery trip costs $200. Minimize total delivery cost."
        ),
        "expected_slots": {
            "NumTrucks": 5,
            "TruckCapacity": 10,
            "NumLocations": 3,
        },
        "expected_schema": "transportation",
        "notes": (
            "'five trucks' → NumTrucks=5 (number-word to integer). "
            "'three delivery locations' → NumLocations=3."
        ),
    },
]

# ---------------------------------------------------------------------------
# Family 3: min/max / lower-vs-upper bound confusion
# ---------------------------------------------------------------------------

MINMAX_CASES: list[dict] = [
    {
        "id": "mm_01_at_least",
        "family": "minmax_bound",
        "sub_type": "at_least_lower_bound",
        "query": (
            "A company must produce at least 200 units of product A per week "
            "and at most 500 units of product B per week. "
            "Each unit of A earns $10 profit and each unit of B earns $15 profit. "
            "Total weekly capacity is 600 units. Maximize profit."
        ),
        "expected_slots": {
            "MinProductA": 200,
            "MaxProductB": 500,
            "ProfitA": 10,
            "ProfitB": 15,
            "TotalCapacity": 600,
        },
        "expected_schema": "production",
        "notes": (
            "'at least 200' → LowerBound=200; 'at most 500' → UpperBound=500. "
            "Swapping these would invert the constraint direction."
        ),
    },
    {
        "id": "mm_02_no_more_no_fewer",
        "family": "minmax_bound",
        "sub_type": "no_more_no_fewer",
        "query": (
            "A hospital needs no fewer than 10 nurses and no more than 25 nurses "
            "on any given shift. The daily budget for nursing staff is $8000. "
            "Senior nurses cost $400 per shift and junior nurses cost $280 per shift. "
            "Minimize total staffing cost while meeting minimum staffing requirements."
        ),
        "expected_slots": {
            "MinNurses": 10,
            "MaxNurses": 25,
            "Budget": 8000,
            "CostSenior": 400,
            "CostJunior": 280,
        },
        "expected_schema": "production",
        "notes": (
            "'no fewer than 10' → LowerBound=10; 'no more than 25' → UpperBound=25. "
            "Confusable because 'no fewer' is a lower bound, 'no more' is an upper bound."
        ),
    },
    {
        "id": "mm_03_between_range",
        "family": "minmax_bound",
        "sub_type": "between_range",
        "query": (
            "An investor wants to allocate between $50,000 and $200,000 to stocks "
            "and between $30,000 and $150,000 to bonds. "
            "Stocks yield 10% annually and bonds yield 5% annually. "
            "The total investment must not exceed $300,000. Maximize annual return."
        ),
        "expected_slots": {
            "MinStocks": 50000,
            "MaxStocks": 200000,
            "MinBonds": 30000,
            "MaxBonds": 150000,
            "StockReturn": 0.10,
            "BondReturn": 0.05,
            "TotalBudget": 300000,
        },
        "expected_schema": "blending",
        "notes": (
            "Range constraints: both lower and upper bounds must be correctly assigned. "
            "The grounder must not swap min/max."
        ),
    },
    {
        "id": "mm_04_lower_upper_reversed_wording",
        "family": "minmax_bound",
        "sub_type": "reversed_wording",
        "query": (
            "The warehouse can hold a maximum of 500 boxes and a minimum of 50 boxes. "
            "Each box of type X weighs 5 kg and each box of type Y weighs 3 kg. "
            "The total weight limit is 2000 kg. "
            "Type X sells for $30 per box and type Y for $20 per box. "
            "Maximize revenue."
        ),
        "expected_slots": {
            "MaxCapacity": 500,
            "MinCapacity": 50,
            "WeightX": 5,
            "WeightY": 3,
            "WeightLimit": 2000,
            "PriceX": 30,
            "PriceY": 20,
        },
        "expected_schema": "production",
        "notes": (
            "'maximum of 500' = upper bound; 'minimum of 50' = lower bound. "
            "Stated in the order max-then-min to test ordering sensitivity."
        ),
    },
]

# ---------------------------------------------------------------------------
# Family 4: total vs per-unit coefficient confusion
# ---------------------------------------------------------------------------

TOTAL_VS_PERUNIT_CASES: list[dict] = [
    {
        "id": "tpu_01_total_budget_vs_cost_per_unit",
        "family": "total_vs_perunit",
        "sub_type": "total_budget_vs_cost_per_unit",
        "query": (
            "A company has a total advertising budget of $120,000. "
            "Each TV ad costs $15,000 and reaches 500,000 viewers. "
            "Each radio ad costs $3,000 and reaches 80,000 viewers. "
            "Maximize total viewership without exceeding the budget."
        ),
        "expected_slots": {
            "TotalBudget": 120000,
            "CostPerTV": 15000,
            "CostPerRadio": 3000,
            "ViewersPerTV": 500000,
            "ViewersPerRadio": 80000,
        },
        "expected_schema": "production",
        "notes": (
            "TotalBudget=120000 is a total resource; "
            "CostPerTV=15000 is a per-unit coefficient. "
            "Swapping these would be the total-vs-per-unit confusion."
        ),
    },
    {
        "id": "tpu_02_available_hours_vs_hours_per_product",
        "family": "total_vs_perunit",
        "sub_type": "available_hours_vs_hours_per_product",
        "query": (
            "A factory produces two products. "
            "Product A requires 3 hours of labor per unit and product B requires 5 hours per unit. "
            "The factory has 480 hours of labor available per week. "
            "Product A earns $40 profit per unit and product B earns $60 profit per unit. "
            "Maximize weekly profit."
        ),
        "expected_slots": {
            "LaborPerA": 3,
            "LaborPerB": 5,
            "TotalLaborHours": 480,
            "ProfitPerA": 40,
            "ProfitPerB": 60,
        },
        "expected_schema": "production",
        "notes": (
            "TotalLaborHours=480 is total availability; "
            "LaborPerA=3 and LaborPerB=5 are per-unit coefficients. "
            "Grounder must not assign 480 to a per-unit slot."
        ),
    },
    {
        "id": "tpu_03_total_demand_vs_profit_per_unit",
        "family": "total_vs_perunit",
        "sub_type": "total_demand_vs_profit_per_unit",
        "query": (
            "A restaurant serves lunch and dinner. "
            "Total daily demand is at most 200 meals. "
            "Lunch meals earn $8 each and dinner meals earn $15 each. "
            "At least 50 lunch meals and at least 30 dinner meals must be served. "
            "Maximize total daily revenue."
        ),
        "expected_slots": {
            "TotalDemand": 200,
            "ProfitLunch": 8,
            "ProfitDinner": 15,
            "MinLunch": 50,
            "MinDinner": 30,
        },
        "expected_schema": "production",
        "notes": (
            "TotalDemand=200 is total capacity; ProfitLunch=8 and ProfitDinner=15 "
            "are per-unit coefficients. MinLunch and MinDinner are lower bounds."
        ),
    },
    {
        "id": "tpu_04_supply_total_vs_per_unit_rate",
        "family": "total_vs_perunit",
        "sub_type": "supply_total_vs_per_unit_rate",
        "query": (
            "A pharmaceutical company produces two drugs using two raw materials. "
            "The total supply of material X is 600 kg and material Y is 400 kg. "
            "Drug A requires 2 kg of X and 1 kg of Y per batch. "
            "Drug B requires 1 kg of X and 3 kg of Y per batch. "
            "Drug A earns $500 per batch and drug B earns $700 per batch. "
            "Maximize total profit."
        ),
        "expected_slots": {
            "SupplyX": 600,
            "SupplyY": 400,
            "XPerA": 2,
            "YPerA": 1,
            "XPerB": 1,
            "YPerB": 3,
            "ProfitA": 500,
            "ProfitB": 700,
        },
        "expected_schema": "production",
        "notes": (
            "SupplyX=600 and SupplyY=400 are total resources; "
            "XPerA=2 is a per-unit resource coefficient. Mixing these is the family error."
        ),
    },
]

# ---------------------------------------------------------------------------
# Family 5: wrong schema / retrieval failure
# ---------------------------------------------------------------------------

RETRIEVAL_CASES: list[dict] = [
    {
        "id": "ret_01_short_sparse_knapsack",
        "family": "retrieval_failure",
        "sub_type": "short_sparse_query",
        "query": "knapsack",
        "expected_slots": {},
        "expected_schema": "knapsack",
        "notes": (
            "Single-word query 'knapsack': should trigger knapsack schema retrieval. "
            "Tests short-query expansion benefit."
        ),
    },
    {
        "id": "ret_02_short_sparse_tsp",
        "family": "retrieval_failure",
        "sub_type": "short_sparse_query",
        "query": "TSP ILP",
        "expected_slots": {},
        "expected_schema": "routing",
        "notes": (
            "Two-word abbreviation query: should retrieve TSP / vehicle routing schema. "
            "Tests short-query expansion for abbreviations."
        ),
    },
    {
        "id": "ret_03_synonym_mismatch_diet",
        "family": "retrieval_failure",
        "sub_type": "synonym_mismatch",
        "query": (
            "Determine the optimal mixture of feed ingredients to satisfy "
            "nutritional requirements at minimum cost."
        ),
        "expected_slots": {},
        "expected_schema": "blending",
        "notes": (
            "'feed ingredients' and 'nutritional requirements' are synonyms for "
            "the blending/diet problem. Tests alias enrichment."
        ),
    },
    {
        "id": "ret_04_confusable_blending_vs_production",
        "family": "retrieval_failure",
        "sub_type": "confusable_schema",
        "query": (
            "A factory blends three raw materials to produce an alloy. "
            "The alloy must contain at least 20% copper, at most 50% zinc, "
            "and the remaining portion can be aluminum. "
            "Copper costs $5/kg, zinc $3/kg, aluminum $2/kg. "
            "Minimize cost to produce 1 tonne of alloy."
        ),
        "expected_slots": {
            "MinCopperFraction": 0.20,
            "MaxZincFraction": 0.50,
            "CostCopper": 5,
            "CostZinc": 3,
            "CostAluminum": 2,
            "TotalAlloy": 1000,
        },
        "expected_schema": "blending",
        "notes": (
            "Query mentions 'factory' and 'produce' which are production cues, "
            "but the problem is actually blending (mixture proportions). "
            "Tests confusable-schema discrimination."
        ),
    },
    {
        "id": "ret_05_alias_sensitive_assignment",
        "family": "retrieval_failure",
        "sub_type": "alias_sensitive",
        "query": (
            "Assign workers to tasks in a one-to-one matching to minimize total cost. "
            "Worker 1 costs $10 for task A, $15 for task B. "
            "Worker 2 costs $12 for task A, $8 for task B."
        ),
        "expected_slots": {},
        "expected_schema": "assignment",
        "notes": (
            "'one-to-one matching' is an alias for the assignment problem. "
            "Tests alias-enriched retrieval."
        ),
    },
    {
        "id": "ret_06_top2_ambiguity",
        "family": "retrieval_failure",
        "sub_type": "top2_ambiguity",
        "query": (
            "A company ships products from warehouses to customers to minimize transportation cost."
        ),
        "expected_slots": {},
        "expected_schema": "transportation",
        "notes": (
            "This query could plausibly match both transportation and facility location. "
            "Tests ambiguity detection: top-2 margin should be flagged if close."
        ),
    },
]

# ---------------------------------------------------------------------------
# Master collection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Final-pass additional cases (targeted at residual subpatterns)
# ---------------------------------------------------------------------------

FINAL_PASS_TOTAL_VS_PERUNIT: list[dict] = [
    {
        "id": "fp_tpu_01_overall_budget",
        "family": "total_vs_perunit",
        "sub_type": "overall_total_cue",
        "query": (
            "The overall budget is 100000. "
            "Each unit generates a profit of 25."
        ),
        "expected_slots": {
            "TotalBudget": 100000,
            "ProfitPerUnit": 25,
        },
        "expected_schema": "production",
        "notes": (
            "'overall budget' uses the new 'overall' total-left cue. "
            "Grounder must not assign 100000 to ProfitPerUnit."
        ),
    },
    {
        "id": "fp_tpu_02_in_total_hours",
        "family": "total_vs_perunit",
        "sub_type": "in_total_phrase",
        "query": (
            "There are 2000 hours in total available. "
            "Each product requires 5 hours to produce."
        ),
        "expected_slots": {
            "TotalHours": 2000,
            "HoursPerProduct": 5,
        },
        "expected_schema": "production",
        "notes": (
            "'in total' phrase triggers total-like on 2000. "
            "5 is per-unit (requires verb)."
        ),
    },
    {
        "id": "fp_tpu_03_generates_per_unit",
        "family": "total_vs_perunit",
        "sub_type": "generates_per_unit_verb",
        "query": (
            "The factory has 5000 units in stock. "
            "Each machine generates 12 units per hour."
        ),
        "expected_slots": {
            "TotalInventory": 5000,
            "UnitsPerMachineHour": 12,
        },
        "expected_schema": "production",
        "notes": (
            "'generates' is a new per-unit verb. 'in stock' is a total-right cue. "
            "Grounder must not mix these."
        ),
    },
    {
        "id": "fp_tpu_04_stock_remaining",
        "family": "total_vs_perunit",
        "sub_type": "stock_remaining_cues",
        "query": (
            "There are 300 units remaining in the warehouse. "
            "Each order requires 3 units."
        ),
        "expected_slots": {
            "TotalStock": 300,
            "UnitsPerOrder": 3,
        },
        "expected_schema": "inventory",
        "notes": (
            "'remaining' is a new total-right cue (final pass). "
            "Grounder must assign 300 to TotalStock."
        ),
    },
]

FINAL_PASS_IMPLICIT_COUNT: list[dict] = [
    {
        "id": "fp_cnt_01_three_varieties",
        "family": "implicit_count",
        "sub_type": "variety_count_noun",
        "query": (
            "The company offers three varieties of coffee. "
            "Each variety costs $5 to produce and sells for $12."
        ),
        "expected_slots": {
            "NumVarieties": 3,
            "CostPerVariety": 5,
            "PricePerVariety": 12,
        },
        "expected_schema": "production",
        "notes": (
            "'variety' is a new count-context noun (final pass). "
            "'three varieties' → NumVarieties=3."
        ),
    },
    {
        "id": "fp_cnt_02_two_services",
        "family": "implicit_count",
        "sub_type": "service_count_noun",
        "query": (
            "The firm provides two services: consulting and training. "
            "Consulting costs $200/hour and training costs $150/hour. "
            "Total available hours are 500. Maximize revenue."
        ),
        "expected_slots": {
            "NumServices": 2,
            "CostConsulting": 200,
            "CostTraining": 150,
            "TotalHours": 500,
        },
        "expected_schema": "production",
        "notes": (
            "'two services' uses the new 'service' count-context noun. "
            "Also has enumeration 'consulting and training' → 2."
        ),
    },
    {
        "id": "fp_cnt_03_four_facilities",
        "family": "implicit_count",
        "sub_type": "facility_count_noun",
        "query": (
            "There are four facilities available for production. "
            "Each facility can produce at most 500 units per week."
        ),
        "expected_slots": {
            "NumFacilities": 4,
            "CapacityPerFacility": 500,
        },
        "expected_schema": "production",
        "notes": (
            "'four facilities' uses the new 'facility' count-context noun. "
        ),
    },
]

FINAL_PASS_MINMAX: list[dict] = [
    {
        "id": "fp_mm_01_minimum_of",
        "family": "minmax_bound",
        "sub_type": "minimum_of_phrase",
        "query": (
            "Production must meet a minimum of 100 units per day. "
            "Output cannot exceed 500 units per day."
        ),
        "expected_slots": {
            "MinProduction": 100,
            "MaxProduction": 500,
        },
        "expected_schema": "production",
        "notes": (
            "'minimum of 100' → LowerBound=100 (new pattern). "
            "'cannot exceed 500' → UpperBound=500 (existing pattern)."
        ),
    },
    {
        "id": "fp_mm_02_maximum_of",
        "family": "minmax_bound",
        "sub_type": "maximum_of_phrase",
        "query": (
            "At least 10 workers are required for the project. "
            "A maximum of 30 workers can be scheduled."
        ),
        "expected_slots": {
            "MinWorkers": 10,
            "MaxWorkers": 30,
        },
        "expected_schema": "scheduling",
        "notes": (
            "'a maximum of 30' → UpperBound=30 (new pattern). "
            "'at least 10' → LowerBound=10 (existing pattern)."
        ),
    },
    {
        "id": "fp_mm_03_bare_x_to_y",
        "family": "minmax_bound",
        "sub_type": "bare_x_to_y_range",
        "query": (
            "The factory can produce 5 to 20 units per day. "
            "Each unit earns $8 profit."
        ),
        "expected_slots": {
            "MinProd": 5,
            "MaxProd": 20,
            "ProfitPerUnit": 8,
        },
        "expected_schema": "production",
        "notes": (
            "Bare 'X to Y' range without 'from' or 'between'. "
            "New final-pass range detection."
        ),
    },
    {
        "id": "fp_mm_04_must_not_exceed",
        "family": "minmax_bound",
        "sub_type": "must_not_exceed",
        "query": (
            "Daily shipments must be at least 50 tons. "
            "Shipments must not exceed 200 tons per day."
        ),
        "expected_slots": {
            "MinShipment": 50,
            "MaxShipment": 200,
        },
        "expected_schema": "transportation",
        "notes": (
            "'must not exceed 200' → UpperBound=200 (new pattern). "
        ),
    },
]

ALL_CASES: list[dict] = (
    PERCENT_CASES
    + COUNT_CASES
    + MINMAX_CASES
    + TOTAL_VS_PERUNIT_CASES
    + RETRIEVAL_CASES
    + FINAL_PASS_TOTAL_VS_PERUNIT
    + FINAL_PASS_IMPLICIT_COUNT
    + FINAL_PASS_MINMAX
)

_FAMILY_SUMMARY: dict[str, int] = {}
for _c in ALL_CASES:
    _FAMILY_SUMMARY[_c["family"]] = _FAMILY_SUMMARY.get(_c["family"], 0) + 1


def get_cases_by_family(family: str) -> list[dict]:
    """Return all synthetic cases for a given family name."""
    return [c for c in ALL_CASES if c["family"] == family]


def get_all_cases() -> list[dict]:
    """Return all synthetic test cases."""
    return list(ALL_CASES)


def summary() -> dict[str, Any]:
    """Return summary statistics."""
    return {
        "total_cases": len(ALL_CASES),
        "by_family": dict(_FAMILY_SUMMARY),
        "families": list(_FAMILY_SUMMARY.keys()),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate synthetic stress-test cases for the 5 easier error families."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Path to write JSON output. "
            "Default: prints summary to stdout and saves to "
            "results/synthetic_easy_families.json"
        ),
    )
    parser.add_argument(
        "--family",
        type=str,
        default="",
        help="If set, only output cases for the specified family.",
    )
    args = parser.parse_args()

    cases = get_cases_by_family(args.family) if args.family else get_all_cases()
    stat = summary()

    print("=" * 60)
    print("SYNTHETIC EASY-FAMILY TEST CASES")
    print("=" * 60)
    print(f"  Total cases: {stat['total_cases']}")
    for fam, cnt in stat["by_family"].items():
        print(f"  {fam}: {cnt} cases")
    print("=" * 60)

    output_path = args.output or str(ROOT / "results" / "synthetic_easy_families.json")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(cases)} cases to: {out}")

    # Also print each case summary
    print("\nCase listing:")
    for c in cases:
        print(
            f"  [{c['id']}] family={c['family']} sub_type={c['sub_type']} "
            f"expected_schema={c['expected_schema']}"
        )


if __name__ == "__main__":
    main()
