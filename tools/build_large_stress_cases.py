"""Large-scale synthetic stress-test case generator.

Produces a broad pool of deterministic test cases across all 9 taxonomy
categories plus mixed/compound cases.  Cases that include ``slots`` and
``expected`` fields can be run end-to-end through
``run_relation_aware_grounding`` exactly like Group 1 / Group 3 cases.
Cases that lack those fields carry only textual evidence (curated/static
analysis, same as the easy-family builder).

Usage::

    python tools/build_large_stress_cases.py
    python tools/build_large_stress_cases.py --output results/large_stress_cases.json

Taxonomy categories
-------------------
easy_percent_type         — percent vs integer / float normalization
easy_count_enumeration    — count/number-word/enumeration-derived count
easy_bounds_minmax        — min/max / lower-vs-upper bound confusion
easy_total_vs_perunit     — total vs per-unit coefficient confusion
easy_retrieval            — wrong schema / retrieval confusion
hard_wrong_assignment     — wrong slot assignment / distractor number (problem 7)
hard_swapped_quantities   — swapped sibling quantities (problem 8)
under_specified_template  — under-specified / template-like
mixed_or_other            — compound cases crossing multiple families

Secondary tags
--------------
sibling_swap, cross_clause_contamination, missed_count_word,
missed_enumeration_count, percent_normalization_error, lower_upper_reversal,
total_to_coeff_confusion, coeff_to_total_confusion, distractor_number,
wrong_entity_family, wrong_measure_family, wrong_role_family,
unresolved_near_tie, missing_value, under_specified, ambiguous,
wrong_schema, numeric_suffix_entity, reversed_clause_order
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Case dataclass
# ---------------------------------------------------------------------------


@dataclass
class StressCase:
    """A single stress-test case."""

    id: str
    category: str              # primary taxonomy category
    secondary_tags: list[str]  # secondary failure tags
    query: str
    slots: list[str]           # slot names (empty for static-analysis-only cases)
    expected: dict[str, float] # slot -> expected value (empty if unknown)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. easy_percent_type
# ---------------------------------------------------------------------------

_PERCENT_CASES: list[StressCase] = [
    StressCase(
        id="pct_s01_forty_percent_min",
        category="easy_percent_type",
        secondary_tags=["percent_normalization_error"],
        query="A factory requires at least 40% of output to be product A. Total capacity is 800 units.",
        slots=["MinFractionA", "TotalCapacity"],
        expected={"MinFractionA": 0.40, "TotalCapacity": 800.0},
        notes="40% must be stored as 0.40, not 40.",
    ),
    StressCase(
        id="pct_s02_thirty_percent_max",
        category="easy_percent_type",
        secondary_tags=["percent_normalization_error", "lower_upper_reversal"],
        query="At most 30% of total production can be product B. Produce at least 500 units overall.",
        slots=["MaxFractionB", "MinTotalProduction"],
        expected={"MaxFractionB": 0.30, "MinTotalProduction": 500.0},
        notes="30% -> 0.30, not 30.",
    ),
    StressCase(
        id="pct_s03_fraction_and_integer",
        category="easy_percent_type",
        secondary_tags=["percent_normalization_error"],
        query="At least 25% of workers must be full-time. There are 200 workers available.",
        slots=["MinFullTimeFraction", "TotalWorkers"],
        expected={"MinFullTimeFraction": 0.25, "TotalWorkers": 200.0},
        notes="25% -> 0.25.",
    ),
    StressCase(
        id="pct_s04_written_percent",
        category="easy_percent_type",
        secondary_tags=["percent_normalization_error"],
        query="At least fifteen percent of beds must be reserved for urgent patients.",
        slots=["MinUrgentFraction"],
        expected={"MinUrgentFraction": 0.15},
        notes="'fifteen percent' -> 0.15.",
    ),
    StressCase(
        id="pct_s05_two_fractions",
        category="easy_percent_type",
        secondary_tags=["percent_normalization_error"],
        query="At most 60% of land is used for crops and at least 20% for orchards.",
        slots=["MaxCropFraction", "MinOrchardFraction"],
        expected={"MaxCropFraction": 0.60, "MinOrchardFraction": 0.20},
        notes="Both percents must be normalised.",
    ),
    StressCase(
        id="pct_s06_mixed_percent_integer",
        category="easy_percent_type",
        secondary_tags=["percent_normalization_error", "total_to_coeff_confusion"],
        query=(
            "A bakery produces three products. At least 50% must be bread. "
            "Each loaf of bread earns 3 dollars profit. Daily capacity is 400 units."
        ),
        slots=["MinBreadFraction", "ProfitPerLoaf", "DailyCapacity"],
        expected={"MinBreadFraction": 0.50, "ProfitPerLoaf": 3.0, "DailyCapacity": 400.0},
        notes="Mixed: 50% fraction + integer profit + integer capacity.",
    ),
    StressCase(
        id="pct_s07_high_fraction",
        category="easy_percent_type",
        secondary_tags=["percent_normalization_error"],
        query="At least 80% of shipments must arrive on time.",
        slots=["MinOnTimeFraction"],
        expected={"MinOnTimeFraction": 0.80},
        notes="80% -> 0.80.",
    ),
    StressCase(
        id="pct_s08_small_fraction",
        category="easy_percent_type",
        secondary_tags=["percent_normalization_error"],
        query="No more than 5% of products can be defective.",
        slots=["MaxDefectFraction"],
        expected={"MaxDefectFraction": 0.05},
        notes="5% -> 0.05.",
    ),
]

# ---------------------------------------------------------------------------
# 2. easy_count_enumeration
# ---------------------------------------------------------------------------

_COUNT_CASES: list[StressCase] = [
    StressCase(
        id="cnt_s01_two_products",
        category="easy_count_enumeration",
        secondary_tags=["missed_enumeration_count"],
        query="A company makes two products: tables and chairs.",
        slots=["NumProducts"],
        expected={"NumProducts": 2.0},
        notes="'two products' -> count = 2.",
    ),
    StressCase(
        id="cnt_s02_three_machines",
        category="easy_count_enumeration",
        secondary_tags=["missed_enumeration_count"],
        query="The factory has three machines: press, lathe, and drill.",
        slots=["NumMachines"],
        expected={"NumMachines": 3.0},
        notes="Enumeration of 3 items -> count = 3.",
    ),
    StressCase(
        id="cnt_s03_four_workers_written",
        category="easy_count_enumeration",
        secondary_tags=["missed_count_word"],
        query="There are four shift workers available.",
        slots=["NumWorkers"],
        expected={"NumWorkers": 4.0},
        notes="'four' (word) -> 4.",
    ),
    StressCase(
        id="cnt_s04_five_warehouses",
        category="easy_count_enumeration",
        secondary_tags=["missed_count_word"],
        query="Five warehouses are available for distribution.",
        slots=["NumWarehouses"],
        expected={"NumWarehouses": 5.0},
        notes="'Five' (word) -> 5.",
    ),
    StressCase(
        id="cnt_s05_count_plus_capacity",
        category="easy_count_enumeration",
        secondary_tags=["missed_enumeration_count"],
        query=(
            "Two types of fuel are used: diesel and gasoline. "
            "Each vehicle uses at most 50 liters per day."
        ),
        slots=["NumFuelTypes", "MaxLitersPerDay"],
        expected={"NumFuelTypes": 2.0, "MaxLitersPerDay": 50.0},
        notes="Count + integer bound.",
    ),
    StressCase(
        id="cnt_s06_six_varieties",
        category="easy_count_enumeration",
        secondary_tags=["missed_enumeration_count"],
        query=(
            "The nursery sells six varieties of plants: "
            "roses, tulips, orchids, cacti, ferns, and daisies."
        ),
        slots=["NumVarieties"],
        expected={"NumVarieties": 6.0},
        notes="Enumeration of 6 items.",
    ),
    StressCase(
        id="cnt_s07_count_word_and_measure",
        category="easy_count_enumeration",
        secondary_tags=["missed_count_word"],
        query=(
            "There are three types of products. "
            "Each unit of product A earns 10 dollars and each unit of product B earns 7 dollars."
        ),
        slots=["NumProductTypes", "ProfitA", "ProfitB"],
        expected={"NumProductTypes": 3.0, "ProfitA": 10.0, "ProfitB": 7.0},
        notes="Count from 'three' + sibling measures.",
    ),
    StressCase(
        id="cnt_s08_implicit_two",
        category="easy_count_enumeration",
        secondary_tags=["missed_enumeration_count"],
        query="A shop sells apples and oranges.",
        slots=["NumFruitTypes"],
        expected={"NumFruitTypes": 2.0},
        notes="Implicit 2-item enumeration.",
    ),
    StressCase(
        id="cnt_s09_two_facilities",
        category="easy_count_enumeration",
        secondary_tags=["missed_enumeration_count"],
        query="There are two facilities available for production.",
        slots=["NumFacilities"],
        expected={"NumFacilities": 2.0},
        notes="'two facilities' -> 2.",
    ),
    StressCase(
        id="cnt_s10_seven_routes",
        category="easy_count_enumeration",
        secondary_tags=["missed_count_word"],
        query="Seven delivery routes connect the warehouse to retailers.",
        slots=["NumRoutes"],
        expected={"NumRoutes": 7.0},
        notes="'Seven' -> 7.",
    ),
]

# ---------------------------------------------------------------------------
# 3. easy_bounds_minmax
# ---------------------------------------------------------------------------

_BOUNDS_CASES: list[StressCase] = [
    StressCase(
        id="bnd_s01_atleast_atmost",
        category="easy_bounds_minmax",
        secondary_tags=["lower_upper_reversal"],
        query="Each product must require at least 2 and at most 8 labor hours.",
        slots=["MinLaborHours", "MaxLaborHours"],
        expected={"MinLaborHours": 2.0, "MaxLaborHours": 8.0},
        notes="Min < Max; correct direction.",
    ),
    StressCase(
        id="bnd_s02_no_less_no_more",
        category="easy_bounds_minmax",
        secondary_tags=["lower_upper_reversal"],
        query="Produce no less than 100 units and no more than 500 units.",
        slots=["MinProduction", "MaxProduction"],
        expected={"MinProduction": 100.0, "MaxProduction": 500.0},
        notes="'no less than' = min, 'no more than' = max.",
    ),
    StressCase(
        id="bnd_s03_minimum_maximum",
        category="easy_bounds_minmax",
        secondary_tags=["lower_upper_reversal"],
        query="Minimum stock level is 50 units. Maximum stock level is 300 units.",
        slots=["MinStock", "MaxStock"],
        expected={"MinStock": 50.0, "MaxStock": 300.0},
        notes="Explicit minimum / maximum labels.",
    ),
    StressCase(
        id="bnd_s04_lower_upper_bound",
        category="easy_bounds_minmax",
        secondary_tags=["lower_upper_reversal"],
        query="The lower bound for machine usage is 10 hours and the upper bound is 40 hours.",
        slots=["LowerBoundHours", "UpperBoundHours"],
        expected={"LowerBoundHours": 10.0, "UpperBoundHours": 40.0},
        notes="Explicit lower/upper labels.",
    ),
    StressCase(
        id="bnd_s05_range_expression",
        category="easy_bounds_minmax",
        secondary_tags=["lower_upper_reversal"],
        query="Production must be between 200 and 800 units.",
        slots=["MinProduction", "MaxProduction"],
        expected={"MinProduction": 200.0, "MaxProduction": 800.0},
        notes="'between X and Y' = [min, max].",
    ),
    StressCase(
        id="bnd_s06_bound_with_percent",
        category="easy_bounds_minmax",
        secondary_tags=["lower_upper_reversal", "percent_normalization_error"],
        query="Machine utilisation must be at least 30% and at most 90%.",
        slots=["MinUtilisation", "MaxUtilisation"],
        expected={"MinUtilisation": 0.30, "MaxUtilisation": 0.90},
        notes="Bounds on a fraction — both directions must be correct.",
    ),
    StressCase(
        id="bnd_s07_three_bounds",
        category="easy_bounds_minmax",
        secondary_tags=["lower_upper_reversal"],
        query=(
            "Product A must be produced between 50 and 150 units. "
            "Product B must be at most 200 units."
        ),
        slots=["MinA", "MaxA", "MaxB"],
        expected={"MinA": 50.0, "MaxA": 150.0, "MaxB": 200.0},
        notes="Multiple bounds across two entities.",
    ),
    StressCase(
        id="bnd_s08_capacity_constraint",
        category="easy_bounds_minmax",
        secondary_tags=["lower_upper_reversal"],
        query="Warehouse capacity is at most 1000 tonnes. Minimum shipment is 10 tonnes.",
        slots=["MaxCapacity", "MinShipment"],
        expected={"MaxCapacity": 1000.0, "MinShipment": 10.0},
        notes="Max capacity + min shipment.",
    ),
]

# ---------------------------------------------------------------------------
# 4. easy_total_vs_perunit
# ---------------------------------------------------------------------------

_TOTAL_PERUNIT_CASES: list[StressCase] = [
    StressCase(
        id="tpu_s01_budget_vs_cost",
        category="easy_total_vs_perunit",
        secondary_tags=["total_to_coeff_confusion"],
        query="The total budget is 10000 dollars. Each unit of product A costs 5 dollars.",
        slots=["TotalBudget", "CostPerUnitA"],
        expected={"TotalBudget": 10000.0, "CostPerUnitA": 5.0},
        notes="10000 = total, 5 = per-unit cost.",
    ),
    StressCase(
        id="tpu_s02_hours_available_vs_required",
        category="easy_total_vs_perunit",
        secondary_tags=["total_to_coeff_confusion"],
        query="There are 2000 labor hours available. Each product requires 3 labor hours.",
        slots=["TotalLaborHours", "LaborHoursPerProduct"],
        expected={"TotalLaborHours": 2000.0, "LaborHoursPerProduct": 3.0},
        notes="2000 = total resource, 3 = per-unit usage.",
    ),
    StressCase(
        id="tpu_s03_revenue_vs_profit",
        category="easy_total_vs_perunit",
        secondary_tags=["coeff_to_total_confusion"],
        query="Product X earns 8 dollars per unit sold. Monthly revenue target is 40000.",
        slots=["ProfitPerUnitX", "MonthlyRevenueTarget"],
        expected={"ProfitPerUnitX": 8.0, "MonthlyRevenueTarget": 40000.0},
        notes="8 = per-unit, 40000 = total target.",
    ),
    StressCase(
        id="tpu_s04_capacity_vs_requirement",
        category="easy_total_vs_perunit",
        secondary_tags=["total_to_coeff_confusion"],
        query="Machine has 500 hours of capacity. Product A needs 2 hours per unit.",
        slots=["MachineCapacity", "HoursPerUnitA"],
        expected={"MachineCapacity": 500.0, "HoursPerUnitA": 2.0},
        notes="500 = total capacity, 2 = per-unit requirement.",
    ),
    StressCase(
        id="tpu_s05_storage_vs_space",
        category="easy_total_vs_perunit",
        secondary_tags=["total_to_coeff_confusion"],
        query="Storage has 1200 cubic metres. Each pallet takes 4 cubic metres.",
        slots=["TotalStorage", "SpacePerPallet"],
        expected={"TotalStorage": 1200.0, "SpacePerPallet": 4.0},
        notes="1200 = total, 4 = per-unit.",
    ),
    StressCase(
        id="tpu_s06_three_coefficients",
        category="easy_total_vs_perunit",
        secondary_tags=["total_to_coeff_confusion"],
        query=(
            "Available raw material is 800 kg. "
            "Product A requires 2 kg per unit, product B requires 5 kg per unit."
        ),
        slots=["TotalRawMaterial", "MaterialPerA", "MaterialPerB"],
        expected={"TotalRawMaterial": 800.0, "MaterialPerA": 2.0, "MaterialPerB": 5.0},
        notes="Total + two per-unit coefficients.",
    ),
    StressCase(
        id="tpu_s07_weight_budget",
        category="easy_total_vs_perunit",
        secondary_tags=["total_to_coeff_confusion"],
        query="The truck can carry at most 3000 kg. Each box weighs 25 kg.",
        slots=["TruckCapacityKg", "WeightPerBox"],
        expected={"TruckCapacityKg": 3000.0, "WeightPerBox": 25.0},
        notes="Total capacity vs per-item weight.",
    ),
    StressCase(
        id="tpu_s08_energy_budget_usage",
        category="easy_total_vs_perunit",
        secondary_tags=["total_to_coeff_confusion"],
        query="Daily energy budget is 600 kWh. Each machine consumes 40 kWh per hour.",
        slots=["DailyEnergyBudget", "EnergyPerMachineHour"],
        expected={"DailyEnergyBudget": 600.0, "EnergyPerMachineHour": 40.0},
        notes="600 = total budget, 40 = per-unit consumption.",
    ),
]

# ---------------------------------------------------------------------------
# 5. easy_retrieval
# ---------------------------------------------------------------------------

_RETRIEVAL_CASES: list[StressCase] = [
    # Static-analysis cases (no slot/expected — purely textual signals)
    StressCase(
        id="ret_s01_correct_schema",
        category="easy_retrieval",
        secondary_tags=[],
        query="Maximize profit subject to labor and budget constraints. Two products.",
        slots=[],
        expected={},
        notes="Standard LP; correct schema expected.",
    ),
    StressCase(
        id="ret_s02_transportation_keywords",
        category="easy_retrieval",
        secondary_tags=["wrong_schema"],
        query="Minimize transportation cost from two warehouses to three retailers.",
        slots=[],
        expected={},
        notes="Transportation problem. Keywords 'transportation' + 'cost' should guide retrieval.",
    ),
    StressCase(
        id="ret_s03_scheduling_keywords",
        category="easy_retrieval",
        secondary_tags=["wrong_schema"],
        query="Schedule machines to minimize total makespan for five jobs.",
        slots=[],
        expected={},
        notes="Scheduling problem.",
    ),
    StressCase(
        id="ret_s04_knapsack_signal",
        category="easy_retrieval",
        secondary_tags=["wrong_schema"],
        query="Select items to maximise value subject to a weight limit of 50 kg.",
        slots=[],
        expected={},
        notes="Knapsack-like problem.",
    ),
    StressCase(
        id="ret_s05_diet_problem",
        category="easy_retrieval",
        secondary_tags=["wrong_schema"],
        query="Mix feeds to meet nutritional requirements at minimum cost.",
        slots=[],
        expected={},
        notes="Diet / blending problem.",
    ),
]

# ---------------------------------------------------------------------------
# 6. hard_wrong_assignment — distractor / wrong-slot assignment (problem 7)
# ---------------------------------------------------------------------------

_WRONG_ASSIGNMENT_CASES: list[StressCase] = [
    # -- Same entity, two different measures --
    StressCase(
        id="ha_s01_cost_profit_same_entity",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_measure_family", "distractor_number"],
        query="Product X costs 4 dollars to make and earns 12 dollars profit.",
        slots=["CostPerUnitX", "ProfitPerUnitX"],
        expected={"CostPerUnitX": 4.0, "ProfitPerUnitX": 12.0},
        notes="Two measures for the same entity; distractor risk.",
    ),
    StressCase(
        id="ha_s02_labor_material_same_entity",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_measure_family", "distractor_number"],
        query="Each chair needs 3 labor hours and 6 board feet of wood.",
        slots=["LaborHoursPerChair", "WoodPerChair"],
        expected={"LaborHoursPerChair": 3.0, "WoodPerChair": 6.0},
        notes="Labor vs material for same product.",
    ),
    # -- Same measure, different entities --
    StressCase(
        id="ha_s03_protein_two_feeds",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_entity_family", "sibling_swap"],
        query="Feed A contains 10 grams of protein. Feed B contains 6 grams of protein.",
        slots=["ProteinFeedA", "ProteinFeedB"],
        expected={"ProteinFeedA": 10.0, "ProteinFeedB": 6.0},
        notes="Same measure, two entities.",
    ),
    StressCase(
        id="ha_s04_profit_two_products",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_entity_family", "sibling_swap"],
        query="Product A yields 15 dollars profit; product B yields 9 dollars profit.",
        slots=["ProfitProductA", "ProfitProductB"],
        expected={"ProfitProductA": 15.0, "ProfitProductB": 9.0},
        notes="Same measure (profit), two products.",
    ),
    # -- Total distractor adjacent to per-unit --
    StressCase(
        id="ha_s05_budget_distractor",
        category="hard_wrong_assignment",
        secondary_tags=["distractor_number", "total_to_coeff_confusion"],
        query=(
            "Total budget is 8000. Product A costs 6 dollars per unit and "
            "product B costs 4 dollars per unit."
        ),
        slots=["TotalBudget", "CostPerA", "CostPerB"],
        expected={"TotalBudget": 8000.0, "CostPerA": 6.0, "CostPerB": 4.0},
        notes="Budget as distractor for per-unit costs.",
    ),
    StressCase(
        id="ha_s06_hours_distractor",
        category="hard_wrong_assignment",
        secondary_tags=["distractor_number", "total_to_coeff_confusion"],
        query=(
            "1000 machine hours are available. "
            "Type 1 requires 4 hours per unit, type 2 requires 7 hours per unit."
        ),
        slots=["TotalMachineHours", "HoursPerType1", "HoursPerType2"],
        expected={"TotalMachineHours": 1000.0, "HoursPerType1": 4.0, "HoursPerType2": 7.0},
        notes="Total as distractor near per-unit values.",
    ),
    # -- Three entities, same measure --
    StressCase(
        id="ha_s07_three_product_profits",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_entity_family"],
        query=(
            "Product 1 earns 5 dollars profit, product 2 earns 8 dollars profit, "
            "and product 3 earns 11 dollars profit."
        ),
        slots=["ProfitProduct1", "ProfitProduct2", "ProfitProduct3"],
        expected={"ProfitProduct1": 5.0, "ProfitProduct2": 8.0, "ProfitProduct3": 11.0},
        notes="Three entities with same measure.",
    ),
    # -- Numeric suffix entity --
    StressCase(
        id="ha_s08_numeric_suffix_entities",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_entity_family", "numeric_suffix_entity"],
        query=(
            "Machine1 has capacity 200. Machine2 has capacity 350. Machine3 has capacity 150."
        ),
        slots=["CapacityMachine1", "CapacityMachine2", "CapacityMachine3"],
        expected={"CapacityMachine1": 200.0, "CapacityMachine2": 350.0, "CapacityMachine3": 150.0},
        notes="Numeric suffix entities — each suffix must anchor its value.",
    ),
    # -- Reversed order in clause --
    StressCase(
        id="ha_s09_reversed_clause_order",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_entity_family", "reversed_clause_order"],
        query="Feed B contains 7 protein and Feed A contains 10 protein.",
        slots=["ProteinFeedA", "ProteinFeedB"],
        expected={"ProteinFeedA": 10.0, "ProteinFeedB": 7.0},
        notes="B appears before A in text; assignment must still be correct.",
    ),
    StressCase(
        id="ha_s10_reversed_product_order",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_entity_family", "reversed_clause_order"],
        query="Product 2 needs 5 labor hours; product 1 needs 2 labor hours.",
        slots=["LaborProduct1", "LaborProduct2"],
        expected={"LaborProduct1": 2.0, "LaborProduct2": 5.0},
        notes="Product 2 appears first in text.",
    ),
    # -- Four sibling slots --
    StressCase(
        id="ha_s11_four_sibling_slots",
        category="hard_wrong_assignment",
        secondary_tags=["wrong_entity_family"],
        query=(
            "Feed A: 12 protein, 8 fat. Feed B: 9 protein, 14 fat."
        ),
        slots=["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
        expected={
            "ProteinFeedA": 12.0, "FatFeedA": 8.0,
            "ProteinFeedB": 9.0, "FatFeedB": 14.0,
        },
        notes="Four sibling slots across two entities and two measures.",
    ),
    # -- Mixed total + per-unit + sibling --
    StressCase(
        id="ha_s12_mixed_total_sibling",
        category="hard_wrong_assignment",
        secondary_tags=["distractor_number", "total_to_coeff_confusion", "wrong_entity_family"],
        query=(
            "Total budget is 5000. Product A costs 3 per unit and product B costs 7 per unit."
        ),
        slots=["TotalBudget", "CostA", "CostB"],
        expected={"TotalBudget": 5000.0, "CostA": 3.0, "CostB": 7.0},
        notes="Total + two per-unit costs for different entities.",
    ),
]

# ---------------------------------------------------------------------------
# 7. hard_swapped_quantities (problem 8)
# ---------------------------------------------------------------------------

_SWAPPED_CASES: list[StressCase] = [
    # -- Classic parallel two-entity two-measure --
    StressCase(
        id="sw_s01_chair_dresser_labor_wood",
        category="hard_swapped_quantities",
        secondary_tags=["sibling_swap", "cross_clause_contamination"],
        query=(
            "A chair requires 3 labor hours and 5 board feet of wood. "
            "A dresser requires 8 labor hours and 2 board feet of wood."
        ),
        slots=["LaborChair", "WoodChair", "LaborDresser", "WoodDresser"],
        expected={
            "LaborChair": 3.0, "WoodChair": 5.0,
            "LaborDresser": 8.0, "WoodDresser": 2.0,
        },
        notes="Classic two-entity two-measure; swap risk between chair and dresser values.",
    ),
    StressCase(
        id="sw_s02_product_ab_heating_cooling",
        category="hard_swapped_quantities",
        secondary_tags=["sibling_swap"],
        query=(
            "Product A needs 2 heating hours and 6 cooling hours. "
            "Product B needs 5 heating hours and 1 cooling hour."
        ),
        slots=["HeatingA", "CoolingA", "HeatingB", "CoolingB"],
        expected={"HeatingA": 2.0, "CoolingA": 6.0, "HeatingB": 5.0, "CoolingB": 1.0},
        notes="Two products, two resource types.",
    ),
    StressCase(
        id="sw_s03_reversed_clause_two_measure",
        category="hard_swapped_quantities",
        secondary_tags=["sibling_swap", "reversed_clause_order"],
        query=(
            "Product B requires 5 heating hours and 1 cooling hour. "
            "Product A requires 2 heating hours and 6 cooling hours."
        ),
        slots=["HeatingA", "CoolingA", "HeatingB", "CoolingB"],
        expected={"HeatingA": 2.0, "CoolingA": 6.0, "HeatingB": 5.0, "CoolingB": 1.0},
        notes="Same as sw_s02 but B clause comes first.",
    ),
    StressCase(
        id="sw_s04_three_entities_same_measure",
        category="hard_swapped_quantities",
        secondary_tags=["sibling_swap"],
        query=(
            "Product 1 earns 5 per unit. "
            "Product 2 earns 10 per unit. "
            "Product 3 earns 8 per unit."
        ),
        slots=["ProfitProduct1", "ProfitProduct2", "ProfitProduct3"],
        expected={"ProfitProduct1": 5.0, "ProfitProduct2": 10.0, "ProfitProduct3": 8.0},
        notes="Three entities, same measure — mid-entity swap is hardest.",
    ),
    StressCase(
        id="sw_s05_feed_protein_fat_full",
        category="hard_swapped_quantities",
        secondary_tags=["sibling_swap", "cross_clause_contamination"],
        query=(
            "Feed A contains 10 protein and 8 fat, "
            "while Feed B contains 7 protein and 15 fat."
        ),
        slots=["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
        expected={
            "ProteinFeedA": 10.0, "FatFeedA": 8.0,
            "ProteinFeedB": 7.0, "FatFeedB": 15.0,
        },
        notes="Classic feed-parallel case from Group 3.",
    ),
    StressCase(
        id="sw_s06_feed_protein_fat_reversed",
        category="hard_swapped_quantities",
        secondary_tags=["sibling_swap", "reversed_clause_order"],
        query=(
            "Feed B contains 7 protein and 15 fat, "
            "while Feed A contains 10 protein and 8 fat."
        ),
        slots=["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
        expected={
            "ProteinFeedA": 10.0, "FatFeedA": 8.0,
            "ProteinFeedB": 7.0, "FatFeedB": 15.0,
        },
        notes="Feed-parallel with B before A.",
    ),
    StressCase(
        id="sw_s07_numeric_suffix_swap",
        category="hard_swapped_quantities",
        secondary_tags=["sibling_swap", "numeric_suffix_entity"],
        query=(
            "Type1 product requires 4 labor hours and costs 3 per unit. "
            "Type2 product requires 7 labor hours and costs 9 per unit."
        ),
        slots=["LaborType1", "CostType1", "LaborType2", "CostType2"],
        expected={"LaborType1": 4.0, "CostType1": 3.0, "LaborType2": 7.0, "CostType2": 9.0},
        notes="Numeric suffix entities with two measures each.",
    ),
    StressCase(
        id="sw_s08_clause_contamination_three",
        category="hard_swapped_quantities",
        secondary_tags=["cross_clause_contamination", "sibling_swap"],
        query=(
            "Feed A has 12 protein. Feed B has 9 protein. "
            "Feed A also has 4 fat, while Feed B has 6 fat."
        ),
        slots=["ProteinFeedA", "ProteinFeedB", "FatFeedA", "FatFeedB"],
        expected={
            "ProteinFeedA": 12.0, "ProteinFeedB": 9.0,
            "FatFeedA": 4.0, "FatFeedB": 6.0,
        },
        notes="Split across two sentences; cross-clause contamination risk.",
    ),
]

# ---------------------------------------------------------------------------
# 8. under_specified_template
# ---------------------------------------------------------------------------

_UNDERSPECIFIED_CASES: list[StressCase] = [
    StressCase(
        id="us_s01_no_numbers",
        category="under_specified_template",
        secondary_tags=["under_specified", "missing_value"],
        query="Maximize profit subject to resource and demand constraints.",
        slots=[],
        expected={},
        notes="No numeric values; grounder cannot fill any slot.",
    ),
    StressCase(
        id="us_s02_partial_values",
        category="under_specified_template",
        secondary_tags=["under_specified", "missing_value"],
        query="Budget is 5000. Each product uses some labor hours.",
        slots=["TotalBudget", "LaborPerUnit"],
        expected={"TotalBudget": 5000.0},
        notes="One slot filled but LaborPerUnit is unspecified.",
    ),
    StressCase(
        id="us_s03_ambiguous_which_entity",
        category="under_specified_template",
        secondary_tags=["under_specified", "ambiguous"],
        query="Product profit is 8 and cost is 4.",
        slots=["ProfitProductA", "CostProductA", "ProfitProductB", "CostProductB"],
        expected={"ProfitProductA": 8.0, "CostProductA": 4.0},
        notes="Only one entity's values given but schema expects two.",
    ),
    StressCase(
        id="us_s04_placeholder_values",
        category="under_specified_template",
        secondary_tags=["under_specified"],
        query="Each product yields a certain amount of profit per unit.",
        slots=[],
        expected={},
        notes="Template language with no concrete numbers.",
    ),
    StressCase(
        id="us_s05_vague_bounds",
        category="under_specified_template",
        secondary_tags=["under_specified", "ambiguous"],
        query="Produce a reasonable amount of each product.",
        slots=[],
        expected={},
        notes="Completely vague; no actionable numeric content.",
    ),
]

# ---------------------------------------------------------------------------
# 9. mixed_or_other — compound cases crossing families
# ---------------------------------------------------------------------------

_MIXED_CASES: list[StressCase] = [
    # percent + sibling entity
    StressCase(
        id="mix_s01_percent_plus_sibling",
        category="mixed_or_other",
        secondary_tags=["percent_normalization_error", "sibling_swap", "wrong_entity_family"],
        query=(
            "At least 30% of total output must be product A. "
            "Product A earns 5 per unit and product B earns 8 per unit."
        ),
        slots=["MinFractionA", "ProfitA", "ProfitB"],
        expected={"MinFractionA": 0.30, "ProfitA": 5.0, "ProfitB": 8.0},
        notes="Fraction + sibling profit values.",
    ),
    # count + sibling measure
    StressCase(
        id="mix_s02_count_plus_two_measures",
        category="mixed_or_other",
        secondary_tags=["missed_enumeration_count", "sibling_swap"],
        query=(
            "Two machine types are available. "
            "Machine 1 processes 20 units per hour. Machine 2 processes 35 units per hour."
        ),
        slots=["NumMachineTypes", "RateMachine1", "RateMachine2"],
        expected={"NumMachineTypes": 2.0, "RateMachine1": 20.0, "RateMachine2": 35.0},
        notes="Enumeration count + sibling throughput values.",
    ),
    # bounds + total distractor
    StressCase(
        id="mix_s03_bounds_plus_total",
        category="mixed_or_other",
        secondary_tags=["lower_upper_reversal", "distractor_number", "total_to_coeff_confusion"],
        query=(
            "Total production capacity is 1000. "
            "Produce between 100 and 400 units of product A."
        ),
        slots=["TotalCapacity", "MinA", "MaxA"],
        expected={"TotalCapacity": 1000.0, "MinA": 100.0, "MaxA": 400.0},
        notes="Total + min/max bounds for one product.",
    ),
    # percent + bounds
    StressCase(
        id="mix_s04_percent_plus_bounds",
        category="mixed_or_other",
        secondary_tags=["percent_normalization_error", "lower_upper_reversal"],
        query=(
            "At least 20% of mix must be ingredient A. "
            "Total mix volume is between 500 and 2000 litres."
        ),
        slots=["MinFractionA", "MinVolume", "MaxVolume"],
        expected={"MinFractionA": 0.20, "MinVolume": 500.0, "MaxVolume": 2000.0},
        notes="Fraction + range bounds.",
    ),
    # total + percent + sibling
    StressCase(
        id="mix_s05_total_percent_sibling",
        category="mixed_or_other",
        secondary_tags=["percent_normalization_error", "total_to_coeff_confusion", "sibling_swap"],
        query=(
            "Total revenue target is 50000. "
            "At least 40% should come from product A. "
            "Product A earns 10 per unit and product B earns 15 per unit."
        ),
        slots=["RevenueTarget", "MinFractionA", "ProfitA", "ProfitB"],
        expected={
            "RevenueTarget": 50000.0, "MinFractionA": 0.40,
            "ProfitA": 10.0, "ProfitB": 15.0,
        },
        notes="Three families combined.",
    ),
    # count + per-unit
    StressCase(
        id="mix_s06_count_plus_perunit",
        category="mixed_or_other",
        secondary_tags=["missed_enumeration_count", "total_to_coeff_confusion"],
        query=(
            "Three product types exist. Each unit of any product requires 2 kg of raw material. "
            "Total raw material available is 600 kg."
        ),
        slots=["NumProductTypes", "MaterialPerUnit", "TotalMaterial"],
        expected={"NumProductTypes": 3.0, "MaterialPerUnit": 2.0, "TotalMaterial": 600.0},
        notes="Count + per-unit usage + total resource.",
    ),
    # sibling swap + numeric suffix
    StressCase(
        id="mix_s07_sibling_swap_numeric_suffix",
        category="mixed_or_other",
        secondary_tags=["sibling_swap", "numeric_suffix_entity", "cross_clause_contamination"],
        query=(
            "Product1 needs 3 labor hours and 4 machine hours. "
            "Product2 needs 6 labor hours and 1 machine hour."
        ),
        slots=["LaborProduct1", "MachineProduct1", "LaborProduct2", "MachineProduct2"],
        expected={
            "LaborProduct1": 3.0, "MachineProduct1": 4.0,
            "LaborProduct2": 6.0, "MachineProduct2": 1.0,
        },
        notes="Numeric suffix + two measures each.",
    ),
    # bounds + sibling
    StressCase(
        id="mix_s08_bounds_plus_sibling",
        category="mixed_or_other",
        secondary_tags=["lower_upper_reversal", "sibling_swap"],
        query=(
            "Produce between 50 and 300 units of product A. "
            "Product A earns 5 per unit and product B earns 9 per unit."
        ),
        slots=["MinA", "MaxA", "ProfitA", "ProfitB"],
        expected={"MinA": 50.0, "MaxA": 300.0, "ProfitA": 5.0, "ProfitB": 9.0},
        notes="Range bounds + sibling profit.",
    ),
    # three-clause compound
    StressCase(
        id="mix_s09_three_clause_compound",
        category="mixed_or_other",
        secondary_tags=["cross_clause_contamination", "sibling_swap", "total_to_coeff_confusion"],
        query=(
            "Total budget is 12000. "
            "Product A requires 3 labor hours and earns 7 profit. "
            "Product B requires 5 labor hours and earns 11 profit."
        ),
        slots=["TotalBudget", "LaborA", "ProfitA", "LaborB", "ProfitB"],
        expected={
            "TotalBudget": 12000.0,
            "LaborA": 3.0, "ProfitA": 7.0,
            "LaborB": 5.0, "ProfitB": 11.0,
        },
        notes="Three clauses: total + two parallel entity blocks.",
    ),
    # near-tie
    StressCase(
        id="mix_s10_near_tie_values",
        category="mixed_or_other",
        secondary_tags=["unresolved_near_tie", "sibling_swap"],
        query="Feed A has 10 protein and Feed B has 11 protein.",
        slots=["ProteinFeedA", "ProteinFeedB"],
        expected={"ProteinFeedA": 10.0, "ProteinFeedB": 11.0},
        notes="Near-identical values; ordering must come from entity anchors not magnitude.",
    ),
]

# ---------------------------------------------------------------------------
# Taxonomy constant (mirrors run_large_failure_audit.TAXONOMY_GROUPS)
# ---------------------------------------------------------------------------

TAXONOMY_GROUPS: list[str] = [
    "easy_percent_type",
    "easy_count_enumeration",
    "easy_bounds_minmax",
    "easy_total_vs_perunit",
    "easy_retrieval",
    "hard_wrong_assignment",
    "hard_swapped_quantities",
    "under_specified_template",
    "mixed_or_other",
]

# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------

ALL_STRESS_CASES: list[StressCase] = (
    _PERCENT_CASES
    + _COUNT_CASES
    + _BOUNDS_CASES
    + _TOTAL_PERUNIT_CASES
    + _RETRIEVAL_CASES
    + _WRONG_ASSIGNMENT_CASES
    + _SWAPPED_CASES
    + _UNDERSPECIFIED_CASES
    + _MIXED_CASES
)


def get_all_stress_cases() -> list[StressCase]:
    """Return all stress-test cases."""
    return list(ALL_STRESS_CASES)


def get_runnable_cases() -> list[StressCase]:
    """Return cases that have slots and expected values (end-to-end runnable)."""
    return [c for c in ALL_STRESS_CASES if c.slots and c.expected]


def get_cases_by_category(category: str) -> list[StressCase]:
    """Return all cases for a given taxonomy category."""
    return [c for c in ALL_STRESS_CASES if c.category == category]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate large synthetic stress-test cases")
    parser.add_argument(
        "--output",
        default="results/large_stress_cases.json",
        help="Output JSON path (default: results/large_stress_cases.json)",
    )
    args = parser.parse_args()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cases = get_all_stress_cases()
    data = {
        "total": len(cases),
        "runnable": len(get_runnable_cases()),
        "by_category": {
            cat: len(get_cases_by_category(cat))
            for cat in sorted({c.category for c in cases})
        },
        "cases": [c.to_dict() for c in cases],
    }

    with out_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(cases)} stress cases ({len(get_runnable_cases())} runnable)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
