"""Group 1 hard-family evaluation — measure/attribute-aware linking.

This module provides a focused ablation evaluation for the Group 1 improvements:
  * camel-case slot-name splitting  → ``_slot_measure_tokens``
  * numeric-token boundary stop     → keeps right context local to each mention
  * narrow_measure_overlap scoring  → rewards mention-slot measure alignment

Usage::

    python tools/evaluate_group1_hard_family.py

The script reports per-case and aggregate results across four ablation levels:
  "basic"  — type + lexical overlap only
  "ops"    — + operator (min/max) matching
  "role"   — + role-family (total/per-unit/bound) matching
  "full"   — + all Group 1 improvements (narrow_measure_overlap, camel-split)

Metrics reported
----------------
family_case_count   — number of test cases in this family
family_fixed_count  — cases where the "full" system is correct
family_fix_rate     — fixed / total
family_regression   — cases where "full" is wrong but "basic" was correct
TypeMatch           — fraction of slots assigned the expected value
Exact20             — fraction within 20% relative error (for numeric slots)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.relation_aware_linking import run_relation_aware_grounding

# ---------------------------------------------------------------------------
# Test-case registry
# ---------------------------------------------------------------------------

@dataclass
class Group1Case:
    """A single evaluation case for Group 1 hard-family improvements."""

    name: str
    query: str
    slots: list[str]
    expected: dict[str, float]
    family: str  # distractor_role | measure_family | total_vs_per_unit | easy_regression


# All Group 1 evaluation cases
_CASES: list[Group1Case] = [
    # ------------------------------------------------------------------
    # Distractor-role mismatch (Step 10 A)
    # ------------------------------------------------------------------
    Group1Case(
        name="distractor_total_budget_labor_workers",
        query="There are 100 workers, each product needs 4 labor hours, and total budget is 5000.",
        slots=["NumWorkers", "LaborHoursPerProduct", "TotalBudget"],
        expected={"NumWorkers": 100.0, "LaborHoursPerProduct": 4.0, "TotalBudget": 5000.0},
        family="distractor_role",
    ),
    Group1Case(
        name="total_available_hours_vs_per_product_hours",
        query="There are 2000 labor hours available, and each product requires 3 labor hours.",
        slots=["TotalLaborHours", "LaborHoursPerProduct"],
        expected={"TotalLaborHours": 2000.0, "LaborHoursPerProduct": 3.0},
        family="distractor_role",
    ),
    # ------------------------------------------------------------------
    # Cost vs profit (Step 10 B)
    # ------------------------------------------------------------------
    Group1Case(
        name="cost_vs_profit_profit_first",
        query="Product X yields 12 dollars profit and costs 5 dollars to produce.",
        slots=["ProfitPerUnit", "CostPerUnit"],
        expected={"ProfitPerUnit": 12.0, "CostPerUnit": 5.0},
        family="measure_family",
    ),
    Group1Case(
        name="cost_vs_profit_cost_first",
        query="It costs 5 dollars to produce product X and earns 12 dollars profit.",
        slots=["ProfitPerUnit", "CostPerUnit"],
        expected={"ProfitPerUnit": 12.0, "CostPerUnit": 5.0},
        family="measure_family",
    ),
    # ------------------------------------------------------------------
    # Protein vs fat (Step 10 C) — both orderings
    # ------------------------------------------------------------------
    Group1Case(
        name="protein_vs_fat_protein_first",
        query="Feed A contains 10 protein and 8 fat.",
        slots=["ProteinFeedA", "FatFeedA"],
        expected={"ProteinFeedA": 10.0, "FatFeedA": 8.0},
        family="measure_family",
    ),
    Group1Case(
        name="protein_vs_fat_fat_first",
        query="Feed A contains 8 fat and 10 protein.",
        slots=["ProteinFeedA", "FatFeedA"],
        expected={"ProteinFeedA": 10.0, "FatFeedA": 8.0},
        family="measure_family",
    ),
    # ------------------------------------------------------------------
    # Heating vs cooling (Step 10 D) — both orderings
    # ------------------------------------------------------------------
    Group1Case(
        name="heating_vs_cooling_heating_first",
        query="Regular glass requires 3 heating hours and 5 cooling hours.",
        slots=["HeatingHours", "CoolingHours"],
        expected={"HeatingHours": 3.0, "CoolingHours": 5.0},
        family="measure_family",
    ),
    Group1Case(
        name="heating_vs_cooling_cooling_first",
        query="Regular glass requires 5 cooling hours and 3 heating hours.",
        slots=["HeatingHours", "CoolingHours"],
        expected={"HeatingHours": 3.0, "CoolingHours": 5.0},
        family="measure_family",
    ),
    # ------------------------------------------------------------------
    # Labor vs material / wood (Step 10 E)
    # ------------------------------------------------------------------
    Group1Case(
        name="labor_vs_wood_labor_first",
        query="Each table requires 4 labor hours and 6 units of wood.",
        slots=["LaborHoursPerTable", "WoodUnitsPerTable"],
        expected={"LaborHoursPerTable": 4.0, "WoodUnitsPerTable": 6.0},
        family="measure_family",
    ),
    Group1Case(
        name="labor_vs_wood_wood_first",
        query="Each table uses 6 wood units and 4 labor hours.",
        slots=["LaborHoursPerTable", "WoodUnitsPerTable"],
        expected={"LaborHoursPerTable": 4.0, "WoodUnitsPerTable": 6.0},
        family="measure_family",
    ),
    # ------------------------------------------------------------------
    # Easy-family regressions (Step 10 F)
    # ------------------------------------------------------------------
    Group1Case(
        name="percent_upper_bound",
        query="At most 40% of workers can be assigned to night shift.",
        slots=["MaxNightFraction"],
        expected={"MaxNightFraction": 0.4},
        family="easy_regression",
    ),
    Group1Case(
        name="derived_count_word_number",
        query="There are two types of jars.",
        slots=["NumJarTypes"],
        expected={"NumJarTypes": 2.0},
        family="easy_regression",
    ),
    Group1Case(
        name="bound_directions",
        query="At least 10 and at most 20 units.",
        slots=["MinUnits", "MaxUnits"],
        expected={"MinUnits": 10.0, "MaxUnits": 20.0},
        family="easy_regression",
    ),
    Group1Case(
        name="total_vs_per_unit",
        query="There are 2000 labor hours available, and each product requires 2 labor hours.",
        slots=["TotalLaborHours", "LaborHoursPerProduct"],
        expected={"TotalLaborHours": 2000.0, "LaborHoursPerProduct": 2.0},
        family="easy_regression",
    ),
]

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _within_20pct(predicted: float, expected: float) -> bool:
    if expected == 0:
        return predicted == 0
    return abs(predicted - expected) / abs(expected) <= 0.20


@dataclass
class CaseResult:
    case: Group1Case
    mode: str
    predictions: dict[str, float]
    correct_slots: int
    total_slots: int
    type_match: float  # = correct_slots / total_slots
    exact20: float
    all_correct: bool


def _evaluate_case(case: Group1Case, mode: str) -> CaseResult:
    _, mentions, _ = run_relation_aware_grounding(
        case.query, "orig", case.slots, ablation_mode=mode
    )
    predictions = {sn: m.value for sn, m in mentions.items()}

    correct = 0
    exact20_count = 0
    total = len(case.expected)
    for slot, exp_val in case.expected.items():
        pred_val = predictions.get(slot)
        if pred_val is None:
            continue
        if abs(pred_val - exp_val) < 1e-6:
            correct += 1
        if _within_20pct(pred_val, exp_val):
            exact20_count += 1

    type_match = correct / total if total > 0 else 0.0
    exact20 = exact20_count / total if total > 0 else 0.0
    return CaseResult(
        case=case,
        mode=mode,
        predictions=predictions,
        correct_slots=correct,
        total_slots=total,
        type_match=type_match,
        exact20=exact20,
        all_correct=(correct == total),
    )


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------

@dataclass
class FamilySummary:
    family: str
    mode: str
    case_count: int
    fixed_count: int
    fix_rate: float
    avg_type_match: float
    avg_exact20: float


def _summarize(results: list[CaseResult], family: str, mode: str) -> FamilySummary:
    family_results = [r for r in results if r.case.family == family and r.mode == mode]
    if not family_results:
        return FamilySummary(family, mode, 0, 0, 0.0, 0.0, 0.0)
    n = len(family_results)
    fixed = sum(1 for r in family_results if r.all_correct)
    return FamilySummary(
        family=family,
        mode=mode,
        case_count=n,
        fixed_count=fixed,
        fix_rate=fixed / n,
        avg_type_match=sum(r.type_match for r in family_results) / n,
        avg_exact20=sum(r.exact20 for r in family_results) / n,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(
    modes: tuple[str, ...] = ("basic", "ops", "semantic", "full"),
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all Group 1 evaluation cases across ablation modes.

    Parameters
    ----------
    modes:
        Ablation levels to evaluate.  Defaults to all four levels:
        ``"basic"``, ``"ops"``, ``"semantic"``, ``"full"``.
    verbose:
        Print a human-readable report to stdout.

    Returns
    -------
    dict with keys:
        ``results``  — list of ``CaseResult`` objects
        ``summaries`` — dict mapping (family, mode) → ``FamilySummary``
        ``overall``  — dict mapping mode → overall fix_rate
    """
    all_results: list[CaseResult] = []
    for mode in modes:
        for case in _CASES:
            result = _evaluate_case(case, mode)
            all_results.append(result)

    families = sorted({c.family for c in _CASES})
    summaries: dict[tuple[str, str], FamilySummary] = {}
    for family in families:
        for mode in modes:
            key = (family, mode)
            summaries[key] = _summarize(all_results, family, mode)

    overall: dict[str, float] = {}
    for mode in modes:
        mode_results = [r for r in all_results if r.mode == mode]
        n = len(mode_results)
        fixed = sum(1 for r in mode_results if r.all_correct)
        overall[mode] = fixed / n if n > 0 else 0.0

    if verbose:
        _print_report(all_results, summaries, overall, modes, families)

    return {"results": all_results, "summaries": summaries, "overall": overall}


def _print_report(
    all_results: list[CaseResult],
    summaries: dict[tuple[str, str], FamilySummary],
    overall: dict[str, float],
    modes: tuple[str, ...],
    families: list[str],
) -> None:
    width = 80
    print("=" * width)
    print("GROUP 1 HARD-FAMILY EVALUATION — measure/attribute-aware linking")
    print("=" * width)

    # Per-case detail for the "full" mode
    print("\n--- Per-case results (full ablation) ---")
    header = f"{'Case':<50} {'Family':<20} {'TypeMatch':>10} {'Exact20':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r.mode != "full":
            continue
        status = "✓" if r.all_correct else "✗"
        print(
            f"{status} {r.case.name:<48} {r.case.family:<20} "
            f"{r.type_match:>10.2f} {r.exact20:>8.2f}"
        )

    # Per-family summary across ablation modes
    print("\n--- Family summaries across ablation levels ---")
    header2 = f"{'Family':<25} {'Mode':<8} {'N':>4} {'Fixed':>6} {'FixRate':>8} {'TypeMatch':>10} {'Exact20':>8}"
    print(header2)
    print("-" * len(header2))
    for family in families:
        for mode in modes:
            s = summaries[(family, mode)]
            print(
                f"  {family:<23} {mode:<8} {s.case_count:>4} {s.fixed_count:>6} "
                f"{s.fix_rate:>8.2f} {s.avg_type_match:>10.2f} {s.avg_exact20:>8.2f}"
            )

    # Overall fix rate per ablation mode
    print("\n--- Overall fix rate by ablation level ---")
    for mode in modes:
        n = len([r for r in all_results if r.mode == mode])
        fixed = sum(1 for r in all_results if r.mode == mode and r.all_correct)
        print(f"  {mode:<8}: {fixed:>3}/{n:<3}  fix_rate={overall[mode]:.2f}")

    # Regression check: full vs basic
    print("\n--- Regressions (basic correct, full wrong) ---")
    basic_correct = {r.case.name for r in all_results if r.mode == "basic" and r.all_correct}
    full_correct = {r.case.name for r in all_results if r.mode == "full" and r.all_correct}
    regressions = basic_correct - full_correct
    if regressions:
        for name in sorted(regressions):
            print(f"  REGRESSION: {name}")
    else:
        print("  None — no regressions detected.")

    print("=" * width)


if __name__ == "__main__":
    run_evaluation()
