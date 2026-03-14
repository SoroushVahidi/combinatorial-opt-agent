"""Group 3 clause-parallel evaluation — clause-local alignment scoring.

This module provides a focused ablation evaluation for Group 3 improvements:
  * left_entity_anchor_overlap  → directional entity-anchor discrimination
  * clause_entity_overlap       → clause-level entity alignment bonus
  * cross_clause_entity_penalty → cross-clause entity contamination penalty
  * detect_and_repair_parallel_swaps → conservative post-assignment swap repair

Usage::

    python tools/evaluate_group3_clause_parallel.py

The script reports per-case and aggregate results across ablation levels and
produces a machine-readable CSV artifact.  A markdown summary is also written.

Ablation levels evaluated
-------------------------
"basic"    — type compat + lexical overlap only
"ops"      — + operator/bound cues
"semantic" — + semantic roles + clause-local alignment (clause features ON)
"full"     — + entity anchoring + clause features + swap repair

Target slices / case families
------------------------------
case7_feed_entity  — parallel entity patterns (Feed A / Feed B style)
case7_product      — numeric-suffix entity patterns (Product 1 / Product 2)
case8_two_measure  — two-entity two-measure patterns (chair/dresser, heating/cooling)
no_overfire        — single-entity cases that must not regress

Metrics
-------
case_count          — number of test cases in family
fixed_count         — cases where system is fully correct
fix_rate            — fixed / case_count
regression_count    — cases correct at basic but broken at this mode
regression_rate     — regression_count / case_count
TypeMatch           — fraction of slots assigned the expected value
Exact20             — fraction within 20 % relative error
"""
from __future__ import annotations

import csv
import json
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
class Group3Case:
    """A single evaluation case for Group 3 clause-parallel improvements."""

    name: str
    query: str
    slots: list[str]
    expected: dict[str, float]
    family: str  # case7_feed_entity | case7_product | case8_two_measure | no_overfire


_CASES: list[Group3Case] = [
    # ------------------------------------------------------------------
    # case7_feed_entity — parallel-entity Feed A / Feed B patterns
    # ------------------------------------------------------------------
    Group3Case(
        name="feed_a_b_protein_only",
        query="Feed A contains 10 protein, while Feed B contains 7 protein.",
        slots=["ProteinFeedA", "ProteinFeedB"],
        expected={"ProteinFeedA": 10.0, "ProteinFeedB": 7.0},
        family="case7_feed_entity",
    ),
    Group3Case(
        name="feed_a_b_protein_reversed",
        query="Feed B contains 7 protein, while Feed A contains 10 protein.",
        slots=["ProteinFeedA", "ProteinFeedB"],
        expected={"ProteinFeedA": 10.0, "ProteinFeedB": 7.0},
        family="case7_feed_entity",
    ),
    Group3Case(
        name="feed_a_b_two_measures",
        query=(
            "Feed A contains 10 protein and 8 fat, "
            "while Feed B contains 7 protein and 15 fat."
        ),
        slots=["ProteinFeedA", "FatFeedA", "ProteinFeedB", "FatFeedB"],
        expected={
            "ProteinFeedA": 10.0,
            "FatFeedA": 8.0,
            "ProteinFeedB": 7.0,
            "FatFeedB": 15.0,
        },
        family="case7_feed_entity",
    ),
    Group3Case(
        name="feed_a_b_cross_clause_contamination",
        query="Feed A has 10 protein. Feed B has 15 fat.",
        slots=["ProteinFeedA", "FatFeedB"],
        expected={"ProteinFeedA": 10.0, "FatFeedB": 15.0},
        family="case7_feed_entity",
    ),
    Group3Case(
        name="labor_hours_a_b_reversed",
        query=(
            "Product B requires 7 labor hours and product A requires 3 labor hours."
        ),
        slots=["LaborHoursA", "LaborHoursB"],
        expected={"LaborHoursA": 3.0, "LaborHoursB": 7.0},
        family="case7_feed_entity",
    ),
    Group3Case(
        name="small_large_sculptures",
        query=(
            "Small sculptures require 2 labor hours and large sculptures require 6."
        ),
        slots=["LaborSmall", "LaborLarge"],
        expected={"LaborSmall": 2.0, "LaborLarge": 6.0},
        family="case7_feed_entity",
    ),
    # ------------------------------------------------------------------
    # case7_product — numeric-suffix entity patterns (Product 1 / Product 2)
    # ------------------------------------------------------------------
    Group3Case(
        name="product1_product2_labor_hours",
        query="Product 1 uses 3 hours of labor. Product 2 uses 7 hours of labor.",
        slots=["LaborProduct1", "LaborProduct2"],
        expected={"LaborProduct1": 3.0, "LaborProduct2": 7.0},
        family="case7_product",
    ),
    Group3Case(
        name="product1_product2_reversed",
        query="Product 2 uses 7 hours of labor. Product 1 uses 3 hours of labor.",
        slots=["LaborProduct1", "LaborProduct2"],
        expected={"LaborProduct1": 3.0, "LaborProduct2": 7.0},
        family="case7_product",
    ),
    Group3Case(
        name="cost_type1_type2",
        query="Type 1 costs 5 dollars and type 2 costs 12 dollars.",
        slots=["CostType1", "CostType2"],
        expected={"CostType1": 5.0, "CostType2": 12.0},
        family="case7_product",
    ),
    # ------------------------------------------------------------------
    # case8_two_measure — two-entity two-measure parallel patterns
    # ------------------------------------------------------------------
    Group3Case(
        name="chair_dresser_wood",
        query=(
            "Chair requires 2 wood units and dresser requires 5 wood units."
        ),
        slots=["ChairWood", "DresserWood"],
        expected={"ChairWood": 2.0, "DresserWood": 5.0},
        family="case8_two_measure",
    ),
    Group3Case(
        name="heating_cooling_single_entity",
        query="Regular glass requires 3 heating hours and 5 cooling hours.",
        slots=["HeatingHours", "CoolingHours"],
        expected={"HeatingHours": 3.0, "CoolingHours": 5.0},
        family="case8_two_measure",
    ),
    Group3Case(
        name="regular_tempered_heating_cooling_four_slots",
        query=(
            "Regular glass requires 3 heating hours and 5 cooling hours, "
            "whereas tempered glass requires 5 heating and 8 cooling."
        ),
        slots=["HeatingRegular", "CoolingRegular", "HeatingTempered", "CoolingTempered"],
        expected={
            "HeatingRegular": 3.0,
            "CoolingRegular": 5.0,
            "HeatingTempered": 5.0,
            "CoolingTempered": 8.0,
        },
        family="case8_two_measure",
    ),
    # ------------------------------------------------------------------
    # no_overfire — single-entity / simple cases that must not regress
    # ------------------------------------------------------------------
    Group3Case(
        name="single_entity_protein_fat",
        query="Feed A contains 10 protein and 8 fat.",
        slots=["ProteinFeedA", "FatFeedA"],
        expected={"ProteinFeedA": 10.0, "FatFeedA": 8.0},
        family="no_overfire",
    ),
    Group3Case(
        name="single_entity_labor_wood",
        query="Each table requires 4 labor hours and 6 units of wood.",
        slots=["LaborHoursPerTable", "WoodUnitsPerTable"],
        expected={"LaborHoursPerTable": 4.0, "WoodUnitsPerTable": 6.0},
        family="no_overfire",
    ),
    Group3Case(
        name="single_entity_bound_directions",
        query="At least 10 and at most 20 units.",
        slots=["MinUnits", "MaxUnits"],
        expected={"MinUnits": 10.0, "MaxUnits": 20.0},
        family="no_overfire",
    ),
    Group3Case(
        name="single_entity_total_vs_per_unit",
        query=(
            "There are 2000 labor hours available, and each product requires 3 labor hours."
        ),
        slots=["TotalLaborHours", "LaborHoursPerProduct"],
        expected={"TotalLaborHours": 2000.0, "LaborHoursPerProduct": 3.0},
        family="no_overfire",
    ),
    Group3Case(
        name="swap_repair_conservative_heating_cooling",
        query="Regular glass requires 3 heating hours and 5 cooling hours.",
        slots=["HeatingHours", "CoolingHours"],
        expected={"HeatingHours": 3.0, "CoolingHours": 5.0},
        family="no_overfire",
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
    case: Group3Case
    mode: str
    predictions: dict[str, float]
    correct_slots: int
    total_slots: int
    type_match: float
    exact20: float
    all_correct: bool


def _evaluate_case(case: Group3Case, mode: str) -> CaseResult:
    filled_values, filled_mentions, _diag = run_relation_aware_grounding(
        case.query, "orig", case.slots, ablation_mode=mode
    )
    predictions = {sn: m.value for sn, m in filled_mentions.items()}

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
    regression_count: int
    regression_rate: float
    avg_type_match: float
    avg_exact20: float


def _summarize(
    all_results: list[CaseResult],
    family: str,
    mode: str,
    basic_correct: set[str],
) -> FamilySummary:
    family_results = [r for r in all_results if r.case.family == family and r.mode == mode]
    if not family_results:
        return FamilySummary(family, mode, 0, 0, 0.0, 0, 0.0, 0.0, 0.0)
    n = len(family_results)
    fixed = sum(1 for r in family_results if r.all_correct)
    mode_correct = {r.case.name for r in family_results if r.all_correct}
    regressions = sum(
        1 for r in family_results
        if r.case.name in basic_correct and not r.all_correct
    )
    return FamilySummary(
        family=family,
        mode=mode,
        case_count=n,
        fixed_count=fixed,
        fix_rate=fixed / n,
        regression_count=regressions,
        regression_rate=regressions / n,
        avg_type_match=sum(r.type_match for r in family_results) / n,
        avg_exact20=sum(r.exact20 for r in family_results) / n,
    )


# ---------------------------------------------------------------------------
# CSV / JSON artifact generation
# ---------------------------------------------------------------------------


def _write_csv(
    all_results: list[CaseResult],
    summaries: dict[tuple[str, str], FamilySummary],
    overall: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a machine-readable CSV summary."""
    rows = []
    for (family, mode), s in summaries.items():
        rows.append({
            "family": family,
            "mode": mode,
            "case_count": s.case_count,
            "fixed_count": s.fixed_count,
            "fix_rate": f"{s.fix_rate:.3f}",
            "regression_count": s.regression_count,
            "regression_rate": f"{s.regression_rate:.3f}",
            "avg_type_match": f"{s.avg_type_match:.3f}",
            "avg_exact20": f"{s.avg_exact20:.3f}",
        })
    # Append overall rows
    for mode, fix_rate in overall.items():
        n = len([r for r in all_results if r.mode == mode])
        fixed = sum(1 for r in all_results if r.mode == mode and r.all_correct)
        rows.append({
            "family": "OVERALL",
            "mode": mode,
            "case_count": n,
            "fixed_count": fixed,
            "fix_rate": f"{fix_rate:.3f}",
            "regression_count": "",
            "regression_rate": "",
            "avg_type_match": "",
            "avg_exact20": "",
        })
    fieldnames = [
        "family", "mode", "case_count", "fixed_count", "fix_rate",
        "regression_count", "regression_rate", "avg_type_match", "avg_exact20",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    all_results: list[CaseResult],
    summaries: dict[tuple[str, str], FamilySummary],
    overall: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a machine-readable JSON summary."""
    data: dict[str, Any] = {
        "overall": {mode: round(v, 4) for mode, v in overall.items()},
        "by_family": {},
        "per_case_full_mode": [],
    }
    for (family, mode), s in summaries.items():
        if family not in data["by_family"]:
            data["by_family"][family] = {}
        data["by_family"][family][mode] = {
            "case_count": s.case_count,
            "fixed_count": s.fixed_count,
            "fix_rate": round(s.fix_rate, 4),
            "regression_count": s.regression_count,
            "regression_rate": round(s.regression_rate, 4),
            "avg_type_match": round(s.avg_type_match, 4),
            "avg_exact20": round(s.avg_exact20, 4),
        }
    for r in all_results:
        if r.mode == "full":
            data["per_case_full_mode"].append({
                "name": r.case.name,
                "family": r.case.family,
                "all_correct": r.all_correct,
                "type_match": round(r.type_match, 4),
                "exact20": round(r.exact20, 4),
                "predictions": {k: round(v, 6) for k, v in r.predictions.items()},
            })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _write_markdown(
    all_results: list[CaseResult],
    summaries: dict[tuple[str, str], FamilySummary],
    overall: dict[str, Any],
    modes: tuple[str, ...],
    families: list[str],
    output_path: Path,
) -> None:
    """Write a markdown summary report."""
    lines = [
        "# Group 3 Clause-Parallel Evaluation Report",
        "",
        "## Summary: Overall fix rate by ablation level",
        "",
        "| Mode | Fixed | Total | FixRate |",
        "|------|------:|------:|--------:|",
    ]
    for mode in modes:
        mode_results = [r for r in all_results if r.mode == mode]
        n = len(mode_results)
        fixed = sum(1 for r in mode_results if r.all_correct)
        lines.append(f"| {mode} | {fixed} | {n} | {overall[mode]:.2f} |")

    lines += [
        "",
        "## Family summaries across ablation levels",
        "",
        "| Family | Mode | N | Fixed | FixRate | Regressions | TypeMatch | Exact20 |",
        "|--------|------|--:|------:|--------:|------------:|----------:|--------:|",
    ]
    for family in families:
        for mode in modes:
            s = summaries[(family, mode)]
            lines.append(
                f"| {family} | {mode} | {s.case_count} | {s.fixed_count} "
                f"| {s.fix_rate:.2f} | {s.regression_count} "
                f"| {s.avg_type_match:.2f} | {s.avg_exact20:.2f} |"
            )

    lines += [
        "",
        "## Per-case results (full ablation)",
        "",
        "| Status | Case | Family | TypeMatch | Exact20 |",
        "|--------|------|--------|----------:|--------:|",
    ]
    for r in all_results:
        if r.mode != "full":
            continue
        status = "✓" if r.all_correct else "✗"
        lines.append(
            f"| {status} | {r.case.name} | {r.case.family} "
            f"| {r.type_match:.2f} | {r.exact20:.2f} |"
        )

    lines += [
        "",
        "## Regressions (basic correct, full wrong)",
        "",
    ]
    basic_correct = {r.case.name for r in all_results if r.mode == "basic" and r.all_correct}
    full_correct = {r.case.name for r in all_results if r.mode == "full" and r.all_correct}
    regressions = basic_correct - full_correct
    if regressions:
        for name in sorted(regressions):
            lines.append(f"- REGRESSION: `{name}`")
    else:
        lines.append("None — no regressions detected.")

    lines += [
        "",
        "## Clause-local scoring integration",
        "",
        "New Group 3 features wired into the main scoring pipeline:",
        "",
        "- **`clause_entity_overlap`** (semantic + full modes): bonus when the mention's "
          "clause entity cues match the slot's entity words.",
        "- **`cross_clause_entity_penalty`** (semantic + full modes): penalty when the "
          "slot's best entity evidence is in a different clause than the mention.",
        "- **`clause_measure_overlap`**: computed but diagnostic-only (not in scoring).",
        "",
        "These features are neutral (0) for single-clause queries, preserving all "
        "existing Group 1/Group 2 behavior.",
        "",
        "Weights used:",
        "",
        "| Weight key | semantic | full |",
        "|-----------|--------:|-----:|",
        "| `clause_entity_alignment_bonus` | 1.2 | 1.2 |",
        "| `cross_clause_entity_penalty_weight` | -1.2 | -1.2 |",
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_evaluation(
    modes: tuple[str, ...] = ("basic", "ops", "semantic", "full"),
    verbose: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all Group 3 evaluation cases across ablation modes.

    Parameters
    ----------
    modes:
        Ablation levels to evaluate.
    verbose:
        Print a human-readable report to stdout.
    output_dir:
        If given, write CSV, JSON, and Markdown artifacts to this directory.

    Returns
    -------
    dict with keys ``results``, ``summaries``, ``overall``.
    """
    all_results: list[CaseResult] = []
    for mode in modes:
        for case in _CASES:
            result = _evaluate_case(case, mode)
            all_results.append(result)

    basic_correct = {r.case.name for r in all_results if r.mode == "basic" and r.all_correct}

    families = sorted({c.family for c in _CASES})
    summaries: dict[tuple[str, str], FamilySummary] = {}
    for family in families:
        for mode in modes:
            key = (family, mode)
            summaries[key] = _summarize(all_results, family, mode, basic_correct)

    overall: dict[str, float] = {}
    for mode in modes:
        mode_results = [r for r in all_results if r.mode == mode]
        n = len(mode_results)
        fixed = sum(1 for r in mode_results if r.all_correct)
        overall[mode] = fixed / n if n > 0 else 0.0

    if verbose:
        _print_report(all_results, summaries, overall, modes, families, basic_correct)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(all_results, summaries, overall, output_dir / "group3_eval.csv")
        _write_json(all_results, summaries, overall, output_dir / "group3_eval.json")
        _write_markdown(
            all_results, summaries, overall, modes, families,
            output_dir / "group3_eval.md",
        )
        if verbose:
            print(f"\nArtifacts written to {output_dir}/")

    return {"results": all_results, "summaries": summaries, "overall": overall}


def _print_report(
    all_results: list[CaseResult],
    summaries: dict[tuple[str, str], FamilySummary],
    overall: dict[str, float],
    modes: tuple[str, ...],
    families: list[str],
    basic_correct: set[str],
) -> None:
    width = 90
    print("=" * width)
    print("GROUP 3 CLAUSE-PARALLEL EVALUATION — clause-local alignment scoring")
    print("=" * width)

    # Per-case detail for the "full" mode
    print("\n--- Per-case results (full ablation) ---")
    header = f"{'Case':<55} {'Family':<22} {'TypeMatch':>10} {'Exact20':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r.mode != "full":
            continue
        status = "✓" if r.all_correct else "✗"
        print(
            f"{status} {r.case.name:<53} {r.case.family:<22} "
            f"{r.type_match:>10.2f} {r.exact20:>8.2f}"
        )

    # Per-family summary across ablation modes
    print("\n--- Family summaries across ablation levels ---")
    header2 = (
        f"{'Family':<25} {'Mode':<8} {'N':>4} {'Fixed':>6} {'FixRate':>8} "
        f"{'Regress':>8} {'TypeMatch':>10} {'Exact20':>8}"
    )
    print(header2)
    print("-" * len(header2))
    for family in families:
        for mode in modes:
            s = summaries[(family, mode)]
            print(
                f"  {family:<23} {mode:<8} {s.case_count:>4} {s.fixed_count:>6} "
                f"{s.fix_rate:>8.2f} {s.regression_count:>8} "
                f"{s.avg_type_match:>10.2f} {s.avg_exact20:>8.2f}"
            )

    # Overall fix rate per ablation mode
    print("\n--- Overall fix rate by ablation level ---")
    for mode in modes:
        n = len([r for r in all_results if r.mode == mode])
        fixed = sum(1 for r in all_results if r.mode == mode and r.all_correct)
        print(f"  {mode:<8}: {fixed:>3}/{n:<3}  fix_rate={overall[mode]:.2f}")

    # Regression check
    print("\n--- Regressions (basic correct, full wrong) ---")
    full_correct = {r.case.name for r in all_results if r.mode == "full" and r.all_correct}
    regressions = basic_correct - full_correct
    if regressions:
        for name in sorted(regressions):
            print(f"  REGRESSION: {name}")
    else:
        print("  None — no regressions detected.")

    print("=" * width)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Group 3 clause-parallel ablation evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write CSV/JSON/Markdown artifacts (default: no files written)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["basic", "ops", "semantic", "full"],
        help="Ablation modes to evaluate",
    )
    args = parser.parse_args()

    run_evaluation(
        modes=tuple(args.modes),
        verbose=True,
        output_dir=args.output_dir,
    )
