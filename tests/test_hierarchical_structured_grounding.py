"""Tests for hierarchical_structured_grounding assignment mode."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.hierarchical_structured_grounding import run_hierarchical_structured_grounding


def _run(query: str, slots: list[str], mode: str = "full"):
    return run_hierarchical_structured_grounding(query, "orig", slots, ablation_mode=mode)


def test_objective_coefficient_vs_resource_bound():
    q = "The goal is to minimize cost where each unit costs 20 dollars. The total budget is 5000 dollars."
    vals, _, diag = _run(q, ["cost_per_unit", "total_budget"])
    assert abs(float(vals["cost_per_unit"]) - 20.0) < 1.0
    assert abs(float(vals["total_budget"]) - 5000.0) < 1.0
    assert diag["regions"]


def test_demand_vs_capacity():
    q = "Demand is 100 units. Warehouse capacity is 500 units."
    vals, _, _ = _run(q, ["minimum_demand", "warehouse_capacity"])
    assert abs(float(vals["minimum_demand"]) - 100.0) < 1.0
    assert abs(float(vals["warehouse_capacity"]) - 500.0) < 1.0


def test_lower_vs_upper_bound():
    q = "At least 10 workers and at most 30 workers are allowed."
    vals, _, _ = _run(q, ["min_workers", "max_workers"])
    assert abs(float(vals["min_workers"]) - 10.0) < 1.0
    assert abs(float(vals["max_workers"]) - 30.0) < 1.0


def test_count_like_vs_quantity_like():
    q = "The factory produces three products and has capacity for 100 units."
    vals, _, _ = _run(q, ["num_products", "capacity"])
    assert abs(float(vals["num_products"]) - 3.0) < 0.1
    assert abs(float(vals["capacity"]) - 100.0) < 1.0


def test_total_vs_per_unit():
    q = "Each radio ad costs 20 dollars. The total advertising budget is 5000 dollars."
    vals, _, _ = _run(q, ["cost_per_unit", "total_budget"])
    assert abs(float(vals["cost_per_unit"]) - 20.0) < 1.0
    assert abs(float(vals["total_budget"]) - 5000.0) < 1.0


def test_long_mixed_prompt_localizes_values():
    q = (
        "Maximize profit with each premium unit earning 12 dollars and each basic unit earning 8 dollars. "
        "Demand for basic units is at least 40. "
        "Machine capacity is 300 hours total, and each premium unit requires 3 hours while each basic unit requires 2 hours. "
        "The total budget is 5000 dollars."
    )
    slots = ["profit_per_premium_unit", "profit_per_basic_unit", "minimum_basic_demand", "machine_capacity", "total_budget"]
    vals, _, diag = _run(q, slots)
    assert abs(float(vals["profit_per_premium_unit"]) - 12.0) < 1.0
    assert abs(float(vals["profit_per_basic_unit"]) - 8.0) < 1.0
    assert abs(float(vals["minimum_basic_demand"]) - 40.0) < 1.0
    assert abs(float(vals["machine_capacity"]) - 300.0) < 1.0
    assert abs(float(vals["total_budget"]) - 5000.0) < 1.0
    assert "slot_roles" in diag and "localized_mentions" in diag and "per_slot_candidates" in diag


def test_mode_is_registered_in_cli_choices():
    source = (ROOT / "tools" / "nlp4lp_downstream_utility.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--assignment-mode":
                    for kw in node.keywords:
                        if kw.arg == "choices" and isinstance(kw.value, ast.Tuple):
                            for el in kw.value.elts:
                                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                                    found.add(el.value)
    assert "hierarchical_structured_grounding" in found
    assert "hierarchical_structured_grounding_no_regions" in found
    assert "hierarchical_structured_grounding_no_search" in found
