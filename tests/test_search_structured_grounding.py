"""Tests for search_structured_grounding assignment mode."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.search_structured_grounding import run_search_structured_grounding


def _run(query: str, slots: list[str], use_global: bool = True):
    return run_search_structured_grounding(query, "orig", slots, use_global=use_global)


def test_total_vs_per_unit_case():
    q = "The total advertising budget is 5000 dollars. Each radio ad costs 20 dollars."
    vals, _, diag = _run(q, ["total_budget", "cost_per_unit"])
    assert abs(float(vals["total_budget"]) - 5000.0) < 1.0
    assert abs(float(vals["cost_per_unit"]) - 20.0) < 1.0
    assert diag["n_slots"] == 2


def test_lower_vs_upper_bound_case():
    q = "Production must be at least 100 units and at most 500 units."
    vals, _, _ = _run(q, ["min_production", "max_production"])
    assert abs(float(vals["min_production"]) - 100.0) < 1.0
    assert abs(float(vals["max_production"]) - 500.0) < 1.0


def test_percent_vs_scalar_case():
    q = "The discount rate is 20% and the unit price is 50."
    vals, _, _ = _run(q, ["discount_percent", "unit_price"])
    assert abs(float(vals["discount_percent"]) - 0.2) < 0.01
    assert abs(float(vals["unit_price"]) - 50.0) < 1.0


def test_count_like_vs_quantity_like_case():
    q = "The company makes three products and has capacity for 100 units."
    vals, _, _ = _run(q, ["product_count", "capacity"])
    assert abs(float(vals["product_count"]) - 3.0) < 0.1
    assert abs(float(vals["capacity"]) - 100.0) < 1.0


def test_duplicate_mention_conflict_avoided():
    q = "The budget is 40 dollars. Profit per unit is 20 dollars."
    _, mentions, _ = _run(q, ["total_budget", "profit_per_unit"])
    mids = [m.mention_id for m in mentions.values()]
    assert len(set(mids)) == len(mids)


def test_abstention_when_evidence_weak():
    q = "The capacity is 90 units."
    vals, _, _ = _run(q, ["capacity", "discount_percent"])
    assert "capacity" in vals
    assert "discount_percent" not in vals


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
    assert "search_structured_grounding" in found
    assert "search_structured_grounding_no_global" in found
