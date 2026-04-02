"""Deterministic tests for counterfactual_grounding_refinement."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.search_structured_grounding import (  # noqa: E402
    build_slot_candidates,
    counterfactual_grounding_refinement,
)


def _prep(query: str, slots: list[str]):
    mentions, slot_irs, candidates, _ = build_slot_candidates(query, "orig", slots)
    slot_by_name = {s.name: s for s in slot_irs}
    mention_by_id = {m.mention_id: m for m in mentions}
    per_slot_candidates = {
        sn: [
            {
                "slot_name": c.slot_name,
                "mention_id": c.mention_id,
                "mention_value": c.mention_value,
                "mention_raw": c.mention_raw,
                "local_score": c.local_score,
                "local_features": c.local_features,
                "is_null": c.is_null,
            }
            for c in lst
        ]
        for sn, lst in candidates.items()
    }
    return mention_by_id, slot_by_name, per_slot_candidates


def _find_mid(mention_by_id, value: float) -> int:
    for mid, m in mention_by_id.items():
        if m.tok.value is not None and abs(float(m.tok.value) - value) < 1e-6:
            return mid
    raise AssertionError(f"mention with value {value} not found")


def _run_refine(query: str, slots: list[str], initial_mid_by_slot: dict[str, int | None]):
    mention_by_id, slot_by_name, per_slot_candidates = _prep(query, slots)
    filled_mentions = {
        s: mention_by_id[mid]
        for s, mid in initial_mid_by_slot.items()
        if mid is not None
    }
    filled_values = {
        s: (mention_by_id[mid].tok.value if mention_by_id[mid].tok.value is not None else mention_by_id[mid].tok.raw)
        for s, mid in initial_mid_by_slot.items()
        if mid is not None
    }
    return counterfactual_grounding_refinement(
        query,
        "orig",
        slots,
        filled_values=filled_values,
        filled_mentions=filled_mentions,
        per_slot_candidates=per_slot_candidates,
        slot_by_name=slot_by_name,
        mention_by_id=mention_by_id,
    )


def test_total_vs_per_unit_refinement_fix():
    q = "The total advertising budget is 5000 dollars. Each radio ad costs 20 dollars."
    mention_by_id, _, _ = _prep(q, ["total_budget", "cost_per_unit"])
    v5000 = _find_mid(mention_by_id, 5000.0)
    v20 = _find_mid(mention_by_id, 20.0)
    vals, _, diag = _run_refine(q, ["total_budget", "cost_per_unit"], {
        "total_budget": v20,
        "cost_per_unit": v5000,
    })
    assert abs(float(vals["total_budget"]) - 5000.0) < 1.0
    assert abs(float(vals["cost_per_unit"]) - 20.0) < 1.0
    assert diag["refined_score"] >= diag["original_score"]


def test_min_max_inversion_refinement_fix():
    q = "Production must be at least 100 units and at most 500 units."
    mention_by_id, _, _ = _prep(q, ["min_production", "max_production"])
    v100 = _find_mid(mention_by_id, 100.0)
    v500 = _find_mid(mention_by_id, 500.0)
    vals, _, _ = _run_refine(q, ["min_production", "max_production"], {
        "min_production": v500,
        "max_production": v100,
    })
    assert abs(float(vals["min_production"]) - 100.0) < 1.0
    assert abs(float(vals["max_production"]) - 500.0) < 1.0


def test_percent_scalar_refinement_fix():
    q = "The discount rate is 20% and the unit price is 50."
    mention_by_id, _, _ = _prep(q, ["discount_percent", "unit_price"])
    v02 = _find_mid(mention_by_id, 0.2)
    v50 = _find_mid(mention_by_id, 50.0)
    vals, _, _ = _run_refine(q, ["discount_percent", "unit_price"], {
        "discount_percent": v50,
        "unit_price": v02,
    })
    assert abs(float(vals["discount_percent"]) - 0.2) < 0.01
    assert abs(float(vals["unit_price"]) - 50.0) < 1.0


def test_duplicate_mention_conflict_refinement_fix():
    q = "The budget is 40 dollars. Profit per unit is 20 dollars."
    mention_by_id, _, _ = _prep(q, ["total_budget", "profit_per_unit"])
    v40 = _find_mid(mention_by_id, 40.0)
    vals, mentions, _ = _run_refine(q, ["total_budget", "profit_per_unit"], {
        "total_budget": v40,
        "profit_per_unit": v40,
    })
    mids = [m.mention_id for m in mentions.values()]
    assert len(set(mids)) == len(mids)
    assert "total_budget" in vals


def test_abstention_refinement_prefers_null_over_bad_fill():
    q = "The capacity is 90 units."
    mention_by_id, _, _ = _prep(q, ["capacity", "discount_percent"])
    v90 = _find_mid(mention_by_id, 90.0)
    vals, _, _ = _run_refine(q, ["capacity", "discount_percent"], {
        "capacity": v90,
        "discount_percent": v90,
    })
    assert abs(float(vals["capacity"]) - 90.0) < 1.0
    assert "discount_percent" not in vals


def test_noop_when_initial_assignment_is_good():
    q = "The total advertising budget is 5000 dollars. Each radio ad costs 20 dollars."
    mention_by_id, _, _ = _prep(q, ["total_budget", "cost_per_unit"])
    v5000 = _find_mid(mention_by_id, 5000.0)
    v20 = _find_mid(mention_by_id, 20.0)
    vals, _, diag = _run_refine(q, ["total_budget", "cost_per_unit"], {
        "total_budget": v5000,
        "cost_per_unit": v20,
    })
    assert abs(float(vals["total_budget"]) - 5000.0) < 1.0
    assert abs(float(vals["cost_per_unit"]) - 20.0) < 1.0
    assert diag["refined_score"] >= diag["original_score"]
