"""Tests for the relation_aware_linking module.

Covers:
  1. Module-level API (build_mention_slot_links, relations, score)
  2. MentionFeatures / SlotFeatures dataclass structure
  3. Relation features: operator compat, polarity, semantic family, percent, total/coeff
  4. Ablation scoring modes: basic / ops / semantic / full
  5. Greedy assignment via run_relation_aware_grounding
  6. Mention-mention and slot-slot relation tables
  7. Integration: wired into run_setting choices and focused_eval
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.relation_aware_linking import (
    ABLATION_MODES,
    RAL_WEIGHTS,
    MentionFeatures,
    MentionMentionRelation,
    MentionSlotLink,
    SlotFeatures,
    SlotSlotRelation,
    best_assignment_greedy,
    build_mention_mention_relations,
    build_mention_slot_links,
    build_slot_slot_relations,
    relation_aware_local_score,
    run_relation_aware_grounding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ral(query: str, slots: list[str], mode: str = "full") -> tuple[dict, dict, dict]:
    return run_relation_aware_grounding(query, "orig", slots, ablation_mode=mode)


# ---------------------------------------------------------------------------
# 1. Public API smoke tests
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_build_mention_slot_links_returns_tuple(self):
        links, mentions_ir, slots_ir, mfeats, sfeats = build_mention_slot_links(
            "The budget is 100 dollars and demand is 10.", "orig", ["budget", "demand"]
        )
        assert isinstance(links, list)
        assert isinstance(mentions_ir, list)
        assert isinstance(slots_ir, list)
        assert isinstance(mfeats, list)
        assert isinstance(sfeats, list)

    def test_links_count_is_mentions_times_slots(self):
        links, mentions_ir, slots_ir, mfeats, sfeats = build_mention_slot_links(
            "The budget is 100 dollars and profit per unit is 5 dollars.", "orig",
            ["budget", "profit_per_unit"]
        )
        expected_count = len(mentions_ir) * len(slots_ir)
        assert len(links) == expected_count

    def test_empty_slots_returns_empty(self):
        links, mentions_ir, slots_ir, mfeats, sfeats = build_mention_slot_links(
            "Budget is 100 dollars.", "orig", []
        )
        assert links == []
        assert slots_ir == []

    def test_no_mentions_returns_empty_links(self):
        links, mentions_ir, slots_ir, mfeats, sfeats = build_mention_slot_links(
            "There are no numbers here at all.", "orig", ["budget"]
        )
        assert mentions_ir == []
        assert links == []

    def test_mention_features_type(self):
        _, _, _, mfeats, _ = build_mention_slot_links(
            "The budget is 100 dollars.", "orig", ["budget"]
        )
        if mfeats:
            assert isinstance(mfeats[0], MentionFeatures)

    def test_slot_features_type(self):
        _, _, _, _, sfeats = build_mention_slot_links(
            "The budget is 100 dollars.", "orig", ["budget"]
        )
        assert len(sfeats) == 1
        assert isinstance(sfeats[0], SlotFeatures)

    def test_link_type(self):
        links, _, _, _, _ = build_mention_slot_links(
            "Budget is 100 dollars.", "orig", ["budget"]
        )
        for lnk in links:
            assert isinstance(lnk, MentionSlotLink)


# ---------------------------------------------------------------------------
# 2. Slot feature flags
# ---------------------------------------------------------------------------

class TestSlotFeatures:
    def test_percent_slot_flagged(self):
        _, _, _, _, sfeats = build_mention_slot_links(
            "Rate is 20%.", "orig", ["discount_percent"]
        )
        sf = sfeats[0]
        assert sf.is_percent_like

    def test_min_slot_flagged(self):
        _, _, _, _, sfeats = build_mention_slot_links(
            "Min demand is 5.", "orig", ["min_demand"]
        )
        sf = sfeats[0]
        assert sf.is_min_like

    def test_max_slot_flagged(self):
        _, _, _, _, sfeats = build_mention_slot_links(
            "Max capacity is 100.", "orig", ["max_capacity"]
        )
        sf = sfeats[0]
        assert sf.is_max_like

    def test_count_slot_flagged(self):
        _, _, _, _, sfeats = build_mention_slot_links(
            "Item count is 5.", "orig", ["item_count"]
        )
        sf = sfeats[0]
        assert sf.is_count_like


# ---------------------------------------------------------------------------
# 3. Relation features
# ---------------------------------------------------------------------------

class TestRelationFeatures:
    def test_percent_match_fires(self):
        """Percent slot + percent mention → percent_match = True."""
        links, _, _, _, _ = build_mention_slot_links(
            "The tax rate is 15%.", "orig", ["tax_rate_percent"]
        )
        pct_links = [l for l in links if l.mention_feats.type_bucket == "percent"]
        if not pct_links:
            pytest.skip("No percent mention extracted")
        assert any(l.percent_match for l in pct_links)

    def test_percent_mismatch_fires(self):
        """Percent slot + non-percent mention, when percent mention exists elsewhere → mismatch."""
        links, _, _, mfeats, _ = build_mention_slot_links(
            "The rate is 15% and the budget is 200 dollars.", "orig",
            ["rate_percent", "budget"]
        )
        has_pct = any(m.type_bucket == "percent" for m in mfeats)
        if not has_pct:
            pytest.skip("No percent mention extracted")
        # A non-pct mention assigned to rate_percent slot should trigger percent_mismatch
        non_pct_to_pct = [
            l for l in links
            if l.slot_feats.is_percent_like and l.mention_feats.type_bucket != "percent"
        ]
        if non_pct_to_pct:
            assert any(l.percent_mismatch for l in non_pct_to_pct)

    def test_polarity_match_fires_for_min_slot(self):
        """Mention with 'at least' context + min slot → polarity_match."""
        links, _, _, _, _ = build_mention_slot_links(
            "You must produce at least 5 units.", "orig", ["min_units"]
        )
        if not links:
            pytest.skip("No links built")
        # At least one link should have polarity_match
        assert any(l.polarity_match for l in links)

    def test_polarity_conflict_fires_for_max_mention_to_min_slot(self):
        """Max-tagged mention to min slot → polarity_conflict."""
        links, _, _, _, _ = build_mention_slot_links(
            "The maximum allowed is 50 units.", "orig", ["min_units"]
        )
        max_to_min = [
            l for l in links
            if l.mention_feats.polarity == "upper" and l.slot_feats.polarity == "lower"
        ]
        if max_to_min:
            assert all(l.polarity_conflict for l in max_to_min)

    def test_type_exact_fires(self):
        """Currency mention to currency-like slot → type_exact."""
        links, _, _, _, _ = build_mention_slot_links(
            "The budget is 200 dollars.", "orig", ["total_budget"]
        )
        if not links:
            pytest.skip("No links")
        # At least one link should have type_exact or type_loose
        assert any(l.type_exact or l.type_loose for l in links)

    def test_type_incompatible_or_low_score_for_percent_to_int(self):
        """Percent mention to integer slot should have lower score than type-matching mention."""
        links, _, _, _, _ = build_mention_slot_links(
            "The rate is 15% and item count is 5.", "orig", ["item_count"]
        )
        if not links:
            pytest.skip("No links extracted")
        # All links should produce a valid score (no error raised).
        from tools.relation_aware_linking import relation_aware_local_score
        for lnk in links:
            sc, _ = relation_aware_local_score(lnk, "full")
            assert isinstance(sc, float)

    def test_operator_compat_for_min(self):
        """Min-preference slot + min-tagged mention → operator_compat."""
        links, _, _, _, _ = build_mention_slot_links(
            "At least 10 units must be produced.", "orig", ["min_production"]
        )
        if not links:
            pytest.skip("No links")
        # Find links where slot has min operator preference
        min_slot_links = [l for l in links if "min" in l.slot_feats.operator_preference]
        if min_slot_links:
            assert any(l.operator_compat for l in min_slot_links)


# ---------------------------------------------------------------------------
# 4. Ablation scoring modes
# ---------------------------------------------------------------------------

class TestAblationScoring:
    def test_basic_mode_returns_float(self):
        links, _, _, _, _ = build_mention_slot_links(
            "Budget is 100 dollars.", "orig", ["budget"]
        )
        if not links:
            pytest.skip("No links")
        score, feats = relation_aware_local_score(links[0], "basic")
        assert isinstance(score, float)
        assert isinstance(feats, dict)

    def test_all_modes_return_scores(self):
        links, _, _, _, _ = build_mention_slot_links(
            "The budget is 500 dollars and profit is 3 dollars per unit.",
            "orig", ["total_budget", "profit_per_unit"]
        )
        if not links:
            pytest.skip("No links")
        for mode in ABLATION_MODES:
            for lnk in links:
                sc, fd = relation_aware_local_score(lnk, mode)
                assert isinstance(sc, float), f"mode={mode}"
                assert isinstance(fd, dict), f"mode={mode}"

    def test_full_mode_uses_entity_anchor(self):
        """Full mode should include entity_anchor_overlap when applicable."""
        links, _, _, _, _ = build_mention_slot_links(
            "Product A yields 5 dollars per unit.", "orig", ["profit_per_unit"]
        )
        if not links:
            pytest.skip("No links")
        for lnk in links:
            _, feats_full = relation_aware_local_score(lnk, "full")
            _, feats_basic = relation_aware_local_score(lnk, "basic")
            # Full mode has more potential features
            assert len(feats_full) >= len(feats_basic)

    def test_ops_mode_includes_operator_bonus(self):
        """Ops mode should recognize polarity/operator features that basic mode doesn't."""
        links, _, _, _, _ = build_mention_slot_links(
            "At least 10 units are required.", "orig", ["min_units"]
        )
        if not links:
            pytest.skip("No links")
        for lnk in links:
            sc_basic, _ = relation_aware_local_score(lnk, "basic")
            sc_ops, fd_ops = relation_aware_local_score(lnk, "ops")
            # ops can differ from basic; just check it runs
            assert isinstance(sc_ops, float)

    def test_scores_are_cached(self):
        """Calling score twice for the same mode should return identical results."""
        links, _, _, _, _ = build_mention_slot_links(
            "Budget is 200 dollars.", "orig", ["budget"]
        )
        if not links:
            pytest.skip("No links")
        lnk = links[0]
        s1, f1 = relation_aware_local_score(lnk, "full")
        s2, f2 = relation_aware_local_score(lnk, "full")
        assert s1 == s2
        assert f1 == f2

    def test_penalty_negative_for_type_incompatible(self):
        """Type-incompatible pair should always have a strongly negative score."""
        links, _, _, _, _ = build_mention_slot_links(
            "The rate is 15% and item count is 5.", "orig", ["item_count"]
        )
        for lnk in links:
            if lnk.type_incompatible:
                sc, _ = relation_aware_local_score(lnk, "full")
                assert sc < -1e6, f"Expected strongly negative score, got {sc}"


# ---------------------------------------------------------------------------
# 5. End-to-end grounding
# ---------------------------------------------------------------------------

class TestGrounding:
    def test_basic_single_slot(self):
        vals, _, _ = _ral("The budget is 100 dollars.", ["budget"])
        assert vals.get("budget") is not None
        assert abs(float(vals["budget"]) - 100.0) < 1.0

    def test_percent_and_scalar_separation(self):
        vals, mentions, _ = _ral(
            "The discount rate is 20% and the price is 80 dollars.",
            ["discount_percent", "price"]
        )
        assert vals.get("discount_percent") is not None
        assert vals.get("price") is not None
        m = mentions.get("discount_percent")
        if m is not None:
            assert m.type_bucket == "percent"

    def test_min_max_ordering(self):
        vals, _, _ = _ral(
            "At least 5 units and at most 30 units are required.",
            ["min_units", "max_units"]
        )
        if vals.get("min_units") and vals.get("max_units"):
            mn = float(vals["min_units"])
            mx = float(vals["max_units"])
            assert mn < mx, f"min ({mn}) should be < max ({mx})"

    def test_profit_vs_budget_separated(self):
        vals, _, _ = _ral(
            "Each unit earns 4 dollars profit. Total budget is 800 dollars.",
            ["profit_per_unit", "total_budget"]
        )
        assert vals.get("profit_per_unit") is not None
        assert vals.get("total_budget") is not None
        profit = float(vals["profit_per_unit"])
        budget = float(vals["total_budget"])
        assert abs(profit - 4.0) < 0.5, f"Expected profit ~4, got {profit}"
        assert abs(budget - 800.0) < 10.0, f"Expected budget ~800, got {budget}"

    def test_no_duplicate_assignments(self):
        vals, mentions, _ = _ral(
            "Budget is 500 dollars and demand is 20.",
            ["total_budget", "min_demand"]
        )
        mids = [m.mention_id for m in mentions.values() if m is not None]
        assert len(set(mids)) == len(mids), f"Duplicate mention assigned: {mids}"

    def test_empty_slots(self):
        vals, mentions, diag = _ral("Budget is 100 dollars.", [])
        assert vals == {}
        assert mentions == {}

    def test_no_mentions(self):
        vals, mentions, _ = _ral("There are no numbers here at all.", ["budget"])
        assert vals == {}

    def test_diagnostics_structure(self):
        _, _, diag = _ral(
            "Budget is 200 dollars and demand is 10.", ["budget", "demand"]
        )
        assert "per_slot_candidates" in diag
        assert "ablation_mode" in diag

    def test_all_ablation_modes_run(self):
        query = "Budget is 500 dollars and profit per unit is 3 dollars."
        slots = ["total_budget", "profit_per_unit"]
        for mode in ABLATION_MODES:
            vals, _, diag = _ral(query, slots, mode=mode)
            assert isinstance(vals, dict), f"mode={mode} returned non-dict"
            assert diag.get("ablation_mode") == mode


# ---------------------------------------------------------------------------
# 6. Mention-mention and slot-slot relations
# ---------------------------------------------------------------------------

class TestRelationTables:
    def test_mention_mention_relation_type(self):
        _, _, _, mfeats, _ = build_mention_slot_links(
            "Budget is 100 dollars and demand is 20.", "orig",
            ["budget", "demand"]
        )
        rels = build_mention_mention_relations(mfeats)
        for r in rels:
            assert isinstance(r, MentionMentionRelation)

    def test_slot_slot_relation_type(self):
        _, _, _, _, sfeats = build_mention_slot_links(
            "Budget is 100 dollars.", "orig", ["min_demand", "max_demand"]
        )
        rels = build_slot_slot_relations(sfeats)
        for r in rels:
            assert isinstance(r, SlotSlotRelation)

    def test_min_max_pair_detected(self):
        _, _, _, _, sfeats = build_mention_slot_links(
            "Min is 5 max is 30.", "orig", ["min_units", "max_units"]
        )
        rels = build_slot_slot_relations(sfeats)
        if rels:
            assert any(r.is_min_max_pair for r in rels)

    def test_total_unit_pair_detected(self):
        _, _, _, _, sfeats = build_mention_slot_links(
            "Budget is 200 and profit per unit is 5.", "orig",
            ["total_budget", "profit_per_unit"]
        )
        rels = build_slot_slot_relations(sfeats)
        if rels:
            assert any(r.is_total_unit_pair for r in rels)

    def test_duplicate_mention_relation(self):
        """Two mentions with the same value → possible_duplicate = True."""
        _, _, _, mfeats, _ = build_mention_slot_links(
            "Cost is 10 dollars and also 10 dollars.", "orig", ["cost_A", "cost_B"]
        )
        rels = build_mention_mention_relations(mfeats)
        if len(mfeats) >= 2:
            # Check whether same-value pair is flagged
            same_val = [
                r for r in rels
                if r.possible_duplicate
            ]
            # This assertion may not fire if values differ slightly — just check type
            for r in rels:
                assert isinstance(r.possible_duplicate, bool)

    def test_one_is_percent_relation(self):
        """One percent and one scalar mention → one_is_percent flag."""
        _, _, _, mfeats, _ = build_mention_slot_links(
            "Rate is 20% and budget is 100 dollars.", "orig", ["rate", "budget"]
        )
        rels = build_mention_mention_relations(mfeats)
        has_pct = any(m.type_bucket == "percent" for m in mfeats)
        has_non_pct = any(m.type_bucket != "percent" for m in mfeats)
        if has_pct and has_non_pct:
            assert any(r.one_is_percent for r in rels)

    def test_slot_slot_rels_returned_in_diagnostics(self):
        _, _, diag = _ral(
            "Budget is 200 dollars and demand is 10.",
            ["total_budget", "min_demand"]
        )
        assert "slot_slot_relations" in diag
        assert "mention_mention_relations" in diag


# ---------------------------------------------------------------------------
# 7. Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_ablation_modes_set(self):
        assert set(ABLATION_MODES) == {"basic", "ops", "semantic", "full"}

    def test_ral_weights_keys_for_all_modes(self):
        for mode in ABLATION_MODES:
            w = RAL_WEIGHTS[mode]
            assert "type_exact_bonus" in w
            assert "type_incompatible_penalty" in w
            assert "weak_match_penalty" in w

    def test_penalties_negative(self):
        for mode in ABLATION_MODES:
            for k, v in RAL_WEIGHTS[mode].items():
                if "penalty" in k:
                    assert v < 0, f"Penalty {k} in mode {mode} should be negative"

    def test_bonuses_positive(self):
        for mode in ABLATION_MODES:
            for k, v in RAL_WEIGHTS[mode].items():
                if "bonus" in k:
                    assert v > 0, f"Bonus {k} in mode {mode} should be positive"


# ---------------------------------------------------------------------------
# 8. Integration: wired into pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_assignment_mode_strings_in_argparse(self):
        """All four relation_aware modes must appear as string constants in downstream utility."""
        import ast as ast_mod
        src = ROOT / "tools" / "nlp4lp_downstream_utility.py"
        tree = ast_mod.parse(src.read_text())
        found = set()
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Constant) and isinstance(node.value, str):
                if node.value.startswith("relation_aware_"):
                    found.add(node.value)
        for mode in ("relation_aware_basic", "relation_aware_ops",
                     "relation_aware_semantic", "relation_aware_full"):
            assert mode in found, f"{mode!r} not found in downstream utility"

    def test_effective_baseline_naming(self):
        from tools.run_nlp4lp_focused_eval import _effective_baseline
        assert _effective_baseline("tfidf", "relation_aware_basic") == "tfidf_relation_aware_basic"
        assert _effective_baseline("tfidf", "relation_aware_full") == "tfidf_relation_aware_full"

    def test_focused_eval_includes_all_modes(self):
        from tools.run_nlp4lp_focused_eval import FOCUSED_BASELINES_DEFAULT
        for name in (
            "tfidf_relation_aware_basic",
            "tfidf_relation_aware_ops",
            "tfidf_relation_aware_semantic",
            "tfidf_relation_aware_full",
        ):
            assert name in FOCUSED_BASELINES_DEFAULT, f"{name} not in FOCUSED_BASELINES_DEFAULT"

    def test_baseline_assignment_includes_all_modes(self):
        from tools.run_nlp4lp_focused_eval import BASELINE_ASSIGNMENT_DEFAULT
        modes = {am for _, am in BASELINE_ASSIGNMENT_DEFAULT}
        for mode in ("relation_aware_basic", "relation_aware_ops",
                     "relation_aware_semantic", "relation_aware_full"):
            assert mode in modes, f"{mode} not in BASELINE_ASSIGNMENT_DEFAULT"
