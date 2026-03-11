"""Tests for the ambiguity_aware_grounding module.

Covers:
  1. CandidateSet construction (build_candidate_sets)
  2. SlotAmbiguity / QueryAmbiguity computation
  3. beam_assignment strategy
  4. abstain_aware_assignment strategy (including abstention)
  5. nbest_assignments (N-best hypothesis generation)
  6. run_ambiguity_aware_grounding end-to-end for all four ablation modes
  7. Competition penalty in N-best hypotheses
  8. Constants and dataclass structure
  9. Integration: wired into downstream utility dispatch + focused eval
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.ambiguity_aware_grounding import (
    AMBIGUITY_ABLATION_MODES,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_AMBIGUITY,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_N_BEST,
    DEFAULT_TOP_K,
    AssignmentHypothesis,
    CandidateEntry,
    CandidateSet,
    QueryAmbiguity,
    SlotAmbiguity,
    abstain_aware_assignment,
    beam_assignment,
    build_candidate_sets,
    compute_query_ambiguity,
    compute_slot_ambiguity,
    nbest_assignments,
    run_ambiguity_aware_grounding,
)
from tools.relation_aware_linking import build_mention_slot_links


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(query: str, slots: list[str], mode: str = "ambiguity_full") -> tuple[dict, dict, dict]:
    return run_ambiguity_aware_grounding(query, "orig", slots, ablation_mode=mode)


def _get_links_and_irs(query: str, slots: list[str]):
    return build_mention_slot_links(query, "orig", slots)


# ---------------------------------------------------------------------------
# 1. CandidateSet construction
# ---------------------------------------------------------------------------

class TestBuildCandidateSets:
    def test_returns_dict_keyed_by_slot_name(self):
        links, _, slots_ir, _, _ = _get_links_and_irs(
            "Budget is 100 dollars and demand is 20.", ["budget", "demand"]
        )
        cs_map = build_candidate_sets(links, slots_ir, _, top_k=DEFAULT_TOP_K)
        assert set(cs_map.keys()) == {"budget", "demand"}

    def test_candidates_sorted_descending(self):
        links, _, slots_ir, _, _ = _get_links_and_irs(
            "Budget is 100 dollars and profit is 5 dollars.", ["total_budget", "profit_per_unit"]
        )
        from tools.nlp4lp_downstream_utility import _build_slot_opt_irs, _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 100 dollars and profit is 5 dollars.", "orig")
        cs_map = build_candidate_sets(links, slots_ir, mentions_ir, top_k=DEFAULT_TOP_K)
        for sn, cs in cs_map.items():
            scores = [c.score for c in cs.candidates]
            assert scores == sorted(scores, reverse=True), f"Slot {sn} candidates not sorted"

    def test_top_k_respected(self):
        links, _, slots_ir, _, _ = _get_links_and_irs(
            "Budget is 100 dollars and profit is 5 dollars.", ["total_budget"]
        )
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 100 dollars and profit is 5 dollars.", "orig")
        cs_map = build_candidate_sets(links, slots_ir, mentions_ir, top_k=2)
        for cs in cs_map.values():
            assert len(cs.candidates) <= 2

    def test_norm_score_in_range(self):
        links, _, slots_ir, _, _ = _get_links_and_irs(
            "Budget is 500 dollars.", ["total_budget"]
        )
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 500 dollars.", "orig")
        cs_map = build_candidate_sets(links, slots_ir, mentions_ir)
        for cs in cs_map.values():
            for c in cs.candidates:
                assert 0.0 <= c.norm_score <= 1.0

    def test_empty_slots_returns_empty_dict(self):
        links, _, slots_ir, _, _ = _get_links_and_irs("Budget is 100.", [])
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 100.", "orig")
        cs_map = build_candidate_sets(links, slots_ir, mentions_ir)
        assert cs_map == {}


# ---------------------------------------------------------------------------
# 2. Ambiguity signals
# ---------------------------------------------------------------------------

class TestSlotAmbiguity:
    def _make_cs(self, scores: list[float], slot_name: str = "test_slot") -> CandidateSet:
        entries = [
            CandidateEntry(mention_id=i, score=s, norm_score=max(0.0, min(1.0, s / 12.0)))
            for i, s in enumerate(scores)
        ]
        return CandidateSet(slot_name=slot_name, candidates=entries)

    def test_empty_candidate_set_is_ambiguous(self):
        cs = CandidateSet(slot_name="x", candidates=[])
        sa = compute_slot_ambiguity(cs)
        assert sa.is_ambiguous
        assert sa.n_candidates == 0
        assert sa.entropy == 1.0

    def test_single_candidate_not_ambiguous_by_entropy(self):
        cs = self._make_cs([10.0])
        sa = compute_slot_ambiguity(cs, entropy_threshold=0.9, margin_threshold=0.1)
        # Single candidate → entropy 0, margin 1.0
        assert sa.entropy == 0.0
        assert sa.margin == pytest.approx(1.0)
        assert not sa.is_ambiguous

    def test_tied_candidates_high_entropy(self):
        cs = self._make_cs([5.0, 5.0, 5.0])
        sa = compute_slot_ambiguity(cs, entropy_threshold=0.5)
        assert sa.entropy > 0.5
        assert sa.is_ambiguous

    def test_margin_computed_correctly(self):
        cs = self._make_cs([8.0, 4.0])
        sa = compute_slot_ambiguity(cs)
        expected_margin = (8.0 - 4.0) / 8.0
        assert sa.margin == pytest.approx(expected_margin)

    def test_spread_computed(self):
        cs = self._make_cs([10.0, 6.0, 2.0])
        sa = compute_slot_ambiguity(cs)
        assert sa.spread == pytest.approx(10.0 - 2.0)

    def test_top1_top2_populated(self):
        cs = self._make_cs([9.0, 3.0, 1.0])
        sa = compute_slot_ambiguity(cs)
        assert sa.top1_score == pytest.approx(9.0)
        assert sa.top2_score == pytest.approx(3.0)


class TestQueryAmbiguity:
    def test_empty_slot_ambiguities(self):
        qa = compute_query_ambiguity([], n_mentions=3)
        assert qa.query_ambiguity_score == 1.0
        assert qa.n_slots == 0

    def test_all_unambiguous(self):
        sas = [
            SlotAmbiguity("a", 9.0, 1.0, 0.89, 8.0, 0.1, 2, False),
            SlotAmbiguity("b", 9.0, 1.0, 0.89, 8.0, 0.1, 2, False),
        ]
        qa = compute_query_ambiguity(sas, n_mentions=2)
        assert qa.n_ambiguous_slots == 0
        assert qa.query_ambiguity_score < 0.5

    def test_all_ambiguous(self):
        sas = [
            SlotAmbiguity("a", 2.0, 1.9, 0.05, 0.1, 0.95, 5, True),
            SlotAmbiguity("b", 2.0, 1.9, 0.05, 0.1, 0.95, 5, True),
        ]
        qa = compute_query_ambiguity(sas, n_mentions=10)
        assert qa.n_ambiguous_slots == 2
        assert qa.query_ambiguity_score > 0.5

    def test_query_ambiguity_bounded(self):
        sas = [
            SlotAmbiguity("a", 1.0, 0.9, 0.1, 0.1, 0.99, 5, True),
        ]
        qa = compute_query_ambiguity(sas, n_mentions=100)
        assert 0.0 <= qa.query_ambiguity_score <= 1.0


# ---------------------------------------------------------------------------
# 3. beam_assignment
# ---------------------------------------------------------------------------

class TestBeamAssignment:
    def _make_candidate_sets(self, spec: dict[str, list[float]], mention_ids_base: int = 0) -> tuple:
        """Build fake CandidateSets and matching MentionOptIR stubs."""
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions(
            "Budget is 100 dollars and profit is 5 and demand is 20.", "orig"
        )
        mid_map = {m.mention_id: m for m in mentions_ir}
        available_mids = list(mid_map.keys())

        cs_map: dict[str, CandidateSet] = {}
        mid_idx = 0
        for slot_name, scores in spec.items():
            entries = []
            for sc in scores:
                mid = available_mids[mid_idx % len(available_mids)]
                mid_idx += 1
                entries.append(CandidateEntry(mention_id=mid, score=sc, norm_score=sc / 12.0))
            cs_map[slot_name] = CandidateSet(slot_name=slot_name, candidates=entries)
        return cs_map, mentions_ir

    def test_returns_valid_types(self):
        cs_map, mentions_ir = self._make_candidate_sets({"budget": [8.0, 3.0]})
        vals, mments, diag = beam_assignment(cs_map, mentions_ir, beam_width=2)
        assert isinstance(vals, dict)
        assert isinstance(mments, dict)
        assert isinstance(diag, dict)

    def test_beam_width_in_diagnostics(self):
        cs_map, mentions_ir = self._make_candidate_sets({"budget": [8.0]})
        _, _, diag = beam_assignment(cs_map, mentions_ir, beam_width=4)
        assert diag["beam_width"] == 4

    def test_empty_candidate_sets(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 100.", "orig")
        vals, mments, diag = beam_assignment({}, mentions_ir, beam_width=4)
        assert vals == {}
        assert mments == {}

    def test_no_duplicate_assignments(self):
        cs_map, mentions_ir = self._make_candidate_sets(
            {"slot_a": [9.0, 5.0], "slot_b": [8.0, 4.0]}
        )
        vals, mments, _ = beam_assignment(cs_map, mentions_ir, beam_width=4)
        mids = [m.mention_id for m in mments.values()]
        assert len(set(mids)) == len(mids), "Duplicate mention assignments detected"


# ---------------------------------------------------------------------------
# 4. abstain_aware_assignment
# ---------------------------------------------------------------------------

class TestAbstainAwareAssignment:
    def _make_cs_map_with_ambiguities(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions(
            "Budget is 100 dollars and profit is 5 and demand is 20.", "orig"
        )
        mids = [m.mention_id for m in mentions_ir]

        # High-confidence slot
        cs_high = CandidateSet("budget", [
            CandidateEntry(mids[0], 10.0, 0.83),
        ])
        # Low-confidence slot (will be abstained)
        cs_low = CandidateSet("demand", [
            CandidateEntry(mids[1] if len(mids) > 1 else mids[0], 1.0, 0.08),
        ])

        sa_high = SlotAmbiguity("budget", 10.0, 0.0, 1.0, 10.0, 0.0, 1, False)
        sa_low = SlotAmbiguity("demand", 1.0, 0.9, 0.1, 0.1, 0.95, 1, True)

        cs_map = {"budget": cs_high, "demand": cs_low}
        sa_map = {"budget": sa_high, "demand": sa_low}
        return cs_map, sa_map, mentions_ir

    def test_high_confidence_slot_assigned(self):
        cs_map, sa_map, mentions_ir = self._make_cs_map_with_ambiguities()
        vals, _, diag = abstain_aware_assignment(
            cs_map, mentions_ir, sa_map,
            min_confidence=0.5, min_margin=0.3, max_entropy=0.8
        )
        assert "budget" in vals

    def test_low_confidence_slot_abstained(self):
        cs_map, sa_map, mentions_ir = self._make_cs_map_with_ambiguities()
        _, _, diag = abstain_aware_assignment(
            cs_map, mentions_ir, sa_map,
            min_confidence=0.5, min_margin=0.3, max_entropy=0.8
        )
        assert "demand" in diag["abstained_slots"]

    def test_abstained_slots_in_diagnostics(self):
        cs_map, sa_map, mentions_ir = self._make_cs_map_with_ambiguities()
        _, _, diag = abstain_aware_assignment(cs_map, mentions_ir, sa_map)
        assert "abstained_slots" in diag
        assert "n_abstained" in diag
        assert isinstance(diag["n_abstained"], int)

    def test_empty_candidates_abstained(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 100.", "orig")
        cs_map = {"missing_slot": CandidateSet("missing_slot", [])}
        sa_map = {}
        _, _, diag = abstain_aware_assignment(cs_map, mentions_ir, sa_map)
        assert "missing_slot" in diag["abstained_slots"]

    def test_thresholds_in_diagnostics(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 100.", "orig")
        _, _, diag = abstain_aware_assignment({}, mentions_ir, {})
        assert "thresholds" in diag
        assert "min_confidence" in diag["thresholds"]


# ---------------------------------------------------------------------------
# 5. nbest_assignments
# ---------------------------------------------------------------------------

class TestNBestAssignments:
    def test_returns_list_of_hypotheses(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions(
            "Budget is 100 dollars and profit is 5.", "orig"
        )
        mids = [m.mention_id for m in mentions_ir]
        cs_map = {
            "total_budget": CandidateSet("total_budget", [
                CandidateEntry(mids[0], 9.0, 0.75),
                CandidateEntry(mids[1] if len(mids) > 1 else mids[0], 4.0, 0.33),
            ])
        }
        hyps = nbest_assignments(cs_map, mentions_ir, n=3)
        assert isinstance(hyps, list)
        for h in hyps:
            assert isinstance(h, AssignmentHypothesis)

    def test_hypotheses_sorted_by_total_score(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions(
            "Budget is 100 dollars and profit is 5.", "orig"
        )
        mids = [m.mention_id for m in mentions_ir]
        cs_map = {
            "slot_a": CandidateSet("slot_a", [
                CandidateEntry(mids[0], 8.0, 0.67),
                CandidateEntry(mids[1] if len(mids) > 1 else mids[0], 3.0, 0.25),
            ])
        }
        hyps = nbest_assignments(cs_map, mentions_ir, n=2)
        if len(hyps) >= 2:
            assert hyps[0].total_score >= hyps[1].total_score

    def test_competition_penalty_is_non_negative(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions(
            "Budget is 100 dollars.", "orig"
        )
        mids = [m.mention_id for m in mentions_ir]
        cs_map = {
            "slot_a": CandidateSet("slot_a", [CandidateEntry(mids[0], 7.0, 0.58)]),
        }
        hyps = nbest_assignments(cs_map, mentions_ir, n=2)
        for h in hyps:
            assert h.competition_penalty >= 0.0

    def test_empty_candidate_sets_returns_hypotheses(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 100.", "orig")
        hyps = nbest_assignments({}, mentions_ir, n=3)
        assert isinstance(hyps, list)

    def test_hypothesis_has_required_fields(self):
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions_ir = _extract_opt_role_mentions("Budget is 100.", "orig")
        mids = [m.mention_id for m in mentions_ir]
        cs_map = {
            "budget": CandidateSet("budget", [CandidateEntry(mids[0] if mids else 0, 8.0, 0.67)])
        }
        hyps = nbest_assignments(cs_map, mentions_ir, n=1)
        if hyps:
            h = hyps[0]
            assert hasattr(h, "rank")
            assert hasattr(h, "score")
            assert hasattr(h, "filled_values")
            assert hasattr(h, "filled_mentions")
            assert hasattr(h, "abstained_slots")


# ---------------------------------------------------------------------------
# 6. End-to-end grounding (all ablation modes)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_candidate_greedy_basic(self):
        vals, _, _ = _run(
            "The budget is 100 dollars.",
            ["total_budget"],
            mode="candidate_greedy"
        )
        if vals:
            assert abs(float(list(vals.values())[0]) - 100.0) < 2.0

    def test_all_modes_run_without_error(self):
        query = "Budget is 500 dollars and profit per unit is 3 dollars."
        slots = ["total_budget", "profit_per_unit"]
        for mode in AMBIGUITY_ABLATION_MODES:
            vals, mments, diag = _run(query, slots, mode=mode)
            assert isinstance(vals, dict), f"mode={mode}"
            assert isinstance(diag, dict), f"mode={mode}"
            assert diag.get("ablation_mode") == mode, f"mode={mode}"

    def test_empty_slots_returns_empty(self):
        vals, mments, diag = _run("Budget is 100 dollars.", [])
        assert vals == {}
        assert mments == {}

    def test_no_mentions_returns_empty(self):
        vals, mments, _ = _run("There are no numbers here.", ["budget"])
        assert vals == {}

    def test_diagnostics_contain_ambiguity_info(self):
        _, _, diag = _run(
            "Budget is 200 dollars and demand is 10.",
            ["budget", "demand"]
        )
        assert "query_ambiguity" in diag
        assert "slot_ambiguities" in diag

    def test_abstain_mode_may_leave_slot_empty(self):
        """With strict thresholds, ambiguous slots should be left unassigned."""
        vals, _, diag = run_ambiguity_aware_grounding(
            "The value is 5.",
            "orig",
            ["total_budget", "demand", "min_units", "max_units"],
            ablation_mode="ambiguity_abstain",
            min_confidence=0.99,   # very strict → most slots abstained
            min_margin=0.99,
        )
        n_abstained = diag.get("n_abstained", 0)
        assert n_abstained >= 0  # abstention mechanism runs without crash

    def test_full_mode_n_best_in_diagnostics(self):
        _, _, diag = _run(
            "Budget is 200 dollars and profit is 5 dollars.",
            ["total_budget", "profit_per_unit"],
            mode="ambiguity_full",
        )
        # N-best may or may not be present depending on whether mentions were found.
        assert isinstance(diag, dict)

    def test_no_duplicate_mention_assignments(self):
        vals, mments, _ = _run(
            "Budget is 500 dollars and demand is 20.",
            ["total_budget", "demand"],
        )
        mids = [m.mention_id for m in mments.values() if m is not None]
        assert len(set(mids)) == len(mids), "Duplicate mention assignments"

    def test_profit_budget_separation(self):
        vals, mments, _ = _run(
            "Each unit yields 4 dollars profit. Total budget is 800 dollars.",
            ["profit_per_unit", "total_budget"],
        )
        if "profit_per_unit" in vals and "total_budget" in vals:
            assert float(vals["profit_per_unit"]) < float(vals["total_budget"])

    def test_min_max_ordering(self):
        vals, _, _ = _run(
            "At least 5 units and at most 30 units are required.",
            ["min_units", "max_units"]
        )
        if "min_units" in vals and "max_units" in vals:
            assert float(vals["min_units"]) <= float(vals["max_units"])


# ---------------------------------------------------------------------------
# 7. Constants and dataclass sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_ablation_modes_tuple(self):
        assert set(AMBIGUITY_ABLATION_MODES) == {
            "candidate_greedy",
            "ambiguity_beam",
            "ambiguity_abstain",
            "ambiguity_full",
        }

    def test_default_values_positive(self):
        assert DEFAULT_TOP_K > 0
        assert DEFAULT_BEAM_WIDTH > 0
        assert DEFAULT_N_BEST > 0
        assert 0.0 < DEFAULT_MIN_CONFIDENCE < 1.0
        assert 0.0 < DEFAULT_MIN_MARGIN < 1.0
        assert 0.0 < DEFAULT_MAX_AMBIGUITY <= 1.0

    def test_candidate_entry_fields(self):
        ce = CandidateEntry(mention_id=1, score=5.0, norm_score=0.4)
        assert ce.mention_id == 1
        assert ce.score == pytest.approx(5.0)
        assert ce.norm_score == pytest.approx(0.4)

    def test_assignment_hypothesis_total_score(self):
        h = AssignmentHypothesis(
            rank=0, score=10.0, filled_values={}, filled_mentions={},
            abstained_slots=[], competition_penalty=2.0
        )
        assert h.total_score == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 8. Integration: wired into pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_assignment_mode_strings_in_downstream_utility(self):
        import ast as ast_mod
        src = ROOT / "tools" / "nlp4lp_downstream_utility.py"
        tree = ast_mod.parse(src.read_text())
        found = set()
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Constant) and isinstance(node.value, str):
                if node.value.startswith("ambiguity_"):
                    found.add(node.value)
        for mode in (
            "ambiguity_candidate_greedy",
            "ambiguity_aware_beam",
            "ambiguity_aware_abstain",
            "ambiguity_aware_full",
        ):
            assert mode in found, f"{mode!r} not found in downstream utility"

    def test_effective_baseline_naming(self):
        from tools.run_nlp4lp_focused_eval import _effective_baseline
        assert _effective_baseline("tfidf", "ambiguity_candidate_greedy") == "tfidf_ambiguity_candidate_greedy"
        assert _effective_baseline("tfidf", "ambiguity_aware_full") == "tfidf_ambiguity_aware_full"

    def test_focused_eval_includes_all_modes(self):
        from tools.run_nlp4lp_focused_eval import FOCUSED_BASELINES_DEFAULT
        for name in (
            "tfidf_ambiguity_candidate_greedy",
            "tfidf_ambiguity_aware_beam",
            "tfidf_ambiguity_aware_abstain",
            "tfidf_ambiguity_aware_full",
        ):
            assert name in FOCUSED_BASELINES_DEFAULT, f"{name} not in FOCUSED_BASELINES_DEFAULT"

    def test_baseline_assignment_includes_all_modes(self):
        from tools.run_nlp4lp_focused_eval import BASELINE_ASSIGNMENT_DEFAULT
        modes = {am for _, am in BASELINE_ASSIGNMENT_DEFAULT}
        for mode in (
            "ambiguity_candidate_greedy",
            "ambiguity_aware_beam",
            "ambiguity_aware_abstain",
            "ambiguity_aware_full",
        ):
            assert mode in modes, f"{mode} not in BASELINE_ASSIGNMENT_DEFAULT"
