"""Search-structured grounding for numeric mention to slot assignment.

`search_structured_grounding` performs deterministic beam search over partial
slot assignments instead of relying on single-pass greedy filling.

Compared with greedy/repair methods, this method explicitly explores
competing hypotheses jointly and scores partial states with local compatibility
plus global consistency penalties (duplicate mention reuse, bound inversion,
percent/scalar mismatch, total-vs-per-unit mismatch, count implausibility),
while allowing abstention through a per-slot NULL option.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (  # reuse existing deterministic IR + extractors
    GCG_PRUNE_THRESHOLD,
    MentionOptIR,
    SlotOptIR,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
)
from tools.relation_aware_linking import build_mention_slot_links, relation_aware_local_score

DEFAULT_BEAM_WIDTH = 16
DEFAULT_TOP_K_PER_SLOT = 5

SEARCH_STRUCTURED_WEIGHTS: dict[str, float] = {
    "duplicate_mention_penalty": -7.0,
    "bound_inversion_penalty": -6.0,
    "total_per_unit_penalty": -3.0,
    "percent_mismatch_penalty": -4.0,
    "count_implausibility_penalty": -2.5,
    "coverage_reward": 0.35,
    "null_penalty": -0.35,
    "abstain_reward": 0.20,
    "coherent_full_reward": 0.50,
}


@dataclass(frozen=True)
class SlotCandidate:
    slot_name: str
    mention_id: int | None
    mention_value: float | None
    mention_raw: str
    local_score: float
    local_features: dict[str, Any]
    is_null: bool


@dataclass
class SearchState:
    assignment: dict[str, SlotCandidate]
    used_mentions: set[int]
    local_score: float
    global_score: float
    score: float
    prune_reasons: list[str]


def _slot_priority(slot: SlotOptIR) -> tuple[int, int, int]:
    bound_like = 0 if ("min" in slot.operator_preference or "max" in slot.operator_preference) else 1
    percent_like = 0 if slot.expected_type == "percent" else 1
    count_like = 0 if slot.is_count_like else 1
    return bound_like, percent_like, count_like


def build_slot_candidates(
    query: str,
    variant: str,
    expected_scalar: list[str],
    top_k_per_slot: int = DEFAULT_TOP_K_PER_SLOT,
) -> tuple[list[MentionOptIR], list[SlotOptIR], dict[str, list[SlotCandidate]], int]:
    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    links, _, _, _, _ = build_mention_slot_links(query, variant, expected_scalar)

    by_slot: dict[str, list[SlotCandidate]] = {s.name: [] for s in slots}
    n_edges = 0
    for lnk in links:
        sc, feats = relation_aware_local_score(lnk, "full")
        if sc <= GCG_PRUNE_THRESHOLD:
            continue
        if lnk.type_incompatible or lnk.percent_mismatch:
            continue

        slot = lnk.slot_feats
        mention = lnk.mention_feats
        if slot.is_percent_like and mention.type_bucket != "percent":
            continue
        if (not slot.is_percent_like) and mention.type_bucket == "percent" and slot.expected_type != "percent":
            continue

        cand = SlotCandidate(
            slot_name=lnk.slot_name,
            mention_id=lnk.mention_id,
            mention_value=mention.value,
            mention_raw=mention.raw_surface,
            local_score=sc,
            local_features=feats,
            is_null=False,
        )
        by_slot[lnk.slot_name].append(cand)
        n_edges += 1

    for s in slots:
        cands = sorted(
            by_slot.get(s.name, []),
            key=lambda c: (-c.local_score, c.mention_id if c.mention_id is not None else 10**9, c.mention_raw),
        )[:top_k_per_slot]
        cands.append(
            SlotCandidate(
                slot_name=s.name,
                mention_id=None,
                mention_value=None,
                mention_raw="<NULL>",
                local_score=SEARCH_STRUCTURED_WEIGHTS["null_penalty"],
                local_features={"null_candidate": True},
                is_null=True,
            )
        )
        by_slot[s.name] = cands

    return mentions, slots, by_slot, n_edges


def order_slots_for_search(slots: list[SlotOptIR], candidates: dict[str, list[SlotCandidate]]) -> list[SlotOptIR]:
    return sorted(
        slots,
        key=lambda s: (
            sum(1 for c in candidates.get(s.name, []) if not c.is_null),
            _slot_priority(s),
            s.name,
        ),
    )


def _pairwise_global_delta(
    slot: SlotOptIR,
    cand: SlotCandidate,
    current: dict[str, SlotCandidate],
    slot_by_name: dict[str, SlotOptIR],
    weights: dict[str, float],
) -> tuple[float, list[str], bool]:
    if cand.is_null:
        return weights["abstain_reward"], ["abstain_reward"], False

    delta = 0.0
    reasons: list[str] = []
    hard_prune = False

    if cand.mention_id is not None:
        for a in current.values():
            if not a.is_null and a.mention_id == cand.mention_id:
                delta += weights["duplicate_mention_penalty"]
                reasons.append("duplicate_mention")
                hard_prune = True
                break

    if slot.expected_type == "percent" and "%" not in cand.mention_raw and cand.mention_value is not None and cand.mention_value > 1.0:
        delta += weights["percent_mismatch_penalty"]
        reasons.append("percent_mismatch")
        hard_prune = True

    if slot.is_count_like and cand.mention_value is not None and cand.mention_value > 1000:
        delta += weights["count_implausibility_penalty"]
        reasons.append("count_implausible")

    for other_slot_name, other_c in current.items():
        if other_c.is_null:
            continue
        other_slot = slot_by_name[other_slot_name]

        if "min" in slot.operator_preference and "max" in other_slot.operator_preference:
            if cand.mention_value is not None and other_c.mention_value is not None and cand.mention_value > other_c.mention_value:
                delta += weights["bound_inversion_penalty"]
                reasons.append("bound_inversion")
                hard_prune = True
        if "max" in slot.operator_preference and "min" in other_slot.operator_preference:
            if cand.mention_value is not None and other_c.mention_value is not None and other_c.mention_value > cand.mention_value:
                delta += weights["bound_inversion_penalty"]
                reasons.append("bound_inversion")
                hard_prune = True

        if (slot.is_total_like and other_slot.is_coefficient_like) or (slot.is_coefficient_like and other_slot.is_total_like):
            total_c = cand if slot.is_total_like else other_c
            coeff_c = other_c if slot.is_total_like else cand
            if total_c.mention_value is not None and coeff_c.mention_value is not None and total_c.mention_value < coeff_c.mention_value:
                delta += weights["total_per_unit_penalty"]
                reasons.append("total_per_unit_mismatch")

    return delta, reasons, hard_prune


def score_partial_state(
    state: SearchState,
    slot: SlotOptIR,
    cand: SlotCandidate,
    slot_by_name: dict[str, SlotOptIR],
    use_global: bool,
    weights: dict[str, float],
) -> tuple[float, float, list[str], bool]:
    local_inc = cand.local_score + (weights["coverage_reward"] if not cand.is_null else 0.0)
    if not use_global:
        return local_inc, 0.0, [], False
    gd, reasons, hard = _pairwise_global_delta(slot, cand, state.assignment, slot_by_name, weights)
    return local_inc, gd, reasons, hard


def is_state_prunable(score: float, best_score: float, margin: float = 10.0) -> bool:
    return score < (best_score - margin)


def expand_state(
    state: SearchState,
    slot: SlotOptIR,
    slot_candidates: list[SlotCandidate],
    slot_by_name: dict[str, SlotOptIR],
    use_global: bool,
    weights: dict[str, float],
    best_so_far: float,
) -> tuple[list[SearchState], int]:
    children: list[SearchState] = []
    pruned = 0
    for cand in slot_candidates:
        local_inc, global_inc, reasons, hard = score_partial_state(
            state, slot, cand, slot_by_name, use_global, weights
        )
        new_score = state.score + local_inc + global_inc
        if hard or is_state_prunable(new_score, best_so_far):
            pruned += 1
            continue
        assign = dict(state.assignment)
        assign[slot.name] = cand
        used = set(state.used_mentions)
        if cand.mention_id is not None:
            used.add(cand.mention_id)
        children.append(
            SearchState(
                assignment=assign,
                used_mentions=used,
                local_score=state.local_score + local_inc,
                global_score=state.global_score + global_inc,
                score=new_score,
                prune_reasons=state.prune_reasons + reasons,
            )
        )
    return children, pruned


def finalize_best_assignment(
    best: SearchState,
    mention_by_id: dict[int, MentionOptIR],
) -> tuple[dict[str, Any], dict[str, Any]]:
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, Any] = {}
    for slot_name, cand in best.assignment.items():
        if cand.is_null or cand.mention_id is None:
            continue
        m = mention_by_id.get(cand.mention_id)
        if m is None:
            continue
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions


def run_search_structured_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
    *,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    top_k_per_slot: int = DEFAULT_TOP_K_PER_SLOT,
    use_global: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not expected_scalar:
        return {}, {}, {"beam_width": beam_width, "top_k_per_slot": top_k_per_slot, "n_slots": 0, "n_mentions": 0}

    mentions, slots, candidates, n_edges = build_slot_candidates(
        query, variant, expected_scalar, top_k_per_slot=top_k_per_slot
    )
    ordered_slots = order_slots_for_search(slots, candidates)
    slot_by_name = {s.name: s for s in slots}
    mention_by_id = {m.mention_id: m for m in mentions}

    beam = [SearchState(assignment={}, used_mentions=set(), local_score=0.0, global_score=0.0, score=0.0, prune_reasons=[])]
    expanded_states = 0
    pruned_states = 0
    best_so_far = float("-inf")

    for slot in ordered_slots:
        next_beam: list[SearchState] = []
        for st in beam:
            children, pruned = expand_state(
                st,
                slot,
                candidates.get(slot.name, []),
                slot_by_name,
                use_global,
                SEARCH_STRUCTURED_WEIGHTS,
                best_so_far,
            )
            expanded_states += len(children)
            pruned_states += pruned
            next_beam.extend(children)

        next_beam.sort(
            key=lambda s: (-s.score, -len([c for c in s.assignment.values() if not c.is_null]), sorted(s.assignment.keys()))
        )
        beam = next_beam[:beam_width] if next_beam else beam
        if beam:
            best_so_far = max(best_so_far, beam[0].score)

    best = beam[0] if beam else SearchState({}, set(), 0.0, 0.0, 0.0, ["empty_beam"])
    if len([c for c in best.assignment.values() if not c.is_null]) == len(slots):
        best.global_score += SEARCH_STRUCTURED_WEIGHTS["coherent_full_reward"]
        best.score += SEARCH_STRUCTURED_WEIGHTS["coherent_full_reward"]

    filled_values, filled_mentions = finalize_best_assignment(best, mention_by_id)

    per_slot_candidates = {
        s.name: [
            {
                "slot_name": c.slot_name,
                "mention_id": c.mention_id,
                "mention_raw": c.mention_raw,
                "mention_value": c.mention_value,
                "local_score": c.local_score,
                "is_null": c.is_null,
                "local_features": c.local_features,
            }
            for c in candidates.get(s.name, [])
        ]
        for s in ordered_slots
    }

    top_assignments = []
    for rank, st in enumerate(beam[: min(beam_width, 5)]):
        top_assignments.append(
            {
                "rank": rank,
                "score": st.score,
                "local_score": st.local_score,
                "global_score": st.global_score,
                "assignment": {
                    k: (None if v.is_null else v.mention_raw) for k, v in sorted(st.assignment.items())
                },
            }
        )

    diagnostics = {
        "ablation_mode": "full" if use_global else "no_global",
        "n_mentions": len(mentions),
        "n_slots": len(slots),
        "n_candidate_edges": n_edges,
        "beam_width": beam_width,
        "top_k_per_slot": top_k_per_slot,
        "expanded_states": expanded_states,
        "pruned_states": pruned_states,
        "best_score": best.score,
        "best_local_score": best.local_score,
        "best_global_score": best.global_score,
        "top_assignments": top_assignments,
        "per_slot_candidates": per_slot_candidates,
        "slot_order": [s.name for s in ordered_slots],
    }
    return filled_values, filled_mentions, diagnostics
