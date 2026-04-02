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

COUNTERFACTUAL_REFINEMENT_DEFAULTS: dict[str, float | int] = {
    "max_refinement_steps": 10,
    "max_counterfactuals_per_slot": 3,
    "min_improvement": 1e-6,
    "unstable_margin": 0.35,
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


@dataclass(frozen=True)
class CounterfactualMove:
    move_type: str
    primary_slot: str
    secondary_slot: str | None
    candidate: SlotCandidate | None
    reason: str


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
    best: SearchState | dict[str, SlotCandidate],
    mention_by_id: dict[int, MentionOptIR],
) -> tuple[dict[str, Any], dict[str, Any]]:
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, Any] = {}
    assignment = best.assignment if isinstance(best, SearchState) else best
    for slot_name, cand in assignment.items():
        if cand.is_null or cand.mention_id is None:
            continue
        m = mention_by_id.get(cand.mention_id)
        if m is None:
            continue
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions


def score_full_assignment(
    assignment: dict[str, SlotCandidate],
    slot_by_name: dict[str, SlotOptIR],
    *,
    weights: dict[str, float],
) -> tuple[float, int, dict[str, Any]]:
    total = 0.0
    contradiction_count = 0
    reasons: dict[str, int] = {}
    slot_penalty_hits: dict[str, int] = {k: 0 for k in slot_by_name}

    for slot_name, cand in assignment.items():
        if cand.is_null:
            total += weights["null_penalty"] + weights["abstain_reward"]
            reasons["null_assignment"] = reasons.get("null_assignment", 0) + 1
            continue
        total += cand.local_score + weights["coverage_reward"]
        if cand.local_score <= 0.0:
            total -= 0.5
            reasons["weak_assignment"] = reasons.get("weak_assignment", 0) + 1
            slot_penalty_hits[slot_name] = slot_penalty_hits.get(slot_name, 0) + 1

    non_null = [(s, c) for s, c in assignment.items() if not c.is_null]
    if non_null and len(non_null) == len(slot_by_name):
        total += weights["coherent_full_reward"]

    mention_to_slots: dict[int, list[str]] = {}
    for slot_name, cand in non_null:
        if cand.mention_id is None:
            continue
        mention_to_slots.setdefault(cand.mention_id, []).append(slot_name)
    for slots in mention_to_slots.values():
        if len(slots) > 1:
            penalty = abs(weights["duplicate_mention_penalty"]) * (len(slots) - 1)
            total -= penalty
            contradiction_count += 1
            reasons["duplicate_mention"] = reasons.get("duplicate_mention", 0) + 1
            for s in slots:
                slot_penalty_hits[s] = slot_penalty_hits.get(s, 0) + 1

    keys = sorted(assignment.keys())
    for i, s1 in enumerate(keys):
        c1 = assignment[s1]
        if c1.is_null:
            continue
        sl1 = slot_by_name[s1]
        for s2 in keys[i + 1 :]:
            c2 = assignment[s2]
            if c2.is_null:
                continue
            sl2 = slot_by_name[s2]

            if ("min" in sl1.operator_preference and "max" in sl2.operator_preference):
                if c1.mention_value is not None and c2.mention_value is not None and c1.mention_value > c2.mention_value:
                    total += weights["bound_inversion_penalty"]
                    contradiction_count += 1
                    reasons["bound_inversion"] = reasons.get("bound_inversion", 0) + 1
                    slot_penalty_hits[s1] = slot_penalty_hits.get(s1, 0) + 1
                    slot_penalty_hits[s2] = slot_penalty_hits.get(s2, 0) + 1
            if ("max" in sl1.operator_preference and "min" in sl2.operator_preference):
                if c1.mention_value is not None and c2.mention_value is not None and c2.mention_value > c1.mention_value:
                    total += weights["bound_inversion_penalty"]
                    contradiction_count += 1
                    reasons["bound_inversion"] = reasons.get("bound_inversion", 0) + 1
                    slot_penalty_hits[s1] = slot_penalty_hits.get(s1, 0) + 1
                    slot_penalty_hits[s2] = slot_penalty_hits.get(s2, 0) + 1

            if (sl1.is_total_like and sl2.is_coefficient_like) or (sl1.is_coefficient_like and sl2.is_total_like):
                total_c = c1 if sl1.is_total_like else c2
                coeff_c = c2 if sl1.is_total_like else c1
                if total_c.mention_value is not None and coeff_c.mention_value is not None and total_c.mention_value < coeff_c.mention_value:
                    total += weights["total_per_unit_penalty"]
                    contradiction_count += 1
                    reasons["total_per_unit_mismatch"] = reasons.get("total_per_unit_mismatch", 0) + 1
                    slot_penalty_hits[s1] = slot_penalty_hits.get(s1, 0) + 1
                    slot_penalty_hits[s2] = slot_penalty_hits.get(s2, 0) + 1

    for slot_name, cand in non_null:
        sl = slot_by_name[slot_name]
        if sl.expected_type == "percent":
            if "%" not in cand.mention_raw and cand.mention_value is not None and cand.mention_value > 1.0:
                total += weights["percent_mismatch_penalty"]
                contradiction_count += 1
                reasons["percent_mismatch"] = reasons.get("percent_mismatch", 0) + 1
                slot_penalty_hits[slot_name] = slot_penalty_hits.get(slot_name, 0) + 1
        if sl.is_count_like and cand.mention_value is not None and cand.mention_value > 1000:
            total += weights["count_implausibility_penalty"]
            contradiction_count += 1
            reasons["count_implausibility"] = reasons.get("count_implausibility", 0) + 1
            slot_penalty_hits[slot_name] = slot_penalty_hits.get(slot_name, 0) + 1

    return total, contradiction_count, {
        "active_reasons": reasons,
        "slot_penalty_hits": slot_penalty_hits,
    }


def identify_unstable_slots(
    assignment: dict[str, SlotCandidate],
    slot_by_name: dict[str, SlotOptIR],
    candidates: dict[str, list[SlotCandidate]],
    score_diag: dict[str, Any],
    *,
    unstable_margin: float,
) -> list[dict[str, Any]]:
    unstable: list[dict[str, Any]] = []
    slot_penalty_hits = score_diag.get("slot_penalty_hits", {})
    for slot_name, slot in slot_by_name.items():
        cur = assignment.get(slot_name)
        cands = [c for c in candidates.get(slot_name, []) if not c.is_null]
        cands_sorted = sorted(cands, key=lambda c: (-c.local_score, c.mention_id if c.mention_id is not None else 10**9))
        reasons: list[str] = []

        if len(cands_sorted) >= 2 and (cands_sorted[0].local_score - cands_sorted[1].local_score) <= unstable_margin:
            reasons.append("top2_close")
        if cur is None or cur.is_null:
            reasons.append("null_or_forced")
        elif cur.local_score <= 0.0:
            reasons.append("weak_assignment")
        if slot_penalty_hits.get(slot_name, 0) > 0:
            reasons.append("global_penalty_trigger")
        if slot.is_count_like or slot.expected_type == "percent":
            reasons.append("ambiguity_family")
        if slot.is_total_like or slot.is_coefficient_like or ("min" in slot.operator_preference) or ("max" in slot.operator_preference):
            reasons.append("ambiguity_family")

        if reasons:
            unstable.append({"slot_name": slot_name, "reasons": sorted(set(reasons))})
    return unstable


def apply_counterfactual_move(
    assignment: dict[str, SlotCandidate],
    move: CounterfactualMove,
) -> dict[str, SlotCandidate]:
    new_assignment = dict(assignment)
    if move.move_type in ("replace", "abstain") and move.candidate is not None:
        new_assignment[move.primary_slot] = move.candidate
    elif move.move_type == "swap" and move.secondary_slot is not None:
        a = new_assignment.get(move.primary_slot)
        b = new_assignment.get(move.secondary_slot)
        if a is not None and b is not None:
            new_assignment[move.primary_slot], new_assignment[move.secondary_slot] = b, a
    return new_assignment


def generate_counterfactual_moves(
    assignment: dict[str, SlotCandidate],
    unstable_slots: list[dict[str, Any]],
    candidates: dict[str, list[SlotCandidate]],
    *,
    max_counterfactuals_per_slot: int,
) -> list[CounterfactualMove]:
    moves: list[CounterfactualMove] = []
    unstable_names = [u["slot_name"] for u in unstable_slots]
    for entry in unstable_slots:
        slot_name = entry["slot_name"]
        cur = assignment.get(slot_name)
        ranked = sorted(
            candidates.get(slot_name, []),
            key=lambda c: (-c.local_score, c.mention_id if c.mention_id is not None else 10**9),
        )
        n_added = 0
        for cand in ranked:
            if cur is not None and cand.mention_id == cur.mention_id and cand.is_null == cur.is_null:
                continue
            mtype = "abstain" if cand.is_null else "replace"
            moves.append(
                CounterfactualMove(
                    move_type=mtype,
                    primary_slot=slot_name,
                    secondary_slot=None,
                    candidate=cand,
                    reason=";".join(entry["reasons"]),
                )
            )
            n_added += 1
            if n_added >= max_counterfactuals_per_slot:
                break

    for i, s1 in enumerate(unstable_names):
        for s2 in unstable_names[i + 1 :]:
            c1 = assignment.get(s1)
            c2 = assignment.get(s2)
            if c1 is None or c2 is None or c1.is_null or c2.is_null:
                continue
            if c1.mention_id == c2.mention_id:
                moves.append(CounterfactualMove("swap", s1, s2, None, "duplicate_mention_conflict"))
            else:
                moves.append(CounterfactualMove("swap", s1, s2, None, "local_neighborhood_swap"))
    return moves


def run_counterfactual_refinement(
    query: str,
    variant: str,
    expected_scalar: list[str],
    initial_assignment: dict[str, SlotCandidate],
    slot_by_name: dict[str, SlotOptIR],
    candidates: dict[str, list[SlotCandidate]],
    *,
    max_refinement_steps: int = 10,
    max_counterfactuals_per_slot: int = 3,
    min_improvement: float = 1e-6,
) -> tuple[dict[str, SlotCandidate], dict[str, Any]]:
    _ = (query, variant, expected_scalar)  # explicit API surface for parity with run_* methods
    current = dict(initial_assignment)
    original_score, original_contradictions, original_diag = score_full_assignment(
        current, slot_by_name, weights=SEARCH_STRUCTURED_WEIGHTS
    )

    diagnostics: dict[str, Any] = {
        "slots_examined": sorted(slot_by_name.keys()),
        "unstable_slots_selected": [],
        "counterfactuals_evaluated": 0,
        "accepted_changes": [],
        "rejected_changes": 0,
        "original_score": original_score,
        "original_contradictions": original_contradictions,
        "change_log": [],
    }

    current_score = original_score
    current_contradictions = original_contradictions
    current_diag = original_diag

    for step in range(max_refinement_steps):
        unstable_slots = identify_unstable_slots(
            current,
            slot_by_name,
            candidates,
            current_diag,
            unstable_margin=float(COUNTERFACTUAL_REFINEMENT_DEFAULTS["unstable_margin"]),
        )
        diagnostics["unstable_slots_selected"].append(
            {"step": step, "slots": unstable_slots}
        )
        if not unstable_slots:
            break

        moves = generate_counterfactual_moves(
            current,
            unstable_slots,
            candidates,
            max_counterfactuals_per_slot=max_counterfactuals_per_slot,
        )
        if not moves:
            break

        best_move: CounterfactualMove | None = None
        best_candidate_assignment: dict[str, SlotCandidate] | None = None
        best_candidate_diag: dict[str, Any] | None = None
        best_candidate_score = current_score
        best_candidate_contradictions = current_contradictions

        for move in moves:
            trial = apply_counterfactual_move(current, move)
            score, contradictions, trial_diag = score_full_assignment(
                trial, slot_by_name, weights=SEARCH_STRUCTURED_WEIGHTS
            )
            diagnostics["counterfactuals_evaluated"] += 1
            improves = score > (current_score + min_improvement)
            lowers_contradiction = (
                contradictions < current_contradictions and score >= (current_score - min_improvement)
            )
            if improves or lowers_contradiction:
                if (
                    best_move is None
                    or score > best_candidate_score + min_improvement
                    or (
                        abs(score - best_candidate_score) <= min_improvement
                        and contradictions < best_candidate_contradictions
                    )
                ):
                    best_move = move
                    best_candidate_assignment = trial
                    best_candidate_diag = trial_diag
                    best_candidate_score = score
                    best_candidate_contradictions = contradictions

        if best_move is None or best_candidate_assignment is None or best_candidate_diag is None:
            diagnostics["rejected_changes"] += len(moves)
            break

        diagnostics["accepted_changes"].append(
            {
                "step": step,
                "move_type": best_move.move_type,
                "slot": best_move.primary_slot,
                "other_slot": best_move.secondary_slot,
                "reason": best_move.reason,
                "score_before": current_score,
                "score_after": best_candidate_score,
                "contradictions_before": current_contradictions,
                "contradictions_after": best_candidate_contradictions,
            }
        )
        diagnostics["change_log"].append(diagnostics["accepted_changes"][-1])
        current = best_candidate_assignment
        current_score = best_candidate_score
        current_contradictions = best_candidate_contradictions
        current_diag = best_candidate_diag

    diagnostics["refined_score"] = current_score
    diagnostics["refined_contradictions"] = current_contradictions
    diagnostics["active_reasons_refined"] = current_diag.get("active_reasons", {})
    return current, diagnostics


def counterfactual_grounding_refinement(
    query: str,
    variant: str,
    expected_scalar: list[str],
    *,
    filled_values: dict[str, Any],
    filled_mentions: dict[str, Any],
    per_slot_candidates: dict[str, list[dict[str, Any]]] | None = None,
    slot_by_name: dict[str, SlotOptIR] | None = None,
    mention_by_id: dict[int, MentionOptIR] | None = None,
    max_refinement_steps: int = 10,
    max_counterfactuals_per_slot: int = 3,
    min_improvement: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if per_slot_candidates is None or slot_by_name is None or mention_by_id is None:
        _, slots, cands, _ = build_slot_candidates(query, variant, expected_scalar)
        slot_by_name = {s.name: s for s in slots}
        mention_by_id = {}
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
            for sn, lst in cands.items()
        }

    assert slot_by_name is not None
    assert mention_by_id is not None
    assert per_slot_candidates is not None

    candidates: dict[str, list[SlotCandidate]] = {
        slot_name: [
            SlotCandidate(
                slot_name=slot_name,
                mention_id=row.get("mention_id"),
                mention_value=row.get("mention_value"),
                mention_raw=str(row.get("mention_raw", "")),
                local_score=float(row.get("local_score", 0.0)),
                local_features=dict(row.get("local_features", {})),
                is_null=bool(row.get("is_null", False)),
            )
            for row in rows
        ]
        for slot_name, rows in per_slot_candidates.items()
    }

    initial_assignment: dict[str, SlotCandidate] = {}
    for slot_name in slot_by_name:
        cur_mid = getattr(filled_mentions.get(slot_name), "mention_id", None)
        selected: SlotCandidate | None = None
        for cand in candidates.get(slot_name, []):
            if cur_mid is not None and cand.mention_id == cur_mid:
                selected = cand
                break
        if selected is None:
            selected = next((c for c in candidates.get(slot_name, []) if c.is_null), None)
        if selected is None:
            selected = SlotCandidate(slot_name, None, None, "<NULL>", SEARCH_STRUCTURED_WEIGHTS["null_penalty"], {}, True)
        initial_assignment[slot_name] = selected

    refined_assignment, diagnostics = run_counterfactual_refinement(
        query,
        variant,
        expected_scalar,
        initial_assignment,
        slot_by_name,
        candidates,
        max_refinement_steps=max_refinement_steps,
        max_counterfactuals_per_slot=max_counterfactuals_per_slot,
        min_improvement=min_improvement,
    )
    refined_values, refined_mentions = finalize_best_assignment(refined_assignment, mention_by_id)
    return refined_values, refined_mentions, diagnostics


def run_search_structured_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
    *,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    top_k_per_slot: int = DEFAULT_TOP_K_PER_SLOT,
    use_global: bool = True,
    use_counterfactual_refinement: bool = False,
    max_refinement_steps: int = 10,
    max_counterfactuals_per_slot: int = 3,
    min_improvement: float = 1e-6,
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
    if use_counterfactual_refinement:
        refined_values, refined_mentions, refinement_diag = counterfactual_grounding_refinement(
            query,
            variant,
            expected_scalar,
            filled_values=filled_values,
            filled_mentions=filled_mentions,
            per_slot_candidates=per_slot_candidates,
            slot_by_name=slot_by_name,
            mention_by_id=mention_by_id,
            max_refinement_steps=max_refinement_steps,
            max_counterfactuals_per_slot=max_counterfactuals_per_slot,
            min_improvement=min_improvement,
        )
        filled_values = refined_values
        filled_mentions = refined_mentions
        diagnostics["counterfactual_grounding_refinement"] = refinement_diag
        diagnostics["use_counterfactual_refinement"] = True
    else:
        diagnostics["use_counterfactual_refinement"] = False
    return filled_values, filled_mentions, diagnostics
