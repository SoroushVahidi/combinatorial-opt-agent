"""Hierarchical structured grounding with query-region decomposition.

This method is additive to existing grounding methods. It reuses mention/slot IR,
relation-aware local scoring, and search-structured assignment while injecting
region-role compatibility signals prior to final assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    GCG_PRUNE_THRESHOLD,
    MentionOptIR,
    SlotOptIR,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
)
from tools.relation_aware_linking import build_mention_slot_links, relation_aware_local_score
from tools.search_structured_grounding import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_TOP_K_PER_SLOT,
    SEARCH_STRUCTURED_WEIGHTS,
    SearchState,
    SlotCandidate,
    expand_state,
    finalize_best_assignment,
    order_slots_for_search,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_REGION_SPLIT_MARKERS = (
    "subject to",
    "at least",
    "at most",
    "while",
    "and",
    "per",
    "each",
    "total",
    "requires",
    "demand",
    "capacity",
    "cost",
    "profit",
    "budget",
    "minimize",
    "maximize",
)

ROLE_TO_SLOT_COMPAT: dict[str, set[str]] = {
    "objective_region": {"objective_coeff_slot", "per_unit_slot"},
    "constraint_region": {"lower_bound_slot", "upper_bound_slot", "resource_slot", "demand_slot"},
    "resource_region": {"resource_slot", "total_slot", "upper_bound_slot"},
    "demand_region": {"demand_slot", "lower_bound_slot"},
    "bound_region": {"lower_bound_slot", "upper_bound_slot"},
    "per_unit_region": {"per_unit_slot", "objective_coeff_slot"},
    "total_region": {"total_slot", "resource_slot"},
    "count_region": {"count_slot"},
    "rate_region": {"rate_slot"},
    "generic_region": {"generic_slot"},
}

INCOMPATIBLE_REGION_SLOT: set[tuple[str, str]] = {
    ("objective_region", "resource_slot"),
    ("objective_region", "total_slot"),
    ("total_region", "per_unit_slot"),
    ("per_unit_region", "total_slot"),
}


@dataclass(frozen=True)
class RegionRecord:
    region_index: int
    sentence_index: int
    text: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class LocalizedMention:
    mention: MentionOptIR
    sentence_index: int
    region_index: int
    region_text: str
    region_roles: tuple[str, ...]


def _split_sentence_into_regions(sentence: str) -> list[str]:
    work = sentence.strip()
    if not work:
        return []
    lowered = work.lower()
    cut_points: list[int] = []
    for marker in _REGION_SPLIT_MARKERS:
        for m_obj in re.finditer(rf"\b{re.escape(marker)}\b", lowered):
            pos = m_obj.start()
            if 0 < pos < len(work) - 1:
                cut_points.append(pos)
    if not cut_points:
        return [work]
    cuts = sorted(set(cut_points))
    parts: list[str] = []
    start = 0
    for cut in cuts:
        piece = work[start:cut].strip(" ,")
        if piece:
            parts.append(piece)
        start = cut
    tail = work[start:].strip(" ,")
    if tail:
        parts.append(tail)
    return parts if parts else [work]


def infer_region_roles(region_text: str) -> tuple[str, ...]:
    t = region_text.lower()
    roles: set[str] = set()
    if any(k in t for k in ("minimize", "maximize", "objective", "profit", "revenue", "cost")):
        roles.add("objective_region")
    if any(k in t for k in ("subject to", "constraint", "must", "allowed", "cannot", "at least", "at most")):
        roles.add("constraint_region")
    if any(k in t for k in ("capacity", "resource", "available", "budget", "supply", "warehouse")):
        roles.add("resource_region")
    if any(k in t for k in ("demand", "requirement", "required", "needs", "must meet")):
        roles.add("demand_region")
    if any(k in t for k in ("at least", "at most", "minimum", "maximum", "lower", "upper", "min", "max")):
        roles.add("bound_region")
    if any(k in t for k in ("per", "each", "per unit", "costs")):
        roles.add("per_unit_region")
    if any(k in t for k in ("total", "overall", "aggregate", "budget")):
        roles.add("total_region")
    if any(k in t for k in ("number of", "count", "products", "types")):
        roles.add("count_region")
    if any(k in t for k in ("percent", "percentage", "rate", "%", "fraction")):
        roles.add("rate_region")
    if not roles:
        roles.add("generic_region")
    return tuple(sorted(roles))


def split_query_into_regions(query: str) -> list[RegionRecord]:
    regions: list[RegionRecord] = []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(query) if s and s.strip()]
    ridx = 0
    for sidx, sent in enumerate(sentences):
        for piece in _split_sentence_into_regions(sent):
            roles = infer_region_roles(piece)
            regions.append(RegionRecord(region_index=ridx, sentence_index=sidx, text=piece, roles=roles))
            ridx += 1
    if not regions:
        roles = infer_region_roles(query)
        regions.append(RegionRecord(region_index=0, sentence_index=0, text=query, roles=roles))
    return regions


def attach_mentions_to_regions(mentions: list[MentionOptIR], regions: list[RegionRecord]) -> dict[int, LocalizedMention]:
    localized: dict[int, LocalizedMention] = {}
    for m in mentions:
        raw = (m.raw_surface or "").lower()
        chosen = regions[0]
        if raw and not raw.startswith("derived:"):
            for r in regions:
                if raw in r.text.lower():
                    chosen = r
                    break
        else:
            best_overlap = -1
            m_ctx = set(m.context_tokens)
            for r in regions:
                overlap = len(m_ctx & set(re.findall(r"[a-zA-Z]+", r.text.lower())))
                if overlap > best_overlap:
                    best_overlap = overlap
                    chosen = r
        localized[m.mention_id] = LocalizedMention(
            mention=m,
            sentence_index=chosen.sentence_index,
            region_index=chosen.region_index,
            region_text=chosen.text,
            region_roles=chosen.roles,
        )
    return localized


def infer_slot_roles(slot_name: str, slot_record: SlotOptIR) -> tuple[str, ...]:
    n = slot_name.lower()
    roles: set[str] = set()
    if slot_record.is_coefficient_like or any(k in n for k in ("cost", "profit", "revenue")):
        roles.add("objective_coeff_slot")
    if slot_record.is_total_like or any(k in n for k in ("total", "budget", "capacity", "available")):
        roles.add("total_slot")
    if any(k in n for k in ("per", "each", "unit")):
        roles.add("per_unit_slot")
    if "min" in slot_record.operator_preference or any(k in n for k in ("minimum", "lower")):
        roles.add("lower_bound_slot")
    if "max" in slot_record.operator_preference or any(k in n for k in ("maximum", "upper")):
        roles.add("upper_bound_slot")
    if slot_record.is_count_like or any(k in n for k in ("count", "num", "number", "products")):
        roles.add("count_slot")
    if slot_record.expected_type == "percent" or any(k in n for k in ("percent", "ratio", "rate", "share")):
        roles.add("rate_slot")
    if any(k in n for k in ("resource", "capacity", "budget", "supply", "warehouse")):
        roles.add("resource_slot")
    if any(k in n for k in ("demand", "require", "need")):
        roles.add("demand_slot")
    if not roles:
        roles.add("generic_slot")
    return tuple(sorted(roles))


def region_slot_compatibility_score(region_roles: tuple[str, ...], slot_roles: tuple[str, ...]) -> tuple[float, list[str]]:
    rr = set(region_roles)
    sr = set(slot_roles)
    bonus = 0.0
    reasons: list[str] = []
    for r in rr:
        compat = ROLE_TO_SLOT_COMPAT.get(r, set())
        if compat & sr:
            bonus += 1.25
            reasons.append(f"region_match:{r}")
    for pair in INCOMPATIBLE_REGION_SLOT:
        if pair[0] in rr and pair[1] in sr:
            bonus -= 1.5
            reasons.append(f"region_incompat:{pair[0]}->{pair[1]}")
    if "generic_region" in rr and sr != {"generic_slot"}:
        bonus -= 0.15
        reasons.append("generic_region_weakness")
    return bonus, reasons


def build_hierarchical_candidates(
    query: str,
    variant: str,
    expected_scalar: list[str],
    *,
    use_regions: bool,
    top_k_per_slot: int,
) -> tuple[list[MentionOptIR], list[SlotOptIR], dict[int, LocalizedMention], dict[str, tuple[str, ...]], dict[str, list[SlotCandidate]], int]:
    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    slot_roles = {s.name: infer_slot_roles(s.name, s) for s in slots}
    regions = split_query_into_regions(query)
    loc_mentions = attach_mentions_to_regions(mentions, regions)

    links, _, _, _, _ = build_mention_slot_links(query, variant, expected_scalar)
    by_slot: dict[str, list[SlotCandidate]] = {s.name: [] for s in slots}
    n_edges = 0

    for lnk in links:
        base_score, base_feats = relation_aware_local_score(lnk, "full")
        if base_score <= GCG_PRUNE_THRESHOLD:
            continue
        if lnk.type_incompatible or lnk.percent_mismatch:
            continue

        mention_loc = loc_mentions.get(lnk.mention_id)
        reg_bonus = 0.0
        reg_reasons: list[str] = []
        if use_regions and mention_loc is not None:
            reg_bonus, reg_reasons = region_slot_compatibility_score(
                mention_loc.region_roles,
                slot_roles.get(lnk.slot_name, ("generic_slot",)),
            )
        score = base_score + reg_bonus
        feats = dict(base_feats)
        feats["base_local_score"] = base_score
        feats["region_bonus"] = reg_bonus
        feats["region_reasons"] = reg_reasons
        feats["slot_roles"] = list(slot_roles.get(lnk.slot_name, ("generic_slot",)))
        feats["mention_region_roles"] = list(mention_loc.region_roles) if mention_loc else ["generic_region"]
        feats["region_index"] = mention_loc.region_index if mention_loc else -1

        cand = SlotCandidate(
            slot_name=lnk.slot_name,
            mention_id=lnk.mention_id,
            mention_value=lnk.mention_feats.value,
            mention_raw=lnk.mention_feats.raw_surface,
            local_score=score,
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

    return mentions, slots, loc_mentions, slot_roles, by_slot, n_edges


def _run_assignment_search(
    slots: list[SlotOptIR],
    candidates: dict[str, list[SlotCandidate]],
    mention_by_id: dict[int, MentionOptIR],
    *,
    beam_width: int,
    use_global: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ordered_slots = order_slots_for_search(slots, candidates)
    slot_by_name = {s.name: s for s in slots}
    beam = [SearchState(assignment={}, used_mentions=set(), local_score=0.0, global_score=0.0, score=0.0, prune_reasons=[])]
    best_so_far = float("-inf")
    expanded_states = 0
    pruned_states = 0

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
            next_beam.extend(children)
            expanded_states += len(children)
            pruned_states += pruned
        next_beam.sort(key=lambda s: (-s.score, -len([c for c in s.assignment.values() if not c.is_null]), sorted(s.assignment.keys())))
        beam = next_beam[:beam_width] if next_beam else beam
        if beam:
            best_so_far = max(best_so_far, beam[0].score)

    best = beam[0] if beam else SearchState({}, set(), 0.0, 0.0, 0.0, ["empty_beam"])
    return (*finalize_best_assignment(best, mention_by_id), {
        "best_score": best.score,
        "best_local_score": best.local_score,
        "best_global_score": best.global_score,
        "expanded_states": expanded_states,
        "pruned_states": pruned_states,
        "slot_order": [s.name for s in ordered_slots],
    })


def _run_assignment_greedy(
    slots: list[SlotOptIR],
    candidates: dict[str, list[SlotCandidate]],
    mention_by_id: dict[int, MentionOptIR],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assigned: dict[str, SlotCandidate] = {}
    used: set[int] = set()
    for slot in order_slots_for_search(slots, candidates):
        best = next((c for c in candidates.get(slot.name, []) if c.is_null), None)
        for cand in candidates.get(slot.name, []):
            if cand.is_null:
                continue
            if cand.mention_id in used:
                continue
            if best is None or cand.local_score > best.local_score:
                best = cand
        if best is None:
            continue
        assigned[slot.name] = best
        if best.mention_id is not None:
            used.add(best.mention_id)
    return (*finalize_best_assignment(assigned, mention_by_id), {"ablation": "no_search", "slot_order": [s.name for s in order_slots_for_search(slots, candidates)]})


def run_hierarchical_structured_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
    *,
    ablation_mode: str = "full",
    beam_width: int = DEFAULT_BEAM_WIDTH,
    top_k_per_slot: int = DEFAULT_TOP_K_PER_SLOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run hierarchical structured grounding.

    ablation_mode:
      - "full"        : region-aware + search-based assignment
      - "no_regions"  : search-based assignment without region bonus/penalty
      - "no_search"   : region-aware scoring with greedy assignment
    """
    if not expected_scalar:
        return {}, {}, {"ablation_mode": ablation_mode, "n_slots": 0, "n_mentions": 0}

    use_regions = ablation_mode != "no_regions"
    mentions, slots, loc_mentions, slot_roles, candidates, n_edges = build_hierarchical_candidates(
        query,
        variant,
        expected_scalar,
        use_regions=use_regions,
        top_k_per_slot=top_k_per_slot,
    )
    mention_by_id = {m.mention_id: m for m in mentions}

    if ablation_mode == "no_search":
        filled_values, filled_mentions, assignment_diag = _run_assignment_greedy(slots, candidates, mention_by_id)
    else:
        filled_values, filled_mentions, assignment_diag = _run_assignment_search(
            slots,
            candidates,
            mention_by_id,
            beam_width=beam_width,
            use_global=True,
        )

    region_rows = []
    for r in split_query_into_regions(query):
        region_rows.append({
            "region_index": r.region_index,
            "sentence_index": r.sentence_index,
            "text": r.text,
            "roles": list(r.roles),
        })

    mention_rows = []
    for mid, lm in sorted(loc_mentions.items()):
        mention_rows.append({
            "mention_id": mid,
            "raw": lm.mention.raw_surface,
            "value": lm.mention.value,
            "sentence_index": lm.sentence_index,
            "region_index": lm.region_index,
            "region_text": lm.region_text,
            "region_roles": list(lm.region_roles),
        })

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
        for s in slots
    }

    diagnostics = {
        "ablation_mode": ablation_mode,
        "n_mentions": len(mentions),
        "n_slots": len(slots),
        "n_candidate_edges": n_edges,
        "beam_width": beam_width,
        "top_k_per_slot": top_k_per_slot,
        "regions": region_rows,
        "slot_roles": {k: list(v) for k, v in slot_roles.items()},
        "localized_mentions": mention_rows,
        "per_slot_candidates": per_slot_candidates,
        "final_assignment": {k: (None if v is None else v.raw_surface) for k, v in filled_mentions.items()},
    }
    diagnostics.update(assignment_diag)
    return filled_values, filled_mentions, diagnostics
