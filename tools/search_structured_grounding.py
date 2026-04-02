"""Search-based structured grounding for numeric slot assignment.

This module implements ``search_structured_grounding``, a beam-search-driven
assignment method that replaces local greedy/repair slot filling with an
explicit search over competing assignment hypotheses, scored by global
consistency and pruned aggressively.

Why this method exists
----------------------
The existing deterministic grounding pipeline (typed-greedy, opt-role repair,
GCG, GCGP) can still fail on several systematic confusion classes:

- **total vs per-unit**: the global budget receives a per-unit price, or vice versa.
- **lower vs upper bound**: min and max slots are swapped.
- **count-like vs quantity-like**: a small cardinality is assigned to a large
  quantity slot, or a large value fills a count slot.
- **percent vs scalar**: a percent-tagged mention fills a plain numeric slot.
- **duplicate reuse**: the same mention fills two incompatible slots because
  each slot's greedy-best is the same number.
- **weak evidence / forced assignment**: a clearly ambiguous slot receives a
  plausible-looking but wrong value rather than being left unfilled.

How it differs from GCG / GCGP
-------------------------------
*Global Consistency Grounding* (GCG) uses threshold-based pruning and
processes all admissible mention-slot pairs during beam expansion.

*Global Compatibility Grounding* (GCGP) adds pairwise slot-slot compatibility
terms but keeps the same threshold-based admission strategy.

``search_structured_grounding`` takes a different approach:

1. **Top-k candidates per slot**: instead of admitting every mention above a
   threshold, it keeps exactly the top-*k* local-scoring candidates per slot
   (configurable, default 5) plus an explicit **null/abstain candidate**.
2. **Slot ordering**: slots are ordered by constraint difficulty before search
   so the hardest, most-constrained slots are processed first, enabling better
   early pruning.
3. **Hard constraint pruning**: illegal partial states are dropped immediately
   during beam expansion (duplicate mention reuse, min > max with values
   assigned, impossible type conflicts).
4. **Explicit null assignment**: abstaining from a slot is a first-class option
   with a small configurable penalty, never a hard failure.
5. **Ablation support**: ``search_structured_grounding_no_global`` runs the
   same search but uses only local scores, making it a fair ablation baseline.

Entry points
------------
- ``search_structured_grounding(query, variant, expected_scalar)``
  → ``(filled_values, filled_mentions, diagnostics)``
- ``search_structured_grounding_no_global(query, variant, expected_scalar)``
  → same signature, local-score-only ablation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (
    MentionOptIR,
    SlotOptIR,
    _bound_swap_repair,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _gcg_global_penalty,
    _gcg_local_score,
    _gcgp_pairwise_score,
)

# ── Configurable constants ─────────────────────────────────────────────────────

# Number of top-scoring mention candidates kept per slot (excluding null).
SSG_TOP_K_PER_SLOT: int = 5

# Beam width (number of partial-assignment states kept at each slot step).
SSG_BEAM_WIDTH: int = 16

# Penalty applied for leaving a slot without any assignment (null / abstain).
# Must be small in magnitude relative to clearly-wrong assignment penalties so
# the method prefers abstaining over forcing a bad value.
SSG_NULL_PENALTY: float = -0.3

# Pruning threshold: mention-slot pairs with local score below this are
# excluded even before the top-k selection.  Hard type-incompatible pairs
# already carry -1e9 from _gcg_local_score, so this only removes weakly
# negative scores.
SSG_PRUNE_THRESHOLD: float = -1.0

# When True, global consistency terms (_gcg_global_penalty + pairwise scores)
# are added to the search.  Flip to False for the no-global ablation.
SSG_ENABLE_GLOBAL: bool = True

# When True, hard constraint pruning drops illegal partial states immediately.
SSG_ENABLE_HARD_PRUNING: bool = True


# ── Slot ordering ──────────────────────────────────────────────────────────────

def _slot_constraint_priority(s: SlotOptIR, candidates: list[tuple[int, float]]) -> float:
    """Lower return value = process this slot earlier (higher priority / harder).

    Heuristics (lower score = harder / more constrained):
    - Percent slots first (strong exclusive constraint with percent mentions).
    - Count-like slots second (small-cardinality prior, easy to violate).
    - Bound slots (min/max explicit operator preference) next.
    - Slots with fewer admissible candidates next (more constrained search branch).
    - Everything else last.
    """
    score = 0.0
    if s.expected_type == "percent":
        score -= 4.0
    if s.is_count_like:
        score -= 3.0
    if s.operator_preference:
        score -= 2.0
    # Fewer admissible candidates → more constrained → process earlier.
    score -= 1.0 / max(1, len(candidates))
    return score


# ── Hard constraint checks ────────────────────────────────────────────────────

def _violates_hard_constraints(
    bundle: frozenset[tuple[int, int]],
    new_j: int,
    new_i: int,
    mentions: list[MentionOptIR],
    slots: list[SlotOptIR],
) -> bool:
    """Return True if adding (new_j, new_i) to bundle creates an illegal state.

    Hard constraints checked:
    1. Same mention reused for two slots (one-to-one violation).
    2. Adding the new assignment would make min-slot value > max-slot value
       when both are already assigned (inverted bound ordering).
    3. Explicit type-incompatible assignment (score would be -1e9 but we
       check it cheaply here to avoid polluting the beam).
    """
    # 1. Duplicate mention.
    used_mentions = {mi for _, mi in bundle}
    if new_i in used_mentions:
        return True

    new_mention = mentions[new_i]
    new_slot = slots[new_j]

    # 2. Min/max inversion: check against already-assigned slots.
    bundle_dict = {j: i for j, i in bundle}
    new_is_min = bool(new_slot.operator_preference and "min" in new_slot.operator_preference)
    new_is_max = bool(new_slot.operator_preference and "max" in new_slot.operator_preference)
    if new_mention.value is not None:
        if new_is_min:
            # Find any already-assigned max-preference slot.
            for prev_j, prev_i in bundle:
                prev_slot = slots[prev_j]
                if prev_slot.operator_preference and "max" in prev_slot.operator_preference:
                    prev_val = mentions[prev_i].value
                    if prev_val is not None and new_mention.value > prev_val:
                        return True  # min > max: inverted
        if new_is_max:
            for prev_j, prev_i in bundle:
                prev_slot = slots[prev_j]
                if prev_slot.operator_preference and "min" in prev_slot.operator_preference:
                    prev_val = mentions[prev_i].value
                    if prev_val is not None and new_mention.value < prev_val:
                        return True  # max < min: inverted

    # 3. Percent / non-percent hard type conflict.
    if new_slot.expected_type == "percent" and new_mention.type_bucket not in ("percent", "unknown"):
        # Only prune if there is a percent mention available.
        has_pct = any(m.type_bucket == "percent" for m in mentions)
        if has_pct:
            return True
    if new_slot.expected_type != "percent" and new_mention.type_bucket == "percent":
        has_non_pct_slot = any(s.expected_type != "percent" for s in slots)
        if has_non_pct_slot:
            return True

    return False


# ── Core beam search ──────────────────────────────────────────────────────────

def _ssg_beam_search(
    mentions: list[MentionOptIR],
    slots: list[SlotOptIR],
    local_scores: list[list[float]],
    local_features: list[list[dict[str, Any]]],
    beam_width: int = SSG_BEAM_WIDTH,
    top_k_per_slot: int = SSG_TOP_K_PER_SLOT,
    null_penalty: float = SSG_NULL_PENALTY,
    enable_global: bool = SSG_ENABLE_GLOBAL,
    enable_hard_pruning: bool = SSG_ENABLE_HARD_PRUNING,
) -> tuple[
    dict[str, MentionOptIR],
    dict[str, float],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Beam search over partial slot assignments with top-k candidates and null options.

    Key improvements over GCG/_gcgp_beam_search:
    - Explicit top-k candidate selection per slot (not just threshold pruning).
    - Null/abstain as a first-class candidate with a small penalty.
    - Strategic slot ordering (hardest-first) before the search.
    - Hard constraint pruning of illegal partial states.
    - Both local + pairwise + global consistency scoring.

    Returns
    -------
    assignments        : slot_name → MentionOptIR (best assignment found)
    slot_scores        : slot_name → local score for the assigned mention
    debug              : slot_name → sorted candidate list (diagnostics)
    top_assignments    : list of top-beam-width assignment dicts (diagnostics)
    search_info        : dict with search statistics
    """
    if not mentions or not slots:
        return {}, {}, {}, [], {}

    m_count = len(mentions)
    s_count = len(slots)
    slots_by_name = {s.name: s for s in slots}

    # ── Build per-slot candidate lists and debug info ──────────────────────────
    # Candidate list for slot j: list of (mention_index, local_score) tuples,
    # limited to top_k_per_slot entries above SSG_PRUNE_THRESHOLD.
    # Index -1 is the null/abstain option.
    candidates_per_slot: list[list[tuple[int, float]]] = []
    debug: dict[str, list[dict[str, Any]]] = {}
    for j, sr in enumerate(slots):
        # Full candidate list for debug output.
        all_cands = []
        for i in range(m_count):
            sc = local_scores[i][j]
            all_cands.append({
                "mention_id": mentions[i].mention_id,
                "mention_raw": mentions[i].raw_surface,
                "score": sc,
                "features": local_features[i][j],
            })
        all_cands.sort(key=lambda x: x["score"], reverse=True)
        debug[sr.name] = all_cands

        # Admissible candidates for search: above threshold, top-k only.
        ranked = sorted(
            (
                (i, local_scores[i][j])
                for i in range(m_count)
                if local_scores[i][j] > SSG_PRUNE_THRESHOLD
            ),
            key=lambda x: -x[1],
        )
        candidates_per_slot.append(ranked[:top_k_per_slot])

    # ── Slot ordering (hardest first) ─────────────────────────────────────────
    slot_order = sorted(
        range(s_count),
        key=lambda j: _slot_constraint_priority(slots[j], candidates_per_slot[j]),
    )

    # ── Beam search ───────────────────────────────────────────────────────────
    # State: (bundle: frozenset[(slot_j, mention_i)], running_score, n_nulls)
    # mention_i == -1 means null/abstain for that slot.
    BeamEntry = tuple[frozenset[tuple[int, int]], float, int]
    beam: list[BeamEntry] = [(frozenset(), 0.0, 0)]

    n_expanded = 0
    for j in slot_order:
        next_beam: list[BeamEntry] = []
        for bundle, running_score, n_nulls in beam:
            used_mentions = {mi for _, mi in bundle if mi >= 0}

            # Option A: null / abstain for slot j.
            null_bundle = bundle | frozenset([(j, -1)])
            next_beam.append((null_bundle, running_score + null_penalty, n_nulls + 1))
            n_expanded += 1

            # Option B: assign an admissible, unused mention.
            for mi, loc_sc in candidates_per_slot[j]:
                if mi in used_mentions:
                    continue
                if enable_hard_pruning and _violates_hard_constraints(
                    bundle, j, mi, mentions, slots
                ):
                    continue
                new_score = running_score + loc_sc
                # Pairwise scoring against already-committed non-null slots.
                if enable_global:
                    for prev_j, prev_mi in bundle:
                        if prev_mi < 0:
                            continue
                        pw_delta, _ = _gcgp_pairwise_score(
                            slots[j], mentions[mi],
                            slots[prev_j], mentions[prev_mi],
                        )
                        new_score += pw_delta
                new_bundle = bundle | frozenset([(j, mi)])
                next_beam.append((new_bundle, new_score, n_nulls))
                n_expanded += 1

        # Sort by running score and prune to beam width.
        next_beam.sort(key=lambda x: -x[1])
        beam = next_beam[:beam_width]

    # ── Final re-scoring with global consistency ───────────────────────────────
    scored_final: list[tuple[float, BeamEntry, float, list[str]]] = []
    for bundle, running_score, n_nulls in beam:
        # Build the non-null assignment for global penalty.
        assignment_candidate: dict[str, MentionOptIR] = {
            slots[j].name: mentions[mi]
            for j, mi in bundle
            if mi >= 0
        }
        if enable_global:
            g_delta, reasons = _gcg_global_penalty(
                assignment_candidate, slots_by_name, mentions
            )
        else:
            g_delta, reasons = 0.0, []
        total_score = running_score + g_delta
        scored_final.append((total_score, (bundle, running_score, n_nulls), g_delta, reasons))

    if not scored_final:
        return {}, {}, debug, [], {}

    scored_final.sort(key=lambda x: -x[0])
    best_total, (best_bundle, best_local_sum, best_nulls), best_g_delta, best_reasons = scored_final[0]

    # ── Build output ──────────────────────────────────────────────────────────
    assignments: dict[str, MentionOptIR] = {}
    slot_scores_out: dict[str, float] = {}
    for j, mi in best_bundle:
        if mi < 0:
            continue  # null/abstain: slot remains unfilled
        slot_name = slots[j].name
        assignments[slot_name] = mentions[mi]
        slot_scores_out[slot_name] = local_scores[mi][j]

    # Top-k diagnostics.
    top_assignments: list[dict[str, Any]] = []
    for rank, (total_sc, (bun, loc_sum, nnulls), g_dl, rsns) in enumerate(
        scored_final[:beam_width]
    ):
        asgn_repr = {}
        for j, mi in bun:
            slot_name = slots[j].name
            asgn_repr[slot_name] = mentions[mi].raw_surface if mi >= 0 else "<null>"
        top_assignments.append({
            "rank": rank,
            "total_score": total_sc,
            "local_sum": loc_sum,
            "global_delta": g_dl,
            "n_nulls": nnulls,
            "active_reasons": rsns,
            "assignment": asgn_repr,
        })

    search_info = {
        "n_mentions": m_count,
        "n_slots": s_count,
        "n_candidate_edges": sum(len(c) for c in candidates_per_slot),
        "beam_width": beam_width,
        "top_k_per_slot": top_k_per_slot,
        "n_expanded_states": n_expanded,
        "best_score": best_total,
        "best_local_sum": best_local_sum,
        "best_global_delta": best_g_delta,
        "n_nulls_in_best": best_nulls,
        "enable_global": enable_global,
        "enable_hard_pruning": enable_hard_pruning,
        "slot_order": [slots[j].name for j in slot_order],
    }

    return assignments, slot_scores_out, debug, top_assignments, search_info


# ── Public API ────────────────────────────────────────────────────────────────

def search_structured_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
    beam_width: int = SSG_BEAM_WIDTH,
    top_k_per_slot: int = SSG_TOP_K_PER_SLOT,
    null_penalty: float = SSG_NULL_PENALTY,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, Any]]:
    """Search-based structured grounding with global consistency scoring.

    Extracts numeric mentions from *query*, builds candidates for each slot in
    *expected_scalar*, and uses beam search with explicit null candidates,
    slot-ordering, hard-constraint pruning, and global consistency scoring to
    find the best overall assignment.

    Parameters
    ----------
    query           : natural-language query string.
    variant         : dataset variant (e.g. ``"orig"``), forwarded to the
                      mention extractor.
    expected_scalar : list of scalar slot names from the predicted schema.
    beam_width      : beam width (default :data:`SSG_BEAM_WIDTH`).
    top_k_per_slot  : maximum candidates per slot, excluding null
                      (default :data:`SSG_TOP_K_PER_SLOT`).
    null_penalty    : score applied for leaving a slot unfilled
                      (default :data:`SSG_NULL_PENALTY`).

    Returns
    -------
    filled_values   : slot_name → assigned numeric value (float or raw string)
    filled_mentions : slot_name → :class:`~tools.nlp4lp_downstream_utility.MentionOptIR`
    diagnostics     : dict containing search statistics and per-slot candidates
    """
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    diagnostics: dict[str, Any] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, diagnostics

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        diagnostics["n_mentions"] = len(mentions)
        diagnostics["n_slots"] = len(slots)
        return filled_values, filled_mentions, diagnostics

    # Precompute local scores for all (mention, slot) pairs.
    m_count, s_count = len(mentions), len(slots)
    local_scores: list[list[float]] = [[0.0] * s_count for _ in range(m_count)]
    local_features: list[list[dict[str, Any]]] = [[{} for _ in range(s_count)] for _ in range(m_count)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, feats = _gcg_local_score(mr, sr)
            local_scores[i][j] = sc
            local_features[i][j] = feats

    assignments, slot_scores, debug, top_assignments, search_info = _ssg_beam_search(
        mentions,
        slots,
        local_scores,
        local_features,
        beam_width=beam_width,
        top_k_per_slot=top_k_per_slot,
        null_penalty=null_penalty,
        enable_global=True,
        enable_hard_pruning=SSG_ENABLE_HARD_PRUNING,
    )

    # Post-assignment bound-swap repair (conservative; only fires on explicit
    # operator evidence + inverted values).
    repair_log: dict[str, str] = {k: "initial" for k in assignments}
    _bound_swap_repair(assignments, repair_log, slots)

    for slot_name, mr in assignments.items():
        filled_values[slot_name] = mr.tok.value if mr.tok.value is not None else mr.tok.raw
        filled_mentions[slot_name] = mr

    diagnostics.update(search_info)
    diagnostics["top_assignments"] = top_assignments
    diagnostics["per_slot_candidates"] = debug
    diagnostics["bound_swap_repairs"] = {
        k: v for k, v in repair_log.items() if v != "initial"
    }
    return filled_values, filled_mentions, diagnostics


def search_structured_grounding_no_global(
    query: str,
    variant: str,
    expected_scalar: list[str],
    beam_width: int = SSG_BEAM_WIDTH,
    top_k_per_slot: int = SSG_TOP_K_PER_SLOT,
    null_penalty: float = SSG_NULL_PENALTY,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, Any]]:
    """Ablation variant: search with local scores only (no global/pairwise terms).

    Identical to :func:`search_structured_grounding` except that
    ``enable_global=False``, disabling global consistency scoring and pairwise
    compatibility terms.  Use this as a controlled ablation baseline.

    Returns the same ``(filled_values, filled_mentions, diagnostics)`` triple.
    """
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    diagnostics: dict[str, Any] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, diagnostics

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        diagnostics["n_mentions"] = len(mentions)
        diagnostics["n_slots"] = len(slots)
        return filled_values, filled_mentions, diagnostics

    m_count, s_count = len(mentions), len(slots)
    local_scores: list[list[float]] = [[0.0] * s_count for _ in range(m_count)]
    local_features: list[list[dict[str, Any]]] = [[{} for _ in range(s_count)] for _ in range(m_count)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, feats = _gcg_local_score(mr, sr)
            local_scores[i][j] = sc
            local_features[i][j] = feats

    assignments, slot_scores, debug, top_assignments, search_info = _ssg_beam_search(
        mentions,
        slots,
        local_scores,
        local_features,
        beam_width=beam_width,
        top_k_per_slot=top_k_per_slot,
        null_penalty=null_penalty,
        enable_global=False,
        enable_hard_pruning=SSG_ENABLE_HARD_PRUNING,
    )

    for slot_name, mr in assignments.items():
        filled_values[slot_name] = mr.tok.value if mr.tok.value is not None else mr.tok.raw
        filled_mentions[slot_name] = mr

    diagnostics.update(search_info)
    diagnostics["top_assignments"] = top_assignments
    diagnostics["per_slot_candidates"] = debug
    return filled_values, filled_mentions, diagnostics
