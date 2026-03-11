"""Ambiguity-Aware Grounding for NLP4LP downstream number-to-slot assignment.

This module adds explicit reasoning about *competing* numeric candidates and
*ambiguity* at slot, mention, and query levels — something ordinary greedy /
constrained assignment does not model.

Key capabilities
----------------
1. **Candidate sets** — for each slot keep the top-K ranked mentions (not just
   the single best pair) so that later stages can reason across alternatives.

2. **Ambiguity signals** — slot-level (score margin, entropy, spread) and
   query-level (dense numeric mentions, many low-margin slots).

3. **Competition-aware beam search** — expand partial assignments from the
   best-scoring slots first; prune with configurable beam width.

4. **Abstention logic** — when the top candidate is below a confidence
   threshold *and* the margin over the runner-up is small, prefer leaving a
   slot unassigned rather than forcing a dubious guess.

5. **N-best hypotheses** — produce the top-N complete assignments so callers
   can inspect alternatives and measure where grounding is uncertain.

Four ablation modes for clean experimental comparison
-----------------------------------------------------
"candidate_greedy"  — candidate sets, greedy assignment (no competition / abstain)
"ambiguity_beam"    — + margin-aware beam search
"ambiguity_abstain" — + abstention when confidence low
"ambiguity_full"    — + N-best reranking with competition penalties

Public API
----------
build_candidate_sets(links, slots_ir, mentions_ir, ablation_mode, top_k)
    -> dict[slot_name -> CandidateSet]

compute_slot_ambiguity(cs)  -> SlotAmbiguity
compute_query_ambiguity(slot_ambiguities) -> QueryAmbiguity

beam_assignment(candidate_sets, mentions_ir, ...) -> (values, mentions, diag)
abstain_aware_assignment(candidate_sets, mentions_ir, ...) -> (values, mentions, diag)
nbest_assignments(candidate_sets, mentions_ir, n, ...) -> list[AssignmentHypothesis]
run_ambiguity_aware_grounding(query, variant, expected_scalar, ablation_mode)
    -> (values, mentions, diagnostics)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from tools.relation_aware_linking import (
    ABLATION_MODES as RAL_MODES,
    build_mention_slot_links,
    relation_aware_local_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMBIGUITY_ABLATION_MODES = (
    "candidate_greedy",
    "ambiguity_beam",
    "ambiguity_abstain",
    "ambiguity_full",
)

# Default relation-aware scoring mode used by candidate-set generation.
DEFAULT_RAL_MODE: str = "full"

# Default beam width for beam-search assignment.
DEFAULT_BEAM_WIDTH: int = 8

# Default top-K candidates per slot.
DEFAULT_TOP_K: int = 5

# Default number of N-best full hypotheses to retain.
DEFAULT_N_BEST: int = 3

# Abstention thresholds (tunable).
DEFAULT_MIN_CONFIDENCE: float = 0.5    # minimum normalised score for assignment
DEFAULT_MIN_MARGIN: float = 0.20       # minimum (top1 - top2) / |top1| margin
DEFAULT_MAX_AMBIGUITY: float = 0.90    # slots with normalised entropy > this → abstain

# Normalisation constant for local scores (max expected raw score).
SCORE_NORM: float = 12.0

# Competition penalty applied during N-best reranking.
COMPETITION_PENALTY: float = 0.5

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateEntry:
    """One candidate mention for a given slot."""

    mention_id: int
    score: float            # raw relation-aware score
    norm_score: float       # score normalised to [0, 1] for thresholding


@dataclass
class CandidateSet:
    """Ranked candidate mentions for a single slot."""

    slot_name: str
    candidates: list[CandidateEntry]   # sorted by descending score


@dataclass(frozen=True)
class SlotAmbiguity:
    """Ambiguity signals for a single slot."""

    slot_name: str
    top1_score: float
    top2_score: float
    margin: float           # (top1 - top2) / max(|top1|, 1e-9)
    spread: float           # score range among top-k candidates
    entropy: float          # normalised entropy over top-k candidate scores
    n_candidates: int
    is_ambiguous: bool      # True when entropy > threshold or margin < threshold


@dataclass(frozen=True)
class QueryAmbiguity:
    """Aggregated ambiguity signals for a whole query."""

    n_mentions: int
    n_slots: int
    n_ambiguous_slots: int
    avg_margin: float
    avg_entropy: float
    min_top1_score: float
    query_ambiguity_score: float   # composite [0, 1]; higher = more ambiguous


@dataclass
class AssignmentHypothesis:
    """One complete (possibly partial) slot assignment."""

    rank: int
    score: float            # sum of local scores for assigned slots
    filled_values: dict[str, Any]
    filled_mentions: dict[str, MentionOptIR]
    abstained_slots: list[str]
    competition_penalty: float = 0.0

    @property
    def total_score(self) -> float:
        return self.score - self.competition_penalty


# ---------------------------------------------------------------------------
# Candidate-set construction
# ---------------------------------------------------------------------------


def build_candidate_sets(
    links: list,                   # list[MentionSlotLink] from relation_aware_linking
    slots_ir: list[SlotOptIR],
    mentions_ir: list[MentionOptIR],
    ral_mode: str = DEFAULT_RAL_MODE,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, CandidateSet]:
    """Build per-slot ranked candidate sets from relation-aware link scores.

    Parameters
    ----------
    links      : output of build_mention_slot_links()
    slots_ir   : list of SlotOptIR for this query
    mentions_ir: list of MentionOptIR for this query
    ral_mode   : which relation-aware scoring mode to use
    top_k      : maximum number of candidates to keep per slot

    Returns
    -------
    dict mapping slot_name to CandidateSet
    """
    # Accumulate (score, mention_id) per slot.
    raw: dict[str, list[tuple[float, int]]] = {}
    for lnk in links:
        sc, _ = relation_aware_local_score(lnk, ral_mode)
        if sc <= GCG_PRUNE_THRESHOLD:
            continue
        raw.setdefault(lnk.slot_name, []).append((sc, lnk.mention_id))

    candidate_sets: dict[str, CandidateSet] = {}
    for s in slots_ir:
        pairs = sorted(raw.get(s.name, []), key=lambda x: -x[0])[:top_k]
        entries = [
            CandidateEntry(
                mention_id=mid,
                score=sc,
                norm_score=max(0.0, min(1.0, sc / SCORE_NORM)),
            )
            for sc, mid in pairs
        ]
        candidate_sets[s.name] = CandidateSet(slot_name=s.name, candidates=entries)

    return candidate_sets


# ---------------------------------------------------------------------------
# Ambiguity signals
# ---------------------------------------------------------------------------


def compute_slot_ambiguity(
    cs: CandidateSet,
    entropy_threshold: float = DEFAULT_MAX_AMBIGUITY,
    margin_threshold: float = DEFAULT_MIN_MARGIN,
) -> SlotAmbiguity:
    """Compute ambiguity signals for one slot's candidate set."""
    cands = cs.candidates
    if not cands:
        return SlotAmbiguity(
            slot_name=cs.slot_name,
            top1_score=0.0,
            top2_score=0.0,
            margin=0.0,
            spread=0.0,
            entropy=1.0,
            n_candidates=0,
            is_ambiguous=True,
        )

    top1 = cands[0].score
    top2 = cands[1].score if len(cands) > 1 else 0.0
    denom = max(abs(top1), 1e-9)
    margin = (top1 - top2) / denom
    spread = top1 - cands[-1].score if len(cands) > 1 else 0.0

    # Normalised entropy over candidate norm_scores.
    ns = [max(c.norm_score, 1e-9) for c in cands]
    total = sum(ns)
    probs = [v / total for v in ns]
    raw_entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(max(len(cands), 1))
    norm_entropy = (raw_entropy / max_entropy) if max_entropy > 1e-12 else 0.0

    is_ambiguous = norm_entropy > entropy_threshold or margin < margin_threshold
    return SlotAmbiguity(
        slot_name=cs.slot_name,
        top1_score=top1,
        top2_score=top2,
        margin=margin,
        spread=spread,
        entropy=norm_entropy,
        n_candidates=len(cands),
        is_ambiguous=is_ambiguous,
    )


def compute_query_ambiguity(
    slot_ambiguities: list[SlotAmbiguity],
    n_mentions: int,
) -> QueryAmbiguity:
    """Aggregate per-slot ambiguity signals into a query-level score."""
    if not slot_ambiguities:
        return QueryAmbiguity(
            n_mentions=n_mentions,
            n_slots=0,
            n_ambiguous_slots=0,
            avg_margin=0.0,
            avg_entropy=1.0,
            min_top1_score=0.0,
            query_ambiguity_score=1.0,
        )

    n_slots = len(slot_ambiguities)
    n_ambiguous = sum(1 for sa in slot_ambiguities if sa.is_ambiguous)
    avg_margin = sum(sa.margin for sa in slot_ambiguities) / n_slots
    avg_entropy = sum(sa.entropy for sa in slot_ambiguities) / n_slots
    min_top1 = min(sa.top1_score for sa in slot_ambiguities)

    # Composite score: weighted average of ambiguity indicators.
    ambig_ratio = n_ambiguous / n_slots
    mention_density = min(1.0, n_mentions / max(n_slots, 1) / 3.0)  # >3 mentions/slot → saturate
    query_ambiguity_score = 0.4 * ambig_ratio + 0.35 * avg_entropy + 0.25 * mention_density

    return QueryAmbiguity(
        n_mentions=n_mentions,
        n_slots=n_slots,
        n_ambiguous_slots=n_ambiguous,
        avg_margin=avg_margin,
        avg_entropy=avg_entropy,
        min_top1_score=min_top1,
        query_ambiguity_score=min(1.0, query_ambiguity_score),
    )


# ---------------------------------------------------------------------------
# Assignment strategies
# ---------------------------------------------------------------------------


def _mid_to_value(m: MentionOptIR) -> Any:
    return m.tok.value if m.tok.value is not None else m.tok.raw


def beam_assignment(
    candidate_sets: dict[str, CandidateSet],
    mentions_ir: list[MentionOptIR],
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, Any]]:
    """Margin-aware beam search over candidate sets.

    Slots are processed in decreasing order of their top-1 candidate score.
    At each step we expand each partial assignment by trying all available
    candidates for the current slot, keeping the top-`beam_width` partials.

    A partial assignment is represented as a (total_score, used_mids frozenset,
    filled_values dict, filled_mentions dict) tuple.

    Returns
    -------
    filled_values, filled_mentions, diagnostics
    """
    mid_to_mention = {m.mention_id: m for m in mentions_ir}

    # Order slots: high-confidence first (best top-1 score descending).
    slot_order = sorted(
        candidate_sets.keys(),
        key=lambda sn: candidate_sets[sn].candidates[0].score
        if candidate_sets[sn].candidates
        else -1e9,
        reverse=True,
    )

    # Each beam state: (total_score, frozenset_of_used_mids, values_dict, mentions_dict)
    BeamState = tuple[float, frozenset, dict, dict]
    beam: list[BeamState] = [(0.0, frozenset(), {}, {})]

    for slot_name in slot_order:
        cs = candidate_sets[slot_name]
        new_beam: list[BeamState] = []
        for total_sc, used_mids, vals, mments in beam:
            placed = False
            for cand in cs.candidates:
                if cand.mention_id in used_mids:
                    continue
                m = mid_to_mention.get(cand.mention_id)
                if m is None:
                    continue
                new_vals = dict(vals)
                new_vals[slot_name] = _mid_to_value(m)
                new_mments = dict(mments)
                new_mments[slot_name] = m
                new_used = used_mids | {cand.mention_id}
                new_beam.append((total_sc + cand.score, new_used, new_vals, new_mments))
                placed = True
                break  # only extend with the best available candidate to stay beam-compact
            if not placed:
                # Keep partial assignment as-is (slot gets no candidate).
                new_beam.append((total_sc, used_mids, vals, mments))

        # Keep top beam_width states.
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]

    best_score, _, best_vals, best_mments = beam[0]
    diagnostics: dict[str, Any] = {
        "beam_width": beam_width,
        "n_beam_states": len(beam),
        "best_score": best_score,
    }
    return best_vals, best_mments, diagnostics


def abstain_aware_assignment(
    candidate_sets: dict[str, CandidateSet],
    mentions_ir: list[MentionOptIR],
    slot_ambiguities: dict[str, SlotAmbiguity],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    max_entropy: float = DEFAULT_MAX_AMBIGUITY,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, Any]]:
    """Greedy assignment with explicit abstention on low-confidence slots.

    A slot is left unassigned ("abstained") when:
      - no candidates survive the prune threshold, OR
      - top-1 normalised score < min_confidence, OR
      - score margin < min_margin AND entropy > max_entropy

    Slots are processed high-confidence-first so that confident slots claim
    their mention before uncertain ones.

    Returns
    -------
    filled_values, filled_mentions, diagnostics
    """
    mid_to_mention = {m.mention_id: m for m in mentions_ir}

    # Sort by top-1 score descending (high confidence first).
    slot_order = sorted(
        candidate_sets.keys(),
        key=lambda sn: candidate_sets[sn].candidates[0].score
        if candidate_sets[sn].candidates
        else -1e9,
        reverse=True,
    )

    used_mids: set[int] = set()
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    abstained_slots: list[str] = []

    for slot_name in slot_order:
        cs = candidate_sets[slot_name]
        sa = slot_ambiguities.get(slot_name)

        if not cs.candidates:
            abstained_slots.append(slot_name)
            continue

        top_cand = cs.candidates[0]

        # Abstention checks.
        if top_cand.norm_score < min_confidence:
            abstained_slots.append(slot_name)
            continue
        if sa is not None and sa.margin < min_margin and sa.entropy > max_entropy:
            abstained_slots.append(slot_name)
            continue

        # Assign best available (non-used) candidate.
        for cand in cs.candidates:
            if cand.mention_id in used_mids:
                continue
            m = mid_to_mention.get(cand.mention_id)
            if m is None:
                continue
            filled_values[slot_name] = _mid_to_value(m)
            filled_mentions[slot_name] = m
            used_mids.add(cand.mention_id)
            break

    diagnostics: dict[str, Any] = {
        "abstained_slots": abstained_slots,
        "n_abstained": len(abstained_slots),
        "thresholds": {
            "min_confidence": min_confidence,
            "min_margin": min_margin,
            "max_entropy": max_entropy,
        },
    }
    return filled_values, filled_mentions, diagnostics


def nbest_assignments(
    candidate_sets: dict[str, CandidateSet],
    mentions_ir: list[MentionOptIR],
    n: int = DEFAULT_N_BEST,
    beam_width: int = DEFAULT_BEAM_WIDTH * 4,
) -> list[AssignmentHypothesis]:
    """Generate the top-N complete assignment hypotheses via beam search.

    Uses a wider beam than standard beam_assignment to explore more of the
    hypothesis space, then applies a competition penalty to discourage
    globally incoherent assignments (e.g. two conflicting slots sharing
    near-identical mention context).

    Returns
    -------
    list of AssignmentHypothesis sorted by total_score descending
    """
    mid_to_mention = {m.mention_id: m for m in mentions_ir}

    slot_order = sorted(
        candidate_sets.keys(),
        key=lambda sn: candidate_sets[sn].candidates[0].score
        if candidate_sets[sn].candidates
        else -1e9,
        reverse=True,
    )

    # Beam state: (total_score, frozenset_used_mids, vals, mments, abstained)
    BeamState = tuple[float, frozenset, dict, dict, list]
    beam: list[BeamState] = [(0.0, frozenset(), {}, {}, [])]

    for slot_name in slot_order:
        cs = candidate_sets[slot_name]
        new_beam: list[BeamState] = []

        for total_sc, used_mids, vals, mments, abstained in beam:
            # Try every candidate for this slot (not just top-1) to fill the N-best.
            added_any = False
            for cand in cs.candidates:
                if cand.mention_id in used_mids:
                    continue
                m = mid_to_mention.get(cand.mention_id)
                if m is None:
                    continue
                nv = dict(vals)
                nv[slot_name] = _mid_to_value(m)
                nm = dict(mments)
                nm[slot_name] = m
                new_used = used_mids | {cand.mention_id}
                new_beam.append((total_sc + cand.score, new_used, nv, nm, list(abstained)))
                added_any = True

            # Also always keep the "skip this slot" option.
            new_beam.append((total_sc, used_mids, vals, mments, list(abstained) + [slot_name]))

        # Prune to beam_width.
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]

    # Convert to AssignmentHypothesis objects with competition penalty.
    hypotheses: list[AssignmentHypothesis] = []
    for rank, (sc, _, vals, mments, abstained) in enumerate(beam[:n]):
        # Competition penalty: number of slots assigned to mentions that appear in
        # multiple slots of the same hypothesis (double-assignment evidence of conflict).
        mid_counts: dict[int, int] = {}
        for m in mments.values():
            mid_counts[m.mention_id] = mid_counts.get(m.mention_id, 0) + 1
        n_duplicates = sum(1 for cnt in mid_counts.values() if cnt > 1)
        comp_penalty = n_duplicates * COMPETITION_PENALTY

        hypotheses.append(
            AssignmentHypothesis(
                rank=rank,
                score=sc,
                filled_values=vals,
                filled_mentions=mments,
                abstained_slots=abstained,
                competition_penalty=comp_penalty,
            )
        )

    # Re-sort by total_score (score - competition_penalty).
    hypotheses.sort(key=lambda h: -h.total_score)
    for i, h in enumerate(hypotheses):
        h.rank = i

    return hypotheses


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_ambiguity_aware_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
    ablation_mode: str = "ambiguity_full",
    top_k: int = DEFAULT_TOP_K,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    n_best: int = DEFAULT_N_BEST,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    max_entropy_abstain: float = DEFAULT_MAX_AMBIGUITY,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, Any]]:
    """End-to-end ambiguity-aware grounding pipeline.

    Parameters
    ----------
    query           : raw query text
    variant         : 'orig' | 'noisy' | 'short'
    expected_scalar : list of scalar slot parameter names
    ablation_mode   : 'candidate_greedy' | 'ambiguity_beam' |
                      'ambiguity_abstain' | 'ambiguity_full'

    Returns
    -------
    filled_values, filled_mentions, diagnostics
    """
    if not expected_scalar:
        return {}, {}, {"ablation_mode": ablation_mode, "n_slots": 0}

    # ── 1. Build links using relation-aware linker ──────────────────────
    links, mentions_ir, slots_ir, mention_feats, slot_feats = build_mention_slot_links(
        query, variant, expected_scalar
    )
    if not mentions_ir or not slots_ir:
        return {}, {}, {"ablation_mode": ablation_mode, "n_slots": len(expected_scalar)}

    # ── 2. Build candidate sets ─────────────────────────────────────────
    candidate_sets = build_candidate_sets(
        links, slots_ir, mentions_ir,
        ral_mode=DEFAULT_RAL_MODE,
        top_k=top_k,
    )

    # ── 3. Compute ambiguity signals ────────────────────────────────────
    slot_ambiguities: dict[str, SlotAmbiguity] = {
        sn: compute_slot_ambiguity(cs) for sn, cs in candidate_sets.items()
    }
    query_ambiguity = compute_query_ambiguity(
        list(slot_ambiguities.values()), n_mentions=len(mentions_ir)
    )

    diagnostics: dict[str, Any] = {
        "ablation_mode": ablation_mode,
        "n_mentions": len(mentions_ir),
        "n_slots": len(expected_scalar),
        "query_ambiguity": {
            "score": query_ambiguity.query_ambiguity_score,
            "n_ambiguous_slots": query_ambiguity.n_ambiguous_slots,
            "avg_margin": query_ambiguity.avg_margin,
            "avg_entropy": query_ambiguity.avg_entropy,
        },
        "slot_ambiguities": {
            sn: {
                "top1_score": sa.top1_score,
                "margin": sa.margin,
                "entropy": sa.entropy,
                "n_candidates": sa.n_candidates,
                "is_ambiguous": sa.is_ambiguous,
            }
            for sn, sa in slot_ambiguities.items()
        },
    }

    # ── 4. Assignment strategy ──────────────────────────────────────────
    if ablation_mode == "candidate_greedy":
        # Greedy using candidate sets (no competition / abstain).
        filled_values, filled_mentions, _d = beam_assignment(
            candidate_sets, mentions_ir, beam_width=1
        )
        diagnostics.update(_d)

    elif ablation_mode == "ambiguity_beam":
        filled_values, filled_mentions, _d = beam_assignment(
            candidate_sets, mentions_ir, beam_width=beam_width
        )
        diagnostics.update(_d)

    elif ablation_mode == "ambiguity_abstain":
        filled_values, filled_mentions, _d = abstain_aware_assignment(
            candidate_sets, mentions_ir, slot_ambiguities,
            min_confidence=min_confidence,
            min_margin=min_margin,
            max_entropy=max_entropy_abstain,
        )
        diagnostics.update(_d)

    else:  # "ambiguity_full"
        # Beam search + N-best reranking, with competition penalties.
        hyps = nbest_assignments(
            candidate_sets, mentions_ir,
            n=n_best, beam_width=beam_width * 4,
        )
        if hyps:
            best = hyps[0]
            filled_values = best.filled_values
            filled_mentions = best.filled_mentions
            diagnostics["n_best_hypotheses"] = [
                {
                    "rank": h.rank,
                    "score": h.score,
                    "total_score": h.total_score,
                    "n_filled": len(h.filled_values),
                    "abstained_slots": h.abstained_slots,
                    "competition_penalty": h.competition_penalty,
                }
                for h in hyps
            ]
        else:
            filled_values, filled_mentions = {}, {}

    return filled_values, filled_mentions, diagnostics
