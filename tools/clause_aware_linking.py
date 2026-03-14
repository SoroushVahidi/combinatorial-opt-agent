"""Group 3 clause-aware linking helpers for NLP4LP downstream grounding.

This module provides lightweight, deterministic clause-level utilities that
complement the local mention-slot scoring in ``relation_aware_linking``.

Group 3 scope: clause-parallel structure + structured reranking / global
coherence.  The primary new signal is the *left-directional entity anchor*
(computed in ``relation_aware_linking`` via ``left_entity_anchor_overlap``),
which discriminates sibling entities in parallel-clause patterns.  The helpers
here provide:

1. Clause splitting (``split_into_clauses``, ``ClauseSpan``)
   Splits a query into lightweight clause spans using sentence boundaries,
   semicolons, and connector words (while, whereas, but).  Does NOT use "and"
   as a standalone boundary (too aggressive for measure-listing patterns like
   "3 heating hours and 5 cooling hours").

2. Clause summaries (``ClauseSummary``, ``build_clause_summaries``)
   For each clause, a lightweight record of its content tokens, which can be
   used for inspection, ablation analysis, and evaluation.

3. Parallel clause detection (``detect_parallel_clauses``)
   Identifies pairs of clauses that share similar measure tokens but differ by
   entity anchor, indicating a repeated parallel pattern.

4. Swap repair (``detect_and_repair_parallel_swaps``)
   After greedy assignment, checks whether pairwise sibling-slot swaps improve
   left-entity-anchor alignment and performs beneficial swaps.

Design principles:
- Deterministic and CPU-only (no learned models).
- Modular: all helpers are standalone functions that can be ablated cleanly.
- Conservative: only trigger when the evidence is clear; neutral otherwise.
- Non-destructive: does not replace the existing matching/repair system;
  it runs as a post-processing layer on top of greedy assignment.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import MentionOptIR, _normalize_tokens
from tools.relation_aware_linking import MentionSlotLink, relation_aware_local_score


# ---------------------------------------------------------------------------
# Clause splitting
# ---------------------------------------------------------------------------

# Connector words that act as clause boundaries when they appear as standalone
# tokens.  "and" is intentionally excluded: it is too common as an intra-clause
# coordinator (e.g. "3 heating hours and 5 cooling hours") and would over-split.
_CLAUSE_CONNECTORS: frozenset[str] = frozenset({"while", "whereas", "but"})

# Words that should not be treated as clause boundaries even when they look like
# connectors (e.g. "no more than", "nothing but").  This is a small guard list.
_CONNECTOR_STOP_BEFORE: frozenset[str] = frozenset({"nothing", "no", "not"})


@dataclass(frozen=True)
class ClauseSpan:
    """Lightweight representation of one clause segment.

    Attributes
    ----------
    clause_idx    : Sequential index (0-based) of this clause in the query.
    text          : Raw text of the clause (stripped).
    content_tokens: Lowercase, stripped content tokens from the clause text.
                    Suitable for token-level overlap comparisons.
    """

    clause_idx: int
    text: str
    content_tokens: frozenset[str]


@dataclass(frozen=True)
class ClauseSummary:
    """Richer per-clause summary including extracted entity and measure cues.

    Suitable for inspection, evaluation, and parallel-clause detection.
    """

    clause_idx: int
    text: str
    content_tokens: frozenset[str]
    # Entity-like tokens: capitalised words that likely name an entity.
    entity_cue_tokens: frozenset[str]
    # Measure-like tokens: words that frequently name resources or attributes
    # (e.g. protein, fat, heating, cooling, labor, wood).  Detected via
    # overlap with known measure/attribute words or by proximity heuristics.
    measure_cue_tokens: frozenset[str]
    # Numeric surface strings of numeric mentions whose left-anchor context
    # falls predominantly within this clause.
    numeric_surfaces: tuple[str, ...]


def split_into_clauses(text: str) -> list[ClauseSpan]:
    """Split *text* into lightweight clause spans.

    Boundaries are detected at:
    - Sentence-ending punctuation followed by an uppercase token.
    - Semicolons.
    - Connector words in ``_CLAUSE_CONNECTORS`` (while, whereas, but).

    Returns a list of :class:`ClauseSpan` objects in order.  The list always
    contains at least one entry (the whole text if no boundary is found).

    Parameters
    ----------
    text : str
        Raw query / sentence string.

    Returns
    -------
    list[ClauseSpan]
        Ordered clause spans covering the whole text.
    """
    if not text or not text.strip():
        return []

    # Tokenize on whitespace while keeping the raw tokens for reconstruction.
    raw_tokens = text.split()
    if not raw_tokens:
        return []

    # Identify split positions (indices into raw_tokens where a NEW clause starts).
    split_starts: list[int] = [0]

    for idx, tok in enumerate(raw_tokens):
        tok_clean = tok.lower().strip(".,;:()[]{}\"'")
        tok_raw_stripped = tok.rstrip()

        # 1. Sentence boundary: token ends with . / ! / ? and next token is uppercase.
        if (
            idx + 1 < len(raw_tokens)
            and tok_raw_stripped.endswith((".", "!", "?"))
            and raw_tokens[idx + 1][:1].isupper()
            and raw_tokens[idx + 1].strip()
        ):
            split_starts.append(idx + 1)
            continue

        # 2. Semicolon boundary: token ends with ";", or consists entirely of
        #    semicolons (e.g. a bare ";" token after tokenisation on whitespace).
        if tok_raw_stripped.endswith(";") or not tok_raw_stripped.replace(";", ""):
            if idx + 1 < len(raw_tokens):
                split_starts.append(idx + 1)
            continue

        # 3. Connector word boundary: the current token IS a connector word and
        #    the previous token does not prevent the split (e.g. "no but").
        if tok_clean in _CLAUSE_CONNECTORS:
            prev_clean = raw_tokens[idx - 1].lower().strip(".,;:()[]{}\"'") if idx > 0 else ""
            if prev_clean not in _CONNECTOR_STOP_BEFORE:
                split_starts.append(idx)

    # Deduplicate and sort.
    split_starts = sorted(set(split_starts))

    # Build clause spans from consecutive split positions.
    spans: list[ClauseSpan] = []
    for i, start in enumerate(split_starts):
        end = split_starts[i + 1] if i + 1 < len(split_starts) else len(raw_tokens)
        clause_tokens = raw_tokens[start:end]
        clause_text = " ".join(clause_tokens).strip()
        # Build content token set: lowercase, stripped.
        content = frozenset(
            t.lower().strip(".,;:()[]{}\"'")
            for t in clause_tokens
            if t.strip(".,;:()[]{}\"'").strip()
        )
        spans.append(ClauseSpan(clause_idx=i, text=clause_text, content_tokens=content))

    return spans


def clause_for_text_position(char_pos: int, text: str, clauses: list[ClauseSpan]) -> int:
    """Return the index of the clause whose text span contains *char_pos*.

    Uses a simple character-search over ``clause.text`` to locate the right
    clause.  Falls back to 0 if no match is found.

    Parameters
    ----------
    char_pos : int
        Character offset in *text*.
    text : str
        Original query string.
    clauses : list[ClauseSpan]
        Clause spans produced by :func:`split_into_clauses`.

    Returns
    -------
    int
        Index of the matching clause.
    """
    cumulative = 0
    for clause in clauses:
        clause_start = text.find(clause.text, cumulative)
        if clause_start == -1:
            continue
        clause_end = clause_start + len(clause.text)
        if clause_start <= char_pos < clause_end:
            return clause.clause_idx
        cumulative = max(cumulative, clause_start)
    return 0


# ---------------------------------------------------------------------------
# Clause summaries
# ---------------------------------------------------------------------------

# Lightweight measure-word list for heuristic measure-cue extraction.
# This is intentionally conservative: only clear measure/attribute nouns.
_MEASURE_WORDS: frozenset[str] = frozenset({
    "protein", "fat", "fiber", "carb", "carbs", "calorie", "calories",
    "heating", "cooling", "manufacturing", "assembly", "finishing",
    "labor", "labour", "material", "wood", "stain", "glass",
    "time", "hours", "hour", "days", "day", "minutes",
    "cost", "profit", "revenue", "price", "wage", "salary",
    "capacity", "demand", "supply", "budget", "resource",
    "weight", "volume", "area", "length", "distance",
})


def build_clause_summaries(
    clauses: list[ClauseSpan],
    mentions: list[MentionOptIR],
) -> list[ClauseSummary]:
    """Build a :class:`ClauseSummary` for each clause.

    Entity cue tokens are uppercase words (likely proper nouns / entity names)
    found in the clause.  Measure cue tokens are words from the clause that
    appear in the known ``_MEASURE_WORDS`` set.  Numeric surfaces are the raw
    surfaces of mentions whose ``narrow_left_tokens`` overlap most with the
    clause's content tokens.

    Parameters
    ----------
    clauses  : list[ClauseSpan]  Clause spans to summarise.
    mentions : list[MentionOptIR]  Extracted numeric mentions.

    Returns
    -------
    list[ClauseSummary]
        One summary per clause, in order.
    """
    summaries: list[ClauseSummary] = []

    # Assign each mention to its most likely clause by left-anchor overlap.
    mention_clause: dict[int, int] = _assign_mentions_to_clauses(mentions, clauses)

    for clause in clauses:
        raw_toks = clause.text.split()

        # Entity cues: capitalised tokens (likely proper nouns / named entities).
        # Include single-letter tokens (e.g. "A" in "Feed A", "B" in "Feed B")
        # because they are key entity discriminators in parallel patterns.
        entity_cues = frozenset(
            t.strip(".,;:()[]{}\"'")
            for t in raw_toks
            if t and t[0].isupper() and t.strip(".,;:()[]{}\"'")
        )

        # Measure cues: content tokens matching the known measure-word list.
        measure_cues = frozenset(
            t for t in clause.content_tokens if t in _MEASURE_WORDS
        )

        # Numeric surfaces for mentions assigned to this clause.
        numeric_surfs = tuple(
            m.raw_surface
            for m in mentions
            if mention_clause.get(m.mention_id, -1) == clause.clause_idx
        )

        summaries.append(
            ClauseSummary(
                clause_idx=clause.clause_idx,
                text=clause.text,
                content_tokens=clause.content_tokens,
                entity_cue_tokens=entity_cues,
                measure_cue_tokens=measure_cues,
                numeric_surfaces=numeric_surfs,
            )
        )

    return summaries


def _assign_mentions_to_clauses(
    mentions: list[MentionOptIR],
    clauses: list[ClauseSpan],
) -> dict[int, int]:
    """Map each mention's id to its best clause index.

    The best clause is the one whose content tokens share the most overlap
    with the mention's ``narrow_left_tokens`` (the directional left anchor).
    Ties are broken by clause index (earlier clause preferred).

    Parameters
    ----------
    mentions : list[MentionOptIR]
    clauses  : list[ClauseSpan]

    Returns
    -------
    dict[int, int]
        mention_id → clause_idx
    """
    result: dict[int, int] = {}
    for m in mentions:
        left_set = set(m.narrow_left_tokens)
        if not left_set or not clauses:
            result[m.mention_id] = 0
            continue
        best_idx = 0
        best_overlap = -1
        for clause in clauses:
            overlap = len(left_set & clause.content_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = clause.clause_idx
        result[m.mention_id] = best_idx
    return result


# ---------------------------------------------------------------------------
# Parallel clause detection
# ---------------------------------------------------------------------------


def detect_parallel_clauses(
    summaries: list[ClauseSummary],
) -> list[tuple[int, int]]:
    """Detect pairs of clauses that form a repeated parallel pattern.

    Two clauses are considered parallel when:
    - They share at least one measure cue token (same measure family), AND
    - They differ in at least one entity cue token (different entity anchors).

    Returns a list of (clause_idx_i, clause_idx_j) pairs.  Only pairs where
    both clauses have at least one measure cue are returned.

    Parameters
    ----------
    summaries : list[ClauseSummary]

    Returns
    -------
    list[tuple[int, int]]
        Ordered pairs of clause indices that form a parallel pattern.
    """
    parallel: list[tuple[int, int]] = []
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            si = summaries[i]
            sj = summaries[j]
            shared_measures = si.measure_cue_tokens & sj.measure_cue_tokens
            distinct_entities = (si.entity_cue_tokens | sj.entity_cue_tokens) - (
                si.entity_cue_tokens & sj.entity_cue_tokens
            )
            if shared_measures and distinct_entities:
                parallel.append((si.clause_idx, sj.clause_idx))
    return parallel


# ---------------------------------------------------------------------------
# Swap repair
# ---------------------------------------------------------------------------


def _left_anchor_overlap_for_link(
    link: MentionSlotLink,
) -> int:
    """Return the pre-computed left_entity_anchor_overlap for *link*."""
    return link.left_entity_anchor_overlap


def _find_link(
    links: list[MentionSlotLink],
    mention_id: int,
    slot_name: str,
) -> MentionSlotLink | None:
    """Return the link for the given (mention_id, slot_name) pair, or None."""
    for lnk in links:
        if lnk.mention_id == mention_id and lnk.slot_name == slot_name:
            return lnk
    return None


def _slot_entity_words_from_link(lnk: MentionSlotLink) -> frozenset[str]:
    """Return entity-discriminating tokens for *lnk*'s slot.

    Mirrors the filter used for ``left_entity_anchor_overlap``: takes the
    slot's full norm_token set, removes measure/unit words (via the same
    exclude-list used in ``relation_aware_linking``), and removes the full
    lower-cased slot name (which duplicates the component tokens).
    """
    from tools.relation_aware_linking import _LEFT_ANCHOR_MEASURE_EXCLUDE  # noqa: PLC0415
    sw = frozenset(lnk.slot_feats.norm_tokens) | lnk.slot_feats.alias_tokens
    return (sw - _LEFT_ANCHOR_MEASURE_EXCLUDE) - {lnk.slot_name.lower()}


def _clause_entity_consistency_gain(
    sn_i: str,
    sn_j: str,
    m_i: MentionOptIR,
    m_j: MentionOptIR,
    links: list[MentionSlotLink],
    mentions_in_assignment: list[MentionOptIR],
    clauses: list["ClauseSpan"],
) -> int:
    """Return the change in clause-entity consistency from swapping sn_i↔sn_j.

    For each of the two slots being considered for a swap, we look at the
    OTHER slots that are assigned to the same clause as the mention.  A
    consistent clause-entity assignment is one where the slot's entity words
    overlap with the entity cue tokens found in that clause.

    Positive return value → swap improves clause-entity consistency.
    Negative return value → swap hurts consistency (do NOT swap).
    Zero → neutral (no clause evidence; leave decision to caller).

    This is ONLY called when left-anchor scores are tied (conservative guard).

    Parameters
    ----------
    sn_i, sn_j      : Names of the two candidate-swap slots.
    m_i, m_j        : Current mention assigned to sn_i, sn_j respectively.
    links           : Full link table from build_mention_slot_links.
    mentions_in_assignment : All mentions currently assigned (values of
                       filled_mentions dict).
    clauses         : Clause spans for the query.
    """
    if not clauses:
        return 0

    mention_to_clause = _assign_mentions_to_clauses(mentions_in_assignment, clauses)

    # Build per-clause entity-cue sets (lowercase, stripped).
    summaries = build_clause_summaries(clauses, mentions_in_assignment)
    clause_ent_map: dict[int, frozenset[str]] = {}
    for s in summaries:
        raw = frozenset(t.lower().strip(".,;:()[]{}\"'") for t in s.entity_cue_tokens)
        # Expand compound tokens ('Type1' → {'type1','type','1'}).
        clause_ent_map[s.clause_idx] = frozenset(_expand_compound_tokens(set(raw)))

    # Entity words for each candidate slot.
    lnk_ii = _find_link(links, m_i.mention_id, sn_i)
    lnk_jj = _find_link(links, m_j.mention_id, sn_j)
    lnk_ij = _find_link(links, m_i.mention_id, sn_j)
    lnk_ji = _find_link(links, m_j.mention_id, sn_i)
    if not all([lnk_ii, lnk_jj, lnk_ij, lnk_ji]):
        return 0

    ent_i = _slot_entity_words_from_link(lnk_ii)   # type: ignore[arg-type]
    ent_j = _slot_entity_words_from_link(lnk_jj)   # type: ignore[arg-type]

    ci = mention_to_clause.get(m_i.mention_id, 0)
    cj = mention_to_clause.get(m_j.mention_id, 0)

    if ci == cj:
        # Same clause: clause evidence does not discriminate.
        return 0

    clause_ent_i = clause_ent_map.get(ci, frozenset())
    clause_ent_j = clause_ent_map.get(cj, frozenset())

    # Current consistency: m_i→sn_i in clause ci, m_j→sn_j in clause cj.
    curr = len(clause_ent_i & ent_i) + len(clause_ent_j & ent_j)
    # Swapped consistency: m_i→sn_j in clause ci, m_j→sn_i in clause cj.
    swap = len(clause_ent_i & ent_j) + len(clause_ent_j & ent_i)

    return swap - curr


def _expand_compound_tokens(tokens: set[str]) -> set[str]:
    """Expand alphanum compound tokens (e.g. 'type1' → {'type1','type','1'}).

    Thin wrapper that re-uses the logic from ``relation_aware_linking``;
    defined here to avoid an import cycle (clause_aware_linking already
    imports from relation_aware_linking).
    """
    import re as _re_local  # noqa: PLC0415
    _compound = _re_local.compile(r"([a-zA-Z]+)(\d+)$")
    expanded = set(tokens)
    for tok in tokens:
        m = _compound.fullmatch(tok)
        if m:
            expanded.add(m.group(1))
            expanded.add(m.group(2))
    return expanded


def detect_and_repair_parallel_swaps(
    filled_values: dict[str, Any],
    filled_mentions: dict[str, MentionOptIR],
    links: list[MentionSlotLink],
    ablation_mode: str = "full",
    query: str = "",
    all_mentions: list[MentionOptIR] | None = None,
) -> tuple[dict[str, Any], dict[str, MentionOptIR], list[str]]:
    """Detect and repair parallel sibling-slot swap errors.

    After an initial greedy assignment, this function checks whether any pair
    of assigned slots can be improved by swapping their mentions.  A swap is
    performed when:

    1. Both slots share at least one common ``norm_token`` (same measure family)
       but have at least one differing token (different entity component).
    2. The swap strictly improves the total ``left_entity_anchor_overlap`` for
       the two-slot pair **or**, when anchor scores are exactly tied, the swap
       strictly improves clause-entity consistency or narrow-measure-overlap
       (the conservative tiebreakers).

    After the swap pass, an optional "derived-occupant rescue" pass looks for
    any slot filled by a low-quality derived mention (nm_ov = 0) that could be
    replaced by an unassigned real mention with better evidence.

    Parameters
    ----------
    filled_values   : dict[str, Any]
        Current slot → value mapping from greedy assignment.
    filled_mentions : dict[str, MentionOptIR]
        Current slot → mention mapping from greedy assignment.
    links           : list[MentionSlotLink]
        All mention-slot links from ``build_mention_slot_links``.
    ablation_mode   : str
        Ablation mode (used only for logging; swap logic uses left_entity_anchor_overlap
        which is mode-independent at the link level).
    query           : str
        Original query string.  When non-empty, enables the clause-entity
        consistency tiebreaker for anchor-tied swap candidates.
    all_mentions    : list[MentionOptIR] | None
        Complete list of all extracted mentions (not just the greedy-assigned
        subset).  When provided, enables the derived-occupant rescue pass that
        replaces low-quality derived occupants with better unassigned mentions.

    Returns
    -------
    new_values   : dict[str, Any]   Updated slot → value mapping.
    new_mentions : dict[str, MentionOptIR]   Updated slot → mention mapping.
    swap_log     : list[str]   Human-readable log of swaps performed.
    """
    new_values = dict(filled_values)
    new_mentions = dict(filled_mentions)
    swap_log: list[str] = []

    # Work with only the filled slots (those that received a mention).
    filled_slot_names = list(new_mentions.keys())
    if len(filled_slot_names) < 2:
        return new_values, new_mentions, swap_log

    # Build a quick lookup: mention_id → slot_name (reverse mapping of current assignment).
    # This lets us verify that a mention is indeed assigned to the expected slot.
    mid_to_slot: dict[int, str] = {
        m.mention_id: sn for sn, m in new_mentions.items()
    }

    # Build a slot-name → set[norm_token] map for sibling detection.
    slot_to_norm: dict[str, frozenset[str]] = {}
    for lnk in links:
        sn = lnk.slot_name
        if sn not in slot_to_norm:
            slot_to_norm[sn] = frozenset(lnk.slot_feats.norm_tokens)

    # Pre-compute clause spans once (used by the clause-entity tiebreaker).
    clauses = split_into_clauses(query) if query else []

    # Track which slots have already been swapped in this pass to avoid
    # double-swapping the same pair.
    swapped: set[str] = set()

    for i in range(len(filled_slot_names)):
        sn_i = filled_slot_names[i]
        if sn_i in swapped or sn_i not in new_mentions:
            continue

        for j in range(i + 1, len(filled_slot_names)):
            sn_j = filled_slot_names[j]
            if sn_j in swapped or sn_j not in new_mentions:
                continue

            norm_i = slot_to_norm.get(sn_i, frozenset())
            norm_j = slot_to_norm.get(sn_j, frozenset())

            # Sibling check: same measure tokens, different entity tokens.
            shared = norm_i & norm_j
            distinct = (norm_i | norm_j) - shared
            if not shared or not distinct:
                continue

            m_i = new_mentions[sn_i]
            m_j = new_mentions[sn_j]

            # Find the four relevant links.
            lnk_ii = _find_link(links, m_i.mention_id, sn_i)
            lnk_jj = _find_link(links, m_j.mention_id, sn_j)
            lnk_ij = _find_link(links, m_i.mention_id, sn_j)
            lnk_ji = _find_link(links, m_j.mention_id, sn_i)

            if not all([lnk_ii, lnk_jj, lnk_ij, lnk_ji]):
                continue  # incomplete link coverage; skip

            current_anchor = (
                lnk_ii.left_entity_anchor_overlap  # type: ignore[union-attr]
                + lnk_jj.left_entity_anchor_overlap  # type: ignore[union-attr]
            )
            swapped_anchor = (
                lnk_ij.left_entity_anchor_overlap  # type: ignore[union-attr]
                + lnk_ji.left_entity_anchor_overlap  # type: ignore[union-attr]
            )

            should_swap = swapped_anchor > current_anchor

            # Clause-entity tiebreaker: when anchor scores are tied, check
            # whether swapping improves clause-entity consistency.  Only
            # active when a query string is available (conservative guard).
            if not should_swap and swapped_anchor == current_anchor and clauses:
                consistency_gain = _clause_entity_consistency_gain(
                    sn_i, sn_j, m_i, m_j, links,
                    list(new_mentions.values()), clauses,
                )
                if consistency_gain > 0:
                    should_swap = True

            # Narrow-measure-overlap tiebreaker: when both anchor and
            # clause-entity scores are tied, check whether swapping improves
            # the combined narrow_measure_overlap for the two-slot pair.
            # This fires when the greedy placed a mention in a slot whose
            # measure tokens have LESS local support than the alternative
            # (e.g. "5 labor hours and earns … profit" where '5' scores
            # higher for ProfitB due to the 'earns' role tag, but its narrow
            # context overlaps more strongly with LaborB).
            # Conservative guard: only fires on an exact anchor tie AND when
            # the nm_gain is strictly positive.
            if not should_swap and swapped_anchor == current_anchor:
                current_nm = (
                    lnk_ii.narrow_measure_overlap  # type: ignore[union-attr]
                    + lnk_jj.narrow_measure_overlap  # type: ignore[union-attr]
                )
                swapped_nm = (
                    lnk_ij.narrow_measure_overlap  # type: ignore[union-attr]
                    + lnk_ji.narrow_measure_overlap  # type: ignore[union-attr]
                )
                if swapped_nm > current_nm:
                    should_swap = True

            if should_swap:
                # Capture original values before modifying so the log is accurate.
                orig_val_i = new_values.get(sn_i)
                orig_val_j = new_values.get(sn_j)
                # Perform the swap.
                new_mentions[sn_i] = m_j
                new_mentions[sn_j] = m_i
                new_values[sn_i] = orig_val_j
                new_values[sn_j] = orig_val_i
                swapped.add(sn_i)
                swapped.add(sn_j)
                swap_log.append(
                    f"swap: {sn_i}({orig_val_i}→{orig_val_j})"
                    f" ↔ {sn_j}({orig_val_j}→{orig_val_i})"
                    f" [anchor {current_anchor}→{swapped_anchor}]"
                )
                break  # move to next i after a swap

    # -----------------------------------------------------------------------
    # Derived-occupant rescue pass.
    # After the swap pass, look for slots occupied by low-quality "derived"
    # mentions (empty narrow_left, nm_ov = 0) when there is an unassigned
    # non-derived mention that has actual narrow-measure evidence for that
    # slot.  Such situations arise when the greedy crowded out the real
    # mention because a different mention greedily took the best slot first.
    #
    # Why opt-in via `all_mentions`:
    #   The rescue pass needs the COMPLETE pool of extracted mentions to find
    #   unassigned candidates.  The greedy only passes back the assigned subset
    #   (filled_mentions), so the complete list must be supplied separately.
    #   Callers that do not supply all_mentions (e.g. unit tests that manually
    #   build partial states) are unaffected.
    #
    # What qualifies as "derived":
    #   A mention whose raw_surface starts with "derived:" — these are synthetic
    #   counts extracted by the enumeration-cue detector (e.g. "derived:2
    #   (hours and earns)").  They are valid when no real numeric token is
    #   available but are poor-quality placeholders when a real mention exists.
    #
    # Why nm_ov = 0 as the threshold:
    #   A derived mention with zero narrow-measure overlap means its local
    #   context provides no measure-word evidence for the slot it occupies.
    #   This is a reliable indicator that it won the slot only by default
    #   (all stronger candidates were already taken).  We only replace it when
    #   a strictly better (nm_ov > 0) real candidate exists, keeping this
    #   pass conservative.
    # -----------------------------------------------------------------------
    if all_mentions is not None:
        assigned_mids: set[int] = {m.mention_id for m in new_mentions.values()}
        unassigned = [m for m in all_mentions if m.mention_id not in assigned_mids]

        if unassigned:
            rescue_log: list[str] = []
            for sn, occupant in list(new_mentions.items()):
                # Only consider slots where the current occupant is derived
                # and has zero narrow-measure overlap (no local evidence).
                occ_link = _find_link(links, occupant.mention_id, sn)
                if occ_link is None:
                    continue
                occupant_nm = occ_link.narrow_measure_overlap
                # "derived" mentions have a raw_surface prefixed "derived:".
                if not (
                    occupant.raw_surface.startswith("derived:")
                    and occupant_nm == 0
                ):
                    continue

                # Find the best unassigned replacement (highest nm_ov > 0).
                best_nm, best_mention, best_lnk = 0, None, None
                for cand in unassigned:
                    cand_lnk = _find_link(links, cand.mention_id, sn)
                    if cand_lnk is not None and cand_lnk.narrow_measure_overlap > best_nm:
                        best_nm = cand_lnk.narrow_measure_overlap
                        best_mention = cand
                        best_lnk = cand_lnk

                if best_mention is not None and best_nm > 0:
                    # Replace the derived occupant with the real candidate.
                    old_val = new_values.get(sn)
                    new_mentions[sn] = best_mention
                    new_values[sn] = best_mention.value
                    unassigned.remove(best_mention)
                    assigned_mids.add(best_mention.mention_id)
                    rescue_log.append(
                        f"rescue: {sn}({old_val}→{best_mention.value})"
                        f" [derived→m{best_mention.mention_id} nm_ov={best_nm}]"
                    )
            if rescue_log:
                swap_log.extend(rescue_log)

    return new_values, new_mentions, swap_log
