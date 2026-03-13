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


def detect_and_repair_parallel_swaps(
    filled_values: dict[str, Any],
    filled_mentions: dict[str, MentionOptIR],
    links: list[MentionSlotLink],
    ablation_mode: str = "full",
) -> tuple[dict[str, Any], dict[str, MentionOptIR], list[str]]:
    """Detect and repair parallel sibling-slot swap errors.

    After an initial greedy assignment, this function checks whether any pair
    of assigned slots can be improved by swapping their mentions.  A swap is
    performed when:

    1. Both slots share at least one common ``norm_token`` (same measure family)
       but have at least one differing token (different entity component).
    2. The swap strictly improves the total ``left_entity_anchor_overlap`` for
       the two-slot pair (i.e. the swapped assignment has a higher combined
       left-anchor overlap than the current one).

    This is conservative: only clearly beneficial swaps are performed.  The
    function runs a single pass over all candidate pairs (no iterative
    re-checking) to avoid cascading changes.

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

            if swapped_anchor > current_anchor:
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

    return new_values, new_mentions, swap_log
