"""
Shared retrieval utilities.

Keeping these separate from ``retrieval.search`` prevents a dependency cycle:
``retrieval.baselines`` needs ``expand_short_query`` but must not import from
``retrieval.search`` (which may in future import from ``retrieval.baselines``).
"""
from __future__ import annotations

# ── Short-query expansion ──────────────────────────────────────────────────────
# Catalog passages are typically 50–300 words long (name + aliases + description).
# A bare keyword such as "knapsack" or "TSP ILP" shares very few tokens with
# those passages, so its cosine similarity is low even when it is a perfect match.
# Appending a small fixed domain-context suffix adds shared vocabulary and
# significantly closes the embedding-space gap.  Only queries of ≤ 5 words are
# expanded; longer natural-language queries already contain enough context.

_EXPANSION_SUFFIX = "optimization problem formulation"
_SHORT_QUERY_WORD_LIMIT = 5


def _is_short_query(query: str) -> bool:
    """Return True when *query* looks like a bare keyword search (≤ 5 words).

    Leading and trailing whitespace is stripped before counting words, so
    ``"  knapsack  "`` is treated as a 1-word query.
    """
    return len(query.strip().split()) <= _SHORT_QUERY_WORD_LIMIT


def expand_short_query(query: str) -> str:
    """Return *query* padded with optimization domain context if it is short.

    Short keyword queries (≤ 5 words) such as ``"knapsack"`` or ``"TSP ILP"``
    embed far from long catalog passages in cosine space because they share few
    tokens with terms like *optimization*, *problem*, *formulation* that appear
    in every passage.  Appending the fixed suffix
    ``"optimization problem formulation"`` adds those shared tokens and
    consistently improves recall for short queries across all retrieval methods
    (BM25, TF-IDF, LSA, SBERT) — without requiring a model retrain.

    Queries of ≥ 6 words are returned unchanged: they already contain enough
    context to embed well on their own.

    Examples::

        >>> expand_short_query("knapsack")
        'knapsack optimization problem formulation'
        >>> expand_short_query("TSP ILP")
        'TSP ILP optimization problem formulation'
        >>> expand_short_query("minimize cost of opening warehouses and assigning customers")
        'minimize cost of opening warehouses and assigning customers'
        >>> expand_short_query("")   # empty / whitespace → returned as empty string
        ''
    """
    stripped = query.strip()
    if not stripped or not _is_short_query(stripped):
        return stripped
    return f"{stripped} {_EXPANSION_SUFFIX}"

