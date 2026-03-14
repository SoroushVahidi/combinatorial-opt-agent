"""
Shared retrieval utilities.

Keeping these separate from ``retrieval.search`` prevents a dependency cycle:
``retrieval.baselines`` needs ``expand_short_query`` but must not import from
``retrieval.search`` (which may in future import from ``retrieval.baselines``).
"""
from __future__ import annotations

import re

# ── Short-query expansion ──────────────────────────────────────────────────────
# Catalog passages are typically 50–300 words long (name + aliases + description).
# A bare keyword such as "knapsack" or "TSP ILP" shares very few tokens with
# those passages, so its cosine similarity is low even when it is a perfect match.
# Appending a small fixed domain-context suffix adds shared vocabulary and
# significantly closes the embedding-space gap.  Only queries of ≤ 5 words are
# expanded; longer natural-language queries already contain enough context.

_EXPANSION_SUFFIX = "optimization problem formulation"
_SHORT_QUERY_WORD_LIMIT = 5

# Domain-specific expansion map: if any key token is found in the query,
# append the corresponding domain context phrase.  Checked before the generic
# suffix so that known-domain queries get richer context.  Long-form queries
# (> _SHORT_QUERY_WORD_LIMIT words) bypass this entirely.
_DOMAIN_EXPANSION_MAP: list[tuple[frozenset[str], str]] = [
    (
        frozenset({"diet", "blend", "blending", "mix", "mixture", "alloy", "ingredient",
                   "nutrient", "feed", "recipe", "composition"}),
        "blending mix mixture ingredient nutrient composition percentage optimization",
    ),
    (
        frozenset({"transport", "transportation", "ship", "shipping", "route",
                   "routing", "delivery", "logistics"}),
        "transportation shipping route supply demand source destination network flow",
    ),
    (
        frozenset({"assign", "assignment", "worker", "task", "job", "employee",
                   "schedule", "scheduling", "shift"}),
        "assignment scheduling worker task job bipartite matching optimization",
    ),
    (
        frozenset({"facility", "location", "depot", "warehouse", "open", "opening",
                   "customer", "median", "hub"}),
        "facility location opening depot warehouse customer serving optimization",
    ),
    (
        frozenset({"knapsack", "pack", "packing", "item", "weight", "capacity"}),
        "knapsack packing items weight capacity maximize value binary optimization",
    ),
    (
        frozenset({"vehicle", "vrp", "tour", "visit"}),
        "vehicle routing tour depot visit capacitated optimization",
    ),
    (
        frozenset({"cover", "covering", "set cover", "dominate"}),
        "set cover covering subsets elements minimum optimization",
    ),
    (
        frozenset({"graph", "vertex", "edge", "node", "clique", "coloring",
                   "chromatic", "independent"}),
        "graph vertex edge node integer programming optimization",
    ),
    (
        frozenset({"production", "produce", "product", "manufacture", "manufacturing",
                   "factory", "output", "profit"}),
        "production planning profit labor machine manufacturing optimization",
    ),
    (
        frozenset({"tsp", "salesman", "traveling", "travelling", "tour"}),
        "traveling salesman problem TSP tour shortest circuit optimization",
    ),
    (
        frozenset({"flow", "network", "path", "shortest"}),
        "network flow shortest path minimum cost arc optimization",
    ),
    (
        frozenset({"bin", "bins", "cutting", "stock", "container"}),
        "bin packing cutting stock container size optimization",
    ),
    # ── LP / MIP / QP formulation families ───────────────────────────────────
    (
        frozenset({"lp", "linear", "ilp", "mip", "milp", "integer",
                   "formulate", "formulation"}),
        "linear program integer programming formulation optimization variables constraints",
    ),
    (
        frozenset({"qp", "quadratic", "socp", "convex", "nonlinear", "nlp"}),
        "quadratic convex nonlinear programming optimization objective constraints",
    ),
    # ── Portfolio / finance ───────────────────────────────────────────────────
    (
        frozenset({"portfolio", "invest", "investment", "stock", "asset",
                   "return", "risk", "markowitz", "sharpe", "frontier"}),
        "portfolio optimization investment return risk asset allocation Markowitz",
    ),
    # ── Matching / bipartite ──────────────────────────────────────────────────
    (
        frozenset({"match", "matching", "bipartite", "marriage", "stable",
                   "allocation", "allocate"}),
        "matching bipartite assignment allocation worker job optimization",
    ),
    # ── Resource / capacity planning ─────────────────────────────────────────
    (
        frozenset({"resource", "capacity", "planning", "demand", "supply",
                   "inventory", "stock", "lot", "sizing"}),
        "resource capacity planning demand supply inventory lot sizing optimization",
    ),
    # ── Cutting / layout / strip packing ─────────────────────────────────────
    (
        frozenset({"cut", "trim", "strip", "layout", "nesting", "2d", "3d",
                   "rectangle", "sheet"}),
        "cutting layout strip packing rectangle nesting optimization",
    ),
]


def _is_short_query(query: str) -> bool:
    """Return True when *query* looks like a bare keyword search (≤ 5 words).

    Leading and trailing whitespace is stripped before counting words, so
    ``"  knapsack  "`` is treated as a 1-word query.
    """
    return len(query.strip().split()) <= _SHORT_QUERY_WORD_LIMIT


def _domain_expansion(query_lower: str) -> str | None:
    """Return a domain-specific expansion phrase if a known domain is detected.

    Returns the *first* matching expansion phrase, or *None* if no domain
    keyword is found in *query_lower*.
    """
    tokens = set(re.sub(r"[^a-z0-9]", " ", query_lower).split())
    for domain_keys, expansion in _DOMAIN_EXPANSION_MAP:
        if tokens & domain_keys:
            return expansion
    return None


def expand_short_query(query: str) -> str:
    """Return *query* padded with optimization domain context if it is short.

    Short keyword queries (≤ 5 words) such as ``"knapsack"`` or ``"TSP ILP"``
    embed far from long catalog passages in cosine space because they share few
    tokens with terms like *optimization*, *problem*, *formulation* that appear
    in every passage.  For known domains (diet, transport, assignment, …), a
    richer domain-specific phrase is appended instead of the generic suffix.
    For unknown-domain short queries, the generic suffix
    ``"optimization problem formulation"`` is used as a fallback.

    Queries of ≥ 6 words are returned unchanged: they already contain enough
    context to embed well on their own.

    Examples::

        >>> expand_short_query("knapsack")  # triggers knapsack domain expansion
        'knapsack knapsack packing items weight capacity maximize value binary optimization'
        >>> expand_short_query("TSP ILP")  # triggers traveling-salesman expansion
        'TSP ILP traveling salesman problem TSP tour shortest circuit optimization'
        >>> expand_short_query("diet protein fat cost")  # triggers blending domain expansion
        'diet protein fat cost blending mix mixture ingredient nutrient composition percentage optimization'
        >>> expand_short_query("ILP formulation")  # unknown domain → generic suffix
        'ILP formulation optimization problem formulation'
        >>> expand_short_query("minimize cost of opening warehouses and assigning customers")
        'minimize cost of opening warehouses and assigning customers'
        >>> expand_short_query("")   # empty / whitespace → returned as empty string
        ''
    """
    stripped = query.strip()
    if not stripped or not _is_short_query(stripped):
        return stripped
    domain_ctx = _domain_expansion(stripped.lower())
    suffix = domain_ctx if domain_ctx is not None else _EXPANSION_SUFFIX
    return f"{stripped} {suffix}"

