"""
Deterministic lexical reranking for schema retrieval (Error Family: wrong schema / retrieval failure).

After first-stage retrieval (BM25 / TF-IDF / SBERT), this module reranks the top-k
candidates using schema-aware lexical features:
  - alias/trigger-phrase overlap
  - slot-name and role-vocabulary overlap
  - quantity-role cue matching (budget, capacity, per-unit, min/max, percent, etc.)
  - optional grounding-consistency second-stage rerank

All logic is deterministic, CPU-only, no external dependencies beyond the standard library.

Toggle reranking via the ``rerank`` flag in ``retrieval.search.search()``.
Toggle ambiguity reporting via ``report_ambiguity=True``.
Toggle grounding-consistency rerank via ``grounding_rerank=True`` in ``retrieval.search.search()``.
"""
from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Quantity-role cue word sets used in scoring
# ---------------------------------------------------------------------------

# Words that signal capacity / budget constraints
_CAPACITY_CUES = frozenset({
    "capacity", "budget", "limit", "availability", "available", "stock", "supply",
    "resource", "quota", "demand", "inventory",
    # network / flow capacity
    "bandwidth", "throughput",
    # discrete resource slots
    "seat", "slot", "slots",
})

# Words that signal per-unit / rate / profit / cost
_PER_UNIT_CUES = frozenset({
    "profit", "cost", "price", "revenue", "rate", "per", "unit", "margin",
    "value", "weight", "coefficient",
    # finance / returns
    "return", "gain", "yield",
    # labour / penalty
    "penalty", "wage", "salary",
})

# Words that signal min/max / bound
_BOUND_CUES = frozenset({
    "min", "max", "minimum", "maximum", "lower", "upper", "bound", "threshold",
    "floor", "ceiling", "atleast", "atmost",
    # natural-language bound phrases: "at least", "at most", "no more"
    "least", "most",
})

# Words that signal percent / share / fraction
_PERCENT_CUES = frozenset({
    "percent", "percentage", "fraction", "share", "ratio", "proportion", "rate",
})

# All role cues combined
_ALL_ROLE_CUES = _CAPACITY_CUES | _PER_UNIT_CUES | _BOUND_CUES | _PERCENT_CUES

# ---------------------------------------------------------------------------
# Domain-specific alias/trigger maps for confusable schema families
# These expand the lexical surface of each schema family beyond the catalog text.
# ---------------------------------------------------------------------------

_DOMAIN_TRIGGER_MAP: dict[str, set[str]] = {
    # Blending / diet / feed mix
    "blending": {
        "blend", "blending", "mix", "mixture", "mixing", "ingredient", "nutrient",
        "composition", "percentage", "diet", "feed", "alloy", "recipe", "formula",
        "proportion", "ratio",
    },
    # Production planning / manufacturing
    "production": {
        "produce", "product", "production", "manufacture", "manufacturing", "profit",
        "labor", "labour", "machine", "process", "processing", "output", "unit",
        "factory", "plant", "assemble", "assembly",
    },
    # Transportation / shipping / logistics
    "transportation": {
        "transport", "transportation", "ship", "shipping", "route", "routing",
        "source", "destination", "supply", "demand", "network", "flow", "arc",
        "node", "origin", "delivery", "logistics", "carrier", "warehouse",
    },
    # Assignment / scheduling
    "assignment": {
        "assign", "assignment", "worker", "task", "job", "employee", "schedule",
        "scheduling", "one-to-one", "bipartite", "matching", "shift",
    },
    # Facility location
    "facility": {
        "facility", "open", "opening", "location", "depot", "warehouse", "customer",
        "serve", "serving", "fixed", "charge",
    },
    # Knapsack / packing
    "knapsack": {
        "knapsack", "pack", "packing", "weight", "capacity", "select", "selection",
        "binary", "0-1", "0/1", "item",
    },
    # Vehicle routing
    "routing": {
        "vehicle", "route", "routing", "tour", "depot", "visit", "vrp",
        "capacitated", "delivery",
    },
    # Covering / set cover
    "covering": {
        "cover", "covering", "set", "element", "subset", "universal", "dominate",
    },
    # Graph problems
    "graph": {
        "graph", "vertex", "edge", "node", "cut", "clique", "matching", "coloring",
        "chromatic", "independent", "path", "cycle", "tree",
    },
    # Scheduling (job shop, flow shop)
    "scheduling": {
        "job", "machine", "schedule", "scheduling", "makespan", "processing",
        "flow", "shop", "sequence", "precedence", "release", "deadline",
    },
    # Integer / binary programming
    "integer": {
        "integer", "binary", "ilp", "mip", "mixed", "integrality",
    },
    # Shortest path / network flow
    "network_flow": {
        "path", "shortest", "flow", "network", "source", "sink", "arc",
        "minimum", "cost", "maximum",
    },
    # Location / P-median / hub
    "location": {
        "median", "center", "hub", "location", "cluster", "p-median", "p-center",
    },
    # Bin packing / cutting stock
    "bin_packing": {
        "bin", "bins", "pack", "packing", "cutting", "stock", "container",
        "strip", "size",
    },
}


# ---------------------------------------------------------------------------
# Confusable-schema discrimination support (Feature 5)
# Maps each schema domain-key to positive discriminative cues that strongly
# distinguish it from its most likely confusable neighbour(s).
# Structure: {domain_key: {"positive": frozenset, "negative": frozenset}}
# positive = cues that should appear for THIS schema
# negative = cues more characteristic of the confusable neighbour
# Only well-justified pairs are listed here.
# ---------------------------------------------------------------------------

_CONFUSABLE_DISCRIMINATION: dict[str, dict[str, frozenset]] = {
    # blending vs production: blending mixes ingredients (proportions), production makes products
    "blending": {
        "positive": frozenset({"blend", "mix", "ingredient", "nutrient", "proportion",
                                "composition", "percentage", "fraction", "diet", "recipe"}),
        "negative": frozenset({"produce", "product", "machine", "labor", "assembly"}),
    },
    "production": {
        "positive": frozenset({"produce", "product", "manufacturing", "profit", "labor",
                                "machine", "assembly", "output", "factory"}),
        "negative": frozenset({"blend", "ingredient", "composition", "nutrient"}),
    },
    # transportation vs assignment: transportation ships goods; assignment matches entities
    "transportation": {
        "positive": frozenset({"ship", "route", "supply", "demand", "source", "destination",
                                "transport", "flow", "arc", "network"}),
        "negative": frozenset({"assign", "worker", "task", "matching", "bipartite"}),
    },
    "assignment": {
        "positive": frozenset({"assign", "worker", "task", "employee", "job", "matching",
                                "bipartite", "one-to-one"}),
        "negative": frozenset({"ship", "route", "supply", "demand", "flow", "network"}),
    },
    # knapsack vs bin_packing: knapsack selects items; bin packing fits items into bins
    "knapsack": {
        "positive": frozenset({"select", "item", "value", "weight", "capacity",
                                "binary", "maximize"}),
        "negative": frozenset({"bin", "bins", "container", "pack", "size"}),
    },
    "bin_packing": {
        "positive": frozenset({"bin", "bins", "container", "strip", "cutting", "stock"}),
        "negative": frozenset({"select", "item", "value", "maximize"}),
    },
    # facility vs location: facility opening is about fixed charge; location is about distances
    "facility": {
        "positive": frozenset({"open", "opening", "fixed", "charge", "depot",
                                "customer", "serve", "warehouse"}),
        "negative": frozenset({"median", "distance", "closest", "nearest"}),
    },
    "location": {
        "positive": frozenset({"median", "center", "distance", "closest", "nearest",
                                "hub", "cluster"}),
        "negative": frozenset({"open", "opening", "fixed", "charge"}),
    },
    # routing (VRP) vs transportation (TP): routing visits customers on tours;
    # transportation ships bulk goods between supply/demand nodes
    "routing": {
        "positive": frozenset({"vehicle", "tour", "depot", "vrp", "capacitated",
                                "visit", "circuit", "cvrp"}),
        "negative": frozenset({"supply", "demand", "ship", "shipping",
                                "source", "destination"}),
    },
    "transportation": {
        "positive": frozenset({"supply", "demand", "ship", "shipping",
                                "source", "destination", "distribution"}),
        "negative": frozenset({"vehicle", "tour", "depot", "vrp",
                                "visit", "circuit"}),
    },
    # scheduling (job shop / flow shop) vs production planning:
    # scheduling minimises makespan / tardiness; production maximises profit / meets demand
    "scheduling": {
        "positive": frozenset({"machine", "makespan", "deadline", "tardiness",
                                "sequence", "flowshop", "jobshop", "release",
                                "preemption", "processing"}),
        "negative": frozenset({"product", "manufacturing", "profit", "assembly",
                                "labor", "labour"}),
    },
    "production": {
        "positive": frozenset({"product", "manufacturing", "profit", "assembly",
                                "labor", "labour", "factory", "output"}),
        "negative": frozenset({"makespan", "deadline", "tardiness", "sequence",
                                "flowshop", "jobshop"}),
    },
    # network_flow (min-cost flow / shortest path) vs transportation (TP):
    # network_flow is path/arc-based; transportation is supply/demand-based
    "network_flow": {
        "positive": frozenset({"path", "shortest", "sink", "arc", "maximum",
                                "minimum", "cut", "augmenting"}),
        "negative": frozenset({"supply", "demand", "ship", "route",
                                "distribution"}),
    },
    # covering (set cover / vertex cover) vs graph (coloring / clique / independent set):
    # covering selects subsets; general graph problems deal with vertex/edge properties
    "covering": {
        "positive": frozenset({"set", "element", "subset", "universe",
                                "cover", "dominate"}),
        "negative": frozenset({"color", "coloring", "chromatic", "clique",
                                "independent"}),
    },
    "graph": {
        "positive": frozenset({"color", "coloring", "chromatic", "clique",
                                "independent", "bipartite", "cycle", "tree"}),
        "negative": frozenset({"subset", "universe", "element", "cover"}),
    },
}


def _confusable_discrimination_score(
    query_tokens: set[str],
    problem: dict,
) -> float:
    """Return a small discrimination bonus/penalty based on confusable-schema cues.

    For schemas that commonly get confused with a neighbour (e.g., blending vs
    production), positive query cues boost the score and negative cues reduce it.
    Score is in [-0.1, +0.1]; added to the reranker total.
    """
    tags = problem.get("tags") or problem.get("categories") or []
    if isinstance(tags, str):
        tags = [tags]
    domain_keys = {(t or "").lower() for t in tags}
    # Also heuristically match by schema name
    name_lower = (problem.get("name") or "").lower()
    for dk in _CONFUSABLE_DISCRIMINATION:
        if dk in name_lower:
            domain_keys.add(dk)

    score = 0.0
    for dk in domain_keys:
        entry = _CONFUSABLE_DISCRIMINATION.get(dk)
        if entry is None:
            continue
        pos_hits = len(query_tokens & entry["positive"])
        neg_hits = len(query_tokens & entry["negative"])
        # +0.02 per positive hit, -0.02 per negative hit
        score += 0.02 * pos_hits - 0.02 * neg_hits
    return max(-0.1, min(0.1, score))


# ---------------------------------------------------------------------------
# Grounding-consistency scoring (Feature 6 — optional second-stage rerank)
# ---------------------------------------------------------------------------

# Quantity-role cue families for slot compatibility checks
_QTY_BUDGET_CUES = frozenset({
    "budget", "cost", "price", "revenue", "profit", "expenditure", "wage", "salary",
    "expense", "payment",
})
_QTY_CAPACITY_CUES = frozenset({
    "capacity", "supply", "availability", "available", "stock", "inventory",
    "resource", "quota", "limit", "limitation",
    # network / flow capacity
    "bandwidth", "throughput",
    # discrete resource slots
    "seat", "slot",
})
_QTY_BOUND_CUES = frozenset({
    "minimum", "maximum", "min", "max", "least", "most", "lower", "upper",
    "bound", "threshold", "floor", "ceiling",
})
_QTY_PERCENT_CUES = frozenset({
    "percent", "percentage", "fraction", "share", "ratio", "proportion", "rate",
    "portion",
})
_QTY_COUNT_CUES = frozenset({
    "number", "count", "quantity", "amount", "units", "items", "types",
    "workers", "employees", "products",
})


def grounding_consistency_score(query: str, problem: dict) -> float:
    """Estimate how well the query's quantity-role cues align with the schema's slots.

    Checks which quantity-role cue families (budget, capacity, bound, percent,
    count) are signalled by the query and whether the schema's slot vocabulary
    also contains those families.  Returns a score in [0, 1].

    This is a lightweight heuristic used as an optional second-stage boost.
    A score of 0.5 means no quantity-role signal was found in the query
    (neutral — no evidence either way).
    """
    q_tokens = _tokenize(query)
    q_budget = bool(q_tokens & _QTY_BUDGET_CUES)
    q_capacity = bool(q_tokens & _QTY_CAPACITY_CUES)
    q_bound = bool(q_tokens & _QTY_BOUND_CUES)
    q_percent = bool(q_tokens & _QTY_PERCENT_CUES)
    q_count = bool(q_tokens & _QTY_COUNT_CUES)

    slot_vocab = _extract_slot_vocabulary(problem)
    s_budget = bool(slot_vocab & _QTY_BUDGET_CUES)
    s_capacity = bool(slot_vocab & _QTY_CAPACITY_CUES)
    s_bound = bool(slot_vocab & _QTY_BOUND_CUES)
    s_percent = bool(slot_vocab & _QTY_PERCENT_CUES)
    s_count = bool(slot_vocab & _QTY_COUNT_CUES)

    families_in_query = int(q_budget) + int(q_capacity) + int(q_bound) + int(q_percent) + int(q_count)
    if families_in_query == 0:
        return 0.5  # no signal → neutral

    matches = (
        int(q_budget and s_budget)
        + int(q_capacity and s_capacity)
        + int(q_bound and s_bound)
        + int(q_percent and s_percent)
        + int(q_count and s_count)
    )
    return matches / families_in_query


def grounding_rerank(
    query: str,
    candidates: list[tuple[dict, float]],
    grounding_lambda: float = 0.15,
    verbose: bool = False,
) -> list[tuple[dict, float]]:
    """Optional second-stage rerank using grounding-consistency scores.

    Combined score::

        combined = retrieval_score + grounding_lambda * grounding_consistency_score

    Applied AFTER first-stage retrieval (and optionally after lexical reranking).
    Toggle via ``grounding_rerank=True`` in ``retrieval.search.search()``.

    Args:
        query: the original search query.
        candidates: list of (problem_dict, score) sorted by retrieval/rerank score.
        grounding_lambda: weight for the grounding-consistency term (default 0.15).
        verbose: if True, print per-candidate grounding scores.

    Returns:
        Re-sorted list of (problem_dict, combined_score).
    """
    if not candidates:
        return candidates
    result: list[tuple[dict, float]] = []
    for problem, retrieval_score in candidates:
        gc = grounding_consistency_score(query, problem)
        combined = retrieval_score + grounding_lambda * gc
        if verbose:
            print(
                f"  [{problem.get('id', '')[:30]}] "
                f"grounding_consistency={gc:.3f} retrieval={retrieval_score:.3f} "
                f"combined={combined:.3f}"
            )
        result.append((problem, combined))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase and split *text* into word tokens, stripping punctuation."""
    return set(re.sub(r"[^a-z0-9]", " ", (text or "").lower()).split())


def _tokenize_seq(text: str) -> list[str]:
    """Return ordered token list (for substring / phrase matching)."""
    return re.sub(r"[^a-z0-9]", " ", (text or "").lower()).split()


# ---------------------------------------------------------------------------
# Per-schema slot vocabulary extraction
# ---------------------------------------------------------------------------

def _extract_slot_vocabulary(problem: dict) -> set[str]:
    """Return a bag of tokens from variable descriptions and constraint descriptions.

    These tokens provide additional retrieval surface: e.g. a transportation
    schema will have variable descriptions like "amount shipped from i to j"
    which contains 'shipped', 'amount', 'route' etc.
    """
    tokens: set[str] = set()
    form = problem.get("formulation") or {}
    for var in (form.get("variables") or []):
        tokens |= _tokenize(var.get("description", ""))
        tokens |= _tokenize(var.get("symbol", ""))
    for c in (form.get("constraints") or []):
        tokens |= _tokenize(c.get("description", ""))
    obj = form.get("objective") or {}
    if isinstance(obj, dict):
        tokens |= _tokenize(obj.get("expression", ""))
    return tokens


def _schema_tokens(problem: dict) -> set[str]:
    """Full token set for a schema: name + aliases + description + slot vocab."""
    tokens = _tokenize(problem.get("name", ""))
    for alias in (problem.get("aliases") or []):
        tokens |= _tokenize(alias)
    tokens |= _tokenize(problem.get("description", ""))
    tokens |= _extract_slot_vocabulary(problem)
    # Add domain trigger words for domains this schema is tagged under (if any)
    tags = problem.get("tags") or problem.get("categories") or []
    if isinstance(tags, list):
        for tag in tags:
            domain = (tag or "").lower()
            tokens |= _DOMAIN_TRIGGER_MAP.get(domain, set())
    return tokens


# ---------------------------------------------------------------------------
# Slot-name overlap score
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "to", "for", "is", "are", "be", "and", "or",
    "that", "this", "with", "on", "at", "by", "from", "it", "as", "no", "not",
    "each", "every", "all", "any", "its", "their", "our", "per", "if", "so",
    "but", "can", "may", "must", "will", "than", "more", "less", "such",
})


def _meaningful_tokens(tokens: set[str]) -> set[str]:
    return tokens - _STOPWORDS - {""}


# ---------------------------------------------------------------------------
# Reranking score computation
# ---------------------------------------------------------------------------

class RerankerFeatures(NamedTuple):
    """Feature breakdown for a single (query, schema) pair (for debugging/ablation)."""
    alias_overlap: float
    slot_overlap: float
    role_cue_overlap: float
    domain_trigger_overlap: float
    total: float


def _rerank_score(
    query_tokens: set[str],
    problem: dict,
    schema_tokens: set[str] | None = None,
) -> RerankerFeatures:
    """Compute deterministic reranking score for a single (query, schema) pair.

    All features are normalised overlap scores in [0, 1].
    Final score is a weighted sum used to re-sort candidates.

    Args:
        query_tokens: pre-tokenised query token set.
        problem: catalog entry dict.
        schema_tokens: pre-computed token set for this schema (optional, avoids recompute).

    Returns:
        RerankerFeatures named tuple with per-feature scores and total.
    """
    if schema_tokens is None:
        schema_tokens = _schema_tokens(problem)

    q = _meaningful_tokens(query_tokens)
    s = _meaningful_tokens(schema_tokens)

    if not q:
        return RerankerFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

    # --- Alias overlap ---
    # Query tokens that match any alias of the schema.
    alias_tokens: set[str] = set()
    for alias in (problem.get("aliases") or []):
        alias_tokens |= _meaningful_tokens(_tokenize(alias))
    alias_overlap = len(q & alias_tokens) / len(q) if q else 0.0

    # --- Slot vocabulary overlap ---
    slot_tokens = _meaningful_tokens(_extract_slot_vocabulary(problem))
    slot_overlap = len(q & slot_tokens) / len(q) if q else 0.0

    # --- Role-cue overlap ---
    # Rewards queries that mention role-specific vocabulary (capacity, profit, etc.)
    q_role = q & _ALL_ROLE_CUES
    s_role = s & _ALL_ROLE_CUES
    if q_role:
        role_cue_overlap = len(q_role & s_role) / len(q_role)
    else:
        role_cue_overlap = 0.0

    # --- Domain trigger overlap ---
    # Sum triggered by mapping query tokens to domain families, then checking
    # whether the schema itself belongs to that family.
    schema_all_tokens = s
    triggered_domains: set[str] = set()
    for domain, triggers in _DOMAIN_TRIGGER_MAP.items():
        if q & triggers:
            triggered_domains.add(domain)
    if triggered_domains:
        # Check how many triggered-domain words appear in the schema
        triggered_words: set[str] = set()
        for d in triggered_domains:
            triggered_words |= _DOMAIN_TRIGGER_MAP[d]
        domain_trigger_overlap = min(
            1.0,
            len(schema_all_tokens & triggered_words) / max(1, len(triggered_words)),
        )
    else:
        domain_trigger_overlap = 0.0

    # --- Confusable-schema discrimination bonus/penalty ---
    discrimination = _confusable_discrimination_score(q, problem)

    # Weighted sum (weights chosen to make alias and slot most influential)
    total = (
        0.35 * alias_overlap
        + 0.25 * slot_overlap
        + 0.20 * role_cue_overlap
        + 0.20 * domain_trigger_overlap
        + discrimination  # small additive term in [-0.1, +0.1]
    )
    total = max(0.0, min(1.1, total))  # clamp to sensible range
    return RerankerFeatures(
        alias_overlap=alias_overlap,
        slot_overlap=slot_overlap,
        role_cue_overlap=role_cue_overlap,
        domain_trigger_overlap=domain_trigger_overlap,
        total=total,
    )


# ---------------------------------------------------------------------------
# Top-level reranking function
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[tuple[dict, float]],
    retrieval_weight: float = 0.7,
    rerank_weight: float = 0.3,
    verbose: bool = False,
) -> list[tuple[dict, float]]:
    """Rerank retrieved candidates using deterministic lexical features.

    Combines the original retrieval score with a schema-aware lexical score::

        combined = retrieval_weight * retrieval_score + rerank_weight * rerank_score

    Args:
        query: the original search query string.
        candidates: list of (problem_dict, retrieval_score) from first-stage retrieval.
        retrieval_weight: weight for the original embedding/BM25 score (default 0.7).
        rerank_weight: weight for the lexical reranker score (default 0.3).
        verbose: if True, print per-candidate feature breakdown.

    Returns:
        Re-sorted list of (problem_dict, combined_score).
    """
    if not candidates:
        return candidates

    query_tokens = _tokenize(query)
    # Pre-compute schema token sets once
    schema_token_cache: dict[str, set[str]] = {}
    for problem, _ in candidates:
        pid = problem.get("id", "")
        if pid not in schema_token_cache:
            schema_token_cache[pid] = _schema_tokens(problem)

    reranked: list[tuple[dict, float]] = []
    for problem, retrieval_score in candidates:
        pid = problem.get("id", "")
        features = _rerank_score(query_tokens, problem, schema_token_cache.get(pid))
        combined = retrieval_weight * retrieval_score + rerank_weight * features.total
        if verbose:
            print(
                f"  [{pid[:30]}] rerank={features.total:.3f} "
                f"(alias={features.alias_overlap:.2f} slot={features.slot_overlap:.2f} "
                f"role={features.role_cue_overlap:.2f} domain={features.domain_trigger_overlap:.2f}) "
                f"retrieval={retrieval_score:.3f} combined={combined:.3f}"
            )
        reranked.append((problem, combined))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# Ambiguity / low-margin detection
# ---------------------------------------------------------------------------

class AmbiguityReport(NamedTuple):
    """Ambiguity metadata for a retrieval result set."""
    top_score: float
    second_score: float
    margin: float
    is_ambiguous: bool
    top_id: str
    second_id: str


def detect_ambiguity(
    results: list[tuple[dict, float]],
    ambiguity_threshold: float = 0.05,
) -> AmbiguityReport | None:
    """Report whether top-1 and top-2 results are very close.

    Args:
        results: sorted (problem, score) list from search or rerank.
        ambiguity_threshold: margin below which we consider results ambiguous
            (default 0.05 on a [0,1] cosine scale).

    Returns:
        AmbiguityReport or None if fewer than 2 results.
    """
    if len(results) < 2:
        return None
    top_score = results[0][1]
    second_score = results[1][1]
    margin = top_score - second_score
    return AmbiguityReport(
        top_score=top_score,
        second_score=second_score,
        margin=margin,
        is_ambiguous=margin < ambiguity_threshold,
        top_id=results[0][0].get("id", ""),
        second_id=results[1][0].get("id", ""),
    )
