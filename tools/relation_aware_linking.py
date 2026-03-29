"""Relation-Aware Mention–Slot Linking for NLP4LP downstream grounding.

This module builds richer mention-slot compatibility representations by computing
three families of features for every (mention, slot) pair:

  A. Mention-side features  — what the mention's context signals
  B. Slot-side features     — what the slot name/type implies
  C. Relation features      — cross-pair compatibility signals

In addition, it builds lightweight relation tables:
  - mention-to-mention relations (for diagnosing duplicate / swap errors)
  - slot-to-slot relations (structural schema priors)

Four ablation scoring modes control which features are active:
  "basic"    — type compatibility + lexical overlap only
  "ops"      — basic + operator/bound cue matching
  "semantic" — ops  + semantic role family + total/per-unit/percent features
  "full"     — semantic + entity anchoring + magnitude plausibility

All scoring is deterministic and CPU-only.  A learned scorer can be plugged in
later by replacing ``relation_aware_local_score`` with a model that accepts the
same feature dicts.

Public API
----------
build_mention_slot_links(query, variant, expected_scalar)
    -> list[MentionSlotLink]

build_mention_mention_relations(mentions)
    -> list[MentionMentionRelation]

build_slot_slot_relations(slots)
    -> list[SlotSlotRelation]

relation_aware_local_score(link, ablation_mode)
    -> (float, dict)

best_assignment_greedy(links, slots, mentions, ablation_mode)
    -> dict[slot_name -> MentionOptIR]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use the optimisation-role IR types and extractors from the downstream utility.
from tools.nlp4lp_downstream_utility import (
    MentionOptIR,
    SlotOptIR,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _is_type_incompatible,
    _normalize_tokens,
    _slot_aliases,
    _slot_measure_tokens,
    _split_camel_case,
    _expected_type,
    _mention_semantic_families,
    _slot_semantic_families,
    _mention_polarity,
    _slot_polarity,
    _mention_style,
    GCG_PRUNE_THRESHOLD,
    OPT_UNIT_PERCENT,
    OPT_UNIT_CURRENCY,
    OPT_UNIT_COUNT,
    OPT_UNIT_TIME,
)

# ---------------------------------------------------------------------------
# Group 3: Left-anchor entity-word filter
# ---------------------------------------------------------------------------

# Measure, unit, and structural words that should be EXCLUDED from the slot
# token set when computing ``left_entity_anchor_overlap``.
#
# Motivation: In patterns like "3 heating hours and **5** cooling hours", the
# left narrow context of 5 contains "heating" and "hours" — contamination from
# the preceding number's measure word.  If slot "HeatingHours" retains "heating"
# in its entity-anchor token set, 5 spuriously receives a left-anchor bonus for
# HeatingHours, causing a wrong assignment.
#
# By filtering these measure/unit words out of the slot tokens before computing
# the overlap, we ensure that only true entity-discriminating tokens (like "a",
# "b", "1", "2", "chair", "dresser", "feed") contribute to the score.
_LEFT_ANCHOR_MEASURE_EXCLUDE: frozenset[str] = frozenset({
    # Nutrients / physical attributes
    "protein", "fat", "fiber", "carb", "carbs", "calorie", "calories",
    # Processing / manufacturing steps
    "heating", "cooling", "manufacturing", "assembly", "finishing", "polishing",
    # Labor / time
    "labor", "labour", "hour", "hours", "time", "day", "days", "minute", "minutes",
    # Materials (measure-like)
    "wood", "stain", "glass", "material", "steel", "plastic", "rubber",
    # Economic quantities
    "cost", "profit", "revenue", "price", "wage", "salary",
    "budget", "demand", "supply", "capacity", "resource",
    # Structural / ratio words
    "per", "total", "rate", "ratio", "count", "number", "amount",
    "min", "max", "minimum", "maximum", "unit", "units", "items",
    # Physical units
    "kg", "lb", "lbs", "ton", "tons",
    # Generic quantity dimensions
    "weight", "volume", "area", "length", "distance", "energy",
})


import re as _re  # noqa: E402 — used by _expand_compound_tokens only

_COMPOUND_SPLIT_RE = _re.compile(r"([a-zA-Z]+)(\d+)$")


def _expand_compound_tokens(tokens: set[str]) -> set[str]:
    """Expand alphanum compound tokens such as 'product2' → {'product2', 'product', '2'}.

    Many entity names in parallel-clause patterns have the form
    ``<word><digit(s)>`` (e.g. 'type1', 'product2', 'machine3').  These
    compound surface forms are not split by ``_normalize_tokens``; they arrive
    in the left-anchor context as a single token.  Without expansion, the
    overlap with slot tokens like 'type' and '1' (from 'Type1') is zero, so
    the left-entity-anchor cannot discriminate 'Type1' from 'Type2'.

    The expansion adds both the alpha prefix and the digit suffix alongside
    the original token, enabling overlap with the split slot norm_tokens
    (e.g. LaborType1 → ['labortype1', 'labor', 'type', '1']).

    Only the pattern ``alpha+ digit+`` is expanded — other tokens are kept
    unchanged.  The original compound form is always retained so that exact
    slot-name matches remain valid.

    Parameters
    ----------
    tokens : set[str]
        Lowercase token set (e.g. from narrow_left_tokens).

    Returns
    -------
    set[str]
        Expanded token set (superset of *tokens*).
    """
    expanded = set(tokens)
    for tok in tokens:
        m = _COMPOUND_SPLIT_RE.fullmatch(tok)
        if m:
            expanded.add(m.group(1))   # alpha prefix  (e.g. 'product')
            expanded.add(m.group(2))   # digit suffix  (e.g. '2')
    return expanded

# ---------------------------------------------------------------------------
# Ablation mode weights
# ---------------------------------------------------------------------------

# Weights for all four ablation levels.  Each level extends the previous.
RAL_WEIGHTS: dict[str, dict[str, float]] = {
    "basic": {
        "type_exact_bonus": 4.0,
        "type_loose_bonus": 1.5,
        "type_incompatible_penalty": -1e9,
        "lex_context_overlap": 0.7,
        "lex_sentence_overlap": 0.3,
        "schema_prior_bonus": 0.4,
        "weak_match_penalty": -0.8,
    },
    "ops": {
        "type_exact_bonus": 4.0,
        "type_loose_bonus": 1.5,
        "type_incompatible_penalty": -1e9,
        "lex_context_overlap": 0.7,
        "lex_sentence_overlap": 0.3,
        "schema_prior_bonus": 0.4,
        "weak_match_penalty": -0.8,
        # extra: operator/bound cues
        "operator_match_bonus": 2.0,
        "polarity_match_bonus": 1.5,
        "polarity_conflict_penalty": -2.0,
        # Stage 4: bound mention → pure objective/coefficient slot mismatch
        "bound_to_objective_mismatch_penalty": -3.5,
    },
    "semantic": {
        "type_exact_bonus": 4.0,
        "type_loose_bonus": 1.5,
        "type_incompatible_penalty": -1e9,
        "lex_context_overlap": 0.7,
        "lex_sentence_overlap": 0.3,
        "schema_prior_bonus": 0.4,
        "weak_match_penalty": -0.8,
        "operator_match_bonus": 2.0,
        "polarity_match_bonus": 1.5,
        "polarity_conflict_penalty": -2.0,
        # extra: semantic role families + total/unit/percent
        "semantic_family_match_bonus": 2.5,
        "total_match_bonus": 1.5,
        "coeff_match_bonus": 1.5,
        "percent_match_bonus": 2.0,
        "percent_mismatch_penalty": -3.0,
        "unit_match_bonus": 1.5,
        "fragment_compat_bonus": 1.2,
        # Stage 3: narrow measure/attribute overlap
        "narrow_measure_overlap_bonus": 2.0,
        # Stage 4: bound mention → pure objective/coefficient slot mismatch
        "bound_to_objective_mismatch_penalty": -3.5,
        # Stage 5: competing-measure conflict (mention context better matches another slot)
        "narrow_measure_conflict_penalty": -1.5,
        # Stage 6: role-family mismatch (cost cue → profit slot, etc.)
        "role_family_mismatch_penalty": -3.0,
        # Group 3: clause-local entity alignment (same weights as full).
        # Note: left_entity_anchor_bonus is intentionally absent in semantic mode
        # (it is only active in full mode) to maintain the ablation boundary between
        # semantic-role scoring and directional entity-anchor scoring.
        "clause_entity_alignment_bonus": 1.2,
        "cross_clause_entity_penalty_weight": -1.2,
    },
    "full": {
        "type_exact_bonus": 4.0,
        "type_loose_bonus": 1.5,
        "type_incompatible_penalty": -1e9,
        "lex_context_overlap": 0.7,
        "lex_sentence_overlap": 0.3,
        "schema_prior_bonus": 0.4,
        "weak_match_penalty": -0.8,
        "operator_match_bonus": 2.0,
        "polarity_match_bonus": 1.5,
        "polarity_conflict_penalty": -2.0,
        "semantic_family_match_bonus": 2.5,
        "total_match_bonus": 1.5,
        "coeff_match_bonus": 1.5,
        "percent_match_bonus": 2.0,
        "percent_mismatch_penalty": -3.0,
        "unit_match_bonus": 1.5,
        "fragment_compat_bonus": 1.2,
        # extra: entity anchoring + magnitude plausibility
        "entity_anchor_overlap_bonus": 1.8,
        "magnitude_pct_gt100_penalty": -2.0,
        "magnitude_decimal_to_int_penalty": -0.5,
        "role_tag_overlap_bonus": 2.0,
        # Stage 3: narrow measure/attribute overlap
        "narrow_measure_overlap_bonus": 2.0,
        # Stage 4: bound mention → pure objective/coefficient slot mismatch
        "bound_to_objective_mismatch_penalty": -3.5,
        # Stage 5: competing-measure conflict (mention context better matches another slot)
        "narrow_measure_conflict_penalty": -1.5,
        # Stage 6: role-family mismatch (cost cue → profit slot, etc.)
        "role_family_mismatch_penalty": -3.0,
        # Group 3: left-directional entity anchor overlap bonus.
        # Rewards mentions whose LEFT context (not bidirectional) shares tokens
        # with the slot name, giving entity-level disambiguation in parallel clauses.
        # Examples: "Product B requires 7" → left={"product","b","requires"} matches
        # LaborHoursB={"labor","hours","b"} better than LaborHoursA={"labor","hours","a"}.
        "left_entity_anchor_bonus": 2.5,
        # Group 3: clause-local entity alignment bonus + cross-clause penalty.
        # clause_entity_alignment_bonus: rewards a mention whose clause's entity cue
        #   tokens match the slot's entity-discriminating words.  Fires only when the
        #   query has ≥2 clauses; stays neutral (0) for single-clause queries.
        # cross_clause_entity_penalty_weight: penalises assignments where the slot's
        #   best entity evidence lives in a DIFFERENT clause than the mention.
        "clause_entity_alignment_bonus": 1.2,
        "cross_clause_entity_penalty_weight": -1.2,
    },
}

ABLATION_MODES = ("basic", "ops", "semantic", "full")

# ---------------------------------------------------------------------------
# Feature dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MentionFeatures:
    """Mention-side features, computed once per mention."""

    mention_id: int
    value: float | None
    type_bucket: str                  # percent / currency / float / int / unknown
    raw_surface: str
    context_tokens: list[str]
    sentence_tokens: list[str]
    operator_tags: frozenset[str]     # min / max / ...
    unit_tags: frozenset[str]         # percent_marker / currency_marker / ...
    role_tags: frozenset[str]         # optimization-role tags
    fragment_type: str                # objective / constraint / resource / ratio / bound / ""
    is_per_unit: bool
    is_total_like: bool
    nearby_entity_tokens: frozenset[str]
    nearby_resource_tokens: frozenset[str]
    nearby_product_tokens: frozenset[str]
    semantic_families: frozenset[str]
    polarity: str                     # lower / upper / neutral
    style: str                        # total / per_unit / percent / scalar
    # Narrow local context for measure/attribute-aware linking (Stage 3).
    # Derived from the directional ±left/right narrow windows used for
    # is_per_unit/is_total_like detection; stays within clause boundaries.
    narrow_context_tokens: tuple[str, ...] = ()
    # Group 3 directional left anchor (LEFT portion of narrow window only).
    # Used for left_entity_anchor_overlap to discriminate sibling entities:
    #   "Product B requires 7"  → narrow_left_tokens = ("product","b","requires")
    #   "product A requires 3"  → narrow_left_tokens = ("and","product","a","requires")
    # Keeping this separate from narrow_context_tokens prevents right-context
    # entity tokens from the following clause from contaminating anchor scoring.
    narrow_left_tokens: tuple[str, ...] = ()
    # Group 1 role-family flags (propagated from MentionOptIR).
    # Derived from ±2-token tight window; used for role_family_mismatch (Step 4).
    is_cost_like: bool = False
    is_profit_like: bool = False
    is_demand_like: bool = False
    is_resource_like: bool = False
    is_time_like: bool = False


@dataclass(frozen=True)
class SlotFeatures:
    """Slot-side features, computed once per slot."""

    name: str
    norm_tokens: list[str]
    expected_type: str
    alias_tokens: frozenset[str]
    slot_role_tags: frozenset[str]
    operator_preference: frozenset[str]
    unit_preference: frozenset[str]
    is_objective_like: bool
    is_bound_like: bool
    is_total_like: bool
    is_coefficient_like: bool
    semantic_families: frozenset[str]
    polarity: str                     # lower / upper / neutral
    # Derived flags from name decomposition
    is_percent_like: bool
    is_count_like: bool
    is_min_like: bool
    is_max_like: bool


@dataclass
class MentionSlotLink:
    """Rich mention-slot compatibility record."""

    mention_id: int
    slot_name: str
    mention_feats: MentionFeatures
    slot_feats: SlotFeatures
    # Relation features
    token_overlap_ctx: int
    token_overlap_sent: int
    operator_compat: bool
    polarity_match: bool
    polarity_conflict: bool
    semantic_family_match: int
    total_match: bool
    coeff_match: bool
    percent_match: bool
    percent_mismatch: bool
    unit_match: bool
    type_exact: bool
    type_loose: bool
    type_incompatible: bool
    entity_anchor_overlap: int
    fragment_compat: bool
    magnitude_pct_suspicious: bool    # percent > 100
    magnitude_decimal_to_int: bool    # decimal mention to integer-only slot
    role_tag_overlap: int
    # Stage 3: narrow measure/attribute-aware overlap.
    # Overlap between mention's narrow clause-local context and slot name tokens.
    # Rewards tight lexical co-occurrence (e.g. "labor hours" near 4 → LaborHoursPerProduct)
    # while suppressing cross-clause contamination from the wider ±14 context window.
    narrow_measure_overlap: int = 0
    # Stage 4: distractor-suppression — lower/upper bound mention filling a
    # purely objective/coefficient slot.  A number tagged as a bound (min/max
    # operator) should not fill a slot that expects a plain coefficient with no
    # bound character (e.g. "at least 50 units" → ProfitPerUnit is a mismatch).
    bound_to_objective_mismatch: bool = False
    # Stage 5: competing-measure conflict.
    # For this (mention, slot) pair, counts how much the mention's narrow context
    # better matches a COMPETING slot than the current one:
    #   narrow_measure_conflict = max(0, max_other_slot_overlap - this_slot_overlap)
    # A positive value means the mention's local cues favour a different slot.
    # Populated by _post_compute_narrow_measure_conflict() after link building.
    narrow_measure_conflict: int = 0
    # Stage 6: role-family mismatch between mention cue and slot role family.
    # True when tight-context evidence clearly disagrees with the slot's role
    # (e.g. cost cue → profit slot, profit cue → cost slot, demand/resource cue
    # → profit slot).  Only fires when the evidence is unambiguous (exactly one
    # of the opposing role flags is active on the mention).
    role_family_mismatch: bool = False
    # Group 3: left-directional entity anchor overlap.
    # Overlap between the mention's narrow_left_tokens (the tokens to the LEFT
    # of the mention in the text) and the slot's norm_tokens.  Since the left
    # context before a number typically names the entity/attribute it belongs to
    # ("Product B requires 7", "Feed A contains 10"), this directional anchor
    # resolves sibling-entity ambiguity that the bidirectional narrow_context
    # cannot distinguish.  Applied in "full" ablation mode only.
    left_entity_anchor_overlap: int = 0
    # Group 3: clause-local alignment features.
    # Populated by _post_compute_clause_alignment() after all links are built.
    # All three fields stay at 0 when the query has only one clause (neutral),
    # preserving existing behavior for single-clause / single-entity queries.
    #
    # clause_entity_overlap:
    #   Count of entity-cue tokens in the mention's clause that match the slot's
    #   entity-discriminating words (slot norm tokens minus measure/unit words).
    #   E.g. for mention in Feed-A clause and slot ProteinFeedA, overlap = 2 ("feed","a").
    # clause_measure_overlap:
    #   Count of measure-cue tokens in the mention's clause that match the slot's
    #   norm tokens.  Diagnostic field; not currently in scoring (symmetric across
    #   parallel clauses), but available for inspection and future ablations.
    # cross_clause_entity_penalty:
    #   Positive integer when the slot's strongest entity evidence lives in a
    #   DIFFERENT clause than this mention.  Value = (best clause overlap) − (this
    #   clause overlap).  Used as a penalty coefficient: multiplied by
    #   cross_clause_entity_penalty_weight in scoring.
    clause_entity_overlap: int = 0
    clause_measure_overlap: int = 0
    cross_clause_entity_penalty: int = 0
    # Cached scores per ablation mode (populated lazily by score functions)
    _scores: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict)


@dataclass(frozen=True)
class MentionMentionRelation:
    """Pairwise relation between two mentions (for diagnostics and pairwise terms)."""

    mention_id_i: int
    mention_id_j: int
    same_type: bool
    one_is_percent: bool
    one_is_total_one_is_unit: bool
    ascending_order: bool | None       # None if either value is None
    possible_duplicate: bool           # same value within small tolerance
    same_sentence_context: bool


@dataclass(frozen=True)
class SlotSlotRelation:
    """Pairwise relation between two slots (structural schema prior)."""

    slot_name_i: str
    slot_name_j: str
    is_min_max_pair: bool
    is_total_unit_pair: bool
    is_budget_cost_pair: bool
    is_demand_capacity_pair: bool
    same_semantic_family: bool
    same_expected_type: bool
    one_is_percent: bool


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _mention_features_from_ir(m: MentionOptIR) -> MentionFeatures:
    """Convert a MentionOptIR to the richer MentionFeatures representation."""
    return MentionFeatures(
        mention_id=m.mention_id,
        value=m.value,
        type_bucket=m.type_bucket,
        raw_surface=m.raw_surface,
        context_tokens=list(m.context_tokens),
        sentence_tokens=list(m.sentence_tokens),
        operator_tags=m.operator_tags,
        unit_tags=m.unit_tags,
        role_tags=m.role_tags,
        fragment_type=m.fragment_type,
        is_per_unit=m.is_per_unit,
        is_total_like=m.is_total_like,
        nearby_entity_tokens=m.nearby_entity_tokens,
        nearby_resource_tokens=m.nearby_resource_tokens,
        nearby_product_tokens=m.nearby_product_tokens,
        semantic_families=_mention_semantic_families(m),
        polarity=_mention_polarity(m),
        style=_mention_style(m),
        narrow_context_tokens=m.narrow_context_tokens,
        narrow_left_tokens=m.narrow_left_tokens,
        is_cost_like=m.is_cost_like,
        is_profit_like=m.is_profit_like,
        is_demand_like=m.is_demand_like,
        is_resource_like=m.is_resource_like,
        is_time_like=m.is_time_like,
    )


def _slot_features_from_ir(s: SlotOptIR) -> SlotFeatures:
    """Convert a SlotOptIR to the richer SlotFeatures representation."""
    n = (s.name or "").lower()
    is_percent_like = s.expected_type == "percent" or any(
        w in n for w in ("percent", "ratio", "fraction", "rate", "share", "pct")
    )
    is_count_like = s.expected_type == "int" or any(
        w in n for w in ("count", "number", "quantity", "items", "units")
    )
    is_min_like = any(w in n for w in ("min", "minimum", "atleast", "least", "lower"))
    is_max_like = any(w in n for w in ("max", "maximum", "atmost", "most", "upper"))
    return SlotFeatures(
        name=s.name,
        norm_tokens=list(s.norm_tokens),
        expected_type=s.expected_type,
        alias_tokens=frozenset(s.alias_tokens),
        slot_role_tags=s.slot_role_tags,
        operator_preference=s.operator_preference,
        unit_preference=s.unit_preference,
        is_objective_like=s.is_objective_like,
        is_bound_like=s.is_bound_like,
        is_total_like=s.is_total_like,
        is_coefficient_like=s.is_coefficient_like,
        semantic_families=_slot_semantic_families(s),
        polarity=_slot_polarity(s),
        is_percent_like=is_percent_like,
        is_count_like=is_count_like,
        is_min_like=is_min_like,
        is_max_like=is_max_like,
    )


def _build_mention_slot_link(
    mf: MentionFeatures,
    sf: SlotFeatures,
    has_pct_mention_in_pool: bool,
) -> MentionSlotLink:
    """Compute all relation features for one (mention, slot) pair."""
    slot_words = set(sf.norm_tokens) | sf.alias_tokens
    ctx_set = set(mf.context_tokens)
    sent_set = set(mf.sentence_tokens)

    token_overlap_ctx = len(ctx_set & slot_words)
    token_overlap_sent = len(sent_set & slot_words)

    operator_compat = bool(mf.operator_tags & sf.operator_preference)
    polarity_match = mf.polarity == sf.polarity and mf.polarity != "neutral"
    polarity_conflict = (
        (mf.polarity == "lower" and sf.polarity == "upper")
        or (mf.polarity == "upper" and sf.polarity == "lower")
    )

    sem_families_m = mf.semantic_families
    sem_families_s = sf.semantic_families
    semantic_family_match = len(sem_families_m & sem_families_s)

    total_match = sf.is_total_like and mf.is_total_like
    coeff_match = sf.is_coefficient_like and mf.is_per_unit
    percent_match = sf.is_percent_like and mf.type_bucket == "percent"
    percent_mismatch = (
        has_pct_mention_in_pool
        and sf.is_percent_like
        and mf.type_bucket != "percent"
    )

    unit_match = bool(mf.unit_tags & sf.unit_preference)

    kind = mf.type_bucket
    expected = sf.expected_type
    type_incompatible = _is_type_incompatible(expected, kind)
    if not type_incompatible and kind != "unknown":
        type_exact = (
            (expected == "percent" and kind == "percent")
            or (expected == "currency" and kind == "currency")
            or (expected == "float" and kind in {"float", "int"})
            or (expected == "int" and kind == "int")
        )
        type_loose = not type_exact and (
            (expected in ("int", "float") and kind == "currency")
            or (expected == "int" and kind == "float")
        )
    else:
        type_exact = False
        type_loose = False

    entity_anchor_tokens = (
        mf.nearby_entity_tokens | mf.nearby_resource_tokens | mf.nearby_product_tokens
    )
    entity_anchor_overlap = len(entity_anchor_tokens & slot_words)

    # Fragment type compatibility
    fragment_compat = (
        (mf.fragment_type == "objective" and sf.is_objective_like)
        or (mf.fragment_type in ("constraint", "bound") and sf.is_bound_like)
        or (mf.fragment_type == "resource" and sf.is_total_like)
        or (
            mf.fragment_type == "ratio"
            and (
                "ratio_constraint" in sf.slot_role_tags
                or "percentage_constraint" in sf.slot_role_tags
            )
        )
    )

    # Magnitude plausibility
    magnitude_pct_suspicious = (
        mf.type_bucket == "percent"
        and mf.value is not None
        and mf.value > 100.0
    )
    magnitude_decimal_to_int = (
        sf.is_count_like
        and mf.value is not None
        and mf.value % 1 != 0  # has fractional part
    )

    role_tag_overlap = len(mf.role_tags & sf.slot_role_tags)

    # Stage 3: narrow measure/attribute-aware overlap.
    # Use the mention's narrow clause-local context (not the full ±14 window) to
    # compute overlap with slot name tokens.  A tight local match (e.g. "labor
    # hours" immediately surrounding "4" matching slot "LaborHoursPerProduct") is a
    # strong signal that this is the correct slot, while preventing spurious matches
    # from cross-clause tokens like "total budget" appearing in the wide context.
    narrow_set = set(mf.narrow_context_tokens)
    narrow_measure_overlap = len(narrow_set & slot_words) if narrow_set else 0

    # Group 3: left-directional entity anchor overlap.
    # Use only the tokens to the LEFT of the mention (not the bidirectional narrow
    # context) to measure entity-slot alignment.  In parallel-clause patterns the
    # entity that "owns" a number almost always appears immediately before it, while
    # the right window often leaks tokens from the sibling clause (e.g. "product b
    # requires 7 … and product a requires 3": left[7]={"product","b","requires"},
    # left[3]={"and","product","a","requires"}).  Using only the left anchor lets the
    # scoring differentiate slot B from slot A without the right-context cross-talk.
    #
    # Critically, the slot's measure/unit words are EXCLUDED before overlap scoring
    # (via _LEFT_ANCHOR_MEASURE_EXCLUDE).  Without this filter, the left anchor of
    # "5" in "3 heating hours and 5 cooling hours" would contain "heating" (leaked
    # from the previous number's context), spuriously boosting HeatingHours for 5.
    # Excluding "heating" from the slot's entity anchor words prevents this.
    left_set = set(mf.narrow_left_tokens)
    # Expand compound alphanum tokens in the left anchor (e.g. 'product2' →
    # {'product2', 'product', '2'}) so they can match the split components of
    # slot norm_tokens such as ['laborproduct2', 'labor', 'product', '2'].
    # Without expansion, 'product2' never overlaps with slot token '2' or
    # 'product' individually, causing numeric-suffix entities to receive zero
    # left-anchor scores and fail to discriminate sibling slots.
    left_set = _expand_compound_tokens(left_set)
    # Slot entity-discriminating tokens: remove measure/unit words and the full
    # lower-cased slot name (which is always a duplicate of the component tokens).
    slot_entity_words = (slot_words - _LEFT_ANCHOR_MEASURE_EXCLUDE) - {sf.name.lower()}
    left_entity_anchor_overlap = (
        len(left_set & slot_entity_words) if (left_set and slot_entity_words) else 0
    )

    # Stage 4: distractor suppression — bound polarity vs. pure objective slot.
    # A lower/upper bound mention (e.g. "at least 50") should not fill a slot
    # that is purely a coefficient/objective (e.g. ProfitPerUnit) with no bound
    # character.  Flag when polarity is non-neutral AND slot is coefficient-like
    # but NOT bound-like.  This penalises the wrong assignment without blocking
    # valid cases like "each unit requires at least 4 hours" → MinHoursPerUnit.
    bound_to_objective_mismatch = (
        mf.polarity != "neutral"          # has min or max polarity
        and sf.is_coefficient_like        # slot expects a per-unit coefficient
        and not sf.is_bound_like          # slot is NOT a bound slot
    )

    # Stage 6: role-family mismatch (Group 1 distractor suppression).
    # Penalise mention-slot pairs where tight-context role cues clearly conflict
    # with the slot's economic role family.  Only fires when the evidence is
    # unambiguous: exactly one of the opposing role flags is set on the mention.
    # Example mismatches:
    #   - cost cue only  → profit slot  (e.g. "costs 5" → ProfitPerUnit)
    #   - profit cue only → cost slot   (e.g. "yields 12 profit" → CostPerUnit)
    #   - demand/need cue → profit slot (e.g. "demand of 100" → ProfitPerUnit)
    #   - resource/time cue → profit slot (e.g. "4 labor hours" → ProfitPerUnit)
    _sf_is_profit = "unit_profit" in sf.slot_role_tags
    _sf_is_cost_only = (
        "unit_cost" in sf.slot_role_tags and "unit_profit" not in sf.slot_role_tags
    )
    role_family_mismatch = (
        # Cost-cued mention → profit slot (unambiguous: no profit cue present)
        (mf.is_cost_like and not mf.is_profit_like and _sf_is_profit)
        # Profit-cued mention → cost-only slot (unambiguous: no cost cue present)
        or (mf.is_profit_like and not mf.is_cost_like and _sf_is_cost_only)
        # Demand/need-cued mention (no economic cue) → profit slot
        or (mf.is_demand_like and not mf.is_cost_like and not mf.is_profit_like and _sf_is_profit)
        # Resource/time-cued mention (no economic cue) → profit slot
        or (
            (mf.is_resource_like or mf.is_time_like)
            and not mf.is_cost_like
            and not mf.is_profit_like
            and _sf_is_profit
        )
    )

    return MentionSlotLink(
        mention_id=mf.mention_id,
        slot_name=sf.name,
        mention_feats=mf,
        slot_feats=sf,
        token_overlap_ctx=token_overlap_ctx,
        token_overlap_sent=token_overlap_sent,
        operator_compat=operator_compat,
        polarity_match=polarity_match,
        polarity_conflict=polarity_conflict,
        semantic_family_match=semantic_family_match,
        total_match=total_match,
        coeff_match=coeff_match,
        percent_match=percent_match,
        percent_mismatch=percent_mismatch,
        unit_match=unit_match,
        type_exact=type_exact,
        type_loose=type_loose,
        type_incompatible=type_incompatible,
        entity_anchor_overlap=entity_anchor_overlap,
        fragment_compat=fragment_compat,
        magnitude_pct_suspicious=magnitude_pct_suspicious,
        magnitude_decimal_to_int=magnitude_decimal_to_int,
        role_tag_overlap=role_tag_overlap,
        narrow_measure_overlap=narrow_measure_overlap,
        bound_to_objective_mismatch=bound_to_objective_mismatch,
        role_family_mismatch=role_family_mismatch,
        left_entity_anchor_overlap=left_entity_anchor_overlap,
    )


def _post_compute_narrow_measure_conflict(links: list[MentionSlotLink]) -> None:
    """Post-compute narrow_measure_conflict for each link (Stage 5).

    For each mention, the conflict for a (mention, slot) pair is::

        narrow_measure_conflict = max(0, max_other_slot_overlap - this_slot_overlap)

    where *max_other_slot_overlap* is the highest ``narrow_measure_overlap``
    among all OTHER slots for the same mention.  A positive conflict means
    the mention's local context matches a different slot better than this one,
    which indicates the current assignment is likely a distractor.

    This is computed in a post-processing pass (after all links are built)
    because it requires global knowledge of all (mention, slot) pairs for each
    mention.  The MentionSlotLink dataclass is not frozen, so mutation is safe.
    The ``_scores`` cache is still empty at this point so no stale entries exist.
    """
    # Group links by mention_id.
    by_mention: dict[int, list[MentionSlotLink]] = {}
    for lnk in links:
        if lnk.mention_id not in by_mention:
            by_mention[lnk.mention_id] = []
        by_mention[lnk.mention_id].append(lnk)

    for m_links in by_mention.values():
        for lnk in m_links:
            max_other = max(
                (
                    other.narrow_measure_overlap
                    for other in m_links
                    if other.slot_name != lnk.slot_name
                ),
                default=0,
            )
            lnk.narrow_measure_conflict = max(0, max_other - lnk.narrow_measure_overlap)


def _post_compute_clause_alignment(
    links: list[MentionSlotLink],
    mentions_ir: list["MentionOptIR"],
    query: str,
) -> None:
    """Post-compute clause-local alignment features for each link (Group 3).

    Adds three soft features to every ``MentionSlotLink``:

    ``clause_entity_overlap``
        Count of entity-cue tokens from the mention's clause that match the
        slot's entity-discriminating words (slot norm tokens minus measure/unit
        words, same filter as ``left_entity_anchor_overlap``).  Rewards an
        assignment where the mention's clause clearly names the slot's entity.

    ``clause_measure_overlap``
        Count of measure-cue tokens from the mention's clause that match the
        slot's norm tokens.  Diagnostic / reserved — not used in scoring by
        default (symmetric across parallel clauses), but exposed for inspection
        and future ablations.

    ``cross_clause_entity_penalty``
        Positive integer when the slot's strongest entity-cue match lives in a
        *different* clause than the mention.  Value equals
        ``best_other_overlap − this_clause_overlap``.  Used as a coefficient
        for ``cross_clause_entity_penalty_weight`` in scoring.

    All three fields remain at 0 when the query has only one clause, fully
    preserving single-clause / single-entity query behavior (neutral).

    Uses a lazy import from ``clause_aware_linking`` to avoid circular imports.
    """
    # Lazy import to avoid circular dependency (clause_aware_linking imports
    # from relation_aware_linking).
    from tools.clause_aware_linking import (  # noqa: PLC0415
        split_into_clauses,
        build_clause_summaries,
        _assign_mentions_to_clauses,
    )

    clauses = split_into_clauses(query)
    if len(clauses) <= 1:
        # Single-clause query: all clause alignment fields stay at 0 (neutral).
        return

    summaries = build_clause_summaries(clauses, mentions_ir)
    if not summaries:
        return

    mention_to_clause: dict[int, int] = _assign_mentions_to_clauses(mentions_ir, clauses)
    summary_by_idx = {s.clause_idx: s for s in summaries}

    # Build per-slot token sets once (avoid recomputing for each link).
    # slot_entity_words: slot tokens that are entity-discriminating (measure/unit
    #   words filtered out, same logic as left_entity_anchor_overlap).
    # slot_all_words: full set of slot tokens (for clause_measure_overlap).
    slot_entity_map: dict[str, frozenset[str]] = {}
    slot_words_map: dict[str, frozenset[str]] = {}
    for lnk in links:
        sn = lnk.slot_name
        if sn not in slot_entity_map:
            sw: frozenset[str] = frozenset(lnk.slot_feats.norm_tokens) | lnk.slot_feats.alias_tokens
            slot_entity_map[sn] = sw - _LEFT_ANCHOR_MEASURE_EXCLUDE - {sn.lower()}
            slot_words_map[sn] = sw

    # For each slot, determine which clause has the best entity-cue overlap.
    # best_clause_for_slot[slot_name] = (best_clause_idx, best_overlap_count)
    best_clause_for_slot: dict[str, tuple[int, int]] = {}
    for sn, ent_words in slot_entity_map.items():
        best_idx, best_ov = 0, 0
        for summary in summaries:
            clause_ent: frozenset[str] = frozenset(
                t.lower().strip(".,;:()[]{}\"'") for t in summary.entity_cue_tokens
            )
            # Expand compound tokens (e.g. 'Type1' → {'type1','type','1'}) so
            # they can match the split components of slot norm_tokens.
            clause_ent = frozenset(_expand_compound_tokens(set(clause_ent)))
            ov = len(clause_ent & ent_words) if ent_words else 0
            if ov > best_ov:
                best_ov = ov
                best_idx = summary.clause_idx
        best_clause_for_slot[sn] = (best_idx, best_ov)

    # Update each link with the three clause-local alignment values.
    for lnk in links:
        m_clause_idx = mention_to_clause.get(lnk.mention_id, 0)
        m_summary = summary_by_idx.get(m_clause_idx)
        if m_summary is None:
            continue

        sn = lnk.slot_name
        ent_words = slot_entity_map.get(sn, frozenset())
        meas_words = slot_words_map.get(sn, frozenset())

        # A. clause_entity_overlap — expand compound tokens so 'Type1' matches
        # slot tokens ['type','1'] from LaborType1, etc.
        clause_ent = frozenset(
            t.lower().strip(".,;:()[]{}\"'") for t in m_summary.entity_cue_tokens
        )
        clause_ent = frozenset(_expand_compound_tokens(set(clause_ent)))
        lnk.clause_entity_overlap = len(clause_ent & ent_words) if ent_words else 0

        # B. clause_measure_overlap (diagnostic; not in scoring)
        lnk.clause_measure_overlap = len(m_summary.measure_cue_tokens & meas_words)

        # C. cross_clause_entity_penalty
        # If the slot's best entity clause is a different clause, the penalty
        # equals how much better that other clause is than this one.
        best_idx, best_ov = best_clause_for_slot.get(sn, (0, 0))
        if best_ov > 0 and best_idx != m_clause_idx:
            lnk.cross_clause_entity_penalty = max(
                0, best_ov - lnk.clause_entity_overlap
            )
        else:
            lnk.cross_clause_entity_penalty = 0


def build_mention_slot_links(
    query: str,
    variant: str,
    expected_scalar: list[str],
) -> tuple[
    list[MentionSlotLink],
    list[MentionOptIR],
    list[SlotOptIR],
    list[MentionFeatures],
    list[SlotFeatures],
]:
    """Build the full relation-aware mention-slot link table.

    For every (mention, slot) pair, computes a MentionSlotLink with all
    A/B/C feature groups populated.

    Returns:
        links         : flat list of all (mention, slot) pairs
        mentions_ir   : raw MentionOptIR list (for assignment)
        slots_ir      : raw SlotOptIR list (for assignment)
        mention_feats : MentionFeatures per mention
        slot_feats    : SlotFeatures per slot
    """
    mentions_ir: list[MentionOptIR] = _extract_opt_role_mentions(query, variant)
    slots_ir: list[SlotOptIR] = _build_slot_opt_irs(expected_scalar)

    mention_feats: list[MentionFeatures] = [_mention_features_from_ir(m) for m in mentions_ir]
    slot_feats: list[SlotFeatures] = [_slot_features_from_ir(s) for s in slots_ir]

    # Compute once; passed to each link builder to avoid O(n²·m) recomputation.
    has_pct_mention_in_pool = any(mf.type_bucket == "percent" for mf in mention_feats)

    links: list[MentionSlotLink] = []
    for mf in mention_feats:
        for sf in slot_feats:
            lnk = _build_mention_slot_link(mf, sf, has_pct_mention_in_pool)
            links.append(lnk)

    # Stage 5: post-compute competing-measure conflict scores.
    # Must run after all links are built (requires global view of all slots per mention).
    _post_compute_narrow_measure_conflict(links)

    # Group 3: post-compute clause-local alignment features.
    # Must run after all links are built (requires global view of all clauses and mentions).
    # Stays neutral (0) for single-clause queries; only enriches multi-clause patterns.
    _post_compute_clause_alignment(links, mentions_ir, query)

    return links, mentions_ir, slots_ir, mention_feats, slot_feats


def build_mention_mention_relations(
    mention_feats: list[MentionFeatures],
) -> list[MentionMentionRelation]:
    """Compute pairwise mention-mention relation records."""
    relations: list[MentionMentionRelation] = []
    for i in range(len(mention_feats)):
        for j in range(i + 1, len(mention_feats)):
            mi = mention_feats[i]
            mj = mention_feats[j]
            same_type = mi.type_bucket == mj.type_bucket
            one_is_percent = (mi.type_bucket == "percent") != (mj.type_bucket == "percent")
            one_is_total_one_is_unit = (mi.is_total_like != mj.is_total_like) or (
                mi.is_per_unit != mj.is_per_unit
            )
            if mi.value is not None and mj.value is not None:
                ascending_order: bool | None = mi.value <= mj.value
                possible_duplicate = abs(mi.value - mj.value) < 1e-6
            else:
                ascending_order = None
                possible_duplicate = False
            ctx_overlap = bool(
                set(mi.context_tokens) & set(mj.context_tokens)
            )
            relations.append(
                MentionMentionRelation(
                    mention_id_i=mi.mention_id,
                    mention_id_j=mj.mention_id,
                    same_type=same_type,
                    one_is_percent=one_is_percent,
                    one_is_total_one_is_unit=one_is_total_one_is_unit,
                    ascending_order=ascending_order,
                    possible_duplicate=possible_duplicate,
                    same_sentence_context=ctx_overlap,
                )
            )
    return relations


def build_slot_slot_relations(
    slot_feats: list[SlotFeatures],
) -> list[SlotSlotRelation]:
    """Compute pairwise slot-slot relation records (structural schema priors)."""
    relations: list[SlotSlotRelation] = []
    for i in range(len(slot_feats)):
        for j in range(i + 1, len(slot_feats)):
            si = slot_feats[i]
            sj = slot_feats[j]
            ni = si.name.lower()
            nj = sj.name.lower()
            is_min_max_pair = (si.is_min_like and sj.is_max_like) or (
                si.is_max_like and sj.is_min_like
            )
            is_total_unit_pair = (si.is_total_like and sj.is_coefficient_like) or (
                si.is_coefficient_like and sj.is_total_like
            )
            # budget/cost pair: one mentions budget, other mentions cost
            _budget_words = {"budget", "available", "total"}
            _cost_words = {"cost", "expense", "price", "profit", "revenue"}
            is_budget_cost_pair = bool(
                (any(w in ni for w in _budget_words) and any(w in nj for w in _cost_words))
                or (any(w in nj for w in _budget_words) and any(w in ni for w in _cost_words))
            )
            # demand/capacity
            _demand_words = {"demand", "require", "need", "minimum"}
            _cap_words = {"capacity", "limit", "supply", "available", "maximum"}
            is_demand_capacity_pair = bool(
                (any(w in ni for w in _demand_words) and any(w in nj for w in _cap_words))
                or (any(w in nj for w in _demand_words) and any(w in ni for w in _cap_words))
            )
            same_sem = bool(si.semantic_families & sj.semantic_families)
            same_expected_type = si.expected_type == sj.expected_type
            one_is_percent = (si.is_percent_like) != (sj.is_percent_like)
            relations.append(
                SlotSlotRelation(
                    slot_name_i=si.name,
                    slot_name_j=sj.name,
                    is_min_max_pair=is_min_max_pair,
                    is_total_unit_pair=is_total_unit_pair,
                    is_budget_cost_pair=is_budget_cost_pair,
                    is_demand_capacity_pair=is_demand_capacity_pair,
                    same_semantic_family=same_sem,
                    same_expected_type=same_expected_type,
                    one_is_percent=one_is_percent,
                )
            )
    return relations


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def relation_aware_local_score(
    link: MentionSlotLink,
    ablation_mode: str = "full",
) -> tuple[float, dict[str, Any]]:
    """Compute the relation-aware local compatibility score for one (mention, slot) pair.

    ablation_mode controls the active feature groups:
      "basic"    — type compat + lexical overlap
      "ops"      — + operator/bound cues
      "semantic" — + semantic roles / total-unit / percent
      "full"     — + entity anchoring / magnitude plausibility

    Returns (score, feature_dict).
    """
    # Return cached result if already computed for this mode.
    if ablation_mode in link._scores:
        return link._scores[ablation_mode]

    w = RAL_WEIGHTS[ablation_mode]
    features: dict[str, Any] = {}
    score = 0.0

    # ── Type compatibility ────────────────────────────────────────────────
    if link.type_incompatible:
        score += w["type_incompatible_penalty"]
        features["type_incompatible"] = True
        result = (score, features)
        link._scores[ablation_mode] = result
        return result

    if link.type_exact:
        score += w["type_exact_bonus"]
        features["type_exact"] = True
    elif link.type_loose:
        score += w["type_loose_bonus"]
        features["type_loose"] = True

    # ── Lexical overlap ───────────────────────────────────────────────────
    if link.token_overlap_ctx:
        score += w["lex_context_overlap"] * link.token_overlap_ctx
        features["ctx_overlap"] = link.token_overlap_ctx
    if link.token_overlap_sent:
        score += w["lex_sentence_overlap"] * link.token_overlap_sent
        features["sent_overlap"] = link.token_overlap_sent

    score += w["schema_prior_bonus"]

    # ── Operator / bound cues (ops and above) ────────────────────────────
    if ablation_mode in ("ops", "semantic", "full"):
        if link.operator_compat:
            score += w["operator_match_bonus"]
            features["operator_match"] = True
        if link.polarity_match:
            score += w["polarity_match_bonus"]
            features["polarity_match"] = True
        if link.polarity_conflict:
            score += w["polarity_conflict_penalty"]
            features["polarity_conflict"] = True
        # Stage 4: bound mention → pure objective/coefficient slot mismatch penalty.
        # Suppresses distractor assignment of constrained values (e.g. "at least 50")
        # to objective/coefficient slots (e.g. ProfitPerUnit) that carry no bound role.
        if link.bound_to_objective_mismatch and "bound_to_objective_mismatch_penalty" in w:
            score += w["bound_to_objective_mismatch_penalty"]
            features["bound_to_objective_mismatch"] = True

    # ── Semantic roles (semantic and above) ──────────────────────────────
    if ablation_mode in ("semantic", "full"):
        if link.semantic_family_match:
            score += w["semantic_family_match_bonus"] * link.semantic_family_match
            features["semantic_family_match"] = link.semantic_family_match
        if link.total_match:
            score += w["total_match_bonus"]
            features["total_match"] = True
        if link.coeff_match:
            score += w["coeff_match_bonus"]
            features["coeff_match"] = True
        if link.percent_match:
            score += w["percent_match_bonus"]
            features["percent_match"] = True
        if link.percent_mismatch:
            score += w["percent_mismatch_penalty"]
            features["percent_mismatch"] = True
        if link.unit_match:
            score += w["unit_match_bonus"]
            features["unit_match"] = True
        if link.fragment_compat:
            score += w["fragment_compat_bonus"]
            features["fragment_compat"] = True
        # Stage 3: narrow measure/attribute-aware overlap bonus
        if link.narrow_measure_overlap and "narrow_measure_overlap_bonus" in w:
            score += w["narrow_measure_overlap_bonus"] * link.narrow_measure_overlap
            features["narrow_measure_overlap"] = link.narrow_measure_overlap
        # Stage 5: competing-measure conflict penalty.
        # The mention's narrow context matches a DIFFERENT slot better than this one.
        if link.narrow_measure_conflict and "narrow_measure_conflict_penalty" in w:
            score += w["narrow_measure_conflict_penalty"] * link.narrow_measure_conflict
            features["narrow_measure_conflict"] = link.narrow_measure_conflict
        # Stage 6: role-family mismatch penalty.
        # Tight-context role cues (cost/profit/demand/resource/time) clearly conflict
        # with the slot's economic role family (e.g. cost cue → profit slot).
        if link.role_family_mismatch and "role_family_mismatch_penalty" in w:
            score += w["role_family_mismatch_penalty"]
            features["role_family_mismatch"] = True
        # Group 3: clause-local entity alignment.
        # clause_entity_overlap: mention's clause entity cues match slot entity words.
        # Only fires in multi-clause queries (single-clause overlap stays 0).
        if link.clause_entity_overlap and "clause_entity_alignment_bonus" in w:
            score += w["clause_entity_alignment_bonus"] * link.clause_entity_overlap
            features["clause_entity_overlap"] = link.clause_entity_overlap
        # Group 3: cross-clause entity penalty.
        # The slot's best entity evidence lives in a different clause from the mention.
        if link.cross_clause_entity_penalty and "cross_clause_entity_penalty_weight" in w:
            score += w["cross_clause_entity_penalty_weight"] * link.cross_clause_entity_penalty
            features["cross_clause_entity_penalty"] = link.cross_clause_entity_penalty

    # ── Entity anchoring + magnitude (full only) ─────────────────────────
    if ablation_mode == "full":
        if link.entity_anchor_overlap:
            score += w["entity_anchor_overlap_bonus"] * link.entity_anchor_overlap
            features["entity_anchor_overlap"] = link.entity_anchor_overlap
        if link.magnitude_pct_suspicious:
            score += w["magnitude_pct_gt100_penalty"]
            features["magnitude_pct_suspicious"] = True
        if link.magnitude_decimal_to_int:
            score += w["magnitude_decimal_to_int_penalty"]
            features["magnitude_decimal_to_int"] = True
        if link.role_tag_overlap:
            score += w["role_tag_overlap_bonus"] * link.role_tag_overlap
            features["role_tag_overlap"] = link.role_tag_overlap
        # Group 3: left-directional entity anchor bonus.
        # Tokens to the LEFT of the mention (entity name that "owns" the number)
        # overlap with the slot name tokens.  Applied after all other features as
        # a tie-breaker that resolves sibling-entity ambiguity in parallel clauses.
        if link.left_entity_anchor_overlap and "left_entity_anchor_bonus" in w:
            score += w["left_entity_anchor_bonus"] * link.left_entity_anchor_overlap
            features["left_entity_anchor_overlap"] = link.left_entity_anchor_overlap

    # Weak match penalty
    if score <= 0.0:
        score += w["weak_match_penalty"]
        features["weak_penalty"] = True

    features["total_score"] = score
    result = (score, features)
    link._scores[ablation_mode] = result
    return result


# ---------------------------------------------------------------------------
# Greedy assignment using relation-aware scores
# ---------------------------------------------------------------------------


def best_assignment_greedy(
    links: list[MentionSlotLink],
    slots_ir: list[SlotOptIR],
    mentions_ir: list[MentionOptIR],
    ablation_mode: str = "full",
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, Any]]:
    """Greedy one-to-one assignment: pick best mention for each slot in score order.

    Slots are processed in decreasing order of their best candidate score so that
    high-confidence slots claim their mention first.

    Returns:
        filled_values   : slot_name -> numeric value (float or raw)
        filled_mentions : slot_name -> MentionOptIR
        diagnostics     : dict with per-slot candidate rankings
    """
    mid_to_mention = {m.mention_id: m for m in mentions_ir}

    # Build per-slot ranked candidate lists.
    slot_candidates: dict[str, list[tuple[float, int]]] = {}  # slot_name -> [(score, mid)]
    for lnk in links:
        sc, _ = relation_aware_local_score(lnk, ablation_mode)
        if sc <= GCG_PRUNE_THRESHOLD:
            continue
        slot_candidates.setdefault(lnk.slot_name, []).append((sc, lnk.mention_id))

    for sn in slot_candidates:
        slot_candidates[sn].sort(key=lambda x: -x[0])

    # Sort slots by the score of their top candidate (descending) so best matches go first.
    slot_order = sorted(
        slot_candidates.keys(),
        key=lambda sn: slot_candidates[sn][0][0] if slot_candidates[sn] else -1e9,
        reverse=True,
    )

    used_mids: set[int] = set()
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    diagnostics: dict[str, Any] = {"per_slot_candidates": {}, "ablation_mode": ablation_mode}

    # Include all slot names in diagnostics even if no candidates.
    all_slot_names = {s.name for s in slots_ir}
    for sn in all_slot_names:
        diagnostics["per_slot_candidates"][sn] = [
            {"mention_id": mid, "score": sc}
            for sc, mid in slot_candidates.get(sn, [])
        ]

    for slot_name in slot_order:
        cands = slot_candidates[slot_name]
        for sc, mid in cands:
            if mid in used_mids:
                continue
            m = mid_to_mention[mid]
            filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
            filled_mentions[slot_name] = m
            used_mids.add(mid)
            break

    return filled_values, filled_mentions, diagnostics


# ---------------------------------------------------------------------------
# Convenience: run full pipeline and return same interface as other grounding fns
# ---------------------------------------------------------------------------


def run_relation_aware_grounding(
    query: str,
    variant: str,
    expected_scalar: list[str],
    ablation_mode: str = "full",
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, Any]]:
    """Full pipeline: extract → link → score → assign → swap-repair.

    Parameters
    ----------
    query         : raw query text
    variant       : 'orig' | 'noisy' | 'short'
    expected_scalar : list of scalar slot parameter names
    ablation_mode : 'basic' | 'ops' | 'semantic' | 'full'

    Returns
    -------
    filled_values, filled_mentions, diagnostics
    """
    if not expected_scalar:
        return {}, {}, {"ablation_mode": ablation_mode}

    links, mentions_ir, slots_ir, mention_feats, slot_feats = build_mention_slot_links(
        query, variant, expected_scalar
    )
    if not mentions_ir or not slots_ir:
        return {}, {}, {"ablation_mode": ablation_mode}

    # Pre-score all links for the requested ablation mode.
    for lnk in links:
        relation_aware_local_score(lnk, ablation_mode)

    filled_values, filled_mentions, diagnostics = best_assignment_greedy(
        links, slots_ir, mentions_ir, ablation_mode
    )
    diagnostics["mention_mention_relations"] = build_mention_mention_relations(mention_feats)
    diagnostics["slot_slot_relations"] = build_slot_slot_relations(slot_feats)

    # Group 3: lightweight parallel-swap repair.
    # After greedy assignment, check whether any pair of sibling-slot assignments
    # can be improved by swapping their mentions (left-entity-anchor gain > 0).
    # Only active in "full" mode to preserve ablation boundaries.
    if ablation_mode == "full" and len(filled_mentions) >= 2:
        from tools.clause_aware_linking import detect_and_repair_parallel_swaps
        filled_values, filled_mentions, swap_log = detect_and_repair_parallel_swaps(
            filled_values, filled_mentions, links, ablation_mode,
            query=query, all_mentions=mentions_ir,
        )
        if swap_log:
            diagnostics["group3_swap_repairs"] = swap_log

    return filled_values, filled_mentions, diagnostics
