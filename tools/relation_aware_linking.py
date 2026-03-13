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
    )


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
    """Full pipeline: extract → link → score → assign.

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
    return filled_values, filled_mentions, diagnostics
