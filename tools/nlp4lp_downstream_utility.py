"""Downstream utility demo for NLP4LP: retrieval enables parameter instantiation from NL.

Deterministic, CPU-only: no LLMs, no torch, no solver dependency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NUM_TOKEN_RE = re.compile(r"[$]?\d[\d,]*(?:\.\d+)?%?")

MONEY_CONTEXT = {"budget", "cost", "price", "profit", "revenue", "dollar", "dollars", "$", "€", "usd", "eur"}
PERCENT_CONTEXT = {"percent", "percentage", "rate", "fraction"}

# Cue words and simple operator markers used by constrained assignment.
CUE_WORDS = {
    "budget",
    "cost",
    "profit",
    "revenue",
    "demand",
    "capacity",
    "limit",
    "requirement",
    "requirements",
    "minimum",
    "maximum",
    "least",
    "most",
    "fraction",
    "ratio",
    "percentage",
    "percent",
    "rate",
    "share",
    "total",
    "available",
}

OPERATOR_MIN_WORDS = {"minimum", "min", "least", "lower", "atleast"}
OPERATOR_MAX_WORDS = {"maximum", "max", "most", "upper", "atmost"}


# Weights for interpretable mention-slot compatibility scoring in constrained assignment.
ASSIGN_WEIGHTS = {
    "type_match_bonus": 3.0,
    "type_mismatch_penalty": -4.0,
    "lex_context_overlap": 0.7,
    "lex_sentence_overlap": 0.3,
    "cue_overlap": 1.5,
    "operator_min_bonus": 1.0,
    "operator_max_bonus": 1.0,
    "unit_percent_bonus": 2.0,
    "unit_currency_bonus": 2.0,
    "weak_match_penalty": -1.0,
}


def _safe_json_loads(s: str | None) -> Any:
    if not s:
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def _load_eval(eval_path: Path) -> list[dict]:
    items = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                {
                    "query_id": obj.get("query_id", ""),
                    "query": (obj.get("query") or "").strip(),
                    "relevant_doc_id": obj.get("relevant_doc_id", ""),
                }
            )
    return items


def _load_catalog_as_problems(catalog_path: Path) -> tuple[list[dict], dict[str, str]]:
    """Load catalog JSONL and return list[problem] for baselines + id->text for snippets."""
    catalog: list[dict] = []
    id_to_text: dict[str, str] = {}
    with open(catalog_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or obj.get("id")
            text = (obj.get("text") or obj.get("description") or "").strip()
            if not doc_id:
                continue
            catalog.append({"id": doc_id, "name": doc_id, "description": text, "aliases": []})
            id_to_text[doc_id] = text
    return catalog, id_to_text


def _load_hf_gold(split: str = "test") -> dict[str, dict]:
    """Load NLP4LP HF split and return doc_id -> parsed fields."""
    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(f"datasets not available: {e}")

    raw = (
        (os.environ.get("HF_TOKEN") or "")
        or (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "")
        or (os.environ.get("HUGGINGFACE_TOKEN") or "")
    ).strip()
    kwargs = {"token": raw} if raw else {}
    ds = load_dataset("udell-lab/NLP4LP", split=split, **kwargs)

    gold: dict[str, dict] = {}
    for i, ex in enumerate(ds):
        doc_id = f"nlp4lp_{split}_{i}"
        params = _safe_json_loads(ex.get("parameters"))
        pinfo = _safe_json_loads(ex.get("problem_info"))
        gold[doc_id] = {
            "parameters": params if isinstance(params, dict) else {},
            "problem_info": pinfo if isinstance(pinfo, dict) else {},
            "optimus_code": ex.get("optimus_code") or "",
            "solution": _safe_json_loads(ex.get("solution")),
        }
    return gold


def _tokens_lower(text: str) -> list[str]:
    return re.findall(r"\w+|<num>|[$]?\d[\d,]*(?:\.\d+)?%?", text.lower())


@dataclass(frozen=True)
class NumTok:
    raw: str
    value: float | None
    kind: str  # percent|currency|int|float|unknown


@dataclass(frozen=True)
class MentionRecord:
    """Numeric mention with local/sentence context for constrained assignment."""

    index: int
    tok: NumTok
    context_tokens: list[str]
    sentence_tokens: list[str]
    cue_words: set[str]


@dataclass(frozen=True)
class SlotRecord:
    """Scalar slot representation."""

    name: str
    norm_tokens: list[str]
    expected_type: str
    aliases: list[str]
    alias_tokens: set[str]


def _parse_num_token(tok: str, context_words: set[str]) -> NumTok:
    t = tok.strip()
    if t == "<num>":
        return NumTok(raw=t, value=None, kind="unknown")
    has_dollar = "$" in t
    is_pct = t.endswith("%")
    num_str = t.replace("$", "").replace("%", "").replace(",", "")
    try:
        val = float(num_str)
    except Exception:
        return NumTok(raw=t, value=None, kind="unknown")

    if is_pct:
        return NumTok(raw=t, value=val / 100.0, kind="percent")

    # Percent context without %: treat e.g. "20 percent" as 0.20.
    if ("percent" in context_words or "percentage" in context_words) and val > 1.0:
        return NumTok(raw=t, value=val / 100.0, kind="percent")

    if 0.0 < val <= 1.0 and (context_words & PERCENT_CONTEXT):
        return NumTok(raw=t, value=val, kind="percent")

    if has_dollar or (context_words & MONEY_CONTEXT) or abs(val) >= 1000:
        # Heuristic: treat large numbers as amounts in many optimization word problems.
        return NumTok(raw=t, value=val, kind="currency")

    # Integer vs float
    if float(int(val)) == val:
        return NumTok(raw=t, value=float(int(val)), kind="int")
    return NumTok(raw=t, value=val, kind="float")


def _extract_num_tokens(query: str, variant: str) -> list[NumTok]:
    toks = query.split()
    out: list[NumTok] = []
    for i, w in enumerate(toks):
        if w == "<num>" and variant in ("noisy", "nonum"):
            out.append(NumTok(raw=w, value=None, kind="unknown"))
            continue
        m = NUM_TOKEN_RE.fullmatch(w.strip())
        if not m:
            continue
        # local context window
        ctx = set(x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 3) : i + 4])
        out.append(_parse_num_token(w, ctx))
    return out


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _slot_aliases(param_name: str) -> list[str]:
    """Rule-based alias expansion for common optimization slot patterns."""
    n = (param_name or "").lower()
    aliases = [param_name]

    def add_many(xs: Iterable[str]) -> None:
        for x in xs:
            if x not in aliases:
                aliases.append(x)

    if "budget" in n:
        add_many(["budget", "total budget", "available amount", "amount available"])
    if "capacity" in n:
        add_many(["capacity", "limit", "maximum", "available", "upper bound"])
    if "cost" in n or "expense" in n or "price" in n:
        add_many(["cost", "expense", "spending", "spend", "price"])
    if "profit" in n or "revenue" in n:
        add_many(["profit", "gain", "revenue", "return"])
    if "demand" in n or "require" in n or "needed" in n:
        add_many(["demand", "required", "needed", "requirement"])
    if any(w in n for w in ("fraction", "ratio", "percent", "percentage", "rate", "share")):
        add_many(["percentage", "fraction", "share", "ratio", "rate", "proportion"])
    if "min" in n or "minimum" in n or "atleast" in n:
        add_many(["minimum", "at least", "lower bound"])
    if "max" in n or "maximum" in n or "atmost" in n:
        add_many(["maximum", "at most", "upper bound"])

    return aliases


def _extract_num_mentions(query: str, variant: str) -> list[MentionRecord]:
    """Extract numeric mentions with richer context for constrained assignment."""
    toks = query.split()
    sent_tokens = [t.lower().strip(".,;:()[]{}") for t in toks]
    mentions: list[MentionRecord] = []
    for i, w in enumerate(toks):
        if w == "<num>" and variant in ("noisy", "nonum"):
            tok = NumTok(raw=w, value=None, kind="unknown")
        else:
            m = NUM_TOKEN_RE.fullmatch(w.strip())
            if not m:
                continue
            # slightly wider context window for constrained assignment
            ctx_tokens = [
                x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 8) : i + 9]
            ]
            ctx_set = {c for c in ctx_tokens if c}
            tok = _parse_num_token(w, ctx_set)
        ctx_tokens = [
            x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 8) : i + 9]
        ]
        ctx_tokens = [c for c in ctx_tokens if c]
        cue_words = set(ctx_tokens) & CUE_WORDS
        mentions.append(
            MentionRecord(
                index=i,
                tok=tok,
                context_tokens=ctx_tokens,
                sentence_tokens=sent_tokens,
                cue_words=cue_words,
            )
        )
    return mentions


def _expected_type(param_name: str) -> str:
    n = (param_name or "").lower()
    if any(s in n for s in ("percent", "percentage", "rate", "fraction")):
        return "percent"
    if any(s in n for s in ("num", "count", "types", "items", "ingredients", "nodes", "edges")):
        return "int"
    if any(
        s in n
        for s in (
            "budget",
            "cost",
            "price",
            "revenue",
            "profit",
            "penalty",
            "investment",
            "demand",
            "capacity",
            "minimum",
            "maximum",
            "limit",
        )
    ):
        return "currency"
    return "float"


def _choose_token(expected: str, candidates: list[NumTok]) -> tuple[int | None, NumTok | None]:
    """Return (index, token) to use, or (None, None) if no candidates."""
    if not candidates:
        return None, None

    def score(tok: NumTok) -> tuple:
        # higher is better; deterministic tie-break via raw
        val = tok.value if tok.value is not None else 0.0
        absval = abs(val)
        has_decimal = (tok.value is not None) and (float(int(val)) != val)
        if expected == "percent":
            pref = 2 if tok.kind == "percent" else (1 if (tok.value is not None and 0.0 < tok.value <= 1.0) else 0)
            return (pref, absval, tok.raw)
        if expected == "int":
            pref = 2 if tok.kind == "int" else (1 if tok.value is not None and float(int(val)) == val else 0)
            return (pref, absval, tok.raw)
        if expected == "currency":
            pref = 2 if tok.kind == "currency" else 0
            return (pref, absval, tok.raw)
        # float
        pref = 2 if tok.kind == "float" and has_decimal else (1 if tok.kind in ("float", "int", "currency") else 0)
        return (pref, absval, tok.raw)

    best_i = 0
    best_s = score(candidates[0])
    for i in range(1, len(candidates)):
        s = score(candidates[i])
        if s > best_s:
            best_s = s
            best_i = i
    return best_i, candidates[best_i]


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_type_incompatible(expected: str, kind: str) -> bool:
    """Hard incompatibilities between expected slot type and mention kind."""
    if expected == "percent" and kind in {"currency"}:
        return True
    if expected == "currency" and kind in {"percent"}:
        return True
    return False


def _build_slot_records(expected_scalar: list[str]) -> list[SlotRecord]:
    slots: list[SlotRecord] = []
    for name in expected_scalar:
        et = _expected_type(name)
        aliases = _slot_aliases(name)
        alias_tokens: set[str] = set()
        for a in aliases:
            alias_tokens.update(_normalize_tokens(a))
        norm_tokens = _normalize_tokens(name)
        slots.append(
            SlotRecord(
                name=name,
                norm_tokens=norm_tokens,
                expected_type=et,
                aliases=aliases,
                alias_tokens=alias_tokens,
            )
        )
    return slots


def _score_mention_slot(m: MentionRecord, s: SlotRecord) -> tuple[float, dict[str, Any]]:
    """Compute interpretable compatibility score and feature breakdown."""
    features: dict[str, Any] = {}
    score = 0.0
    kind = m.tok.kind
    expected = s.expected_type

    # Hard type incompatibility.
    if _is_type_incompatible(expected, kind):
        features["type_incompatible"] = True
        return -1e9, features

    # Type compatibility.
    if kind != "unknown":
        if (expected == "percent" and kind == "percent") or (
            expected == "currency" and kind == "currency"
        ):
            score += ASSIGN_WEIGHTS["type_match_bonus"]
            features["type_match"] = True
        elif expected == "int" and kind in {"int", "currency", "float"}:
            score += ASSIGN_WEIGHTS["type_match_bonus"] * 0.5
            features["type_loose_match"] = True
        elif expected == "float" and kind in {"float", "int", "currency"}:
            score += ASSIGN_WEIGHTS["type_match_bonus"] * 0.5
            features["type_loose_match"] = True

    # Lexical overlap: context vs slot name/aliases.
    ctx_set = set(m.context_tokens)
    sent_set = set(m.sentence_tokens)
    slot_words = set(s.norm_tokens) | s.alias_tokens

    ctx_overlap = len(ctx_set & slot_words)
    if ctx_overlap:
        score += ASSIGN_WEIGHTS["lex_context_overlap"] * float(ctx_overlap)
        features["ctx_overlap"] = ctx_overlap

    sent_overlap = len(sent_set & slot_words)
    if sent_overlap:
        score += ASSIGN_WEIGHTS["lex_sentence_overlap"] * float(sent_overlap)
        features["sent_overlap"] = sent_overlap

    # Cue-word overlap.
    cue_overlap = len(m.cue_words & slot_words)
    if cue_overlap:
        score += ASSIGN_WEIGHTS["cue_overlap"] * float(cue_overlap)
        features["cue_overlap"] = cue_overlap

    # Operator semantics.
    ctx_tokens = set(m.context_tokens)
    if ctx_tokens & OPERATOR_MIN_WORDS and any(
        w in "".join(s.norm_tokens) for w in ("min", "minimum")
    ):
        score += ASSIGN_WEIGHTS["operator_min_bonus"]
        features["operator_min"] = True
    if ctx_tokens & OPERATOR_MAX_WORDS and any(
        w in "".join(s.norm_tokens) for w in ("max", "maximum")
    ):
        score += ASSIGN_WEIGHTS["operator_max_bonus"]
        features["operator_max"] = True

    # Unit markers.
    if kind == "percent":
        score += ASSIGN_WEIGHTS["unit_percent_bonus"]
        features["unit_percent"] = True
    if kind == "currency":
        score += ASSIGN_WEIGHTS["unit_currency_bonus"]
        features["unit_currency"] = True

    # Weak-match penalty if score is still very small.
    if score <= 0.0:
        score += ASSIGN_WEIGHTS["weak_match_penalty"]
        features["weak_penalty"] = True

    features["total_score"] = score
    return score, features


# --- Semantic IR + repair assignment (Stage A–D) ---

# Rule-based semantic role words -> tag names (easy to extend).
SEMANTIC_ROLE_WORDS: dict[str, str] = {}
for tag, words in [
    ("budget", ["budget", "budgets"]),
    ("cost", ["cost", "costs"]),
    ("expense", ["expense", "expenses"]),
    ("spend", ["spend", "spending", "spent"]),
    ("profit", ["profit", "profits"]),
    ("revenue", ["revenue", "revenues"]),
    ("return", ["return", "returns"]),
    ("demand", ["demand", "demands"]),
    ("requirement", ["requirement", "requirements"]),
    ("need", ["need", "needed", "needs"]),
    ("capacity", ["capacity"]),
    ("limit", ["limit", "limits"]),
    ("available", ["available", "availability"]),
    ("supply", ["supply", "supplies"]),
    ("minimum", ["minimum", "min"]),
    ("lower_bound", ["lower", "least"]),
    ("maximum", ["maximum", "max"]),
    ("upper_bound", ["upper", "most"]),
    ("at_least", ["atleast", "at_least"]),
    ("at_most", ["atmost", "at_most"]),
    ("ratio", ["ratio", "ratios"]),
    ("fraction", ["fraction", "fractions"]),
    ("percentage", ["percentage", "percent"]),
    ("rate", ["rate", "rates"]),
    ("share", ["share", "shares"]),
    ("total", ["total", "totals"]),
    ("fixed_cost", ["fixed", "fixed_cost"]),
    ("variable_cost", ["variable", "variable_cost"]),
    ("penalty", ["penalty", "penalties"]),
    ("resource", ["resource", "resources"]),
    ("time", ["time", "minutes", "hours", "weeks"]),
    ("quantity", ["quantity", "quantities", "amount", "amounts"]),
    ("item_count", ["number", "count", "items", "units"]),
]:
    for w in words:
        SEMANTIC_ROLE_WORDS[w.lower()] = tag

# Operator phrase tokens (min/max).
OPERATOR_MIN_PHRASES = {"at", "least", "minimum", "min", "no", "less", "than", "lower"}
OPERATOR_MAX_PHRASES = {"at", "most", "maximum", "max", "no", "more", "than", "upper", "up", "to"}

# Unit/marker detection.
PERCENT_MARKER_TOKENS = {"%", "percent", "percentage", "pct"}
CURRENCY_MARKER_TOKENS = {"$", "€", "dollar", "dollars", "usd", "eur", "cost", "price", "budget"}

# Weights for semantic IR scoring (interpretable, additive).
SEMANTIC_IR_WEIGHTS = {
    "type_exact_bonus": 4.0,
    "type_loose_bonus": 1.5,
    "type_incompatible_penalty": -1e9,
    "semantic_tag_overlap": 2.5,
    "lex_context_overlap": 0.6,
    "lex_sentence_overlap": 0.25,
    "operator_min_bonus": 1.2,
    "operator_max_bonus": 1.2,
    "unit_percent_bonus": 2.0,
    "unit_currency_bonus": 2.0,
    "entity_target_bonus": 0.8,
    "weak_match_penalty": -1.0,
}
# Repair / validation weights.
REPAIR_WEIGHTS = {
    "validation_bonus": 0.5,
    "inconsistency_penalty": -2.0,
    "coverage_repair_bonus": 1.5,
    "min_semantic_support": 0.5,
}


@dataclass(frozen=True)
class MentionIR:
    """Lightweight semantic intermediate representation for a numeric mention."""

    mention_id: int
    value: float | None
    type_bucket: str
    raw_surface: str
    context_tokens: list[str]
    sentence_tokens: list[str]
    semantic_role_tags: frozenset[str]
    operator_tags: frozenset[str]  # "min", "max", or empty
    unit_tags: frozenset[str]  # "percent_marker", "currency_marker", "integer_like", "decimal_like"
    polarity_or_bound: str  # "min", "max", or ""
    target_entity_tokens: frozenset[str]
    tok: NumTok


@dataclass(frozen=True)
class SlotIR:
    """Slot with semantic target tags for IR-based matching."""

    name: str
    norm_tokens: list[str]
    expected_type: str
    alias_tokens: set[str]
    semantic_target_tags: frozenset[str]
    operator_preference: frozenset[str]  # "min", "max", or empty
    unit_preference: frozenset[str]


def _context_to_semantic_tags(context_tokens: list[str]) -> frozenset[str]:
    """Rule-based semantic tagging from context tokens."""
    tags: set[str] = set()
    ctx_set = {t.lower().strip(".,;:()[]{}") for t in context_tokens if t}
    for w, tag in SEMANTIC_ROLE_WORDS.items():
        if w in ctx_set:
            tags.add(tag)
    return frozenset(tags)


def _detect_operator_tags(context_tokens: list[str]) -> frozenset[str]:
    out: set[str] = set()
    ctx = set(t.lower() for t in context_tokens if t)
    if ctx & OPERATOR_MIN_PHRASES or "atleast" in ctx or "at_least" in ctx:
        out.add("min")
    if ctx & OPERATOR_MAX_PHRASES or "atmost" in ctx or "at_most" in ctx or "upto" in ctx:
        out.add("max")
    return frozenset(out)


def _detect_unit_tags(tok: NumTok, context_tokens: list[str]) -> frozenset[str]:
    out: set[str] = set()
    ctx = set(t.lower() for t in context_tokens if t)
    if tok.kind == "percent" or ctx & PERCENT_MARKER_TOKENS:
        out.add("percent_marker")
    if tok.kind == "currency" or "$" in (tok.raw or "") or ctx & CURRENCY_MARKER_TOKENS:
        out.add("currency_marker")
    if tok.kind == "int" or (tok.value is not None and float(int(tok.value)) == tok.value):
        out.add("integer_like")
    if tok.kind == "float" or (tok.value is not None and float(int(tok.value)) != tok.value):
        out.add("decimal_like")
    return frozenset(out)


def _extract_enriched_mentions(query: str, variant: str) -> list[MentionIR]:
    """Stage A: Enriched mention extraction with semantic tagging."""
    toks = query.split()
    sent_tokens = [t.lower().strip(".,;:()[]{}") for t in toks]
    mentions_ir: list[MentionIR] = []
    mention_id = 0
    for i, w in enumerate(toks):
        if w == "<num>" and variant in ("noisy", "nonum"):
            tok = NumTok(raw=w, value=None, kind="unknown")
        else:
            m = NUM_TOKEN_RE.fullmatch(w.strip())
            if not m:
                continue
            ctx_tokens = [
                x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 12) : i + 13]
            ]
            ctx_tokens = [c for c in ctx_tokens if c]
            ctx_set = set(ctx_tokens)
            tok = _parse_num_token(w, ctx_set)

        ctx_tokens = [
            x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 12) : i + 13]
        ]
        ctx_tokens = [c for c in ctx_tokens if c]
        semantic_role_tags = _context_to_semantic_tags(ctx_tokens)
        operator_tags = _detect_operator_tags(ctx_tokens)
        unit_tags = _detect_unit_tags(tok, ctx_tokens)
        polarity = "min" if "min" in operator_tags else ("max" if "max" in operator_tags else "")
        target_entity_tokens = frozenset(
            t for t in ctx_tokens if t in SEMANTIC_ROLE_WORDS or len(t) > 2
        )  # simple heuristic: longer tokens / role words

        mentions_ir.append(
            MentionIR(
                mention_id=mention_id,
                value=tok.value,
                type_bucket=tok.kind,
                raw_surface=tok.raw,
                context_tokens=ctx_tokens,
                sentence_tokens=sent_tokens,
                semantic_role_tags=semantic_role_tags,
                operator_tags=operator_tags,
                unit_tags=unit_tags,
                polarity_or_bound=polarity,
                target_entity_tokens=target_entity_tokens,
                tok=tok,
            )
        )
        mention_id += 1
    return mentions_ir


def _slot_semantic_expansion(param_name: str) -> frozenset[str]:
    """Rule-based slot semantic target tags."""
    n = (param_name or "").lower()
    tags: set[str] = set()
    if "budget" in n:
        tags.update(["budget", "total", "available", "spending_limit", "resource_limit"])
    if "capacity" in n or "limit" in n:
        tags.update(["capacity", "limit", "maximum", "available", "upper_bound"])
    if "demand" in n or "require" in n or "need" in n:
        tags.update(["demand", "requirement", "need", "required"])
    if "profit" in n or "revenue" in n:
        tags.update(["profit", "revenue", "gain", "return"])
    if "cost" in n or "expense" in n or "price" in n:
        tags.update(["cost", "expense", "spend", "price"])
    if "min" in n or "minimum" in n or "atleast" in n:
        tags.update(["minimum", "at_least", "lower_bound"])
    if "max" in n or "maximum" in n or "atmost" in n:
        tags.update(["maximum", "at_most", "upper_bound"])
    if any(w in n for w in ("percent", "percentage", "rate", "fraction", "ratio", "share")):
        tags.update(["percentage", "ratio", "share", "rate", "proportion"])
    if not tags:
        tags.add("quantity")
    return frozenset(tags)


def _build_slot_irs(expected_scalar: list[str]) -> list[SlotIR]:
    """Stage B: Build slot IRs with semantic target tags."""
    slot_irs: list[SlotIR] = []
    for name in expected_scalar:
        et = _expected_type(name)
        aliases = _slot_aliases(name)
        alias_tokens = set()
        for a in aliases:
            alias_tokens.update(_normalize_tokens(a))
        norm_tokens = _normalize_tokens(name)
        semantic_target_tags = _slot_semantic_expansion(name)
        op_pref: set[str] = set()
        if any(x in name.lower() for x in ("min", "minimum", "atleast")):
            op_pref.add("min")
        if any(x in name.lower() for x in ("max", "maximum", "atmost")):
            op_pref.add("max")
        unit_pref: set[str] = set()
        if et == "percent":
            unit_pref.add("percent_marker")
        if et == "currency":
            unit_pref.add("currency_marker")
        if et == "int":
            unit_pref.add("integer_like")
        slot_irs.append(
            SlotIR(
                name=name,
                norm_tokens=norm_tokens,
                expected_type=et,
                alias_tokens=alias_tokens,
                semantic_target_tags=semantic_target_tags,
                operator_preference=frozenset(op_pref),
                unit_preference=frozenset(unit_pref),
            )
        )
    return slot_irs


def _score_mention_slot_ir(m: MentionIR, s: SlotIR) -> tuple[float, dict[str, Any]]:
    """Stage C: Interpretable compatibility score (mention IR vs slot IR)."""
    features: dict[str, Any] = {}
    score = 0.0
    kind = m.type_bucket
    expected = s.expected_type

    if _is_type_incompatible(expected, kind):
        features["type_incompatible"] = True
        return SEMANTIC_IR_WEIGHTS["type_incompatible_penalty"], features

    if kind != "unknown":
        if (expected == "percent" and kind == "percent") or (
            expected == "currency" and kind == "currency"
        ):
            score += SEMANTIC_IR_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected in ("int", "float") and kind in {"int", "currency", "float"}:
            score += SEMANTIC_IR_WEIGHTS["type_loose_bonus"]
            features["type_loose"] = True

    semantic_overlap = len(m.semantic_role_tags & s.semantic_target_tags)
    if semantic_overlap:
        score += SEMANTIC_IR_WEIGHTS["semantic_tag_overlap"] * float(semantic_overlap)
        features["semantic_tag_overlap"] = semantic_overlap

    slot_words = set(s.norm_tokens) | s.alias_tokens
    ctx_set = set(m.context_tokens)
    sent_set = set(m.sentence_tokens)
    ctx_overlap = len(ctx_set & slot_words)
    if ctx_overlap:
        score += SEMANTIC_IR_WEIGHTS["lex_context_overlap"] * float(ctx_overlap)
        features["ctx_overlap"] = ctx_overlap
    sent_overlap = len(sent_set & slot_words)
    if sent_overlap:
        score += SEMANTIC_IR_WEIGHTS["lex_sentence_overlap"] * float(sent_overlap)
        features["sent_overlap"] = sent_overlap

    if m.operator_tags & s.operator_preference:
        if "min" in (m.operator_tags & s.operator_preference):
            score += SEMANTIC_IR_WEIGHTS["operator_min_bonus"]
        else:
            score += SEMANTIC_IR_WEIGHTS["operator_max_bonus"]
        features["operator_match"] = True

    if m.unit_tags & s.unit_preference:
        if "percent_marker" in (m.unit_tags & s.unit_preference):
            score += SEMANTIC_IR_WEIGHTS["unit_percent_bonus"]
        else:
            score += SEMANTIC_IR_WEIGHTS["unit_currency_bonus"]
        features["unit_match"] = True

    entity_overlap = len(m.target_entity_tokens & slot_words)
    if entity_overlap:
        score += SEMANTIC_IR_WEIGHTS["entity_target_bonus"] * float(entity_overlap)
        features["entity_overlap"] = entity_overlap

    if score <= 0.0:
        score += SEMANTIC_IR_WEIGHTS["weak_match_penalty"]
        features["weak_penalty"] = True

    features["total_score"] = score
    return score, features


def _semantic_ir_global_assignment(
    mentions_ir: list[MentionIR],
    slots_ir: list[SlotIR],
) -> tuple[dict[str, MentionIR], dict[str, float], dict[str, list[dict[str, Any]]]]:
    """Stage C: Maximum-weight bipartite matching (deterministic)."""
    assignments: dict[str, MentionIR] = {}
    scores_out: dict[str, float] = {}
    debug: dict[str, list[dict[str, Any]]] = {}

    if not mentions_ir or not slots_ir:
        return assignments, scores_out, debug

    m, s = len(mentions_ir), len(slots_ir)
    cost = [[0.0 for _ in range(s)] for _ in range(m)]
    for i, mr in enumerate(mentions_ir):
        for j, sr in enumerate(slots_ir):
            sc, feats = _score_mention_slot_ir(mr, sr)
            cost[i][j] = -sc if sc > -1e8 else 1e9
            if sr.name not in debug:
                debug[sr.name] = []
            debug[sr.name].append(
                {"mention_id": mr.mention_id, "mention_raw": mr.raw_surface, "score": sc, "features": feats}
            )

    for name in debug:
        debug[name].sort(key=lambda x: x["score"], reverse=True)

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        cost_arr = np.array(cost)
        row_ind, col_ind = linear_sum_assignment(cost_arr)
        for ri, cj in zip(row_ind, col_ind):
            if cost_arr[ri, cj] < 1e8:
                sr = slots_ir[cj]
                mr = mentions_ir[ri]
                sc, _ = _score_mention_slot_ir(mr, sr)
                assignments[sr.name] = mr
                scores_out[sr.name] = sc
    except ImportError:
        from math import inf
        num_states = 1 << s
        dp = [[-inf] * num_states for _ in range(m + 1)]
        parent = [[(-1, None)] * num_states for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(m):
            for mask in range(num_states):
                if dp[i][mask] == -inf:
                    continue
                if dp[i][mask] > dp[i + 1][mask]:
                    dp[i + 1][mask] = dp[i][mask]
                    parent[i + 1][mask] = (mask, None)
                for j in range(s):
                    if (mask >> j) & 1:
                        continue
                    sc = -cost[i][j]
                    if sc <= 0:
                        continue
                    new_mask = mask | (1 << j)
                    val = dp[i][mask] + sc
                    if val > dp[i + 1][new_mask]:
                        dp[i + 1][new_mask] = val
                        parent[i + 1][new_mask] = (mask, j)
        best_mask = max(range(num_states), key=lambda mk: dp[m][mk])
        i, mask = m, best_mask
        used_slot_for_mention = {}
        while i > 0:
            prev_mask, j = parent[i][mask]
            if j is not None:
                used_slot_for_mention[i - 1] = j
            mask = prev_mask
            i -= 1
        for mi, cj in used_slot_for_mention.items():
            sr = slots_ir[cj]
            mr = mentions_ir[mi]
            sc, _ = _score_mention_slot_ir(mr, sr)
            assignments[sr.name] = mr
            scores_out[sr.name] = sc

    return assignments, scores_out, debug


def _validate_slot_assignment(
    slot_name: str,
    m: MentionIR,
    s: SlotIR,
    score: float,
) -> tuple[bool, float]:
    """Stage D: Plausibility check for one assignment. Returns (valid, adjustment)."""
    if _is_type_incompatible(s.expected_type, m.type_bucket):
        return False, REPAIR_WEIGHTS["inconsistency_penalty"]
    adj = 0.0
    if m.semantic_role_tags & s.semantic_target_tags:
        adj += REPAIR_WEIGHTS["validation_bonus"]
    if (m.operator_tags & s.operator_preference) or (m.unit_tags & s.unit_preference):
        adj += REPAIR_WEIGHTS["validation_bonus"]
    if score < 0.5 and not (m.semantic_role_tags & s.semantic_target_tags):
        adj += REPAIR_WEIGHTS["inconsistency_penalty"]
    return adj >= 0 or score > 1.0, adj


def _validation_and_repair(
    mentions_ir: list[MentionIR],
    slots_ir: list[SlotIR],
    initial_assignments: dict[str, MentionIR],
    initial_scores: dict[str, float],
    debug: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, MentionIR], dict[str, str]]:
    """Stage D: Deterministic validation and coverage-preserving repair."""
    filled = dict(initial_assignments)
    filled_in_repair: dict[str, str] = {}  # slot -> "initial" | "repair"

    for slot in filled:
        filled_in_repair[slot] = "initial"

    used_mention_ids = {m.mention_id for m in filled.values()}
    slot_list = list(slots_ir)
    slot_by_name = {s.name: s for s in slot_list}

    for s in slot_list:
        if s.name in filled:
            m = filled[s.name]
            score = initial_scores.get(s.name, 0.0)
            valid, _ = _validate_slot_assignment(s.name, m, s, score)
            if not valid and score < REPAIR_WEIGHTS["min_semantic_support"]:
                del filled[s.name]
                del filled_in_repair[s.name]
                used_mention_ids.discard(m.mention_id)

    unfilled = [s for s in slot_list if s.name not in filled]
    unfilled.sort(key=lambda x: len(x.semantic_target_tags) + len(x.alias_tokens), reverse=True)

    for s in unfilled:
        candidates = debug.get(s.name, [])
        for cand in candidates:
            mid = cand.get("mention_id")
            if mid is None:
                continue
            mr = next((m for m in mentions_ir if m.mention_id == mid), None)
            if not mr or mr.mention_id in used_mention_ids:
                continue
            sc = cand.get("score", 0.0)
            if sc <= 0 or _is_type_incompatible(s.expected_type, mr.type_bucket):
                continue
            valid, adj = _validate_slot_assignment(s.name, mr, s, sc)
            if valid or sc + adj > 0.5:
                filled[s.name] = mr
                filled_in_repair[s.name] = "repair"
                used_mention_ids.add(mr.mention_id)
                break

    return filled, filled_in_repair


def _run_semantic_ir_repair(
    query: str,
    variant: str,
    expected_scalar: list[str],
) -> tuple[dict[str, Any], dict[str, MentionIR], dict[str, str]]:
    """Run full pipeline A→B→C→D. Returns (filled_values, filled_mentions, filled_in_repair)."""
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionIR] = {}
    filled_in_repair: dict[str, str] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, filled_in_repair

    mentions_ir = _extract_enriched_mentions(query, variant)
    slots_ir = _build_slot_irs(expected_scalar)
    if not mentions_ir or not slots_ir:
        return filled_values, filled_mentions, filled_in_repair

    initial_assignments, initial_scores, debug = _semantic_ir_global_assignment(
        mentions_ir, slots_ir
    )
    filled, filled_in_repair = _validation_and_repair(
        mentions_ir, slots_ir, initial_assignments, initial_scores, debug
    )

    for slot_name, m in filled.items():
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions, filled_in_repair


# --- Optimization-role-aware assignment (optimization_role_repair) ---

# Words/phrases that imply optimization roles (word or phrase -> role tag). Modular, easy to extend.
OPT_ROLE_WORDS: dict[str, str] = {}
for _tag, _words in [
    ("objective_coeff", ["maximize", "minimize", "profit", "revenue", "return", "earns", "yields", "gain"]),
    ("unit_profit", ["profit", "revenue", "return", "per", "each", "unit", "yields", "earns"]),
    ("unit_revenue", ["revenue", "return", "per", "each", "unit"]),
    ("unit_return", ["return", "per", "each", "unit"]),
    ("unit_cost", ["cost", "costs", "expense", "spend", "per", "each", "unit", "price"]),
    ("resource_consumption", ["requires", "uses", "consumes", "per", "each", "unit"]),
    ("capacity_limit", ["capacity", "available", "can supply", "at most", "no more than", "limit", "limits"]),
    ("demand_requirement", ["demand", "required", "requirement", "must meet", "must satisfy", "needed", "at least", "minimum"]),
    ("total_budget", ["budget", "total budget", "available budget", "spending limit", "available money", "total amount"]),
    ("total_available", ["available", "total capacity", "hours available", "supply available", "available amount"]),
    ("lower_bound", ["at least", "minimum", "no less than", "least", "lower", "min"]),
    ("upper_bound", ["at most", "maximum", "no more than", "most", "upper", "max", "up to", "limited to", "cannot exceed"]),
    ("ratio_constraint", ["ratio", "fraction", "proportion", "share"]),
    ("percentage_constraint", ["percent", "percentage", "rate", "%", "pct"]),
    ("share_constraint", ["share", "fraction", "ratio", "proportion"]),
    ("fixed_cost", ["fixed", "fixed cost", "fixed charge", "setup", "setup cost"]),
    ("penalty", ["penalty", "penalties"]),
    ("setup_cost", ["setup", "setup cost", "fixed charge"]),
    ("time_requirement", ["time", "hours", "days", "minutes", "weeks"]),
    ("quantity_limit", ["quantity", "amount", "number", "count", "limit"]),
    ("cardinality_limit", ["number", "count", "items", "units", "at most", "at least"]),
    ("minimum_requirement", ["minimum", "at least", "least", "min", "required"]),
    ("maximum_allowance", ["maximum", "at most", "most", "max", "up to", "limited"]),
]:
    for _w in _words:
        OPT_ROLE_WORDS[_w.lower().replace(" ", "_")] = _tag
        OPT_ROLE_WORDS[_w.lower().replace(" ", "")] = _tag
        if " " in _w:
            for _t in _w.lower().split():
                OPT_ROLE_WORDS[_t] = _tag

# Per-unit patterns (token sequences).
PER_UNIT_PATTERNS = ["each", "for each", "per", "per unit", "each unit", "each item", "each product", "for every"]
OBJECTIVE_PATTERNS = ["maximize", "minimize", "profit", "revenue", "return", "cost"]
BOUND_PATTERNS = ["at least", "at most", "no less than", "no more than", "minimum", "maximum", "up to", "limited to", "cannot exceed"]
TOTAL_RESOURCE_PATTERNS = ["total budget", "available budget", "hours available", "total capacity", "available amount", "supply available"]
RATIO_PATTERNS = ["percent", "percentage", "ratio", "fraction", "share", "proportion"]
REQUIREMENT_PATTERNS = ["demand", "required", "requirement", "needed", "must satisfy", "must meet"]

# Unit markers for optimization context.
OPT_UNIT_PERCENT = {"%", "percent", "percentage", "pct"}
OPT_UNIT_CURRENCY = {"$", "€", "dollar", "dollars", "usd", "eur", "cost", "price", "budget"}
OPT_UNIT_COUNT = {"number", "count", "items", "units", "quantity"}
OPT_UNIT_TIME = {"hour", "hours", "day", "days", "minute", "minutes", "week", "weeks", "time"}
OPT_UNIT_DECIMAL = set()  # decimal form inferred from value

# Weights for optimization-role scoring (role overlap stronger than plain lexical).
OPT_ROLE_WEIGHTS = {
    "type_exact_bonus": 4.0,
    "type_loose_bonus": 1.5,
    "type_incompatible_penalty": -1e9,
    "opt_role_overlap": 3.0,
    "fragment_compat_bonus": 1.5,
    "operator_match_bonus": 1.2,
    "lex_context_overlap": 0.5,
    "lex_sentence_overlap": 0.2,
    "unit_match_bonus": 2.0,
    "entity_resource_overlap": 0.8,
    "coefficient_vs_total_bonus": 1.2,
    "weak_match_penalty": -1.0,
    "schema_prior_bonus": 0.5,
}
OPT_REPAIR_WEIGHTS = {
    "role_plausibility_bonus": 0.6,
    "total_vs_coeff_penalty": -1.5,
    "bound_plausibility_bonus": 0.5,
    "coverage_repair_bonus": 1.2,
    "min_role_support": 0.4,
}


@dataclass(frozen=True)
class MentionOptIR:
    """Optimization-role intermediate representation for a numeric mention."""

    mention_id: int
    value: float | None
    type_bucket: str
    raw_surface: str
    role_tags: frozenset[str]
    operator_tags: frozenset[str]
    unit_tags: frozenset[str]
    fragment_type: str  # "objective" | "constraint" | "resource" | "ratio" | "bound" | ""
    is_per_unit: bool
    is_total_like: bool
    nearby_entity_tokens: frozenset[str]
    nearby_resource_tokens: frozenset[str]
    nearby_product_tokens: frozenset[str]
    context_tokens: list[str]
    sentence_tokens: list[str]
    tok: NumTok


@dataclass(frozen=True)
class SlotOptIR:
    """Slot with optimization-role priors for matching."""

    name: str
    norm_tokens: list[str]
    expected_type: str
    alias_tokens: set[str]
    slot_role_tags: frozenset[str]
    operator_preference: frozenset[str]
    unit_preference: frozenset[str]
    is_objective_like: bool
    is_bound_like: bool
    is_total_like: bool
    is_coefficient_like: bool


def _context_to_opt_role_tags(context_tokens: list[str]) -> frozenset[str]:
    """Infer optimization-role tags from context (rule-based)."""
    tags: set[str] = set()
    ctx_lower = [t.lower().strip(".,;:()[]{}") for t in context_tokens if t]
    ctx_set = set(ctx_lower)
    ctx_str = " ".join(ctx_lower)
    for w, tag in OPT_ROLE_WORDS.items():
        if w in ctx_set or w.replace("_", " ") in ctx_str or w in ctx_str:
            tags.add(tag)
    return frozenset(tags)


def _classify_fragment_type(context_tokens: list[str]) -> str:
    """Classify mention context as objective-like, constraint-like, resource-like, ratio-like, or bound-like."""
    ctx = set(t.lower() for t in context_tokens if t)
    ctx_str = " ".join(t.lower() for t in context_tokens if t)
    if ctx & {"maximize", "minimize", "profit", "revenue", "return", "earns", "yields"} or any(
        p in ctx_str for p in OBJECTIVE_PATTERNS
    ):
        return "objective"
    if ctx & {"at most", "at least", "cannot exceed", "limited", "must be", "no more", "no less"} or any(
        p in ctx_str for p in BOUND_PATTERNS
    ):
        return "constraint"
    if ctx & {"available", "capacity", "hours available", "budget", "supply"} or any(
        p in ctx_str for p in TOTAL_RESOURCE_PATTERNS
    ):
        return "resource"
    if ctx & {"percent", "percentage", "ratio", "fraction", "share", "proportion", "%"} or any(
        p in ctx_str for p in RATIO_PATTERNS
    ):
        return "ratio"
    if ctx & {"at least", "at most", "minimum", "maximum", "lower", "upper", "min", "max"}:
        return "bound"
    return ""


def _detect_opt_unit_tags(tok: NumTok, context_tokens: list[str]) -> frozenset[str]:
    """Unit markers for optimization: percent_marker, currency_marker, count_marker, time_marker, decimal_marker."""
    out: set[str] = set()
    ctx = set(t.lower() for t in context_tokens if t)
    if tok.kind == "percent" or ctx & OPT_UNIT_PERCENT or (tok.raw or "").endswith("%"):
        out.add("percent_marker")
    if tok.kind == "currency" or "$" in (tok.raw or "") or ctx & OPT_UNIT_CURRENCY:
        out.add("currency_marker")
    if ctx & OPT_UNIT_COUNT or (tok.kind == "int" and tok.value is not None):
        out.add("count_marker")
    if ctx & OPT_UNIT_TIME:
        out.add("time_marker")
    if tok.kind == "float" or (tok.value is not None and float(int(tok.value)) != tok.value):
        out.add("decimal_marker")
    return frozenset(out)


def _extract_opt_role_mentions(query: str, variant: str) -> list[MentionOptIR]:
    """Stage 1: Optimization-aware mention extraction with role tags."""
    toks = query.split()
    sent_tokens = [t.lower().strip(".,;:()[]{}") for t in toks]
    mentions: list[MentionOptIR] = []
    mention_id = 0
    for i, w in enumerate(toks):
        if w == "<num>" and variant in ("noisy", "nonum"):
            tok = NumTok(raw=w, value=None, kind="unknown")
        else:
            m = NUM_TOKEN_RE.fullmatch(w.strip())
            if not m:
                continue
            ctx_tokens = [
                x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 14) : i + 15]
            ]
            ctx_tokens = [c for c in ctx_tokens if c]
            tok = _parse_num_token(w, set(ctx_tokens))

        ctx_tokens = [
            x.lower().strip(".,;:()[]{}") for x in toks[max(0, i - 14) : i + 15]
        ]
        ctx_tokens = [c for c in ctx_tokens if c]
        ctx_set = set(ctx_tokens)
        ctx_str = " ".join(ctx_tokens)
        role_tags = _context_to_opt_role_tags(ctx_tokens)
        operator_tags = _detect_operator_tags(ctx_tokens)
        unit_tags = _detect_opt_unit_tags(tok, ctx_tokens)
        fragment_type = _classify_fragment_type(ctx_tokens)
        is_per_unit = any(
            p.replace(" ", "") in ctx_str.replace(" ", "") or p in ctx_str
            for p in ["each", "per unit", "per item", "for each", "per"]
        )
        is_total_like = bool(
            ctx_set & {"total", "budget", "available", "capacity", "limit", "amount"}
            or "total " in ctx_str or "available " in ctx_str
        )
        entity_tokens = frozenset(t for t in ctx_tokens if len(t) > 2 and t in OPT_ROLE_WORDS)
        resource_tokens = frozenset(
            t for t in ctx_tokens if t in {"capacity", "budget", "available", "limit", "resource", "hours", "time"}
        )
        product_tokens = frozenset(
            t for t in ctx_tokens if t in {"item", "product", "unit", "each", "demand", "quantity"}
        )

        mentions.append(
            MentionOptIR(
                mention_id=mention_id,
                value=tok.value,
                type_bucket=tok.kind,
                raw_surface=tok.raw,
                role_tags=role_tags,
                operator_tags=operator_tags,
                unit_tags=unit_tags,
                fragment_type=fragment_type,
                is_per_unit=is_per_unit,
                is_total_like=is_total_like,
                nearby_entity_tokens=entity_tokens,
                nearby_resource_tokens=resource_tokens,
                nearby_product_tokens=product_tokens,
                context_tokens=ctx_tokens,
                sentence_tokens=sent_tokens,
                tok=tok,
            )
        )
        mention_id += 1
    return mentions


def _slot_opt_role_expansion(param_name: str) -> frozenset[str]:
    """Slot name -> optimization-role tags (schema-side priors)."""
    n = (param_name or "").lower()
    tags: set[str] = set()
    if "budget" in n:
        tags.update(["total_budget", "upper_bound", "capacity_limit"])
    if "capacity" in n or "limit" in n:
        tags.update(["capacity_limit", "upper_bound", "total_available"])
    if "demand" in n or "require" in n or "need" in n:
        tags.update(["demand_requirement", "lower_bound", "minimum_requirement"])
    if "profit" in n or "revenue" in n or "return" in n:
        tags.update(["objective_coeff", "unit_profit", "unit_revenue", "unit_return"])
    if "cost" in n or "expense" in n or "price" in n:
        if "total" in n or "budget" in n:
            tags.update(["total_budget", "unit_cost"])
        else:
            tags.update(["unit_cost", "objective_coeff"])
    if "percent" in n or "ratio" in n or "fraction" in n or "share" in n or "rate" in n:
        tags.update(["ratio_constraint", "percentage_constraint", "share_constraint"])
    if "min" in n or "minimum" in n or "atleast" in n:
        tags.update(["lower_bound", "minimum_requirement", "demand_requirement"])
    if "max" in n or "maximum" in n or "atmost" in n:
        tags.update(["upper_bound", "maximum_allowance", "capacity_limit"])
    if "penalty" in n or "setup" in n or "fixed" in n:
        tags.update(["penalty", "fixed_cost", "setup_cost"])
    if "time" in n or "hour" in n or "day" in n:
        tags.update(["time_requirement", "resource_consumption"])
    if "number" in n or "count" in n or "item" in n or "quantity" in n:
        tags.update(["quantity_limit", "cardinality_limit"])
    if not tags:
        tags.add("quantity_limit")
    return frozenset(tags)


def _build_slot_opt_irs(expected_scalar: list[str]) -> list[SlotOptIR]:
    """Stage 3: Build slot IRs with optimization-role priors."""
    slot_irs: list[SlotOptIR] = []
    for name in expected_scalar:
        et = _expected_type(name)
        aliases = _slot_aliases(name)
        alias_tokens_flat: set[str] = set()
        for a in aliases:
            alias_tokens_flat.update(_normalize_tokens(a))
        norm_tokens = _normalize_tokens(name)
        slot_role_tags = _slot_opt_role_expansion(name)
        op_pref: set[str] = set()
        if any(x in name.lower() for x in ("min", "minimum", "atleast")):
            op_pref.add("min")
        if any(x in name.lower() for x in ("max", "maximum", "atmost")):
            op_pref.add("max")
        unit_pref: set[str] = set()
        if et == "percent":
            unit_pref.add("percent_marker")
        if et == "currency":
            unit_pref.add("currency_marker")
        if et == "int":
            unit_pref.add("count_marker")
        is_objective = bool(slot_role_tags & {"objective_coeff", "unit_profit", "unit_revenue", "unit_return", "unit_cost"})
        is_bound = bool(slot_role_tags & {"lower_bound", "upper_bound", "capacity_limit", "demand_requirement", "minimum_requirement", "maximum_allowance"})
        is_total = bool(slot_role_tags & {"total_budget", "total_available"}) or "budget" in name.lower() or "total" in name.lower()
        is_coeff = bool(slot_role_tags & {"unit_cost", "unit_profit", "unit_revenue", "resource_consumption"})

        slot_irs.append(
            SlotOptIR(
                name=name,
                norm_tokens=norm_tokens,
                expected_type=et,
                alias_tokens=alias_tokens_flat,
                slot_role_tags=slot_role_tags,
                operator_preference=frozenset(op_pref),
                unit_preference=frozenset(unit_pref),
                is_objective_like=is_objective,
                is_bound_like=is_bound,
                is_total_like=is_total,
                is_coefficient_like=is_coeff,
            )
        )
    return slot_irs


def _score_mention_slot_opt(m: MentionOptIR, s: SlotOptIR) -> tuple[float, dict[str, Any]]:
    """Stage 5: Optimization-role compatibility score."""
    features: dict[str, Any] = {}
    score = 0.0
    kind = m.type_bucket
    expected = s.expected_type

    if _is_type_incompatible(expected, kind):
        features["type_incompatible"] = True
        return OPT_ROLE_WEIGHTS["type_incompatible_penalty"], features

    if kind != "unknown":
        if (expected == "percent" and kind == "percent") or (expected == "currency" and kind == "currency"):
            score += OPT_ROLE_WEIGHTS["type_exact_bonus"]
            features["type_exact"] = True
        elif expected in ("int", "float") and kind in {"int", "currency", "float"}:
            score += OPT_ROLE_WEIGHTS["type_loose_bonus"]
            features["type_loose"] = True

    role_overlap = len(m.role_tags & s.slot_role_tags)
    if role_overlap:
        score += OPT_ROLE_WEIGHTS["opt_role_overlap"] * float(role_overlap)
        features["opt_role_overlap"] = role_overlap

    if m.fragment_type == "objective" and s.is_objective_like:
        score += OPT_ROLE_WEIGHTS["fragment_compat_bonus"]
        features["fragment_objective"] = True
    if m.fragment_type in ("constraint", "bound") and s.is_bound_like:
        score += OPT_ROLE_WEIGHTS["fragment_compat_bonus"]
        features["fragment_bound"] = True
    if m.fragment_type == "resource" and s.is_total_like:
        score += OPT_ROLE_WEIGHTS["fragment_compat_bonus"]
        features["fragment_resource"] = True
    if m.fragment_type == "ratio" and ("ratio_constraint" in s.slot_role_tags or "percentage_constraint" in s.slot_role_tags):
        score += OPT_ROLE_WEIGHTS["fragment_compat_bonus"]
        features["fragment_ratio"] = True

    if m.operator_tags & s.operator_preference:
        score += OPT_ROLE_WEIGHTS["operator_match_bonus"]
        features["operator_match"] = True

    slot_words = set(s.norm_tokens) | s.alias_tokens
    ctx_set = set(m.context_tokens)
    sent_set = set(m.sentence_tokens)
    ctx_overlap = len(ctx_set & slot_words)
    if ctx_overlap:
        score += OPT_ROLE_WEIGHTS["lex_context_overlap"] * float(ctx_overlap)
        features["ctx_overlap"] = ctx_overlap
    sent_overlap = len(sent_set & slot_words)
    if sent_overlap:
        score += OPT_ROLE_WEIGHTS["lex_sentence_overlap"] * float(sent_overlap)
        features["sent_overlap"] = sent_overlap

    if m.unit_tags & s.unit_preference:
        score += OPT_ROLE_WEIGHTS["unit_match_bonus"]
        features["unit_match"] = True

    entity_resource = len(
        (m.nearby_entity_tokens | m.nearby_resource_tokens | m.nearby_product_tokens) & slot_words
    )
    if entity_resource:
        score += OPT_ROLE_WEIGHTS["entity_resource_overlap"] * float(entity_resource)
        features["entity_resource_overlap"] = entity_resource

    if s.is_total_like and m.is_total_like:
        score += OPT_ROLE_WEIGHTS["coefficient_vs_total_bonus"]
        features["total_match"] = True
    if s.is_coefficient_like and m.is_per_unit:
        score += OPT_ROLE_WEIGHTS["coefficient_vs_total_bonus"]
        features["coefficient_match"] = True

    score += OPT_ROLE_WEIGHTS["schema_prior_bonus"]
    features["schema_prior"] = True

    if score <= 0.0:
        score += OPT_ROLE_WEIGHTS["weak_match_penalty"]
        features["weak_penalty"] = True

    features["total_score"] = score
    return score, features


def _opt_role_global_assignment(
    mentions: list[MentionOptIR],
    slots: list[SlotOptIR],
) -> tuple[dict[str, MentionOptIR], dict[str, float], dict[str, list[dict[str, Any]]]]:
    """Stage 5: Maximum-weight bipartite matching for optimization-role assignment."""
    assignments: dict[str, MentionOptIR] = {}
    scores_out: dict[str, float] = {}
    debug: dict[str, list[dict[str, Any]]] = {}

    if not mentions or not slots:
        return assignments, scores_out, debug

    m, s = len(mentions), len(slots)
    cost = [[0.0 for _ in range(s)] for _ in range(m)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, feats = _score_mention_slot_opt(mr, sr)
            cost[i][j] = -sc if sc > -1e8 else 1e9
            if sr.name not in debug:
                debug[sr.name] = []
            debug[sr.name].append(
                {"mention_id": mr.mention_id, "mention_raw": mr.raw_surface, "score": sc, "features": feats}
            )

    for name in debug:
        debug[name].sort(key=lambda x: x["score"], reverse=True)

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        cost_arr = np.array(cost)
        row_ind, col_ind = linear_sum_assignment(cost_arr)
        for ri, cj in zip(row_ind, col_ind):
            if cost_arr[ri, cj] < 1e8:
                sr = slots[cj]
                mr = mentions[ri]
                sc, _ = _score_mention_slot_opt(mr, sr)
                assignments[sr.name] = mr
                scores_out[sr.name] = sc
    except ImportError:
        from math import inf
        num_states = 1 << s
        dp = [[-inf] * num_states for _ in range(m + 1)]
        parent = [[(-1, None)] * num_states for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(m):
            for mask in range(num_states):
                if dp[i][mask] == -inf:
                    continue
                if dp[i][mask] > dp[i + 1][mask]:
                    dp[i + 1][mask] = dp[i][mask]
                    parent[i + 1][mask] = (mask, None)
                for j in range(s):
                    if (mask >> j) & 1:
                        continue
                    sc = -cost[i][j]
                    if sc <= 0:
                        continue
                    new_mask = mask | (1 << j)
                    val = dp[i][mask] + sc
                    if val > dp[i + 1][new_mask]:
                        dp[i + 1][new_mask] = val
                        parent[i + 1][new_mask] = (mask, j)
        best_mask = max(range(num_states), key=lambda mk: dp[m][mk])
        i, mask = m, best_mask
        used_slot_for_mention = {}
        while i > 0:
            prev_mask, j = parent[i][mask]
            if j is not None:
                used_slot_for_mention[i - 1] = j
            mask = prev_mask
            i -= 1
        for mi, cj in used_slot_for_mention.items():
            sr = slots[cj]
            mr = mentions[mi]
            sc, _ = _score_mention_slot_opt(mr, sr)
            assignments[sr.name] = mr
            scores_out[sr.name] = sc

    return assignments, scores_out, debug


def _opt_role_validate_one(slot_name: str, m: MentionOptIR, s: SlotOptIR, score: float) -> tuple[bool, float]:
    """Stage 6: Plausibility check using optimization logic."""
    if _is_type_incompatible(s.expected_type, m.type_bucket):
        return False, OPT_REPAIR_WEIGHTS["total_vs_coeff_penalty"]
    adj = 0.0
    if m.role_tags & s.slot_role_tags:
        adj += OPT_REPAIR_WEIGHTS["role_plausibility_bonus"]
    if s.is_bound_like:
        if ("min" in s.operator_preference and "min" in m.operator_tags) or ("max" in s.operator_preference and "max" in m.operator_tags):
            adj += OPT_REPAIR_WEIGHTS["bound_plausibility_bonus"]
    if s.is_total_like and not m.is_total_like and m.is_per_unit:
        adj += OPT_REPAIR_WEIGHTS["total_vs_coeff_penalty"]
    if s.is_coefficient_like and m.is_total_like and not m.is_per_unit:
        adj += OPT_REPAIR_WEIGHTS["total_vs_coeff_penalty"] * 0.5
    return adj >= -0.5 or score > 1.0, adj


def _opt_role_validate_and_repair(
    mentions: list[MentionOptIR],
    slots: list[SlotOptIR],
    initial_assignments: dict[str, MentionOptIR],
    initial_scores: dict[str, float],
    debug: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, MentionOptIR], dict[str, str]]:
    """Stage 6: Optimization-aware validation and coverage-preserving repair."""
    filled = dict(initial_assignments)
    filled_in_repair: dict[str, str] = {}

    for slot in filled:
        filled_in_repair[slot] = "initial"

    used_mention_ids = {m.mention_id for m in filled.values()}
    slot_list = list(slots)

    for s in slot_list:
        if s.name not in filled:
            continue
        m = filled[s.name]
        score = initial_scores.get(s.name, 0.0)
        valid, _ = _opt_role_validate_one(s.name, m, s, score)
        if not valid and score < OPT_REPAIR_WEIGHTS["min_role_support"]:
            del filled[s.name]
            if s.name in filled_in_repair:
                del filled_in_repair[s.name]
            used_mention_ids.discard(m.mention_id)

    unfilled = [s for s in slot_list if s.name not in filled]
    unfilled.sort(key=lambda x: len(x.slot_role_tags) + len(x.alias_tokens), reverse=True)

    for s in unfilled:
        for cand in debug.get(s.name, []):
            mid = cand.get("mention_id")
            if mid is None:
                continue
            mr = next((x for x in mentions if x.mention_id == mid), None)
            if not mr or mr.mention_id in used_mention_ids:
                continue
            sc = cand.get("score", 0.0)
            if sc <= 0 or _is_type_incompatible(s.expected_type, mr.type_bucket):
                continue
            valid, adj = _opt_role_validate_one(s.name, mr, s, sc)
            if valid or sc + adj > OPT_REPAIR_WEIGHTS["min_role_support"]:
                filled[s.name] = mr
                filled_in_repair[s.name] = "repair"
                used_mention_ids.add(mr.mention_id)
                break

    return filled, filled_in_repair


def _run_optimization_role_repair(
    query: str,
    variant: str,
    expected_scalar: list[str],
) -> tuple[dict[str, Any], dict[str, MentionOptIR], dict[str, str]]:
    """Run full optimization-role pipeline. Returns (filled_values, filled_mentions, filled_in_repair)."""
    filled_values: dict[str, Any] = {}
    filled_mentions: dict[str, MentionOptIR] = {}
    filled_in_repair: dict[str, str] = {}

    if not expected_scalar:
        return filled_values, filled_mentions, filled_in_repair

    mentions = _extract_opt_role_mentions(query, variant)
    slots = _build_slot_opt_irs(expected_scalar)
    if not mentions or not slots:
        return filled_values, filled_mentions, filled_in_repair

    initial_assignments, initial_scores, debug = _opt_role_global_assignment(mentions, slots)
    filled, filled_in_repair = _opt_role_validate_and_repair(
        mentions, slots, initial_assignments, initial_scores, debug
    )

    for slot_name, m in filled.items():
        filled_values[slot_name] = m.tok.value if m.tok.value is not None else m.tok.raw
        filled_mentions[slot_name] = m
    return filled_values, filled_mentions, filled_in_repair


def _constrained_assignment(
    mentions: list[MentionRecord],
    slots: list[SlotRecord],
) -> tuple[dict[str, MentionRecord], dict[str, list[dict[str, Any]]]]:
    """
    Deterministic global assignment: at most one mention per slot, one slot per mention.

    Uses simple DP over subsets of slots (sufficient for small numbers of scalar params).
    Returns:
      - mapping slot_name -> assigned MentionRecord (if any)
      - debug info: slot_name -> list of candidate dicts with scores/features
    """
    assignments: dict[str, MentionRecord] = {}
    debug: dict[str, list[dict[str, Any]]] = {}

    if not mentions or not slots:
        return assignments, debug

    m = len(mentions)
    s = len(slots)

    # Precompute scores and feature breakdowns.
    scores: list[list[float]] = [[0.0 for _ in range(s)] for _ in range(m)]
    feat_mat: list[list[dict[str, Any]]] = [[{} for _ in range(s)] for _ in range(m)]
    for i, mr in enumerate(mentions):
        for j, sr in enumerate(slots):
            sc, feats = _score_mention_slot(mr, sr)
            scores[i][j] = sc
            feat_mat[i][j] = feats

    # Build debug info: per-slot candidate list sorted by score.
    for j, sr in enumerate(slots):
        cand: list[dict[str, Any]] = []
        for i, mr in enumerate(mentions):
            cand.append(
                {
                    "mention_index": mr.index,
                    "mention_raw": mr.tok.raw,
                    "mention_kind": mr.tok.kind,
                    "score": scores[i][j],
                    "features": feat_mat[i][j],
                }
            )
        cand.sort(key=lambda x: x["score"], reverse=True)
        debug[sr.name] = cand

    # Dynamic programming over subsets of slots.
    # This is exponential in number of slots; fallback to greedy if too many.
    if s > 15:
        used_mentions: set[int] = set()
        for j, sr in enumerate(slots):
            best_i = None
            best_score = float("-inf")
            for i in range(m):
                if i in used_mentions:
                    continue
                sc = scores[i][j]
                if sc > best_score:
                    best_score = sc
                    best_i = i
            if best_i is not None and best_score > 0.0:
                assignments[slots[j].name] = mentions[best_i]
                used_mentions.add(best_i)
        return assignments, debug

    from math import inf

    num_states = 1 << s
    dp = [[-inf for _ in range(num_states)] for _ in range(m + 1)]
    parent: list[list[tuple[int, int | None]]] = [
        [(-1, None) for _ in range(num_states)] for _ in range(m + 1)
    ]
    dp[0][0] = 0.0

    for i in range(m):
        for mask in range(num_states):
            cur = dp[i][mask]
            if cur == -inf:
                continue
            # Option 1: skip mention i.
            if cur > dp[i + 1][mask]:
                dp[i + 1][mask] = cur
                parent[i + 1][mask] = (mask, None)
            # Option 2: assign mention i to an unused slot j.
            for j in range(s):
                if (mask >> j) & 1:
                    continue
                sc = scores[i][j]
                if sc <= 0.0:
                    continue
                new_mask = mask | (1 << j)
                new_val = cur + sc
                if new_val > dp[i + 1][new_mask]:
                    dp[i + 1][new_mask] = new_val
                    parent[i + 1][new_mask] = (mask, j)

    # Choose best final mask.
    best_mask = 0
    best_val = -inf
    for mask in range(num_states):
        if dp[m][mask] > best_val:
            best_val = dp[m][mask]
            best_mask = mask

    # Backtrack.
    i = m
    mask = best_mask
    used_slot_for_mention: dict[int, int] = {}
    while i > 0:
        prev_mask, chosen_j = parent[i][mask]
        if chosen_j is not None:
            used_slot_for_mention[i - 1] = chosen_j
        mask = prev_mask
        i -= 1

    for mi, sj in used_slot_for_mention.items():
        sr = slots[sj]
        assignments[sr.name] = mentions[mi]

    return assignments, debug


def _rel_err(pred: float, gold: float) -> float:
    return abs(pred - gold) / max(1.0, abs(gold))


def _md5_seed(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _upsert_summary_row(summary_path: Path, row: dict) -> None:
    cols = [
        "variant",
        "baseline",
        "schema_R1",
        "param_coverage",
        "type_match",
        "exact5_on_hits",
        "exact20_on_hits",
        "param_coverage_hits",
        "param_coverage_miss",
        "type_match_hits",
        "type_match_miss",
        "key_overlap",
        "key_overlap_hits",
        "key_overlap_miss",
        "instantiation_ready",
        "n",
    ]
    rows: list[dict] = []
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for rr in r:
                rows.append(rr)
    d = {(r.get("variant"), r.get("baseline")): r for r in rows}
    d[(row["variant"], row["baseline"])] = {k: row.get(k, "") for k in cols}

    # deterministic order
    ordered = sorted(d.values(), key=lambda x: (x.get("variant", ""), x.get("baseline", "")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ordered)


def _upsert_types_rows(types_path: Path, types_agg: dict[str, dict]) -> None:
    cols = [
        "variant",
        "baseline",
        "param_type",
        "n_expected",
        "n_filled",
        "param_coverage",
        "type_match",
        "exact5_on_hits",
        "exact20_on_hits",
        "n_queries",
    ]
    rows: list[dict] = []
    if types_path.exists():
        with open(types_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for rr in r:
                rows.append(rr)
    d = {(r.get("variant"), r.get("baseline"), r.get("param_type")): r for r in rows}
    for t, info in types_agg.items():
        key = (info["variant"], info["baseline"], info["param_type"])
        d[key] = {k: info.get(k, "") for k in cols}

    ordered = sorted(d.values(), key=lambda x: (x.get("variant", ""), x.get("baseline", ""), x.get("param_type", "")))
    types_path.parent.mkdir(parents=True, exist_ok=True)
    with open(types_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ordered)


# --- Acceptance reranking: schema acceptance scoring + optional hierarchy ---
# Weights for interpretable acceptance scoring (additive).
ACCEPTANCE_RERANK_WEIGHTS = {
    "type_coverage_bonus": 1.5,
    "type_missing_penalty": -2.0,
    "role_coverage_bonus": 1.2,
    "operator_compat_bonus": 0.8,
    "fillability_bonus": 1.0,
    "family_consistency_bonus": 1.5,
    "family_mismatch_penalty": -1.5,
    "missing_critical_penalty": -2.5,
    "weak_fit_penalty": -0.8,
}

# Coarse optimization families (rule-based from schema name + slot names).
FAMILY_KEYWORDS: dict[str, list[str]] = {
    "allocation": ["budget", "investment", "allocate", "portfolio", "dollar", "condo", "detached"],
    "production": ["produce", "sandwich", "ingredient", "blend", "production", "recipe"],
    "transportation": ["transport", "flow", "route", "ship", "delivery", "cable", "gold"],
    "scheduling": ["schedule", "assignment", "assign", "worker", "wage", "radio", "ad"],
    "packing": ["pack", "knapsack", "selection", "item", "weight", "capacity"],
    "covering": ["cover", "facility", "location", "warehouse", "customer"],
    "network": ["network", "graph", "path", "edge", "node"],
    "inventory": ["inventory", "supply", "capacity", "available", "stock"],
    "ratio": ["ratio", "fraction", "percentage", "percent", "minimum percentage", "ratio"],
    "generic_lp": ["maximize", "minimize", "subject", "constraint", "linear"],
}


def _schema_family(schema_id: str, schema_text: str, slot_names: list[str]) -> str:
    """Assign coarse optimization family from schema id, description, and slot names."""
    combined = " ".join([schema_id, schema_text] + slot_names).lower()
    best = "generic_lp"
    best_count = 0
    for family, keywords in FAMILY_KEYWORDS.items():
        if family == "generic_lp":
            continue
        c = sum(1 for kw in keywords if kw in combined)
        if c > best_count:
            best_count = c
            best = family
    return best


def _schema_subgroup(schema_id: str, slot_names: list[str]) -> str:
    """Optional subgroup from slot name patterns."""
    slot_str = " ".join(slot_names).lower()
    if any(x in slot_str for x in ["totalbudget", "total budget", "profitper", "profit per"]):
        return "total_budget_per_unit_profit"
    if any(x in slot_str for x in ["capacity", "demand", "available", "required"]):
        return "capacity_demand"
    if any(x in slot_str for x in ["minimum", "maximum", "min", "max", "at least", "at most"]):
        return "min_max_bounds"
    if any(x in slot_str for x in ["ratio", "percentage", "fraction", "percent"]):
        return "ratio_fraction"
    if any(x in slot_str for x in ["fixed", "penalty", "cost"]):
        return "fixed_cost_penalty"
    if any(x in slot_str for x in ["time", "hour", "day", "week", "wage"]):
        return "time_resource"
    if any(x in slot_str for x in ["num", "count", "number", "quantity"]):
        return "item_count"
    return ""


def _query_family_hints(query: str) -> set[str]:
    """Infer likely optimization family from query text."""
    q = query.lower()
    hints: set[str] = set()
    for family, keywords in FAMILY_KEYWORDS.items():
        if family == "generic_lp":
            continue
        if any(kw in q for kw in keywords):
            hints.add(family)
    if not hints:
        hints.add("generic_lp")
    return hints


def _query_subgroup_hints(query: str) -> set[str]:
    """Infer subgroup signals from query."""
    q = query.lower()
    hints: set[str] = set()
    if any(w in q for w in ["budget", "total", "investment", "allocate"]):
        hints.add("total_budget_per_unit_profit")
    if any(w in q for w in ["capacity", "demand", "available", "required", "requirement"]):
        hints.add("capacity_demand")
    if any(w in q for w in ["minimum", "maximum", "at least", "at most", "no less", "no more"]):
        hints.add("min_max_bounds")
    if any(w in q for w in ["ratio", "percentage", "fraction", "percent", "share"]):
        hints.add("ratio_fraction")
    if any(w in q for w in ["fixed", "penalty", "cost"]):
        hints.add("fixed_cost_penalty")
    if any(w in q for w in ["hour", "day", "week", "wage", "time"]):
        hints.add("time_resource")
    if any(w in q for w in ["number of", "count", "quantity", "each type"]):
        hints.add("item_count")
    return hints


def _extract_query_acceptance_features(query: str, variant: str) -> dict[str, Any]:
    """Extract optimization-aware evidence from query for acceptance scoring."""
    toks = query.split()
    q_lower = query.lower()
    tokens_lower = [t.lower().strip(".,;:()[]{}") for t in toks]
    token_set = set(tokens_lower)
    num_toks = _extract_num_tokens(query, variant)
    # Type counts
    type_counts = {"percent": 0, "currency": 0, "int": 0, "float": 0, "unknown": 0}
    for nt in num_toks:
        type_counts[nt.kind] = type_counts.get(nt.kind, 0) + 1
    # Role evidence (cue words present)
    role_evidence = {}
    for role, words in [
        ("budget", ["budget", "total budget", "investment", "available amount"]),
        ("cost", ["cost", "expense", "spend", "price"]),
        ("profit", ["profit", "revenue", "return", "gain"]),
        ("demand", ["demand", "requirement", "required", "needed"]),
        ("capacity", ["capacity", "limit", "maximum", "available"]),
        ("min_max", ["minimum", "maximum", "at least", "at most", "lower", "upper"]),
        ("ratio", ["ratio", "percentage", "fraction", "share", "percent"]),
        ("fixed_penalty", ["fixed", "penalty"]),
        ("time", ["time", "hour", "day", "week", "wage"]),
        ("quantity", ["quantity", "count", "number", "amount"]),
    ]:
        role_evidence[role] = 1 if any(w in q_lower for w in words) else 0
    # Operator evidence
    operator_evidence = {
        "min_like": 1 if (token_set & OPERATOR_MIN_WORDS) else 0,
        "max_like": 1 if (token_set & OPERATOR_MAX_WORDS) else 0,
        "per_unit": 1 if any(p in q_lower for p in ["per ", "each ", "per unit", "for each"]) else 0,
        "total_like": 1 if any(w in token_set for w in ["total", "available", "overall"]) else 0,
    }
    # Structural (objective/constraint/resource language)
    objective_words = {"maximize", "minimize", "maximise", "minimise", "objective", "maximize total", "minimize total"}
    constraint_words = {"subject", "constraint", "must not exceed", "at least", "at most", "cannot exceed"}
    resource_words = {"budget", "capacity", "available", "limit", "resource"}
    structural = {
        "objective_like": 1 if any(w in q_lower for w in objective_words) else 0,
        "constraint_like": 1 if any(w in q_lower for w in constraint_words) else 0,
        "resource_like": 1 if any(w in q_lower for w in resource_words) else 0,
    }
    return {
        "type_counts": type_counts,
        "role_evidence": role_evidence,
        "operator_evidence": operator_evidence,
        "structural": structural,
        "n_numeric": len(num_toks),
        "cue_words": token_set & CUE_WORDS,
    }


def _get_expected_scalar_for_schema(schema_id: str, gold_by_id: dict[str, dict]) -> list[str]:
    """Return list of scalar parameter names for a schema (for acceptance profile)."""
    gold = gold_by_id.get(schema_id) or {}
    pred_params = gold.get("parameters") or {}
    pred_info = gold.get("problem_info") or {}
    expected_params: list[str] = []
    if isinstance(pred_info, dict) and isinstance(pred_info.get("parameters"), dict):
        expected_params = list(pred_info["parameters"].keys())
    elif isinstance(pred_params, dict):
        expected_params = list(pred_params.keys())
    gold_params = gold.get("parameters") or {}
    scalar_keys = [p for p in expected_params if _is_scalar(gold_params.get(p))]
    return scalar_keys


def _build_schema_acceptance_profile(
    schema_id: str,
    gold_entry: dict,
    catalog_entry: dict,
    gold_by_id: dict[str, dict],
) -> dict[str, Any]:
    """Build acceptance profile for a schema: what evidence it expects."""
    slot_names = _get_expected_scalar_for_schema(schema_id, gold_by_id)
    schema_text = (catalog_entry.get("description") or catalog_entry.get("name") or "").lower()
    # Slot/type expectations
    type_expectations = {"percent": 0, "currency": 0, "int": 0, "float": 0}
    expects_percent = False
    expects_currency = False
    expects_count = False
    for p in slot_names:
        et = _expected_type(p)
        if et == "percent":
            type_expectations["percent"] += 1
            expects_percent = True
        elif et == "currency":
            type_expectations["currency"] += 1
            expects_currency = True
        elif et == "int":
            type_expectations["int"] += 1
            expects_count = True
        else:
            type_expectations["float"] += 1
    # Role expectations from slot names
    slot_str = " ".join(slot_names).lower()
    role_expectations = {
        "budget": any(x in slot_str for x in ["budget", "total", "investment"]),
        "demand": any(x in slot_str for x in ["demand", "required", "requirement"]),
        "capacity": any(x in slot_str for x in ["capacity", "available", "limit"]),
        "objective_coeff": any(x in slot_str for x in ["profit", "cost", "revenue", "per", "wage"]),
        "ratio": any(x in slot_str for x in ["ratio", "percent", "fraction", "percentage"]),
        "fixed_penalty": any(x in slot_str for x in ["fixed", "penalty"]),
        "time": any(x in slot_str for x in ["time", "wage", "hour", "day"]),
    }
    # Structural
    structural = {
        "objective_like": "maximize" in schema_text or "minimize" in schema_text,
        "bound_heavy": sum(1 for s in slot_names if "min" in s.lower() or "max" in s.lower()) >= 2,
        "total_budget_like": "budget" in schema_text or "total" in schema_text,
    }
    family = _schema_family(schema_id, schema_text, slot_names)
    subgroup = _schema_subgroup(schema_id, slot_names)
    return {
        "schema_id": schema_id,
        "slot_names": slot_names,
        "type_expectations": type_expectations,
        "expects_percent": expects_percent,
        "expects_currency": expects_currency,
        "expects_count": expects_count,
        "role_expectations": role_expectations,
        "structural": structural,
        "family": family,
        "subgroup": subgroup,
    }


def _acceptance_score(
    query_features: dict[str, Any],
    schema_profile: dict[str, Any],
    query_family_hints: set[str],
    query_subgroup_hints: set[str],
) -> tuple[float, dict[str, Any]]:
    """Compute additive acceptance score and debug breakdown."""
    w = ACCEPTANCE_RERANK_WEIGHTS
    score = 0.0
    debug: dict[str, Any] = {}
    q_types = query_features.get("type_counts") or {}
    s_types = schema_profile.get("type_expectations") or {}
    # Type coverage
    for kind in ("percent", "currency", "int", "float"):
        q_has = (q_types.get(kind) or 0) > 0
        s_expects = (s_types.get(kind) or 0) > 0
        if s_expects and q_has:
            score += w["type_coverage_bonus"]
            debug[f"type_{kind}_ok"] = True
        elif s_expects and not q_has:
            score += w["type_missing_penalty"]
            debug[f"type_{kind}_missing"] = True
    if schema_profile.get("expects_percent") and not (q_types.get("percent") or 0):
        score += w["type_missing_penalty"]
    if schema_profile.get("expects_currency") and not (q_types.get("currency") or 0):
        score += w["type_missing_penalty"]
    # Role coverage
    q_roles = query_features.get("role_evidence") or {}
    s_roles = schema_profile.get("role_expectations") or {}
    for role in s_roles:
        if s_roles.get(role) and q_roles.get(role):
            score += w["role_coverage_bonus"]
            debug[f"role_{role}"] = True
    # Operator compatibility
    q_op = query_features.get("operator_evidence") or {}
    if q_op.get("min_like") and ("min" in str(schema_profile.get("slot_names", [])).lower() or "minimum" in str(schema_profile.get("slot_names", [])).lower()):
        score += w["operator_compat_bonus"]
        debug["op_min"] = True
    if q_op.get("max_like") and ("max" in str(schema_profile.get("slot_names", [])).lower() or "maximum" in str(schema_profile.get("slot_names", [])).lower()):
        score += w["operator_compat_bonus"]
        debug["op_max"] = True
    # Fillability (approximate: we have n_numeric and n slots)
    n_slots = len(schema_profile.get("slot_names") or [])
    n_num = query_features.get("n_numeric") or 0
    if n_slots > 0 and n_num >= n_slots * 0.5:
        score += w["fillability_bonus"]
        debug["fillability_ok"] = True
    elif n_slots > 0 and n_num == 0:
        score += w["missing_critical_penalty"]
        debug["fillability_poor"] = True
    # Family/subgroup consistency
    s_family = schema_profile.get("family") or "generic_lp"
    s_subgroup = schema_profile.get("subgroup") or ""
    if s_family in query_family_hints:
        score += w["family_consistency_bonus"]
        debug["family_match"] = True
    elif query_family_hints and "generic_lp" not in query_family_hints and s_family not in query_family_hints:
        score += w["family_mismatch_penalty"]
        debug["family_mismatch"] = True
    if s_subgroup and s_subgroup in query_subgroup_hints:
        score += w["family_consistency_bonus"] * 0.5
        debug["subgroup_match"] = True
    # Weak fit: only lexical support
    if score <= 0 and n_slots > 0:
        score += w["weak_fit_penalty"]
        debug["weak_fit"] = True
    debug["total_acceptance"] = score
    return score, debug


def make_rerank_rank_fn(
    base_rank_fn,
    gold_by_id: dict[str, dict],
    catalog: list[dict],
    k_retrieval: int = 10,
    use_hierarchy: bool = False,
    variant: str = "orig",
) -> Any:
    """Return a rank_fn(query, top_k) that gets top-k from base, reranks by acceptance, returns top_k."""

    id_to_catalog: dict[str, dict] = {p.get("id"): p for p in catalog if p.get("id")}
    profile_cache: dict[str, dict] = {}

    def get_profile(schema_id: str) -> dict[str, Any]:
        if schema_id in profile_cache:
            return profile_cache[schema_id]
        gold_entry = gold_by_id.get(schema_id) or {}
        catalog_entry = id_to_catalog.get(schema_id) or {"id": schema_id, "name": schema_id, "description": ""}
        profile = _build_schema_acceptance_profile(schema_id, gold_entry, catalog_entry, gold_by_id)
        profile_cache[schema_id] = profile
        return profile

    def rank_fn(query: str, top_k: int) -> list[tuple[str, float]]:
        # Stage A: get top-k candidates from base retriever
        candidates = base_rank_fn(query, top_k=k_retrieval)
        if not candidates:
            return []
        query_features = _extract_query_acceptance_features(query, variant)
        q_family = _query_family_hints(query)
        q_subgroup = _query_subgroup_hints(query)
        # Normalize retrieval scores to [0,1] for combination
        raw_scores = [s for _, s in candidates]
        min_s, max_s = min(raw_scores), max(raw_scores)
        norm = (max_s - min_s) or 1.0
        scored: list[tuple[str, float, float, dict]] = []
        for pid, ret_score in candidates:
            profile = get_profile(pid)
            accept_score, debug = _acceptance_score(
                query_features, profile, q_family, q_subgroup
            )
            norm_ret = (ret_score - min_s) / norm if norm else 0.0
            # Combine: retrieval weight 0.5, acceptance weight 0.5, optional family bonus already in accept_score
            final = 0.5 * norm_ret + 0.5 * max(0.0, (accept_score + 5.0) / 10.0)  # shift accept to non-negative scale
            if use_hierarchy and profile.get("family") not in q_family and "generic_lp" not in q_family:
                final -= 0.2  # extra penalty for hierarchy mismatch
            scored.append((pid, final, accept_score, debug))
        scored.sort(key=lambda x: -x[1])
        return [(pid, score) for pid, score, _, _ in scored[:top_k]]

    return rank_fn


def run_setting(
    variant: str,
    baseline_name: str,
    eval_items: list[dict],
    gold_by_id: dict[str, dict],
    rank_fn,
    doc_ids: list[str],
    random_control: bool,
    assignment_mode: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_path = out_dir / f"nlp4lp_downstream_per_query_{variant}_{baseline_name}.csv"
    json_path = out_dir / f"nlp4lp_downstream_{variant}_{baseline_name}.json"
    summary_path = out_dir / "nlp4lp_downstream_summary.csv"

    def run_one(label: str, mode: str) -> tuple[list[dict], dict, dict]:
        rows: list[dict] = []
        hit_flags: list[int] = []
        cov_vals: list[float] = []
        type_vals: list[float] = []
        exact5_vals: list[float] = []
        exact20_vals: list[float] = []
        inst_ready_flags: list[int] = []
        cov_hits: list[float] = []
        cov_miss: list[float] = []
        type_hits: list[float] = []
        type_miss: list[float] = []
        ko_all: list[float] = []
        ko_hits: list[float] = []
        ko_miss: list[float] = []

        type_names = ["percent", "integer", "currency", "float"]
        type_expected_total = {t: 0 for t in type_names}
        type_filled_total = {t: 0 for t in type_names}
        type_correct_total = {t: 0 for t in type_names}
        type_exact5_num = {t: 0 for t in type_names}
        type_exact5_den = {t: 0 for t in type_names}
        type_exact20_num = {t: 0 for t in type_names}
        type_exact20_den = {t: 0 for t in type_names}

        for ex in eval_items:
            qid = ex["query_id"]
            query = ex["query"]
            gold_id = ex["relevant_doc_id"]
            if mode == "oracle":
                pred_id = gold_id
            elif mode == "random":
                rng = random.Random(_md5_seed(qid))
                pred_id = doc_ids[rng.randrange(len(doc_ids))] if doc_ids else ""
            else:  # retrieval
                ranked = rank_fn(query, top_k=1)
                pred_id = ranked[0][0] if ranked else ""
            schema_hit = 1 if pred_id == gold_id else 0

            # Gold problem (for evaluation) and predicted problem (for schema)
            gold = gold_by_id.get(gold_id) or {}
            gold_params = gold.get("parameters") or {}
            pred = gold_by_id.get(pred_id) or {}
            pred_params = pred.get("parameters") or {}
            pred_info = pred.get("problem_info") or {}

            expected_params: list[str] = []
            if isinstance(pred_info, dict) and isinstance(pred_info.get("parameters"), dict):
                expected_params = list(pred_info["parameters"].keys())
            elif isinstance(pred_params, dict):
                expected_params = list(pred_params.keys())

            # scalar keys based on gold values
            gold_scalar_keys = {p for p, v in (gold_params or {}).items() if _is_scalar(v)}

            def _bucket_type(pname: str) -> str:
                et = _expected_type(pname)
                if et == "percent":
                    return "percent"
                if et == "int":
                    return "integer"
                if et == "currency":
                    return "currency"
                return "float"

            # count expected scalar params per type from gold schema
            for p in gold_scalar_keys:
                t = _bucket_type(p)
                type_expected_total[t] += 1

            pred_scalar_keys = {
                p for p in expected_params if _is_scalar(gold_params.get(p))
            } if isinstance(gold_params, dict) else set()
            # expected scalar params (for filling) are intersection of predicted keys with scalar gold
            expected_scalar = list(pred_scalar_keys)
            n_expected_scalar = len(expected_scalar)
            # schema key overlap (relative to gold scalar keys)
            if gold_scalar_keys:
                key_overlap = len(pred_scalar_keys & gold_scalar_keys) / float(len(gold_scalar_keys))
            else:
                key_overlap = 0.0

            num_toks = _extract_num_tokens(query, variant)
            candidates = list(num_toks)

            filled = {}
            type_matches = 0
            n_filled = 0
            comparable_errs = []

            # per-query per-type counts (optional, used for per-query CSV enrichment if desired later)
            type_expected_q = {t: 0 for t in type_names}
            type_filled_q = {t: 0 for t in type_names}
            type_correct_q = {t: 0 for t in type_names}

            for p in gold_scalar_keys:
                t = _bucket_type(p)
                type_expected_q[t] += 1

            if assignment_mode == "optimization_role_repair" and expected_scalar:
                # Optimization-role-aware assignment.
                filled_values, filled_mentions, _ = _run_optimization_role_repair(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if tok.kind == et:
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "semantic_ir_repair" and expected_scalar:
                # Semantic IR + validation-and-repair assignment.
                filled_values, filled_mentions, _ = _run_semantic_ir_repair(
                    query, variant, expected_scalar
                )
                for p in expected_scalar:
                    if p not in filled_values:
                        continue
                    m_ir = filled_mentions.get(p)
                    tok = m_ir.tok if m_ir else None
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = filled_values[p]
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if tok.kind == et:
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            elif assignment_mode == "constrained" and expected_scalar:
                # Global constrained assignment over mention-slot pairs.
                mention_records = _extract_num_mentions(query, variant)
                slot_records = _build_slot_records(expected_scalar)
                constrained_assignments, _debug = _constrained_assignment(
                    mention_records, slot_records
                )
                # Apply assignments.
                for p in expected_scalar:
                    sr_type = _bucket_type(p)
                    mrec = constrained_assignments.get(p)
                    if mrec is None:
                        continue
                    tok = mrec.tok
                    if tok is None:
                        continue
                    n_filled += 1
                    filled[p] = tok.value if tok.value is not None else tok.raw
                    btype = sr_type
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    et = _expected_type(p)
                    if tok.kind == et:
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1
            else:
                # Original greedy typed / untyped assignment.
                for p in expected_scalar:
                    et = _expected_type(p)
                    if assignment_mode == "untyped":
                        idx, tok = (0, candidates[0]) if candidates else (None, None)
                    else:
                        idx, tok = _choose_token(et, candidates)
                    if tok is None:
                        continue
                    # remove used token
                    if idx is not None and 0 <= idx < len(candidates):
                        candidates.pop(idx)
                    n_filled += 1
                    filled[p] = tok.value if tok.value is not None else tok.raw
                    btype = _bucket_type(p)
                    type_filled_total[btype] += 1
                    type_filled_q[btype] += 1
                    if tok.kind == et:
                        type_matches += 1
                        type_correct_total[btype] += 1
                        type_correct_q[btype] += 1
                    # error only if numeric and schema hit
                    if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                        gold_val = float(gold_params[p])
                        err = _rel_err(float(tok.value), gold_val)
                        comparable_errs.append(err)
                        if btype in type_names:
                            type_exact5_den[btype] += 1
                            type_exact20_den[btype] += 1
                            if err <= 0.05:
                                type_exact5_num[btype] += 1
                            if err <= 0.20:
                                type_exact20_num[btype] += 1

            param_coverage = (n_filled / max(1, n_expected_scalar)) if n_expected_scalar else 0.0
            type_match = (type_matches / max(1, n_filled)) if n_filled else 0.0
            ko_all.append(key_overlap)

            exact5 = ""
            exact20 = ""
            if schema_hit:
                if comparable_errs:
                    exact5 = sum(1 for e in comparable_errs if e <= 0.05) / len(comparable_errs)
                    exact20 = sum(1 for e in comparable_errs if e <= 0.20) / len(comparable_errs)
                else:
                    exact5 = ""
                    exact20 = ""

            rows.append(
                {
                    "query_id": qid,
                    "variant": variant,
                    "baseline": label,
                    "predicted_doc_id": pred_id,
                    "gold_doc_id": gold_id,
                    "schema_hit": schema_hit,
                    "n_expected_scalar": n_expected_scalar,
                    "n_filled": n_filled,
                    "param_coverage": param_coverage,
                    "type_match": type_match,
                    "exact5": exact5,
                    "exact20": exact20,
                    "key_overlap": key_overlap,
                }
            )

            hit_flags.append(schema_hit)
            cov_vals.append(param_coverage)
            type_vals.append(type_match)
            if schema_hit:
                cov_hits.append(param_coverage)
                type_hits.append(type_match)
                ko_hits.append(key_overlap)
            else:
                cov_miss.append(param_coverage)
                type_miss.append(type_match)
                ko_miss.append(key_overlap)
            if isinstance(exact5, float):
                exact5_vals.append(exact5)
            if isinstance(exact20, float):
                exact20_vals.append(exact20)
            inst_ready_flags.append(1 if (param_coverage >= 0.8 and type_match >= 0.8) else 0)

        n = len(rows)
        def _mean(xs: list[float]) -> float | str:
            return (sum(xs) / len(xs)) if xs else ""

        agg = {
            "variant": variant,
            "baseline": label,
            "schema_R1": sum(hit_flags) / n if n else 0.0,
            "param_coverage": sum(cov_vals) / n if n else 0.0,
            "type_match": sum(type_vals) / n if n else 0.0,
            "exact5_on_hits": (sum(exact5_vals) / len(exact5_vals)) if exact5_vals else "",
            "exact20_on_hits": (sum(exact20_vals) / len(exact20_vals)) if exact20_vals else "",
            "param_coverage_hits": _mean(cov_hits),
            "param_coverage_miss": _mean(cov_miss),
            "type_match_hits": _mean(type_hits),
            "type_match_miss": _mean(type_miss),
            "key_overlap": _mean(ko_all),
            "key_overlap_hits": _mean(ko_hits),
            "key_overlap_miss": _mean(ko_miss),
            "instantiation_ready": sum(inst_ready_flags) / n if n else 0.0,
            "n": n,
        }
        # per-type aggregate summary for this (variant, baseline)
        types_agg: dict[str, dict] = {}
        n_queries = len(rows)
        for t in type_names:
            n_exp_t = type_expected_total[t]
            n_fill_t = type_filled_total[t]
            cov_t = (n_fill_t / max(1, n_exp_t)) if n_exp_t else 0.0
            tm_t = (
                (type_correct_total[t] / n_fill_t) if n_fill_t else ""
            )
            e5 = (
                (type_exact5_num[t] / type_exact5_den[t]) if type_exact5_den[t] else ""
            )
            e20 = (
                (type_exact20_num[t] / type_exact20_den[t]) if type_exact20_den[t] else ""
            )
            types_agg[t] = {
                "variant": variant,
                "baseline": label,
                "param_type": t,
                "n_expected": n_exp_t,
                "n_filled": n_fill_t,
                "param_coverage": cov_t,
                "type_match": tm_t,
                "exact5_on_hits": e5,
                "exact20_on_hits": e20,
                "n_queries": n_queries,
            }

        return rows, agg, types_agg

    rows_main, agg_main, types_main = run_one(baseline_name, "oracle" if baseline_name.startswith("oracle") else "retrieval")

    cols = [
        "query_id",
        "variant",
        "baseline",
        "predicted_doc_id",
        "gold_doc_id",
        "schema_hit",
        "n_expected_scalar",
        "n_filled",
        "param_coverage",
        "type_match",
        "exact5",
        "exact20",
        "key_overlap",
    ]
    with open(per_query_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_main)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {"variant": variant, "baseline": baseline_name, "k": 1, "random_control": random_control},
                "aggregate": agg_main,
            },
            f,
            indent=2,
        )

    _upsert_summary_row(summary_path, agg_main)

    # Update per-type summary
    types_summary_path = out_dir / "nlp4lp_downstream_types_summary.csv"
    _upsert_types_rows(types_summary_path, types_main)

    if random_control:
        rand_label = "random_untyped" if baseline_name.endswith("_untyped") else "random"
        _, agg_rand, types_rand = run_one(rand_label, "random")
        _upsert_summary_row(summary_path, agg_rand)
        _upsert_types_rows(types_summary_path, types_rand)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="NLP4LP downstream utility demo (retrieval -> parameter instantiation)")
    p.add_argument("--variant", type=str, default="orig", choices=("orig", "noisy", "short"))
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    p.add_argument("--eval", type=Path, default=None)
    p.add_argument(
        "--baseline",
        type=str,
        default="tfidf",
        help="One of: bm25, tfidf, lsa, oracle; or tfidf_acceptance_rerank, tfidf_hierarchical_acceptance_rerank, bm25_acceptance_rerank, bm25_hierarchical_acceptance_rerank",
    )
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--acceptance-k", type=int, default=10, help="Top-k retrieval candidates for acceptance reranking (default 10)")
    p.add_argument("--random-control", action="store_true")
    p.add_argument(
        "--assignment-mode",
        type=str,
        default="typed",
        choices=("typed", "untyped", "constrained", "semantic_ir_repair", "optimization_role_repair"),
    )
    args = p.parse_args()

    if args.k != 1:
        raise SystemExit("This demo currently supports --k 1 only (top-1 schema selection).")

    eval_path = args.eval or (ROOT / "data" / "processed" / f"nlp4lp_eval_{args.variant}.jsonl")
    eval_items = _load_eval(Path(eval_path))
    if not eval_items:
        raise SystemExit(f"No eval instances loaded from {eval_path}")

    # Gold schema/parameters from HF
    gold_by_id = _load_hf_gold(split="test")

    catalog, _id_to_text = _load_catalog_as_problems(args.catalog)
    doc_ids = [p["id"] for p in catalog if p.get("id")]

    rank_fn = None
    effective_baseline = args.baseline
    if args.baseline == "oracle":
        rank_fn = None
    elif args.baseline in ("tfidf_acceptance_rerank", "tfidf_hierarchical_acceptance_rerank"):
        from retrieval.baselines import get_baseline
        base = get_baseline("tfidf")
        base.fit(catalog)
        rank_fn = make_rerank_rank_fn(
            base.rank,
            gold_by_id,
            catalog,
            k_retrieval=args.acceptance_k,
            use_hierarchy=(args.baseline == "tfidf_hierarchical_acceptance_rerank"),
            variant=args.variant,
        )
    elif args.baseline in ("bm25_acceptance_rerank", "bm25_hierarchical_acceptance_rerank"):
        from retrieval.baselines import get_baseline
        base = get_baseline("bm25")
        base.fit(catalog)
        rank_fn = make_rerank_rank_fn(
            base.rank,
            gold_by_id,
            catalog,
            k_retrieval=args.acceptance_k,
            use_hierarchy=(args.baseline == "bm25_hierarchical_acceptance_rerank"),
            variant=args.variant,
        )
    else:
        if args.baseline not in ("bm25", "tfidf", "lsa"):
            raise SystemExit(f"Unknown baseline: {args.baseline}. Use bm25, tfidf, lsa, oracle, or *_acceptance_rerank / *_hierarchical_acceptance_rerank.")
        from retrieval.baselines import get_baseline
        baseline = get_baseline(args.baseline)
        baseline.fit(catalog)
        rank_fn = baseline.rank
    if args.assignment_mode == "untyped":
        effective_baseline = f"{args.baseline}_untyped"
    elif args.assignment_mode == "constrained":
        effective_baseline = f"{args.baseline}_constrained"
    elif args.assignment_mode == "semantic_ir_repair":
        effective_baseline = f"{args.baseline}_semantic_ir_repair"
    elif args.assignment_mode == "optimization_role_repair":
        effective_baseline = f"{args.baseline}_optimization_role_repair"

    out_dir = ROOT / "results" / "paper"
    run_setting(
        variant=args.variant,
        baseline_name=effective_baseline,
        eval_items=eval_items,
        gold_by_id=gold_by_id,
        rank_fn=rank_fn,
        doc_ids=doc_ids,
        random_control=bool(args.random_control),
        assignment_mode=args.assignment_mode,
        out_dir=out_dir,
    )

    print(f"Wrote {out_dir / f'nlp4lp_downstream_{args.variant}_{effective_baseline}.json'}")
    print(f"Wrote {out_dir / f'nlp4lp_downstream_per_query_{args.variant}_{effective_baseline}.csv'}")
    print(f"Updated {out_dir / 'nlp4lp_downstream_summary.csv'}")


if __name__ == "__main__":
    main()

