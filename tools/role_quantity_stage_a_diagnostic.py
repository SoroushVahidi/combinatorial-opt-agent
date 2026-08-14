#!/usr/bin/env python3
"""Stage-A diagnostic for role/quantity signal in typed-greedy failures.

This script is intentionally read-only with respect to the production method:
it replays the current typed-greedy path, emits per-slot diagnostics, and asks
whether deterministic role/quantity features separate the gold candidate from
the currently selected wrong candidate. It does not change grounding behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.baselines import get_baseline
from tools import nlp4lp_downstream_utility as u


OUT_DIR = ROOT / "results" / "role_quantity_stage_a"
DEFAULT_GOLD_CACHE = ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"

PUNCT = ".,;:()[]{}\"'"
MONEY_WORDS = {"$", "dollar", "dollars", "usd", "budget", "cost", "costs", "price", "prices", "pay", "paid"}
PERCENT_WORDS = {"%", "percent", "percentage", "fraction", "ratio", "share"}
PER_UNIT_WORDS = {
    "each", "per", "every", "requires", "require", "uses", "use", "takes", "take",
    "costs", "cost", "earns", "earn", "yields", "yield", "contains", "contain",
    "produces", "produce", "provides", "provide", "consumes", "consume",
}
TOTAL_WORDS = {
    "total", "overall", "aggregate", "available", "capacity", "limit", "budget",
    "supply", "stock", "stockpile", "on-hand", "remaining",
}
DEMAND_WORDS = {"demand", "demands", "required", "requirement", "requirements", "needed", "need", "needs"}
COST_WORDS = {"cost", "costs", "expense", "expenses", "price", "prices", "pay", "paid", "wage", "salary"}
PROFIT_WORDS = {"profit", "profits", "revenue", "revenues", "return", "returns", "earn", "earns", "yield", "yields"}
COUNT_NOUNS = {
    "type", "types", "kind", "kinds", "product", "products", "item", "items",
    "machine", "machines", "worker", "workers", "resource", "resources",
    "ingredient", "ingredients", "vehicle", "vehicles", "task", "tasks",
}
LOWER_PATTERNS = (
    "at least", "no less than", "not less than", "no fewer than", "not fewer than",
    "minimum", "min", "lower", "greater than or equal", "more than",
)
UPPER_PATTERNS = (
    "at most", "no more than", "not more than", "cannot exceed", "maximum", "max",
    "upper", "less than or equal", "fewer than", "less than", "capacity",
)
RATE_WORDS = {"rate", "rates", "per", "hour", "hours", "minute", "minutes", "day", "days"}
ENTITY_STOP = (
    MONEY_WORDS | PERCENT_WORDS | PER_UNIT_WORDS | TOTAL_WORDS | DEMAND_WORDS
    | COST_WORDS | PROFIT_WORDS | COUNT_NOUNS | RATE_WORDS
    | {"and", "or", "the", "a", "an", "of", "to", "in", "for", "with", "must", "can", "be", "is", "are"}
)


@dataclass(frozen=True)
class MentionDiag:
    mention_id: int
    raw: str
    value: float | None
    kind: str
    token_start: int
    token_end: int
    sentence_index: int
    clause_index: int
    nearby_tokens: tuple[str, ...]
    left_tokens: tuple[str, ...]
    right_tokens: tuple[str, ...]
    quantity_forms: frozenset[str]
    roles: frozenset[str]
    entity_anchor: str


@dataclass(frozen=True)
class SlotMeta:
    name: str
    expected_type: str
    expected_quantity_forms: frozenset[str]
    expected_roles: frozenset[str]
    entity_tokens: frozenset[str]


def _clean(tok: str) -> str:
    return tok.lower().strip(PUNCT)


def _rel_close(a: Any, b: Any, tol: float = 0.20) -> bool:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return False
    return abs(af - bf) / max(1.0, abs(bf)) <= tol


def _sentence_clause_indices(tokens: list[str]) -> tuple[list[int], list[int]]:
    sent: list[int] = []
    clause: list[int] = []
    s_idx = 0
    c_idx = 0
    for tok in tokens:
        sent.append(s_idx)
        clause.append(c_idx)
        stripped = tok.rstrip()
        if stripped.endswith((".", "!", "?")):
            s_idx += 1
            c_idx = 0
        elif stripped.endswith((",", ";", ":")) or _clean(tok) in {"and", "but", "while", "whereas"}:
            c_idx += 1
    return sent, clause


def _context(tokens: list[str], start: int, end: int, left: int = 6, right: int = 8) -> tuple[list[str], list[str], list[str]]:
    left_raw: list[str] = []
    for tok in reversed(tokens[max(0, start - left):start]):
        stripped = tok.rstrip()
        if stripped.endswith((".", "!", "?")):
            break
        left_raw.append(tok)
    left_raw.reverse()
    right_raw: list[str] = []
    for tok in tokens[end:min(len(tokens), end + right)]:
        right_raw.append(tok)
        if tok.rstrip().endswith((".", "!", "?")):
            break
    left_tokens = [_clean(t) for t in left_raw if _clean(t)]
    right_tokens = [_clean(t) for t in right_raw if _clean(t)]
    nearby = left_tokens + [_clean(t) for t in tokens[start:end] if _clean(t)] + right_tokens
    return nearby, left_tokens, right_tokens


def _phrase_in(text: str, patterns: tuple[str, ...]) -> bool:
    return any(p in text for p in patterns)


def _nearest_entity(left_tokens: list[str], right_tokens: list[str]) -> str:
    for tok in reversed(left_tokens[-4:]):
        if tok and tok not in ENTITY_STOP and not re.fullmatch(r"\d+(\.\d+)?", tok):
            return tok
    for tok in right_tokens[:4]:
        if tok and tok not in ENTITY_STOP and not re.fullmatch(r"\d+(\.\d+)?", tok):
            return tok
    return ""


def _mention_features(tok: u.NumTok, nearby: list[str], left: list[str], right: list[str]) -> tuple[frozenset[str], frozenset[str], str]:
    forms: set[str] = set()
    roles: set[str] = set()
    ctx = set(nearby)
    left_str = " ".join(left)
    right_str = " ".join(right)
    ctx_str = " ".join(nearby)

    if tok.kind == "percent" or ctx & PERCENT_WORDS:
        forms.add("percent")
        roles.add("rate")
    if tok.kind == "currency" or "$" in (tok.raw or "") or ctx & MONEY_WORDS:
        forms.add("currency")
    if tok.kind == "int" and ctx & COUNT_NOUNS:
        forms.add("count")
        roles.add("cardinality")
    if ctx & RATE_WORDS or " per " in f" {ctx_str} ":
        forms.add("rate")
        roles.add("rate")
    if ctx & PER_UNIT_WORDS or " per " in f" {left_str} {right_str} ":
        forms.add("per_unit")
        roles.add("constraint_coefficient")
    if ctx & TOTAL_WORDS or "in total" in ctx_str:
        forms.add("total")
        roles.add("rhs_capacity")
    if ctx & DEMAND_WORDS:
        forms.add("demand")
        roles.add("rhs_capacity")
    if ctx & COST_WORDS:
        forms.add("cost")
        roles.add("objective_coefficient")
    if ctx & PROFIT_WORDS:
        forms.add("profit")
        roles.add("objective_coefficient")
    if _phrase_in(ctx_str, LOWER_PATTERNS):
        forms.add("bound")
        roles.add("lower_bound")
    if _phrase_in(ctx_str, UPPER_PATTERNS):
        forms.add("bound")
        roles.add("upper_bound")
    if not forms:
        forms.add("generic_scalar")
    if not roles:
        roles.add("unknown")
    return frozenset(forms), frozenset(roles), _nearest_entity(left, right)


def extract_mention_diags(query: str, variant: str) -> list[MentionDiag]:
    """Extract typed-greedy numeric mentions with positions and diagnostic features."""
    tokens = query.split()
    sent_idx, clause_idx = _sentence_clause_indices(tokens)
    mentions: list[MentionDiag] = []
    mention_id = 0
    i = 0
    while i < len(tokens):
        raw = tokens[i]
        span = 1
        tok: u.NumTok | None = None
        if raw == "<num>" and variant in ("noisy", "nonum"):
            tok = u.NumTok(raw=raw, value=None, kind="unknown")
        elif u.NUM_TOKEN_RE.fullmatch(raw.strip().rstrip(",;:()[]{}").rstrip(".")):
            nearby, _, _ = _context(tokens, i, i + 1)
            tok = u._parse_num_token(raw, set(nearby))
        else:
            frac = u._WORD_FRACTIONS.get(_clean(raw))
            if frac is not None:
                tok = u.NumTok(raw=_clean(raw), value=frac, kind="percent")
            else:
                value, consumed = u._parse_word_num_span(tokens, i)
                if value is not None:
                    span = consumed
                    j = i + consumed
                    nearby, _, _ = _context(tokens, i, j)
                    tok = u._classify_word_num_tok(" ".join(tokens[i:j]), value, set(nearby), tokens, j)
        if tok is None:
            i += 1
            continue
        nearby, left, right = _context(tokens, i, i + span)
        forms, roles, entity = _mention_features(tok, nearby, left, right)
        mentions.append(
            MentionDiag(
                mention_id=mention_id,
                raw=tok.raw,
                value=tok.value,
                kind=tok.kind,
                token_start=i,
                token_end=i + span,
                sentence_index=sent_idx[i] if i < len(sent_idx) else 0,
                clause_index=clause_idx[i] if i < len(clause_idx) else 0,
                nearby_tokens=tuple(nearby),
                left_tokens=tuple(left),
                right_tokens=tuple(right),
                quantity_forms=forms,
                roles=roles,
                entity_anchor=entity,
            )
        )
        mention_id += 1
        i += span
    return mentions


def slot_metadata(slot_name: str) -> SlotMeta:
    n = slot_name.lower()
    expected_type = u._expected_type(slot_name)
    forms: set[str] = set()
    roles: set[str] = set()
    if expected_type == "percent" or any(x in n for x in ("percent", "percentage", "ratio", "fraction", "share")):
        forms.add("percent")
        roles.add("rate")
    if expected_type == "currency" or any(x in n for x in ("budget", "cost", "price", "profit", "revenue", "wage", "salary")):
        forms.add("currency")
    if u._is_count_like_slot(slot_name):
        forms.add("count")
        roles.add("cardinality")
    if any(x in n for x in ("per", "each", "rate")):
        forms.add("per_unit")
        forms.add("rate")
        roles.add("constraint_coefficient")
    if any(x in n for x in ("profit", "revenue", "return", "cost", "price", "wage")) and not any(x in n for x in ("total", "budget", "available", "capacity")):
        roles.add("objective_coefficient")
    if any(x in n for x in ("total", "budget", "available", "availability", "capacity", "limit", "supply", "max")):
        forms.add("total")
        roles.add("rhs_capacity")
    if any(x in n for x in ("demand", "require", "need", "minimum")):
        forms.add("demand")
        roles.add("rhs_capacity")
    if any(x in n for x in ("cost", "price", "wage", "salary")):
        forms.add("cost")
    if any(x in n for x in ("profit", "revenue", "return")):
        forms.add("profit")
    if any(x in n for x in ("min", "minimum", "lower", "atleast")):
        forms.add("bound")
        roles.add("lower_bound")
    if any(x in n for x in ("max", "maximum", "upper", "atmost")):
        forms.add("bound")
        roles.add("upper_bound")
    if not forms:
        forms.add("generic_scalar")
    if not roles:
        roles.add("unknown")
    entity_tokens = {
        t for t in u._split_camel_case(slot_name)
        if t not in ENTITY_STOP and len(t) > 1 and not t.isdigit()
    }
    return SlotMeta(slot_name, expected_type, frozenset(forms), frozenset(roles), frozenset(entity_tokens))


def choose_preference(expected: str, tok: u.NumTok) -> tuple[int, float, str]:
    val = tok.value if tok.value is not None else 0.0
    absval = abs(val)
    if expected == "percent":
        pref = 2 if tok.kind == "percent" else (1 if tok.value is not None and 0.0 < tok.value <= 1.0 else 0)
    elif expected == "int":
        pref = 2 if tok.kind == "int" else (1 if tok.value is not None and float(int(val)) == val else 0)
    elif expected == "currency":
        pref = 2 if tok.kind == "currency" else (1 if tok.kind in {"int", "float"} else 0)
    else:
        pref = 2 if tok.kind in {"float", "int"} else (1 if tok.kind == "currency" else 0)
    return pref, absval, tok.raw


def _diag_for_tok(diags: list[MentionDiag], tok: u.NumTok, used_ids: set[int]) -> MentionDiag | None:
    for d in diags:
        if d.mention_id in used_ids:
            continue
        if d.raw == tok.raw and d.kind == tok.kind and (d.value == tok.value or (d.value is not None and tok.value is not None and abs(d.value - tok.value) < 1e-12)):
            return d
    for d in diags:
        if d.mention_id not in used_ids and d.value == tok.value and d.kind == tok.kind:
            return d
    return None


def compatibility_score(d: MentionDiag, s: SlotMeta) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0
    form_overlap = sorted(d.quantity_forms & s.expected_quantity_forms)
    role_overlap = sorted(d.roles & s.expected_roles)
    entity_overlap = sorted(set(d.nearby_tokens) & set(s.entity_tokens))
    if form_overlap:
        score += 2 * len(form_overlap)
        reasons.extend(f"form:{x}" for x in form_overlap)
    if role_overlap:
        score += 2 * len(role_overlap)
        reasons.extend(f"role:{x}" for x in role_overlap)
    if entity_overlap:
        score += len(entity_overlap)
        reasons.extend(f"entity:{x}" for x in entity_overlap)
    if u._is_type_match(s.expected_type, d.kind):
        score += 1
        reasons.append("type_match")
    return score, reasons


def classify_case(slot: SlotMeta, chosen: MentionDiag | None, gold: MentionDiag | None) -> str:
    forms = set(slot.expected_quantity_forms)
    roles = set(slot.expected_roles)
    if "per_unit" in forms or "total" in forms:
        if chosen and gold and (("per_unit" in chosen.quantity_forms) != ("per_unit" in gold.quantity_forms) or ("total" in chosen.quantity_forms) != ("total" in gold.quantity_forms)):
            return "total_perunit"
    if "lower_bound" in roles or "upper_bound" in roles:
        return "bound_role"
    if "objective_coefficient" in roles or (chosen and gold and (("objective_coefficient" in chosen.roles) != ("objective_coefficient" in gold.roles))):
        return "objective_constraint"
    if chosen and gold and chosen.kind == gold.kind:
        return "same_type_ambiguity"
    return "other_role_quantity"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def run_diagnostic(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    if "NLP4LP_GOLD_CACHE" not in os.environ and DEFAULT_GOLD_CACHE.exists():
        os.environ["NLP4LP_GOLD_CACHE"] = str(DEFAULT_GOLD_CACHE)

    eval_items = u._load_eval(ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl")
    gold_by_id = u._load_hf_gold(split="test")
    catalog, _ = u._load_catalog_as_problems(ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    tfidf = get_baseline("tfidf")
    tfidf.fit(catalog)

    per_slot_rows: list[dict[str, Any]] = []
    targeted_rows: list[dict[str, Any]] = []
    separability_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    current_ready = 0
    schema_hit_not_ready = 0
    rescued_queries: set[str] = set()
    triggered_queries: set[str] = set()
    triggered_ready_queries: set[str] = set()
    trigger_sets: dict[str, set[str]] = {
        "verification_failure": set(),
        "low_retrieval_margin": set(),
        "multiple_compatible_mentions": set(),
        "same_type_multiplicity": set(),
        "low_assignment_margin": set(),
        "role_quantity_conflict": set(),
        "any_pre_verification": set(),
        "verification_or_low_retrieval_margin": set(),
    }
    correctable_query_ids: set[str] = set()
    all_query_ids: set[str] = set()
    ready_query_ids: set[str] = set()

    for ex in eval_items:
        qid = ex["query_id"]
        all_query_ids.add(qid)
        query = ex["query"]
        gold_id = ex["relevant_doc_id"]
        ranked = tfidf.rank(query, top_k=2)
        pred_id = ranked[0][0] if ranked else ""
        retrieval_margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 0.0
        schema_hit = pred_id == gold_id
        gold = gold_by_id.get(gold_id) or {}
        gold_params = gold.get("parameters") or {}
        pred = gold_by_id.get(pred_id) or {}
        pred_params = pred.get("parameters") or {}
        pred_info = pred.get("problem_info") or {}
        if isinstance(pred_info, dict) and isinstance(pred_info.get("parameters"), dict):
            expected_params = list(pred_info["parameters"].keys())
        elif isinstance(pred_params, dict):
            expected_params = list(pred_params.keys())
        else:
            expected_params = []
        expected_scalar = list({p for p in expected_params if u._is_scalar(gold_params.get(p))})
        n_expected = len(expected_scalar)
        mention_diags = extract_mention_diags(query, "orig")
        candidates = list(u._extract_num_tokens(query, "orig"))
        used_diag_ids: set[int] = set()
        filled: dict[str, Any] = {}
        filled_kind: dict[str, str] = {}
        slot_debug: dict[str, dict[str, Any]] = {}
        retrieval_trigger = retrieval_margin <= 0.05
        multi_compatible_trigger = False
        same_type_trigger = False
        low_assignment_margin_trigger = False
        role_quantity_conflict_trigger = False
        pre_verification_trigger = retrieval_trigger
        pending_targets: list[dict[str, Any]] = []

        for slot_name in expected_scalar:
            smeta = slot_metadata(slot_name)
            before = list(candidates)
            candidate_diags: list[tuple[u.NumTok, MentionDiag | None, tuple[int, float, str]]] = []
            for c in before:
                candidate_diags.append((c, _diag_for_tok(mention_diags, c, used_diag_ids), choose_preference(smeta.expected_type, c)))
            idx, tok = u._choose_token(smeta.expected_type, candidates)
            chosen_diag = _diag_for_tok(mention_diags, tok, used_diag_ids) if tok is not None else None
            if tok is not None and idx is not None and 0 <= idx < len(candidates):
                candidates.pop(idx)
            if chosen_diag is not None:
                used_diag_ids.add(chosen_diag.mention_id)
            if tok is not None:
                filled[slot_name] = tok.value if tok.value is not None else tok.raw
                filled_kind[slot_name] = tok.kind

            sorted_candidates = sorted(candidate_diags, key=lambda x: x[2], reverse=True)
            gold_value = gold_params.get(slot_name)
            gold_candidates = [
                (rank + 1, c, d, pref) for rank, (c, d, pref) in enumerate(sorted_candidates)
                if c.value is not None and u._is_scalar(gold_value) and _rel_close(c.value, gold_value)
            ]
            gold_rank = gold_candidates[0][0] if gold_candidates else ""
            gold_diag = gold_candidates[0][2] if gold_candidates else None
            selected_value = tok.value if tok is not None else None
            selected_is_gold = bool(tok is not None and tok.value is not None and u._is_scalar(gold_value) and _rel_close(tok.value, gold_value))
            selected_type_ok = bool(tok is not None and u._is_type_match(smeta.expected_type, tok.kind))
            compatible_count = sum(1 for c in before if not u._is_type_incompatible(smeta.expected_type, c.kind))
            same_type_count = sum(1 for c in before if u._is_type_match(smeta.expected_type, c.kind))
            margin = ""
            if len(sorted_candidates) >= 2:
                top = sorted_candidates[0][2]
                second = sorted_candidates[1][2]
                margin = (top[0] - second[0]) * 1_000_000 + (top[1] - second[1])
            multi_compatible_trigger = multi_compatible_trigger or compatible_count > 1
            same_type_trigger = same_type_trigger or same_type_count > 1
            low_assignment_margin_trigger = low_assignment_margin_trigger or margin == 0
            slot_trigger = compatible_count > 1 or same_type_count > 1 or margin == 0 or retrieval_margin <= 0.05
            if chosen_diag and gold_diag:
                if chosen_diag.quantity_forms != gold_diag.quantity_forms:
                    role_quantity_conflict_trigger = True
                    slot_trigger = True
                if ("lower_bound" in chosen_diag.roles) != ("lower_bound" in gold_diag.roles):
                    role_quantity_conflict_trigger = True
                    slot_trigger = True
                if ("upper_bound" in chosen_diag.roles) != ("upper_bound" in gold_diag.roles):
                    role_quantity_conflict_trigger = True
                    slot_trigger = True
            pre_verification_trigger = pre_verification_trigger or slot_trigger

            chosen_score, chosen_reasons = compatibility_score(chosen_diag, smeta) if chosen_diag else (0, [])
            gold_score, gold_reasons = compatibility_score(gold_diag, smeta) if gold_diag else (0, [])
            separability = ""
            if schema_hit and gold_diag and tok is not None and not selected_is_gold and compatible_count > 1:
                if gold_score > chosen_score:
                    separability = "separable"
                elif gold_score == chosen_score:
                    separability = "ambiguous"
                else:
                    separability = "not_separable"
                case_type = classify_case(smeta, chosen_diag, gold_diag)
                could_fix_type = u._is_type_match(smeta.expected_type, gold_diag.kind) and not selected_type_ok
                target = {
                    "problem_id": qid,
                    "predicted_schema": pred_id,
                    "slot_name": slot_name,
                    "slot_type": smeta.expected_type,
                    "case_type": case_type,
                    "selected_raw": tok.raw,
                    "selected_value": selected_value,
                    "selected_type": tok.kind,
                    "selected_forms": ";".join(sorted(chosen_diag.quantity_forms)) if chosen_diag else "",
                    "selected_roles": ";".join(sorted(chosen_diag.roles)) if chosen_diag else "",
                    "gold_value": gold_value,
                    "gold_raw": gold_diag.raw,
                    "gold_type": gold_diag.kind,
                    "gold_forms": ";".join(sorted(gold_diag.quantity_forms)),
                    "gold_roles": ";".join(sorted(gold_diag.roles)),
                    "gold_rank": gold_rank,
                    "same_type_candidate_count": same_type_count,
                    "compatible_candidate_count": compatible_count,
                    "selected_role_quantity_score": chosen_score,
                    "gold_role_quantity_score": gold_score,
                    "selected_reasons": ";".join(chosen_reasons),
                    "gold_reasons": ";".join(gold_reasons),
                    "separability": separability,
                    "could_fix_type_gate": int(could_fix_type),
                    "assignment_margin": margin,
                    "retrieval_margin": retrieval_margin,
                }
                pending_targets.append(target)

            per_slot_rows.append({
                "problem_id": qid,
                "predicted_schema": pred_id,
                "gold_schema": gold_id,
                "schema_hit": int(schema_hit),
                "slot_name": slot_name,
                "slot_inferred_type": smeta.expected_type,
                "slot_expected_forms": ";".join(sorted(smeta.expected_quantity_forms)),
                "slot_expected_roles": ";".join(sorted(smeta.expected_roles)),
                "slot_entity_tokens": ";".join(sorted(smeta.entity_tokens)),
                "all_mentions": json.dumps([
                    {
                        "raw": d.raw,
                        "value": d.value,
                        "type": d.kind,
                        "sentence": d.sentence_index,
                        "clause": d.clause_index,
                        "nearby": list(d.nearby_tokens),
                        "forms": sorted(d.quantity_forms),
                        "roles": sorted(d.roles),
                        "entity": d.entity_anchor,
                    } for d in mention_diags
                ], sort_keys=True),
                "selected_mention": tok.raw if tok else "",
                "selected_value": selected_value if tok else "",
                "selected_type": tok.kind if tok else "",
                "gold_value": gold_value if u._is_scalar(gold_value) else "",
                "gold_value_among_candidates": int(bool(gold_candidates)),
                "gold_candidate_rank": gold_rank,
                "same_type_candidate_count": same_type_count,
                "compatible_candidate_count": compatible_count,
                "assignment_margin": margin,
                "selected_is_gold": int(selected_is_gold),
                "selected_type_ok": int(selected_type_ok),
                "separability": separability,
            })

            slot_debug[slot_name] = {
                "selected_type_ok": selected_type_ok,
                "corrected_type_ok": bool(gold_diag and u._is_type_match(smeta.expected_type, gold_diag.kind)),
                "targeted_separable": separability == "separable",
            }

        n_filled = len(filled)
        type_matches = sum(1 for p, kind in filled_kind.items() if u._is_type_match(u._expected_type(p), kind))
        coverage = n_filled / max(1, n_expected) if n_expected else 0.0
        type_match = type_matches / max(1, n_filled) if n_filled else 0.0
        ready = coverage >= 0.8 and type_match >= 0.8
        query_trigger = (not ready) or pre_verification_trigger
        if not ready:
            trigger_sets["verification_failure"].add(qid)
        if retrieval_trigger:
            trigger_sets["low_retrieval_margin"].add(qid)
        if multi_compatible_trigger:
            trigger_sets["multiple_compatible_mentions"].add(qid)
        if same_type_trigger:
            trigger_sets["same_type_multiplicity"].add(qid)
        if low_assignment_margin_trigger:
            trigger_sets["low_assignment_margin"].add(qid)
        if role_quantity_conflict_trigger:
            trigger_sets["role_quantity_conflict"].add(qid)
        if pre_verification_trigger:
            trigger_sets["any_pre_verification"].add(qid)
        if (not ready) or retrieval_trigger:
            trigger_sets["verification_or_low_retrieval_margin"].add(qid)
        if ready:
            current_ready += 1
            ready_query_ids.add(qid)
        if schema_hit and not ready:
            schema_hit_not_ready += 1
            targeted_rows.extend(pending_targets)
            separability_rows.extend(pending_targets)

        targeted_separable_slots = {
            r["slot_name"] for r in pending_targets
            if r["separability"] == "separable"
        } if schema_hit and not ready else set()
        corrected_type_matches = type_matches
        for slot_name, dbg in slot_debug.items():
            if slot_name in targeted_separable_slots and not dbg["selected_type_ok"] and dbg["corrected_type_ok"]:
                corrected_type_matches += 1
        corrected_type_match = corrected_type_matches / max(1, n_filled) if n_filled else 0.0
        corrected_ready = coverage >= 0.8 and corrected_type_match >= 0.8
        if not ready and corrected_ready:
            rescued_queries.add(qid)
        if any(r["separability"] == "separable" for r in pending_targets) and schema_hit and not ready:
            correctable_query_ids.add(qid)
        if query_trigger:
            triggered_queries.add(qid)
            if ready:
                triggered_ready_queries.add(qid)

        query_rows.append({
            "problem_id": qid,
            "schema_hit": int(schema_hit),
            "ready": int(ready),
            "coverage": coverage,
            "type_match": type_match,
            "corrected_ready_upper_bound": int(corrected_ready),
            "retrieval_margin": retrieval_margin,
            "cascade_trigger": int(query_trigger),
            "trigger_verification_failure": int(not ready),
            "trigger_low_retrieval_margin": int(retrieval_trigger),
            "trigger_multiple_compatible_mentions": int(multi_compatible_trigger),
            "trigger_same_type_multiplicity": int(same_type_trigger),
            "trigger_low_assignment_margin": int(low_assignment_margin_trigger),
            "trigger_role_quantity_conflict": int(role_quantity_conflict_trigger),
            "trigger_verification_or_low_retrieval_margin": int((not ready) or retrieval_trigger),
        })

    target_count = len(targeted_rows)
    separable_count = sum(1 for r in targeted_rows if r["separability"] == "separable")
    ambiguous_count = sum(1 for r in targeted_rows if r["separability"] == "ambiguous")
    not_sep_count = sum(1 for r in targeted_rows if r["separability"] == "not_separable")
    case_counts: dict[str, int] = {}
    sep_by_case: dict[str, int] = {}
    for r in targeted_rows:
        case_counts[r["case_type"]] = case_counts.get(r["case_type"], 0) + 1
        if r["separability"] == "separable":
            sep_by_case[r["case_type"]] = sep_by_case.get(r["case_type"], 0) + 1

    projected_ready = current_ready + len(rescued_queries)
    trigger_rate = len(triggered_queries) / len(all_query_ids)
    false_trigger_rate_ready = len(triggered_ready_queries) / max(1, len(ready_query_ids))
    correctable_trigger_recall = len(correctable_query_ids & triggered_queries) / max(1, len(correctable_query_ids))
    trigger_analysis: dict[str, dict[str, Any]] = {}
    for name, ids in trigger_sets.items():
        trigger_analysis[name] = {
            "triggered_queries": len(ids),
            "trigger_rate": len(ids) / len(all_query_ids),
            "correctable_failure_recall": len(correctable_query_ids & ids) / max(1, len(correctable_query_ids)),
            "false_trigger_rate_on_ready_queries": len(ready_query_ids & ids) / max(1, len(ready_query_ids)),
        }

    decision = "STAGE_A_NO_GO"
    if separable_count >= 10 and len(rescued_queries) >= 7:
        decision = "STAGE_A_GO"
    elif separable_count >= 10 and len(rescued_queries) > 0:
        decision = "STAGE_A_WEAK_GO"

    summary = {
        "n_total_queries": len(eval_items),
        "current_ready": current_ready,
        "current_instantiation_ready": current_ready / len(eval_items),
        "schema_hit_not_ready": schema_hit_not_ready,
        "targeted_wrong_assignments": target_count,
        "separable_assignments": separable_count,
        "ambiguous_assignments": ambiguous_count,
        "not_separable_assignments": not_sep_count,
        "case_counts": case_counts,
        "separable_by_case": sep_by_case,
        "potentially_rescued_queries": len(rescued_queries),
        "rescued_query_ids": sorted(rescued_queries),
        "projected_ready": projected_ready,
        "projected_instantiation_ready": projected_ready / len(eval_items),
        "absolute_pp_gain": (projected_ready - current_ready) / len(eval_items),
        "cascade_triggered_queries": len(triggered_queries),
        "cascade_trigger_rate": trigger_rate,
        "correctable_failure_recall": correctable_trigger_recall,
        "false_trigger_rate_on_ready_queries": false_trigger_rate_ready,
        "trigger_analysis": trigger_analysis,
        "api_oracle": "NOT_USED",
        "decision": decision,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_slot.csv", per_slot_rows, [
        "problem_id", "predicted_schema", "gold_schema", "schema_hit", "slot_name",
        "slot_inferred_type", "slot_expected_forms", "slot_expected_roles",
        "slot_entity_tokens", "all_mentions", "selected_mention", "selected_value",
        "selected_type", "gold_value", "gold_value_among_candidates",
        "gold_candidate_rank", "same_type_candidate_count",
        "compatible_candidate_count", "assignment_margin", "selected_is_gold",
        "selected_type_ok", "separability",
    ])
    fields = [
        "problem_id", "predicted_schema", "slot_name", "slot_type", "case_type",
        "selected_raw", "selected_value", "selected_type", "selected_forms",
        "selected_roles", "gold_value", "gold_raw", "gold_type", "gold_forms",
        "gold_roles", "gold_rank", "same_type_candidate_count",
        "compatible_candidate_count", "selected_role_quantity_score",
        "gold_role_quantity_score", "selected_reasons", "gold_reasons",
        "separability", "could_fix_type_gate", "assignment_margin",
        "retrieval_margin",
    ]
    _write_csv(out_dir / "targeted_failures.csv", targeted_rows, fields)
    _write_csv(out_dir / "separability.csv", separability_rows, fields)
    _write_csv(out_dir / "cascade_analysis.csv", query_rows, [
        "problem_id", "schema_hit", "ready", "coverage", "type_match",
        "corrected_ready_upper_bound", "retrieval_margin", "cascade_trigger",
        "trigger_verification_failure", "trigger_low_retrieval_margin",
        "trigger_multiple_compatible_mentions", "trigger_same_type_multiplicity",
        "trigger_low_assignment_margin", "trigger_role_quantity_conflict",
        "trigger_verification_or_low_retrieval_margin",
    ])
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    summary = run_diagnostic(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
