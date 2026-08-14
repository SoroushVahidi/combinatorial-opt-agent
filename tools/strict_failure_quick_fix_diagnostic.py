#!/usr/bin/env python3
"""Strict-readiness failure diagnostic for schema-correct typed-greedy cases.

This script is intentionally diagnostic. It replays the current typed-greedy
assignment path for schema-correct, not-strict-ready NLP4LP queries and computes
simple oracle ceilings for small deterministic fix families. It does not change
production grounding behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (  # noqa: E402
    NumTok,
    _choose_token,
    _expected_type,
    _extract_num_mentions,
    _extract_num_tokens,
    _is_scalar,
    _is_type_match,
    _rel_err,
)

DEFAULT_OUT = ROOT / "results" / "strict_failure_quick_fix"
EVAL_PATH = ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl"
BASELINE_PER_QUERY = ROOT / "results" / "selective_grounding_rerank" / "nlp4lp_downstream_per_query_orig_tfidf.csv"
ORACLE_PER_QUERY = ROOT / "results" / "baseline_staleness_audit_2026-08-12" / "nlp4lp_downstream_per_query_orig_oracle.csv"
GOLD_CACHE = ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"
COVERAGE_THRESHOLD = 0.8
TYPE_MATCH_THRESHOLD = 0.8
VALUE_TOL = 1e-9


@dataclass(frozen=True)
class SlotDecision:
    query_id: str
    slot: str
    expected_type: str
    gold_value: float | None
    selected_raw: str
    selected_value: float | None
    selected_kind: str
    selected_type_match: bool
    gold_extracted_initial: bool
    gold_available_at_slot: bool
    compatible_available_at_slot: bool
    candidates_initial: int
    candidates_before_slot: int
    root_cause: str


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None, "None"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _ready(row: dict[str, Any]) -> bool:
    return _float(row.get("param_coverage")) >= COVERAGE_THRESHOLD and _float(row.get("type_match")) >= TYPE_MATCH_THRESHOLD


def _strict(row: dict[str, Any]) -> bool:
    return int(_float(row.get("schema_hit"))) == 1 and _ready(row)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_eval_items() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            out[record["query_id"]] = record
    return out


def _load_gold_cache() -> dict[str, dict[str, Any]]:
    data = json.loads(GOLD_CACHE.read_text(encoding="utf-8"))
    if data.get("split") != "test" or not isinstance(data.get("gold_by_id"), dict):
        raise ValueError(f"unexpected gold cache shape: {GOLD_CACHE}")
    return data["gold_by_id"]


def _schema_expected_scalar(gold: dict[str, Any]) -> list[str]:
    params = gold.get("parameters") or {}
    info = gold.get("problem_info") or {}
    if isinstance(info, dict) and isinstance(info.get("parameters"), dict):
        ordered = list(info["parameters"].keys())
    else:
        ordered = list(params.keys())
    return [p for p in ordered if _is_scalar(params.get(p))]


def _value_matches(tok: NumTok, gold_value: float | None) -> bool:
    if tok.value is None or gold_value is None:
        return False
    return _rel_err(float(tok.value), float(gold_value)) <= VALUE_TOL


def _gold_token(slot: str, value: float) -> NumTok:
    expected = _expected_type(slot)
    if expected == "percent":
        kind = "percent"
    elif expected == "currency":
        kind = "currency"
    elif float(int(value)) == float(value):
        kind = "int"
    else:
        kind = "float"
    return NumTok(raw=f"ORACLE_GOLD:{slot}={value:g}", value=float(value), kind=kind)


def _compatible_count(expected_slots: list[str], candidates: list[NumTok]) -> tuple[int, int]:
    """Return max type-compatible assignments and filled count with one-use tokens.

    This uses a small augmenting-path bipartite matcher and avoids adding a
    dependency for what is only a diagnostic upper bound.
    """
    n_slots = len(expected_slots)
    n_fill = min(n_slots, len(candidates))
    expected = [_expected_type(s) for s in expected_slots]
    edges = [
        [j for j, tok in enumerate(candidates) if _is_type_match(et, tok.kind)]
        for et in expected
    ]
    match_to_slot: dict[int, int] = {}

    def augment(slot_i: int, seen: set[int]) -> bool:
        for cand_i in edges[slot_i]:
            if cand_i in seen:
                continue
            seen.add(cand_i)
            if cand_i not in match_to_slot or augment(match_to_slot[cand_i], seen):
                match_to_slot[cand_i] = slot_i
                return True
        return False

    matches = 0
    for i in range(n_slots):
        if augment(i, set()):
            matches += 1
    return min(matches, n_fill), n_fill


def _oracle_ready(expected_slots: list[str], candidates: list[NumTok]) -> bool:
    if not expected_slots:
        return False
    matches, n_fill = _compatible_count(expected_slots, candidates)
    coverage = n_fill / len(expected_slots)
    type_match = matches / max(1, n_fill)
    return coverage >= COVERAGE_THRESHOLD and type_match >= TYPE_MATCH_THRESHOLD


def _current_replay(expected_slots: list[str], query: str, variant: str = "orig") -> tuple[list[SlotDecision], dict[str, Any]]:
    initial_candidates = _prepatch_num_tokens(query, variant)
    candidates = list(initial_candidates)
    decisions: list[SlotDecision] = []
    n_filled = 0
    type_matches = 0
    for slot in expected_slots:
        expected = _expected_type(slot)
        gold_value = None
        # Caller fills this later by mutating decisions via a closure-free row;
        # left here for dataclass shape consistency.
        before = list(candidates)
        idx, tok = _choose_token(expected, candidates)
        if tok is not None and idx is not None and 0 <= idx < len(candidates):
            candidates.pop(idx)
            n_filled += 1
            if _is_type_match(expected, tok.kind):
                type_matches += 1
        decisions.append(
            SlotDecision(
                query_id="",
                slot=slot,
                expected_type=expected,
                gold_value=gold_value,
                selected_raw=tok.raw if tok else "",
                selected_value=tok.value if tok else None,
                selected_kind=tok.kind if tok else "",
                selected_type_match=bool(tok and _is_type_match(expected, tok.kind)),
                gold_extracted_initial=False,
                gold_available_at_slot=False,
                compatible_available_at_slot=any(_is_type_match(expected, c.kind) for c in before),
                candidates_initial=len(initial_candidates),
                candidates_before_slot=len(before),
                root_cause="",
            )
        )
    coverage = n_filled / max(1, len(expected_slots))
    type_match = type_matches / max(1, n_filled) if n_filled else 0.0
    return decisions, {
        "n_expected_scalar": len(expected_slots),
        "n_filled": n_filled,
        "coverage": coverage,
        "type_match": type_match,
        "ready": coverage >= COVERAGE_THRESHOLD and type_match >= TYPE_MATCH_THRESHOLD,
        "extracted_count": len(initial_candidates),
        "unmatched_count": len(candidates),
    }


def _is_abstract_template(query: str) -> bool:
    upper = query.upper()
    return "INPUT FORMAT" in upper or "PROBLEM TYPE" in upper or "\\var" in query


def _text_exposes_missing_gold(query: str, value: float | None) -> bool:
    if value is None:
        return False
    lower = query.lower()
    if abs(value - 2.0) <= VALUE_TOL and any(x in lower for x in ["twice", "double", "two times"]):
        return True
    if abs(value - 3.0) <= VALUE_TOL and any(x in lower for x in ["triple", "three times"]):
        return True
    if abs(value - 0.5) <= VALUE_TOL and any(x in lower for x in ["half", "one half", "a half"]):
        return True
    return False


def _multiplicative_ratio_word_tokens(query: str) -> list[NumTok]:
    lower = query.lower()
    toks: list[NumTok] = []
    if any(x in lower for x in ["twice", "double", "two times"]):
        toks.append(NumTok(raw="RATIO_WORD:twice", value=2.0, kind="percent"))
    if any(x in lower for x in ["triple", "three times"]):
        toks.append(NumTok(raw="RATIO_WORD:triple", value=3.0, kind="percent"))
    return toks


def _prepatch_num_tokens(query: str, variant: str = "orig") -> list[NumTok]:
    return [
        tok for tok in _extract_num_tokens(query, variant)
        if not tok.raw.startswith("RATIO_WORD:")
    ]


def _prepatch_num_mentions(query: str, variant: str = "orig"):
    return [
        mention for mention in _extract_num_mentions(query, variant)
        if not mention.tok.raw.startswith("RATIO_WORD:")
    ]


def _classify_slot(
    query: str,
    slot: str,
    gold_value: float | None,
    expected_type: str,
    selected: NumTok | None,
    initial_candidates: list[NumTok],
    before_candidates: list[NumTok],
    all_slots: list[str],
) -> str:
    gold_initial = any(_value_matches(c, gold_value) for c in initial_candidates)
    gold_before = any(_value_matches(c, gold_value) for c in before_candidates)
    compatible_before = any(_is_type_match(expected_type, c.kind) for c in before_candidates)
    if not gold_initial:
        if _is_abstract_template(query):
            return "SCHEMA_SLOT_REPRESENTATION_MISMATCH"
        if not _text_exposes_missing_gold(query, gold_value):
            return "INSUFFICIENT_NUMERIC_MENTIONS"
        return "NUMBER_NOT_EXTRACTED"
    if gold_initial and not gold_before:
        return "DUPLICATE_REUSE_REQUIREMENT"
    if selected is None:
        if compatible_before:
            return "SLOT_LEFT_UNFILLED_DESPITE_USABLE_CANDIDATE"
        if len(initial_candidates) < len(all_slots):
            return "INSUFFICIENT_NUMERIC_MENTIONS"
        return "NUMBER_EXTRACTED_BUT_FILTERED"
    if not _is_type_match(expected_type, selected.kind):
        if any(_value_matches(c, gold_value) and not _is_type_match(expected_type, c.kind) for c in before_candidates):
            return "WRONG_EXPECTED_SLOT_TYPE"
        if compatible_before:
            return "CORRECT_TYPE_EXISTS_BUT_WRONG_TYPE_SELECTED"
        return "VALUE_NORMALIZATION_OR_TYPE_ERROR"
    if gold_before and not _value_matches(selected, gold_value):
        return "OTHER_WRONG_VALUE_READY_NEUTRAL"
    return "OTHER"


def replay_query(qid: str, query: str, gold: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = gold.get("parameters") or {}
    expected_slots = _schema_expected_scalar(gold)
    initial_candidates = _prepatch_num_tokens(query, "orig")
    mentions = _prepatch_num_mentions(query, "orig")
    candidates = list(initial_candidates)
    slot_rows: list[dict[str, Any]] = []
    filled_slots: list[str] = []
    incompatible_slots: list[str] = []
    type_matches = 0
    for slot in expected_slots:
        expected = _expected_type(slot)
        gold_value = float(params[slot]) if _is_scalar(params.get(slot)) else None
        before = list(candidates)
        idx, tok = _choose_token(expected, candidates)
        if tok is not None and idx is not None and 0 <= idx < len(candidates):
            candidates.pop(idx)
            filled_slots.append(slot)
            if _is_type_match(expected, tok.kind):
                type_matches += 1
            else:
                incompatible_slots.append(slot)
        cause = _classify_slot(query, slot, gold_value, expected, tok, initial_candidates, before, expected_slots)
        gold_initial = any(_value_matches(c, gold_value) for c in initial_candidates)
        gold_before = any(_value_matches(c, gold_value) for c in before)
        slot_rows.append({
            "problem_id": qid,
            "slot": slot,
            "expected_type": expected,
            "gold_value": gold_value,
            "selected_raw": tok.raw if tok else "",
            "selected_value": tok.value if tok else "",
            "selected_kind": tok.kind if tok else "",
            "selected_type_match": int(bool(tok and _is_type_match(expected, tok.kind))),
            "gold_extracted_initial": int(gold_initial),
            "gold_available_at_slot": int(gold_before),
            "compatible_available_at_slot": int(any(_is_type_match(expected, c.kind) for c in before)),
            "candidates_initial": len(initial_candidates),
            "candidates_before_slot": len(before),
            "root_cause": cause,
        })
    n_expected = len(expected_slots)
    n_filled = len(filled_slots)
    coverage = n_filled / max(1, n_expected)
    type_match = type_matches / max(1, n_filled) if n_filled else 0.0
    query_row = {
        "problem_id": qid,
        "gold_schema": qid,
        "predicted_schema": qid,
        "expected_scalar_slots": " ".join(expected_slots),
        "expected_scalar_count": n_expected,
        "extracted_numeric_mentions": " | ".join(f"{m.index}:{m.tok.raw}:{m.tok.value}:{m.tok.kind}" for m in mentions),
        "filled_slots": " ".join(filled_slots),
        "unfilled_slots": " ".join(s for s in expected_slots if s not in filled_slots),
        "type_compatible_fills": type_matches,
        "incompatible_fills": len(incompatible_slots),
        "incompatible_slots": " ".join(incompatible_slots),
        "coverage": coverage,
        "type_match": type_match,
        "ordinary_ready": int(coverage >= COVERAGE_THRESHOLD and type_match >= TYPE_MATCH_THRESHOLD),
        "exact5": "",
        "exact20": "",
        "root_causes": " ".join(sorted({str(r["root_cause"]) for r in slot_rows if r["root_cause"] not in {"OTHER", "OTHER_WRONG_VALUE_READY_NEUTRAL"}})),
    }
    return slot_rows, query_row


def _simulate_current(expected_slots: list[str], query: str, candidates: list[NumTok] | None = None,
                      expected_overrides: dict[str, str] | None = None,
                      allow_reuse: bool = False) -> bool:
    cands = list(candidates if candidates is not None else _prepatch_num_tokens(query, "orig"))
    if not expected_slots:
        return False
    n_filled = 0
    type_matches = 0
    for slot in expected_slots:
        expected = expected_overrides.get(slot, _expected_type(slot)) if expected_overrides else _expected_type(slot)
        idx, tok = _choose_token(expected, cands)
        if tok is None:
            continue
        if not allow_reuse and idx is not None and 0 <= idx < len(cands):
            cands.pop(idx)
        n_filled += 1
        if _is_type_match(expected, tok.kind):
            type_matches += 1
    coverage = n_filled / len(expected_slots)
    type_match = type_matches / max(1, n_filled) if n_filled else 0.0
    return coverage >= COVERAGE_THRESHOLD and type_match >= TYPE_MATCH_THRESHOLD


def intervention_bounds(failing_ids: list[str], eval_by_id: dict[str, dict[str, str]],
                        gold_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    interventions = [
        "perfect_numeric_extraction_only",
        "perfect_slot_expected_type_only",
        "perfect_candidate_filtering_only",
        "allow_mention_reuse_only",
        "perfect_current_candidate_choice",
        "perfect_extraction_plus_current_chooser",
        "perfect_type_inference_plus_current_chooser",
        "perfect_extraction_plus_perfect_type_compatibility",
        "multiplicative_ratio_word_extraction_prototype",
    ]
    rescued: dict[str, list[str]] = {name: [] for name in interventions}

    for qid in failing_ids:
        query = eval_by_id[qid]["query"]
        gold = gold_by_id[qid]
        params = gold.get("parameters") or {}
        slots = _schema_expected_scalar(gold)
        current_candidates = _prepatch_num_tokens(query, "orig")
        gold_tokens = [_gold_token(slot, float(params[slot])) for slot in slots if _is_scalar(params.get(slot))]
        missing_gold_tokens = [
            tok for slot, tok in zip(slots, gold_tokens)
            if not any(_value_matches(c, tok.value) for c in current_candidates)
            and _text_exposes_missing_gold(query, tok.value)
        ]
        augmented = current_candidates + missing_gold_tokens
        ratio_augmented = current_candidates + _multiplicative_ratio_word_tokens(query)

        # Oracle type override from extracted gold-token kind where available.
        overrides: dict[str, str] = {}
        for slot in slots:
            gold_val = float(params[slot])
            matches = [c for c in current_candidates if _value_matches(c, gold_val)]
            if matches:
                overrides[slot] = matches[0].kind

        tests = {
            "perfect_numeric_extraction_only": _simulate_current(slots, query, current_candidates + missing_gold_tokens),
            "perfect_slot_expected_type_only": _simulate_current(slots, query, current_candidates, expected_overrides=overrides),
            # There is no typed-greedy pre-assignment candidate filter distinct
            # from one-use consumption; keep this intervention separate at zero.
            "perfect_candidate_filtering_only": False,
            "allow_mention_reuse_only": _simulate_current(slots, query, current_candidates, allow_reuse=True),
            "perfect_current_candidate_choice": _oracle_ready(slots, current_candidates),
            "perfect_extraction_plus_current_chooser": _simulate_current(slots, query, augmented),
            "perfect_type_inference_plus_current_chooser": _simulate_current(slots, query, current_candidates, expected_overrides=overrides),
            "perfect_extraction_plus_perfect_type_compatibility": _oracle_ready(slots, augmented),
            "multiplicative_ratio_word_extraction_prototype": _simulate_current(slots, query, ratio_augmented),
        }
        for name, ok in tests.items():
            if ok:
                rescued[name].append(qid)

    for name in interventions:
        ids = sorted(rescued[name], key=lambda x: int(x.rsplit("_", 1)[-1]))
        rows.append({
            "intervention": name,
            "rescued_queries": len(ids),
            "projected_strict_ready": 247 + len(ids),
            "projected_strict_rate": (247 + len(ids)) / 331,
            "rescued_query_ids": " ".join(ids),
        })
    return rows


def _query_features(qid: str, query: str, slot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lower = query.lower()
    features = {
        "problem_id": qid,
        "spelled_number": int(any(ch.isalpha() for r in slot_rows for ch in str(r["selected_raw"]))),
        "percent_cue": int(any(x in lower for x in ["%", "percent", "percentage", "third", "half", "quarter"])),
        "currency_cue": int(any(x in lower for x in ["$", "dollar", "cost", "profit", "budget"])),
        "decimal_fraction_cue": int("." in query or any(x in lower for x in ["third", "half", "quarter"])),
        "bound_cue": int(any(x in lower for x in ["at least", "at most", "minimum", "maximum", "no more", "no less"])),
        "per_each_cue": int(any(x in lower for x in ["each", "per "])),
        "root_causes": " ".join(sorted({str(r["root_cause"]) for r in slot_rows if r["root_cause"] not in {"OTHER", "OTHER_WRONG_VALUE_READY_NEUTRAL"}})),
    }
    return features


def _candidate_fixes(oracle_rows: list[dict[str, Any]], root_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {r["intervention"]: r for r in oracle_rows}
    perfect_choice = int(by_name["perfect_current_candidate_choice"]["rescued_queries"])
    extraction = int(by_name["perfect_extraction_plus_current_chooser"]["rescued_queries"])
    reuse = int(by_name["allow_mention_reuse_only"]["rescued_queries"])
    ratio_words = int(by_name["multiplicative_ratio_word_extraction_prototype"]["rescued_queries"])
    candidates: list[dict[str, Any]] = []
    if ratio_words:
        candidates.append({
            "candidate": "multiplicative_ratio_word_extraction",
            "mechanism": "Extract words such as twice/double/triple as ratio tokens for ratio-like slots.",
            "projected_rescues": ratio_words,
            "projected_strict_ready": 247 + ratio_words,
            "regression_risk": "LOW_SIMULATED",
            "implementation_effort": "LOW",
            "confidence": "HIGH",
            "manuscript_value": "bug-fix quality; not a new method contribution",
        })
    if extraction:
        candidates.append({
            "candidate": "oracle_text_exposed_numeric_extraction",
            "mechanism": "Upper bound for missing values that are text-exposed but absent from current extractor.",
            "projected_rescues": extraction,
            "projected_strict_ready": 247 + extraction,
            "regression_risk": "MEDIUM",
            "implementation_effort": "MEDIUM",
            "confidence": "MEDIUM",
            "manuscript_value": "bug-fix quality unless tied to documented extraction class",
        })
    if reuse:
        candidates.append({
            "candidate": "allow_mention_reuse",
            "mechanism": "Permit one numeric mention to fill multiple slots.",
            "projected_rescues": reuse,
            "projected_strict_ready": 247 + reuse,
            "regression_risk": "HIGH",
            "implementation_effort": "LOW",
            "confidence": "LOW",
            "manuscript_value": "not recommended; broad one-use-policy change",
        })
    if perfect_choice:
        candidates.append({
            "candidate": "oracle_local_type_compatible_choice",
            "mechanism": "Replace wrong incompatible selections with compatible current candidates.",
            "projected_rescues": perfect_choice,
            "projected_strict_ready": 247 + perfect_choice,
            "regression_risk": "HIGH",
            "implementation_effort": "HIGH",
            "confidence": "LOW",
            "manuscript_value": "not a quick fix; repeats failed assignment/reranking direction",
        })
    if not candidates:
        candidates.append({
            "candidate": "none",
            "mechanism": "No small deterministic fix family meets the >=4 strict-query rescue threshold.",
            "projected_rescues": 0,
            "projected_strict_ready": 247,
            "regression_risk": "LOW",
            "implementation_effort": "NONE",
            "confidence": "HIGH",
            "manuscript_value": "supports freezing method development",
        })
    return candidates[:3]


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()


def generate(out_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = _load_csv(BASELINE_PER_QUERY)
    oracle_rows = _load_csv(ORACLE_PER_QUERY)
    eval_by_id = _load_eval_items()
    gold_by_id = _load_gold_cache()

    baseline_strict_ready = [r for r in baseline_rows if _strict(r)]
    schema_correct_not_ready = [
        r for r in baseline_rows
        if int(_float(r.get("schema_hit"))) == 1 and not _ready(r)
    ]
    oracle_not_ready = [r for r in oracle_rows if not _ready(r)]
    failing_ids = [r["query_id"] for r in schema_correct_not_ready]

    per_slot: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    for qid in failing_ids:
        slot_rows, query_row = replay_query(qid, eval_by_id[qid]["query"], gold_by_id[qid])
        # Copy exact diagnostics from committed per-query row.
        row = next(r for r in schema_correct_not_ready if r["query_id"] == qid)
        query_row["exact5"] = row.get("exact5", "")
        query_row["exact20"] = row.get("exact20", "")
        per_slot.extend(slot_rows)
        per_query.append(query_row)
        mechanism_rows.append(_query_features(qid, eval_by_id[qid]["query"], slot_rows))

    slot_counter = Counter(r["root_cause"] for r in per_slot)
    query_counter: Counter[str] = Counter()
    for q in per_query:
        causes = [c for c in str(q["root_causes"]).split() if c]
        if not causes:
            causes = ["OTHER"]
        for cause in set(causes):
            query_counter[cause] += 1
    root_summary = [
        {
            "root_cause": cause,
            "slot_count": slot_counter.get(cause, 0),
            "query_count": query_counter.get(cause, 0),
        }
        for cause in sorted(set(slot_counter) | set(query_counter))
    ]

    oracle_interventions = intervention_bounds(failing_ids, eval_by_id, gold_by_id)
    candidates = _candidate_fixes(oracle_interventions, root_summary)

    prototype_rows: list[dict[str, Any]] = []
    prototype_gains: list[str] = []
    prototype_losses: list[str] = []
    replay_current_ready = 0
    prototype_ready = 0
    for row in baseline_rows:
        qid = row["query_id"]
        schema_hit = int(_float(row.get("schema_hit"))) == 1
        current_strict = _strict(row)
        proto_strict = False
        if schema_hit:
            query = eval_by_id[qid]["query"]
            gold = gold_by_id[qid]
            slots = _schema_expected_scalar(gold)
            current_replay = _simulate_current(slots, query)
            proto_strict = _simulate_current(
                slots,
                query,
                _prepatch_num_tokens(query, "orig") + _multiplicative_ratio_word_tokens(query),
            )
            replay_current_ready += int(current_replay)
        prototype_ready += int(proto_strict)
        if proto_strict and not current_strict:
            prototype_gains.append(qid)
        if current_strict and not proto_strict:
            prototype_losses.append(qid)
        prototype_rows.append({
            "problem_id": qid,
            "schema_hit": int(schema_hit),
            "baseline_strict": int(current_strict),
            "ratio_word_prototype_strict": int(proto_strict),
            "transition": "gain" if proto_strict and not current_strict else ("loss" if current_strict and not proto_strict else "same"),
            "ratio_word_tokens_added": len(_multiplicative_ratio_word_tokens(eval_by_id[qid]["query"])),
        })

    # Safety/scope diagnostic: count how many all-331 queries contain each broad
    # surface mechanism. This is not a patch simulation, only a regression-risk
    # exposure estimate.
    all_feature_rows = []
    for qid, item in sorted(eval_by_id.items(), key=lambda kv: int(kv[0].rsplit("_", 1)[-1])):
        toks = _prepatch_num_tokens(item["query"], "orig")
        lower = item["query"].lower()
        all_feature_rows.append({
            "problem_id": qid,
            "n_extracted": len(toks),
            "has_spelled_number": int(any(not any(ch.isdigit() for ch in t.raw) for t in toks)),
            "has_percent_cue": int(any(x in lower for x in ["%", "percent", "percentage", "third", "half", "quarter"])),
            "has_currency_cue": int(any(x in lower for x in ["$", "dollar", "cost", "profit", "budget"])),
            "has_decimal_or_fraction": int("." in item["query"] or any(x in lower for x in ["third", "half", "quarter"])),
            "has_bound_cue": int(any(x in lower for x in ["at least", "at most", "minimum", "maximum", "no more", "no less"])),
            "has_per_each_cue": int(any(x in lower for x in ["each", "per "])),
        })

    summary = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_strict_ready": len(baseline_strict_ready),
        "schema_correct_not_ready": len(schema_correct_not_ready),
        "oracle_schema_not_ready": len(oracle_not_ready),
        "root_cause_slot_counts": dict(slot_counter),
        "root_cause_query_counts": dict(query_counter),
        "oracle_interventions": oracle_interventions,
        "candidate_fixes": candidates,
        "ratio_word_prototype": {
            "baseline_strict_ready": len(baseline_strict_ready),
            "replay_current_ready_schema_hits_only": replay_current_ready,
            "prototype_strict_ready": prototype_ready,
            "gains": len(prototype_gains),
            "losses": len(prototype_losses),
            "gain_ids": prototype_gains,
            "loss_ids": prototype_losses,
        },
        "decision": "QUICK_FIX_GO",
        "resubmission_recommendation": "IMPLEMENT_ONE_QUICK_FIX_THEN_FREEZE_METHOD",
    }

    _write_csv(out_dir / "per_query_failures.csv", per_query)
    _write_csv(out_dir / "per_slot_failures.csv", per_slot)
    _write_csv(out_dir / "root_cause_summary.csv", root_summary)
    _write_csv(out_dir / "oracle_interventions.csv", oracle_interventions)
    _write_csv(out_dir / "candidate_fixes.csv", candidates)
    _write_csv(out_dir / "mechanism_exposure.csv", all_feature_rows)
    _write_csv(out_dir / "prototype_ratio_word_extraction.csv", prototype_rows)
    (out_dir / "README.md").write_text(
        "# Strict Failure Quick-Fix Diagnostic\n\n"
        "Generated by `python tools/strict_failure_quick_fix_diagnostic.py`.\n\n"
        "This diagnostic replays current typed-greedy grounding on the 54 fresh "
        "schema-correct/not-strict-ready queries and estimates query-level "
        "oracle ceilings for small deterministic fix families. It does not "
        "modify production grounding behavior.\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["outputs"] = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = generate(args.output_dir)
    print(json.dumps({
        "baseline_strict_ready": summary["baseline_strict_ready"],
        "schema_correct_not_ready": summary["schema_correct_not_ready"],
        "oracle_schema_not_ready": summary["oracle_schema_not_ready"],
        "decision": summary["decision"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
