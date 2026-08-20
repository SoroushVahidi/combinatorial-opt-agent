#!/usr/bin/env python3
"""Phase 4: full-scale mechanism + error analysis for max_weight_matching.

Re-derives, per query, exactly what the canonical CLI's max_weight_matching
branch computes (same retrieval, same expected_scalar construction, same
_run_max_weight_matching_grounding call) but additionally keeps the
per-slot filled_mentions so residual failures can be categorized at the
slot level -- something the canonical per-query CSV does not expose.

Also re-runs typed greedy, search_structured_grounding, and
hierarchical_structured_grounding through the same per-query loop so a
query-level transition matrix (typed greedy right/wrong x MWM right/wrong)
can be computed without relying on possibly-misaligned separately-generated
CSVs.

No changes to tools/nlp4lp_downstream_utility.py. Only its already-public
module-level functions are imported and reused.

Usage (from repo root):
    NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \
    python3 scripts/analysis/mwm_full_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (  # noqa: E402
    _load_eval,
    _load_hf_gold,
    _load_catalog_as_problems,
    _expected_type,
    _is_type_match,
    _is_scalar,
    _rel_err,
    _run_max_weight_matching_grounding,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _choose_token,
    _extract_num_tokens,
)

OUT_DIR = ROOT / "results" / "max_weight_matching_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bucket_type(pname: str) -> str:
    et = _expected_type(pname)
    if et == "percent":
        return "percent"
    if et == "int":
        return "integer"
    if et == "currency":
        return "currency"
    return "float"


def run_typed_greedy(query: str, variant: str, expected_scalar: list[str]) -> tuple[dict, dict]:
    """Reproduce the canonical 'else' (typed greedy) branch faithfully."""
    candidates = list(_extract_num_tokens(query, variant))
    filled: dict[str, float] = {}
    filled_kind: dict[str, str] = {}
    for p in expected_scalar:
        et = _expected_type(p)
        idx, tok = _choose_token(et, candidates)
        if tok is None:
            continue
        if idx is not None and 0 <= idx < len(candidates):
            candidates.pop(idx)
        filled[p] = tok.value if tok.value is not None else tok.raw
        filled_kind[p] = tok.kind
    return filled, filled_kind


def main() -> None:
    eval_path = ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl"
    eval_items = _load_eval(eval_path)
    assert len(eval_items) == 331, f"expected 331 eval items, got {len(eval_items)}"

    gold_by_id = _load_hf_gold(split="test")
    catalog, _id_to_text = _load_catalog_as_problems(ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")

    from retrieval.baselines import get_baseline
    tfidf = get_baseline("tfidf")
    tfidf.fit(catalog)

    per_query_rows = []
    taxonomy_counts: dict[str, int] = {}
    transition_counts = {
        "both_ready": 0,
        "tg_ready_mwm_not": 0,
        "mwm_ready_tg_not": 0,
        "neither_ready": 0,
    }

    for ex in eval_items:
        qid = ex["query_id"]
        query = ex["query"]
        gold_id = ex["relevant_doc_id"]
        ranked = tfidf.rank(query, top_k=1)
        pred_id = ranked[0][0] if ranked else ""
        schema_hit = 1 if pred_id == gold_id else 0

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

        pred_scalar_keys = {
            p for p in expected_params if _is_scalar(gold_params.get(p))
        } if isinstance(gold_params, dict) else set()
        expected_scalar = list(pred_scalar_keys)
        n_expected = len(expected_scalar)

        # ---- MWM ----
        filled_values, filled_mentions, _diag = _run_max_weight_matching_grounding(
            query, "orig", expected_scalar
        )
        mwm_n_filled = 0
        mwm_type_matches = 0
        slot_records = []
        # candidate pool for same-type-ambiguity detection
        all_mentions = _extract_opt_role_mentions(query, "orig")
        slots_irs = {s.name: s for s in _build_slot_opt_irs(expected_scalar)} if expected_scalar else {}

        for p in expected_scalar:
            et = _expected_type(p)
            slot_ir = slots_irs.get(p)
            if p not in filled_values:
                slot_records.append({"slot": p, "status": "missing"})
                continue
            m_ir = filled_mentions.get(p)
            tok = m_ir.tok if m_ir else None
            if tok is None:
                slot_records.append({"slot": p, "status": "missing"})
                continue
            mwm_n_filled += 1
            type_ok = _is_type_match(et, tok.kind)
            if type_ok:
                mwm_type_matches += 1
            value_ok = None
            category = None
            if schema_hit and tok.value is not None and _is_scalar(gold_params.get(p)):
                gold_val = float(gold_params[p])
                err = _rel_err(float(tok.value), gold_val)
                value_ok = err <= 0.20
                if not type_ok:
                    category = "type_mismatch"
                elif not value_ok:
                    if slot_ir is not None and slot_ir.expected_type == "percent":
                        category = "percent_ambiguity"
                    elif slot_ir is not None and (
                        slot_ir.is_total_like != m_ir.is_total_like
                        or (slot_ir.is_coefficient_like and not m_ir.is_per_unit and m_ir.is_total_like)
                    ):
                        category = "total_perunit_confusion"
                    elif slot_ir is not None and slot_ir.is_bound_like and slot_ir.operator_preference and not (
                        m_ir.operator_tags & slot_ir.operator_preference
                    ):
                        category = "minmax_polarity"
                    elif slot_ir is not None and slot_ir.is_objective_like and m_ir.fragment_type not in ("objective",):
                        category = "objective_constraint_confusion"
                    else:
                        same_type_pool = [
                            mm for mm in all_mentions
                            if mm.type_bucket == m_ir.type_bucket and mm.mention_id != m_ir.mention_id
                        ]
                        category = "same_type_ambiguity" if same_type_pool else "wrong_value_other"
                else:
                    category = "correct"
            elif not type_ok:
                category = "type_mismatch"
            else:
                category = "not_comparable"  # filled, type ok, no gold scalar to check against
            slot_records.append({"slot": p, "status": "filled", "type_ok": type_ok, "value_ok": value_ok, "category": category})

        for rec in slot_records:
            cat = rec.get("category") or rec["status"]
            if cat in ("correct", "not_comparable"):
                continue
            taxonomy_counts[cat] = taxonomy_counts.get(cat, 0) + 1

        mwm_coverage = mwm_n_filled / max(1, n_expected) if n_expected else 0.0
        mwm_typematch = mwm_type_matches / max(1, mwm_n_filled) if mwm_n_filled else 0.0
        mwm_ready = 1 if (mwm_coverage >= 0.8 and mwm_typematch >= 0.8) else 0

        if not schema_hit:
            taxonomy_counts["schema_retrieval_miss"] = taxonomy_counts.get("schema_retrieval_miss", 0) + 1
        if n_expected == 0:
            taxonomy_counts["zero_expected_scalar"] = taxonomy_counts.get("zero_expected_scalar", 0) + 1

        # ---- typed greedy (for transition matrix only) ----
        tg_filled, tg_kind = run_typed_greedy(query, "orig", expected_scalar)
        tg_n_filled = len(tg_filled)
        tg_type_matches = sum(1 for p, k in tg_kind.items() if _is_type_match(_expected_type(p), k))
        tg_coverage = tg_n_filled / max(1, n_expected) if n_expected else 0.0
        tg_typematch = tg_type_matches / max(1, tg_n_filled) if tg_n_filled else 0.0
        tg_ready = 1 if (tg_coverage >= 0.8 and tg_typematch >= 0.8) else 0

        if tg_ready and mwm_ready:
            transition_counts["both_ready"] += 1
        elif tg_ready and not mwm_ready:
            transition_counts["tg_ready_mwm_not"] += 1
        elif mwm_ready and not tg_ready:
            transition_counts["mwm_ready_tg_not"] += 1
        else:
            transition_counts["neither_ready"] += 1

        per_query_rows.append({
            "query_id": qid,
            "schema_hit": schema_hit,
            "n_expected_scalar": n_expected,
            "tg_ready": tg_ready,
            "mwm_ready": mwm_ready,
            "mwm_coverage": round(mwm_coverage, 4),
            "mwm_type_match": round(mwm_typematch, 4),
        })

    with open(OUT_DIR / "per_query_transition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        w.writerows(per_query_rows)

    n = len(per_query_rows)
    mwm_ready_total = sum(r["mwm_ready"] for r in per_query_rows)
    tg_ready_total = sum(r["tg_ready"] for r in per_query_rows)

    summary = {
        "n_queries": n,
        "mwm_instantiation_ready_recomputed": mwm_ready_total / n,
        "typed_greedy_instantiation_ready_recomputed": tg_ready_total / n,
        "transition_matrix": transition_counts,
        "residual_failure_taxonomy_counts": dict(
            sorted(taxonomy_counts.items(), key=lambda kv: -kv[1])
        ),
        "n_mwm_not_ready": n - mwm_ready_total,
    }
    with open(OUT_DIR / "mechanism_and_error_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
