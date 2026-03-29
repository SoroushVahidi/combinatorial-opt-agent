#!/usr/bin/env python3
"""Build per-instance comparison: gold vs pred schema, assignments from opt_repair and relation_repair.

Default: only opt_repair and relation_repair (4-method pipeline). Use --experimental to also run
anchor_linking, bottomup_beam, entity_semantic_beam and add those columns.

Uses TF-IDF retrieval so downstream methods see the same predicted schema.
Output: results/paper/nlp4lp_focused_per_instance_comparison.csv

Use --safe on Wulver/low-resource nodes to set thread env and (if NLP4LP_GOLD_CACHE is set) load gold from cache.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Columns for experimental methods (only included with --experimental).
COLS_EXPERIMENTAL = [
    "pred_anchor_linking", "pred_bottomup_beam", "pred_entity_semantic_beam",
    "n_filled_anchor", "n_filled_beam", "n_filled_entity_semantic_beam",
    "exact_anchor", "exact_beam", "exact_entity_semantic_beam",
    "inst_ready_anchor", "inst_ready_beam", "inst_ready_entity_semantic_beam",
]


def _apply_safe_env() -> None:
    """Set env for low-resource execution (avoids 'can't start new thread')."""
    for k, v in {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    }.items():
        if k not in os.environ:
            os.environ[k] = v


def _get_expected_scalar(pred: dict, gold_params: dict, is_scalar) -> list[str]:
    pred_info = pred.get("problem_info") or {}
    if isinstance(pred_info.get("parameters"), dict):
        expected_params = list(pred_info["parameters"].keys())
    else:
        expected_params = list((pred.get("parameters") or {}).keys())
    return [p for p in expected_params if is_scalar(gold_params.get(p))]


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build per-instance comparison (default: opt_repair + relation_repair only)")
    p.add_argument("--variant", type=str, default="orig", choices=("orig", "noisy", "short"))
    p.add_argument("--eval", type=Path, default=None)
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper")
    p.add_argument("--safe", action="store_true", help="Low-resource: set thread env; use NLP4LP_GOLD_CACHE if set")
    p.add_argument("--experimental", action="store_true", help="Include anchor_linking, bottomup_beam, entity_semantic_beam columns and runs")
    args = p.parse_args()

    if args.safe:
        _apply_safe_env()
    from tools import nlp4lp_downstream_utility as _dutil
    _dutil._apply_low_resource_env()
    _load_eval = _dutil._load_eval
    _load_hf_gold = _dutil._load_hf_gold
    _load_catalog_as_problems = _dutil._load_catalog_as_problems
    _run_optimization_role_repair = _dutil._run_optimization_role_repair
    _run_optimization_role_relation_repair = _dutil._run_optimization_role_relation_repair
    _extract_num_tokens = _dutil._extract_num_tokens
    _is_scalar = _dutil._is_scalar
    _rel_err = _dutil._rel_err
    _expected_type = _dutil._expected_type
    if args.experimental:
        _run_optimization_role_anchor_linking = _dutil._run_optimization_role_anchor_linking
        _run_optimization_role_bottomup_beam_repair = _dutil._run_optimization_role_bottomup_beam_repair
        _run_optimization_role_entity_semantic_beam_repair = _dutil._run_optimization_role_entity_semantic_beam_repair

    eval_path = args.eval or (ROOT / "data" / "processed" / f"nlp4lp_eval_{args.variant}.jsonl")
    eval_items = _load_eval(Path(eval_path))
    if not eval_items:
        raise SystemExit(f"No eval items from {eval_path}")

    gold_by_id = _load_hf_gold(split="test")
    catalog, _ = _load_catalog_as_problems(args.catalog)
    doc_ids = [p["id"] for p in catalog if p.get("id")]

    from retrieval.baselines import get_baseline
    baseline = get_baseline("tfidf")
    baseline.fit(catalog)
    rank_fn = baseline.rank

    out_path = args.out_dir / "nlp4lp_focused_per_instance_comparison.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cols_base = [
        "query_id", "gold_doc_id", "pred_doc_id", "schema_hit",
        "mentions_summary", "gold_assignments", "pred_opt_repair", "pred_relation_repair",
        "n_expected_scalar", "n_filled_opt", "n_filled_relation",
        "exact_opt_repair", "exact_relation_repair",
        "inst_ready_opt_repair", "inst_ready_relation_repair",
    ]
    cols = cols_base + (COLS_EXPERIMENTAL if args.experimental else [])
    rows: list[dict] = []

    for ex in eval_items:
        qid = ex["query_id"]
        query = ex["query"]
        gold_id = ex["relevant_doc_id"]
        ranked = rank_fn(query, top_k=1)
        pred_id = ranked[0][0] if ranked else ""
        schema_hit = 1 if pred_id == gold_id else 0

        gold = gold_by_id.get(gold_id) or {}
        pred = gold_by_id.get(pred_id) or {}
        gold_params = gold.get("parameters") or {}
        expected_scalar = _get_expected_scalar(pred, gold_params, _is_scalar)
        n_expected = len(expected_scalar)

        mentions_summary = ""
        try:
            num_toks = _extract_num_tokens(query, args.variant)
            mentions_summary = json.dumps([{"raw": t.raw, "value": t.value, "kind": t.kind} for t in num_toks])
        except Exception:
            pass

        gold_assignments = {}
        if expected_scalar:
            gold_assignments = {p: gold_params.get(p) for p in expected_scalar if p in gold_params}

        filled_opt: dict[str, str | float] = {}
        filled_relation: dict[str, str | float] = {}
        filled_anchor: dict[str, str | float] = {}
        filled_beam: dict[str, str | float] = {}
        filled_entity_semantic_beam: dict[str, str | float] = {}
        if expected_scalar:
            filled_opt_v, _, _ = _run_optimization_role_repair(query, args.variant, expected_scalar)
            filled_relation_v, _, _ = _run_optimization_role_relation_repair(query, args.variant, expected_scalar)
            filled_opt = dict(filled_opt_v)
            filled_relation = dict(filled_relation_v)
            if args.experimental:
                filled_anchor_v, _, _ = _run_optimization_role_anchor_linking(query, args.variant, expected_scalar)
                filled_beam_v, _, _ = _run_optimization_role_bottomup_beam_repair(query, args.variant, expected_scalar)
                filled_entity_semantic_v, _, _ = _run_optimization_role_entity_semantic_beam_repair(query, args.variant, expected_scalar)
                filled_anchor = dict(filled_anchor_v)
                filled_beam = dict(filled_beam_v)
                filled_entity_semantic_beam = dict(filled_entity_semantic_v)

        n_filled_opt = len(filled_opt)
        n_filled_relation = len(filled_relation)
        n_filled_anchor = len(filled_anchor)
        n_filled_beam = len(filled_beam)
        n_filled_entity_semantic_beam = len(filled_entity_semantic_beam)

        def _type_match_ratio(filled: dict, expected: list[str]) -> float:
            if not filled:
                return 0.0
            correct = 0
            for p in expected:
                if p not in filled:
                    continue
                et = _expected_type(p)
                # We don't have mention kind in filled; use value type as proxy
                val = filled[p]
                if et == "percent" and (isinstance(val, str) and "%" in str(val) or val is None):
                    correct += 1
                elif et in ("int", "float", "currency") and (isinstance(val, (int, float)) or (isinstance(val, str) and val.replace(".", "").replace("-", "").isdigit())):
                    correct += 1
                else:
                    correct += 1  # lenient
            return correct / len(filled) if filled else 0.0

        comparable_errs_opt: list[float] = []
        comparable_errs_relation: list[float] = []
        comparable_errs_anchor: list[float] = []
        comparable_errs_beam: list[float] = []
        comparable_errs_entity_semantic_beam: list[float] = []
        if schema_hit and gold_params:
            pairs: list[tuple[dict, list[float]]] = [
                (filled_opt, comparable_errs_opt),
                (filled_relation, comparable_errs_relation),
            ]
            if args.experimental:
                pairs.extend([
                    (filled_anchor, comparable_errs_anchor),
                    (filled_beam, comparable_errs_beam),
                    (filled_entity_semantic_beam, comparable_errs_entity_semantic_beam),
                ])
            for p in expected_scalar:
                gv = gold_params.get(p)
                if not _is_scalar(gv):
                    continue
                gold_val = float(gv)
                for filled, errs in pairs:
                    if p not in filled:
                        continue
                    try:
                        pv = filled[p]
                        pv_float = float(pv) if isinstance(pv, (int, float)) else None
                        if pv_float is not None:
                            errs.append(_rel_err(pv_float, gold_val))
                    except (TypeError, ValueError):
                        pass

        exact_opt = sum(1 for e in comparable_errs_opt if e <= 0.20) / len(comparable_errs_opt) if comparable_errs_opt else ""
        exact_relation = sum(1 for e in comparable_errs_relation if e <= 0.20) / len(comparable_errs_relation) if comparable_errs_relation else ""
        exact_anchor = sum(1 for e in comparable_errs_anchor if e <= 0.20) / len(comparable_errs_anchor) if comparable_errs_anchor else ""
        exact_beam = sum(1 for e in comparable_errs_beam if e <= 0.20) / len(comparable_errs_beam) if comparable_errs_beam else ""
        exact_entity_semantic_beam = sum(1 for e in comparable_errs_entity_semantic_beam if e <= 0.20) / len(comparable_errs_entity_semantic_beam) if comparable_errs_entity_semantic_beam else ""

        param_cov_opt = (n_filled_opt / n_expected) if n_expected else 0.0
        param_cov_relation = (n_filled_relation / n_expected) if n_expected else 0.0
        param_cov_anchor = (n_filled_anchor / n_expected) if n_expected else 0.0
        param_cov_beam = (n_filled_beam / n_expected) if n_expected else 0.0
        param_cov_entity_semantic_beam = (n_filled_entity_semantic_beam / n_expected) if n_expected else 0.0
        type_match_opt = _type_match_ratio(filled_opt, expected_scalar)
        type_match_relation = _type_match_ratio(filled_relation, expected_scalar)
        type_match_anchor = _type_match_ratio(filled_anchor, expected_scalar)
        type_match_beam = _type_match_ratio(filled_beam, expected_scalar)
        type_match_entity_semantic_beam = _type_match_ratio(filled_entity_semantic_beam, expected_scalar)
        inst_ready_opt = 1 if (param_cov_opt >= 0.8 and type_match_opt >= 0.8) else 0
        inst_ready_relation = 1 if (param_cov_relation >= 0.8 and type_match_relation >= 0.8) else 0
        inst_ready_anchor = 1 if (param_cov_anchor >= 0.8 and type_match_anchor >= 0.8) else 0
        inst_ready_beam = 1 if (param_cov_beam >= 0.8 and type_match_beam >= 0.8) else 0
        inst_ready_entity_semantic_beam = 1 if (param_cov_entity_semantic_beam >= 0.8 and type_match_entity_semantic_beam >= 0.8) else 0

        row: dict[str, str | int | float] = {
            "query_id": qid,
            "gold_doc_id": gold_id,
            "pred_doc_id": pred_id,
            "schema_hit": schema_hit,
            "mentions_summary": mentions_summary[:2000] if len(mentions_summary) > 2000 else mentions_summary,
            "gold_assignments": json.dumps(gold_assignments),
            "pred_opt_repair": json.dumps(filled_opt),
            "pred_relation_repair": json.dumps(filled_relation),
            "n_expected_scalar": n_expected,
            "n_filled_opt": n_filled_opt,
            "n_filled_relation": n_filled_relation,
            "exact_opt_repair": exact_opt if exact_opt != "" else "",
            "exact_relation_repair": exact_relation if exact_relation != "" else "",
            "inst_ready_opt_repair": inst_ready_opt,
            "inst_ready_relation_repair": inst_ready_relation,
        }
        if args.experimental:
            row["pred_anchor_linking"] = json.dumps(filled_anchor)
            row["pred_bottomup_beam"] = json.dumps(filled_beam)
            row["pred_entity_semantic_beam"] = json.dumps(filled_entity_semantic_beam)
            row["n_filled_anchor"] = n_filled_anchor
            row["n_filled_beam"] = n_filled_beam
            row["n_filled_entity_semantic_beam"] = n_filled_entity_semantic_beam
            row["exact_anchor"] = exact_anchor if exact_anchor != "" else ""
            row["exact_beam"] = exact_beam if exact_beam != "" else ""
            row["exact_entity_semantic_beam"] = exact_entity_semantic_beam if exact_entity_semantic_beam != "" else ""
            row["inst_ready_anchor"] = inst_ready_anchor
            row["inst_ready_beam"] = inst_ready_beam
            row["inst_ready_entity_semantic_beam"] = inst_ready_entity_semantic_beam
        rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
