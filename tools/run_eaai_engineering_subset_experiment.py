#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.baselines import get_baseline
from tools import nlp4lp_downstream_utility as dutil


def _load_eval(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _render_parametrized_description(problem_info: dict[str, Any], filled: dict[str, Any]) -> str:
    text = str(problem_info.get("parametrized_description") or "")
    for k, v in sorted(filled.items(), key=lambda x: -len(x[0])):
        text = text.replace(k, str(v))
    return text


def _pick_subset(eval_rows: list[dict[str, Any]], gold_by_id: dict[str, dict], n: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for ex in eval_rows:
        gold = gold_by_id.get(ex["relevant_doc_id"]) or {}
        params = gold.get("parameters") or {}
        n_scalar = sum(1 for v in params.values() if _is_scalar(v))
        if 2 <= n_scalar <= 8:
            chosen.append(ex)
        if len(chosen) >= n:
            break
    return chosen


def _predict_schema_id(baseline_name: str, rank_fn, query: str, gold_id: str) -> str:
    if baseline_name == "oracle":
        return gold_id
    ranked = rank_fn(query, top_k=1)
    return ranked[0][0] if ranked else ""


def _ground_values(query: str, variant: str, expected_scalar: list[str], assignment_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not expected_scalar:
        return {}, {}
    if assignment_mode == "optimization_role_relation_repair":
        vals, mentions, _ = dutil._run_optimization_role_relation_repair(query, variant, expected_scalar)
        return vals, mentions
    if assignment_mode == "typed":
        candidates = list(dutil._extract_num_tokens(query, variant))
        filled: dict[str, Any] = {}
        mentions: dict[str, Any] = {}
        for p in expected_scalar:
            et = dutil._expected_type(p)
            idx, tok = dutil._choose_token(et, candidates)
            if tok is None:
                continue
            if idx is not None and 0 <= idx < len(candidates):
                candidates.pop(idx)
            filled[p] = tok.value if tok.value is not None else tok.raw
            mentions[p] = tok
        return filled, mentions
    raise ValueError(f"Unsupported assignment mode: {assignment_mode}")


def run_experiment(variant: str, subset_size: int, assignment_mode: str, out_dir: Path) -> None:
    dutil._apply_low_resource_env()
    eval_path = ROOT / "data" / "processed" / f"nlp4lp_eval_{variant}.jsonl"
    catalog_path = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"

    eval_rows = _load_eval(eval_path)
    gold_by_id = dutil._load_hf_gold("test")
    subset = _pick_subset(eval_rows, gold_by_id, subset_size)

    catalog, _ = dutil._load_catalog_as_problems(catalog_path)
    tfidf = get_baseline("tfidf").fit(catalog)
    bm25 = get_baseline("bm25").fit(catalog)

    retrievers = {
        "tfidf": tfidf.rank,
        "bm25": bm25.rank,
        "oracle": None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = out_dir / "subset_instances.jsonl"
    with open(subset_path, "w", encoding="utf-8") as f:
        for r in subset:
            f.write(json.dumps(r) + "\n")

    instance_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for baseline_name, rank_fn in retrievers.items():
        failures = Counter()
        for ex in subset:
            qid = ex["query_id"]
            query = ex["query"]
            gold_id = ex["relevant_doc_id"]

            pred_id = _predict_schema_id(baseline_name, rank_fn, query, gold_id)
            schema_hit = int(pred_id == gold_id)
            if not schema_hit:
                failures["schema_miss"] += 1

            pred = gold_by_id.get(pred_id) or {}
            pred_params = pred.get("parameters") or {}
            pred_info = pred.get("problem_info") or {}
            pred_param_keys = list((pred_info.get("parameters") or {}).keys()) if isinstance(pred_info.get("parameters"), dict) else list(pred_params.keys())
            expected_scalar = [p for p in pred_param_keys if _is_scalar(pred_params.get(p))]

            filled, mentions = _ground_values(query, variant, expected_scalar, assignment_mode)

            n_expected = len(expected_scalar)
            n_filled = len(filled)
            coverage = (n_filled / n_expected) if n_expected else 0.0

            type_match_count = 0
            for p, m in mentions.items():
                tok_kind = getattr(m, "kind", None)
                if tok_kind is None and hasattr(m, "tok"):
                    tok_kind = m.tok.kind
                if dutil._is_type_match(dutil._expected_type(p), tok_kind or "unknown"):
                    type_match_count += 1
            type_match = (type_match_count / n_filled) if n_filled else 0.0

            structural_valid = int((n_expected == 0) or (coverage >= 0.8 and type_match >= 0.8))
            if not structural_valid:
                if coverage < 0.8:
                    failures["missing_scalar_slots"] += 1
                if type_match < 0.8:
                    failures["type_mismatch"] += 1

            parsed_artifact = bool(pred_info.get("objective")) and bool(pred_info.get("constraints"))
            if not parsed_artifact:
                failures["missing_template_structure"] += 1

            rendered = _render_parametrized_description(pred_info, filled)
            placeholders_remaining = any(param in rendered for param in expected_scalar)
            instantiation_complete = int(parsed_artifact and not placeholders_remaining and (n_expected == 0 or n_filled == n_expected))
            if not instantiation_complete:
                failures["incomplete_instantiation"] += 1

            solver_attempted = 0
            solver_feasible = 0
            solver_reason = "no_pyomo_code_path"
            failures["no_solver_code"] += 1

            instance_rows.append(
                {
                    "baseline": baseline_name,
                    "query_id": qid,
                    "gold_doc_id": gold_id,
                    "predicted_doc_id": pred_id,
                    "schema_hit": schema_hit,
                    "n_expected_scalar": n_expected,
                    "n_filled": n_filled,
                    "param_coverage": round(coverage, 4),
                    "type_match": round(type_match, 4),
                    "structural_valid": structural_valid,
                    "parseable_model_artifact": int(parsed_artifact),
                    "instantiation_complete": instantiation_complete,
                    "solver_attempted": solver_attempted,
                    "solver_feasible": solver_feasible,
                    "solver_reason": solver_reason,
                    "expected_scalar": json.dumps(expected_scalar),
                    "filled_values": json.dumps(filled),
                    "rendered_description": rendered,
                }
            )

        b_rows = [r for r in instance_rows if r["baseline"] == baseline_name]
        n = len(b_rows)
        summary_rows.append(
            {
                "baseline": baseline_name,
                "subset_size": n,
                "schema_hit_rate": round(sum(r["schema_hit"] for r in b_rows) / n, 4),
                "mean_param_coverage": round(sum(r["param_coverage"] for r in b_rows) / n, 4),
                "mean_type_match": round(sum(r["type_match"] for r in b_rows) / n, 4),
                "structural_valid_rate": round(sum(r["structural_valid"] for r in b_rows) / n, 4),
                "parseable_rate": round(sum(r["parseable_model_artifact"] for r in b_rows) / n, 4),
                "instantiation_complete_rate": round(sum(r["instantiation_complete"] for r in b_rows) / n, 4),
                "solver_feasible_rate": round(sum(r["solver_feasible"] for r in b_rows) / n, 4),
                "failure_counts": json.dumps(dict(failures), sort_keys=True),
            }
        )

    instance_csv = out_dir / "engineering_subset_instances.csv"
    with open(instance_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(instance_rows[0].keys()))
        w.writeheader()
        w.writerows(instance_rows)

    summary_csv = out_dir / "engineering_subset_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # 5 case studies: first two successes and three failures from tfidf.
    tfidf_rows = [r for r in instance_rows if r["baseline"] == "tfidf"]
    successes = [r for r in tfidf_rows if r["instantiation_complete"] == 1][:2]
    failures = [r for r in tfidf_rows if r["instantiation_complete"] == 0][:3]
    case_rows = successes + failures
    case_csv = out_dir / "engineering_case_studies.csv"
    with open(case_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "baseline", "query_id", "gold_doc_id", "predicted_doc_id", "schema_hit",
            "n_expected_scalar", "n_filled", "param_coverage", "type_match",
            "structural_valid", "instantiation_complete", "solver_reason", "filled_values",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in case_rows:
            w.writerow({k: r[k] for k in fields})

    meta = {
        "variant": variant,
        "subset_size": subset_size,
        "assignment_mode": assignment_mode,
        "subset_file": str(subset_path),
        "summary_file": str(summary_csv),
        "instances_file": str(instance_csv),
        "case_studies_file": str(case_csv),
    }
    with open(out_dir / "engineering_subset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="orig", choices=["orig", "short", "noisy"])
    parser.add_argument("--subset-size", type=int, default=60)
    parser.add_argument("--assignment-mode", default="optimization_role_relation_repair", choices=["optimization_role_relation_repair", "typed"])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper" / "eaai_engineering_subset")
    parser.add_argument("--gold-cache", type=Path, default=ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json")
    args = parser.parse_args()

    os.environ["NLP4LP_GOLD_CACHE"] = str(args.gold_cache)
    run_experiment(args.variant, args.subset_size, args.assignment_mode, args.out_dir)
