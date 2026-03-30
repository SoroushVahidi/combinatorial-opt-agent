#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.baselines import get_baseline
from tools import nlp4lp_downstream_utility as dutil


def _load_eval(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _render_parametrized_description(problem_info: dict[str, Any], filled: dict[str, Any]) -> str:
    text = str(problem_info.get("parametrized_description") or "")
    for k, v in sorted(filled.items(), key=lambda x: -len(x[0])):
        text = text.replace(k, str(v))
    return text


def _pick_executable_subset(eval_rows: list[dict[str, Any]], gold_by_id: dict[str, dict], limit: int | None) -> list[dict[str, Any]]:
    subset = [ex for ex in eval_rows if (gold_by_id.get(ex["relevant_doc_id"]) or {}).get("optimus_code")]
    if limit is not None:
        subset = subset[:limit]
    return subset


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
    raise ValueError(f"Unsupported assignment mode: {assignment_mode}")


def _build_exec_params(pred_params: dict[str, Any], filled: dict[str, Any]) -> dict[str, Any]:
    params = dict(pred_params)
    params.update(filled)
    return params


def _patch_optimus_code(code: str, params_path: str) -> str:
    pattern = r'with open\("[^"]*parameters\.json"\s*,\s*"r"\) as f:'
    repl = f'with open("{params_path}", "r") as f:'
    return re.sub(pattern, repl, code, count=1)


def _attempt_exec(optimus_code: str, params: dict[str, Any]) -> dict[str, Any]:
    if not optimus_code:
        return {
            "executable_model": 0,
            "solver_run_success": 0,
            "feasible_solution": 0,
            "objective_produced": 0,
            "exec_error": "missing_optimus_code",
        }

    tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_parameters.json", delete=False, encoding="utf-8") as f:
            json.dump(params, f)
            tmp = f.name

        patched = _patch_optimus_code(optimus_code, tmp)
        namespace: dict[str, Any] = {}
        exec(patched, namespace)  # noqa: S102

        model = namespace.get("model")
        objective_value = None
        feasible = 0
        if model is not None:
            try:
                objective_value = float(model.objVal)
            except Exception:
                objective_value = None
            try:
                status = int(model.Status)
                feasible = int(status in {2, 9})
            except Exception:
                feasible = 0

        return {
            "executable_model": int(model is not None),
            "solver_run_success": int(model is not None),
            "feasible_solution": feasible,
            "objective_produced": int(objective_value is not None),
            "exec_error": "",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "executable_model": 0,
            "solver_run_success": 0,
            "feasible_solution": 0,
            "objective_produced": 0,
            "exec_error": f"{type(e).__name__}: {e}",
        }
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def run_experiment(variant: str, assignment_mode: str, out_dir: Path, limit: int | None = None) -> None:
    dutil._apply_low_resource_env()
    eval_path = ROOT / "data" / "processed" / f"nlp4lp_eval_{variant}.jsonl"
    catalog_path = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"

    eval_rows = _load_eval(eval_path)
    gold_by_id = dutil._load_hf_gold("test")
    subset = _pick_executable_subset(eval_rows, gold_by_id, limit)

    catalog, _ = dutil._load_catalog_as_problems(catalog_path)
    tfidf = get_baseline("tfidf").fit(catalog)
    bm25 = get_baseline("bm25").fit(catalog)
    retrievers = {"tfidf": tfidf.rank, "bm25": bm25.rank, "oracle": None}

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "subset_instances.jsonl", "w", encoding="utf-8") as f:
        for row in subset:
            f.write(json.dumps(row) + "\n")

    instance_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for baseline_name, rank_fn in retrievers.items():
        failures = Counter()
        for ex in subset:
            query = ex["query"]
            qid = ex["query_id"]
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
            rendered = _render_parametrized_description(pred_info, filled)
            placeholders_remaining = any(param in rendered for param in expected_scalar)
            instantiation_complete = int(parsed_artifact and not placeholders_remaining and (n_expected == 0 or n_filled == n_expected))
            if not instantiation_complete:
                failures["incomplete_instantiation"] += 1

            exec_params = _build_exec_params(pred_params, filled)
            exec_out = _attempt_exec(pred.get("optimus_code") or "", exec_params)

            if not exec_out["executable_model"]:
                failures["model_build_fail"] += 1
            if not exec_out["solver_run_success"]:
                failures["solver_run_fail"] += 1
            if exec_out["exec_error"]:
                failures[exec_out["exec_error"].split(":", 1)[0]] += 1

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
                    "executable_model": exec_out["executable_model"],
                    "solver_run_success": exec_out["solver_run_success"],
                    "feasible_solution": exec_out["feasible_solution"],
                    "objective_produced": exec_out["objective_produced"],
                    "exec_error": exec_out["exec_error"],
                }
            )

        b_rows = [r for r in instance_rows if r["baseline"] == baseline_name]
        n = len(b_rows)
        summary_rows.append(
            {
                "baseline": baseline_name,
                "subset_size": n,
                "schema_hit_rate": round(sum(r["schema_hit"] for r in b_rows) / n, 4),
                "structural_valid_rate": round(sum(r["structural_valid"] for r in b_rows) / n, 4),
                "instantiation_complete_rate": round(sum(r["instantiation_complete"] for r in b_rows) / n, 4),
                "executable_model_rate": round(sum(r["executable_model"] for r in b_rows) / n, 4),
                "solver_run_success_rate": round(sum(r["solver_run_success"] for r in b_rows) / n, 4),
                "feasible_solution_rate": round(sum(r["feasible_solution"] for r in b_rows) / n, 4),
                "objective_produced_rate": round(sum(r["objective_produced"] for r in b_rows) / n, 4),
                "failure_counts": json.dumps(dict(failures), sort_keys=True),
            }
        )

    with open(out_dir / "executable_subset_instances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(instance_rows[0].keys()))
        w.writeheader()
        w.writerows(instance_rows)

    with open(out_dir / "executable_subset_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # case studies: success (if any) + retrieval fail + grounding fail + exec fail
    tfidf_rows = [r for r in instance_rows if r["baseline"] == "tfidf"]
    case_rows: list[dict[str, Any]] = []

    success = next((r for r in tfidf_rows if r["solver_run_success"] == 1), None)
    if success:
        case_rows.append(success)
    retrieval_fail = next((r for r in tfidf_rows if r["schema_hit"] == 0), None)
    if retrieval_fail:
        case_rows.append(retrieval_fail)
    grounding_fail = next((r for r in tfidf_rows if r["schema_hit"] == 1 and r["instantiation_complete"] == 0), None)
    if grounding_fail:
        case_rows.append(grounding_fail)
    exec_fail = next((r for r in tfidf_rows if r["exec_error"]), None)
    if exec_fail and exec_fail not in case_rows:
        case_rows.append(exec_fail)
    for r in tfidf_rows:
        if len(case_rows) >= 5:
            break
        if r not in case_rows:
            case_rows.append(r)

    with open(out_dir / "executable_case_studies.csv", "w", newline="", encoding="utf-8") as f:
        fields = [
            "baseline", "query_id", "gold_doc_id", "predicted_doc_id", "schema_hit",
            "structural_valid", "instantiation_complete", "executable_model", "solver_run_success",
            "feasible_solution", "objective_produced", "exec_error",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in case_rows[:5]:
            w.writerow({k: row[k] for k in fields})

    with open(out_dir / "executable_subset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": variant,
                "assignment_mode": assignment_mode,
                "selection_rule": "All eval instances whose gold schema has non-empty optimus_code.",
                "subset_size": len(subset),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="orig", choices=["orig", "short", "noisy"])
    parser.add_argument("--assignment-mode", default="optimization_role_relation_repair", choices=["optimization_role_relation_repair"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper" / "eaai_executable_subset")
    parser.add_argument("--gold-cache", type=Path, default=ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json")
    args = parser.parse_args()

    os.environ["NLP4LP_GOLD_CACHE"] = str(args.gold_cache)
    run_experiment(args.variant, args.assignment_mode, args.out_dir, args.limit)
