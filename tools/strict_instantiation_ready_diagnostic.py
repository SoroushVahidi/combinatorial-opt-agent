#!/usr/bin/env python3
"""Strict schema-gated readiness diagnostic for fresh NLP4LP results.

The existing InstantiationReady metric is:

    Coverage >= threshold AND TypeMatch >= threshold

over the selected schema's scalar slots. It does not require the selected
schema to be the gold schema. This diagnostic adds that missing gate without
changing any production evaluator or historical result table.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.comparison.statistics import build_transition_table, mcnemar_exact, wilson_interval

DEFAULT_OUT = ROOT / "results" / "strict_instantiation_ready"
GOLD_CACHE = ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"

COVERAGE_THRESHOLD = 0.8
TYPE_MATCH_THRESHOLD = 0.8

FRESH_METHODS: dict[str, Path] = {
    "tfidf_typed_greedy": ROOT / "results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv",
    "tfidf_selective_grounding_rerank": ROOT / "results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf_selective_grounding_rerank.csv",
    "bm25_typed_greedy": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_bm25.csv",
    "lsa_typed_greedy": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_lsa.csv",
    "oracle_typed_greedy": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_oracle.csv",
    "tfidf_acceptance_rerank": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_tfidf_acceptance_rerank.csv",
    "tfidf_constrained": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_tfidf_constrained.csv",
    "tfidf_hierarchical_acceptance_rerank": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_tfidf_hierarchical_acceptance_rerank.csv",
    "tfidf_hierarchical_structured_grounding": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_tfidf_hierarchical_structured_grounding.csv",
    "tfidf_max_weight_matching": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_tfidf_max_weight_matching.csv",
    "tfidf_optimization_role_repair": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_tfidf_optimization_role_repair.csv",
    "tfidf_search_structured_grounding": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_tfidf_search_structured_grounding.csv",
    "tfidf_semantic_ir_repair": ROOT / "results/baseline_staleness_audit_2026-08-12/nlp4lp_downstream_per_query_orig_tfidf_semantic_ir_repair.csv",
}

HISTORICAL_STRICT = ROOT / "results/eswa_revision/18_strict_instready/strict_instantiation_ready.csv"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None, "None"):
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _bool_int(value: object) -> bool:
    return int(_float(value, 0.0)) == 1


def ordinary_ready(row: dict[str, object], coverage_threshold: float = COVERAGE_THRESHOLD,
                   type_match_threshold: float = TYPE_MATCH_THRESHOLD) -> bool:
    return (
        _float(row.get("param_coverage")) >= coverage_threshold
        and _float(row.get("type_match")) >= type_match_threshold
    )


def strict_ready(row: dict[str, object], coverage_threshold: float = COVERAGE_THRESHOLD,
                 type_match_threshold: float = TYPE_MATCH_THRESHOLD) -> bool:
    return _bool_int(row.get("schema_hit")) and ordinary_ready(row, coverage_threshold, type_match_threshold)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_gold_scalar_counts(path: Path = GOLD_CACHE) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))["gold_by_id"]
    counts: dict[str, int] = {}
    for problem_id, record in data.items():
        params = record.get("parameters", {})
        counts[problem_id] = sum(1 for value in params.values() if isinstance(value, (int, float)) and not isinstance(value, bool))
    return counts


def annotate_method(method: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        is_ready = ordinary_ready(row)
        is_strict = strict_ready(row)
        schema_correct = _bool_int(row.get("schema_hit"))
        exact5 = row.get("exact5", "")
        exact20 = row.get("exact20", "")
        out.append({
            "method": method,
            "query_id": row["query_id"],
            "predicted_doc_id": row.get("predicted_doc_id", ""),
            "gold_doc_id": row.get("gold_doc_id", ""),
            "schema_correct": int(schema_correct),
            "ordinary_ready": int(is_ready),
            "strict_ready": int(is_strict),
            "false_ready": int(is_ready and not schema_correct),
            "param_coverage": _float(row.get("param_coverage")),
            "type_match": _float(row.get("type_match")),
            "n_expected_scalar": int(_float(row.get("n_expected_scalar"))),
            "n_filled": int(_float(row.get("n_filled"))),
            "exact5": exact5,
            "exact20": exact20,
            "key_overlap": row.get("key_overlap", ""),
        })
    return out


def summarize_method(method: str, rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    ready = sum(int(r["ordinary_ready"]) for r in rows)
    strict = sum(int(r["strict_ready"]) for r in rows)
    schema = sum(int(r["schema_correct"]) for r in rows)
    false_ready = sum(int(r["false_ready"]) for r in rows)
    correct_not_ready = sum(1 for r in rows if int(r["schema_correct"]) and not int(r["ordinary_ready"]))
    wrong_not_ready = sum(1 for r in rows if not int(r["schema_correct"]) and not int(r["ordinary_ready"]))
    ci = wilson_interval(strict, n)
    exact5_vals = [_float(r["exact5"], -1.0) for r in rows if r.get("exact5", "") not in ("", "None", None)]
    exact20_vals = [_float(r["exact20"], -1.0) for r in rows if r.get("exact20", "") not in ("", "None", None)]
    return {
        "method": method,
        "n": n,
        "schema_correct_count": schema,
        "schema_R1": schema / n,
        "ordinary_ready_count": ready,
        "instantiation_ready": ready / n,
        "strict_ready_count": strict,
        "strict_instantiation_ready": strict / n,
        "strict_wilson95_low": ci.lower,
        "strict_wilson95_high": ci.upper,
        "false_ready_count": false_ready,
        "correct_schema_not_ready_count": correct_not_ready,
        "wrong_schema_not_ready_count": wrong_not_ready,
        "coverage": sum(_float(r["param_coverage"]) for r in rows) / n,
        "type_match": sum(_float(r["type_match"]) for r in rows) / n,
        "exact5_on_schema_hits": (sum(exact5_vals) / len(exact5_vals)) if exact5_vals else "",
        "exact20_on_schema_hits": (sum(exact20_vals) / len(exact20_vals)) if exact20_vals else "",
    }


def transition_rows(a_name: str, a_rows: list[dict[str, object]], b_name: str,
                    b_rows: list[dict[str, object]], metric: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_a = {str(r["query_id"]): r for r in a_rows}
    by_b = {str(r["query_id"]): r for r in b_rows}
    ids = sorted(set(by_a) & set(by_b), key=lambda x: int(x.rsplit("_", 1)[-1]))
    counts = Counter()
    details: list[dict[str, object]] = []
    for qid in ids:
        av = bool(int(by_a[qid][metric]))
        bv = bool(int(by_b[qid][metric]))
        if av and bv:
            bucket = "both"
        elif av:
            bucket = f"{a_name}_only"
        elif bv:
            bucket = f"{b_name}_only"
        else:
            bucket = "neither"
        counts[bucket] += 1
        if bucket != "both" and bucket != "neither":
            details.append({
                "query_id": qid,
                "metric": metric,
                f"{a_name}_{metric}": int(av),
                f"{b_name}_{metric}": int(bv),
                f"{a_name}_schema_correct": by_a[qid]["schema_correct"],
                f"{b_name}_schema_correct": by_b[qid]["schema_correct"],
                f"{a_name}_ordinary_ready": by_a[qid]["ordinary_ready"],
                f"{b_name}_ordinary_ready": by_b[qid]["ordinary_ready"],
                f"{a_name}_predicted_doc_id": by_a[qid]["predicted_doc_id"],
                f"{b_name}_predicted_doc_id": by_b[qid]["predicted_doc_id"],
                "gold_doc_id": by_a[qid]["gold_doc_id"],
            })
    pairs = [(bool(int(by_a[qid][metric])), bool(int(by_b[qid][metric]))) for qid in ids]
    mc = mcnemar_exact(build_transition_table(pairs))
    summary = {
        "metric": metric,
        "system_a": a_name,
        "system_b": b_name,
        "both": counts["both"],
        f"{a_name}_only": counts[f"{a_name}_only"],
        f"{b_name}_only": counts[f"{b_name}_only"],
        "neither": counts["neither"],
        "mcnemar_p": mc.p_value if mc.p_value is not None else "",
        "mcnemar_note": mc.note,
        f"{a_name}_only_ids": " ".join(d["query_id"] for d in details if d[f"{a_name}_{metric}"] == 1),
        f"{b_name}_only_ids": " ".join(d["query_id"] for d in details if d[f"{b_name}_{metric}"] == 1),
    }
    return summary, details


def threshold_diagnostic(method_rows: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, mrows in method_rows.items():
        for cov_t in (0.8, 0.9, 1.0):
            for tm_t in (0.8, 0.9, 1.0):
                ready = 0
                strict = 0
                false_ready = 0
                for row in mrows:
                    is_ready = (
                        _float(row["param_coverage"]) >= cov_t
                        and _float(row["type_match"]) >= tm_t
                    )
                    is_schema = bool(int(row["schema_correct"]))
                    ready += int(is_ready)
                    strict += int(is_ready and is_schema)
                    false_ready += int(is_ready and not is_schema)
                rows.append({
                    "method": method,
                    "coverage_threshold": cov_t,
                    "type_match_threshold": tm_t,
                    "ordinary_ready_count": ready,
                    "strict_ready_count": strict,
                    "false_ready_count": false_ready,
                })
    return rows


def classify_false_ready(row: dict[str, object], gold_scalar_counts: dict[str, int]) -> str:
    gold_count = gold_scalar_counts.get(str(row["gold_doc_id"]), 0)
    predicted_count = int(row["n_expected_scalar"])
    key_overlap = _float(row.get("key_overlap"), 0.0)
    coverage = _float(row["param_coverage"])
    type_match = _float(row["type_match"])
    if predicted_count < gold_count:
        return "fewer_or_easier_scalar_slots"
    if key_overlap >= 0.75:
        return "structurally_similar_schema"
    if 0.0 < key_overlap < 0.75:
        return "partial_slot_overlap"
    if coverage < 1.0 or type_match < 1.0:
        return "threshold_permissiveness"
    return "coincidental_full_compatibility"


def false_ready_cases(method_rows: dict[str, list[dict[str, object]]],
                      gold_scalar_counts: dict[str, int]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for method, rows in method_rows.items():
        for row in rows:
            if int(row["false_ready"]):
                out.append({
                    **row,
                    "gold_scalar_count": gold_scalar_counts.get(str(row["gold_doc_id"]), ""),
                    "taxonomy": classify_false_ready(row, gold_scalar_counts),
                })
    return out


def exact_relationship(method_rows: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for method, rows in method_rows.items():
        ordinary_ready_rows = [r for r in rows if int(r["ordinary_ready"])]
        strict_ready_rows = [r for r in rows if int(r["strict_ready"])]
        schema_correct_ready = strict_ready_rows
        exact5_full = [r for r in strict_ready_rows if r.get("exact5", "") not in ("", "None", None) and _float(r["exact5"]) >= 1.0]
        exact20_full = [r for r in strict_ready_rows if r.get("exact20", "") not in ("", "None", None) and _float(r["exact20"]) >= 1.0]
        exact20_wrong = [r for r in schema_correct_ready if r.get("exact20", "") not in ("", "None", None) and _float(r["exact20"]) < 1.0]
        out.append({
            "method": method,
            "ordinary_ready_count": len(ordinary_ready_rows),
            "strict_ready_count": len(strict_ready_rows),
            "ordinary_ready_wrong_schema_count": sum(1 for r in ordinary_ready_rows if not int(r["schema_correct"])),
            "strict_ready_exact5_full_count": len(exact5_full),
            "strict_ready_exact20_full_count": len(exact20_full),
            "strict_ready_exact20_not_full_count": len(exact20_wrong),
            "exact_metrics_schema_gated_by_construction": True,
            "note": "exact5/exact20 are only populated when schema_hit=1 in tools/nlp4lp_downstream_utility.py",
        })
    return out


def bottleneck_taxonomy(method_rows: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for method, rows in method_rows.items():
        counts = Counter()
        for row in rows:
            schema = bool(int(row["schema_correct"]))
            cov_ok = _float(row["param_coverage"]) >= COVERAGE_THRESHOLD
            type_ok = _float(row["type_match"]) >= TYPE_MATCH_THRESHOLD
            exact20_full = row.get("exact20", "") not in ("", "None", None) and _float(row["exact20"]) >= 1.0
            if not schema:
                counts["schema_wrong"] += 1
            elif not cov_ok and type_ok:
                counts["schema_correct_coverage_failure"] += 1
            elif cov_ok and not type_ok:
                counts["schema_correct_typematch_failure"] += 1
            elif not cov_ok and not type_ok:
                counts["schema_correct_both_coverage_and_typematch_failure"] += 1
            elif cov_ok and type_ok and not exact20_full:
                counts["schema_correct_ready_but_exact20_not_full"] += 1
            elif cov_ok and type_ok and exact20_full:
                counts["schema_correct_ready_and_exact20_full"] += 1
        for bucket, count in sorted(counts.items()):
            out.append({"method": method, "bucket": bucket, "count": count})
    return out


def historical_compatibility() -> list[dict[str, object]]:
    if not HISTORICAL_STRICT.exists():
        return [{
            "source": str(HISTORICAL_STRICT.relative_to(ROOT)),
            "status": "NOT_RECONSTRUCTABLE",
            "reason": "Historical strict summary file is absent.",
        }]
    rows = load_rows(HISTORICAL_STRICT)
    return [{
        "source": str(HISTORICAL_STRICT.relative_to(ROOT)),
        "status": "RECONSTRUCTABLE_FROM_HISTORICAL_ARTIFACT",
        "method": row.get("method", ""),
        "historical_instantiation_ready": row.get("InstantiationReady", ""),
        "historical_strict_instantiation_ready": row.get("StrictInstantiationReady", ""),
        "note": "Historical artifact is 0.5287-era and must not be mixed with fresh 257/331 metrics.",
    } for row in rows]


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()


def generate(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_scalar_counts = load_gold_scalar_counts()
    method_rows: dict[str, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []
    for method, path in FRESH_METHODS.items():
        rows = annotate_method(method, load_rows(path))
        method_rows[method] = rows
        all_rows.extend(rows)

    summaries = [summarize_method(method, rows) for method, rows in method_rows.items()]
    summaries.sort(key=lambda r: (-float(r["strict_instantiation_ready"]), str(r["method"])))

    baseline = method_rows["tfidf_typed_greedy"]
    selective = method_rows["tfidf_selective_grounding_rerank"]
    strict_transition, strict_details = transition_rows("tfidf_typed_greedy", baseline, "tfidf_selective_grounding_rerank", selective, "strict_ready")
    ordinary_transition, ordinary_details = transition_rows("tfidf_typed_greedy", baseline, "tfidf_selective_grounding_rerank", selective, "ordinary_ready")
    schema_transition, schema_details = transition_rows("tfidf_typed_greedy", baseline, "tfidf_selective_grounding_rerank", selective, "schema_correct")
    transition_summaries = [strict_transition, ordinary_transition, schema_transition]
    transition_details = strict_details + ordinary_details + schema_details

    false_cases = false_ready_cases(method_rows, gold_scalar_counts)
    threshold_rows = threshold_diagnostic({
        "tfidf_typed_greedy": baseline,
        "tfidf_selective_grounding_rerank": selective,
        "oracle_typed_greedy": method_rows["oracle_typed_greedy"],
    })
    exact_rows = exact_relationship(method_rows)
    bottleneck_rows = bottleneck_taxonomy(method_rows)
    historical_rows = historical_compatibility()

    oracle_summary = next(r for r in summaries if r["method"] == "oracle_typed_greedy")
    baseline_summary = next(r for r in summaries if r["method"] == "tfidf_typed_greedy")
    selective_summary = next(r for r in summaries if r["method"] == "tfidf_selective_grounding_rerank")
    oracle_decomp = [{
        "retrieval_ceiling_schema_R1": 1.0,
        "oracle_strict_ready_count": oracle_summary["strict_ready_count"],
        "oracle_strict_instantiation_ready": oracle_summary["strict_instantiation_ready"],
        "baseline_strict_ready_count": baseline_summary["strict_ready_count"],
        "baseline_strict_gap_to_oracle_count": int(oracle_summary["strict_ready_count"]) - int(baseline_summary["strict_ready_count"]),
        "selective_strict_ready_count": selective_summary["strict_ready_count"],
        "selective_strict_gap_to_oracle_count": int(oracle_summary["strict_ready_count"]) - int(selective_summary["strict_ready_count"]),
    }]

    write_csv(out_dir / "per_query.csv", all_rows)
    write_csv(out_dir / "method_summary.csv", summaries)
    write_csv(out_dir / "baseline_vs_selective_transitions.csv", transition_summaries)
    write_csv(out_dir / "baseline_vs_selective_transition_details.csv", transition_details)
    write_csv(out_dir / "false_ready_cases.csv", false_cases)
    write_csv(out_dir / "threshold_diagnostic.csv", threshold_rows)
    write_csv(out_dir / "exact_metric_relationship.csv", exact_rows)
    write_csv(out_dir / "bottleneck_taxonomy.csv", bottleneck_rows)
    write_csv(out_dir / "oracle_decomposition.csv", oracle_decomp)
    write_csv(out_dir / "historical_compatibility.csv", historical_rows)

    false_taxonomy = Counter(str(r["taxonomy"]) for r in false_cases)
    baseline_false_taxonomy = Counter(str(r["taxonomy"]) for r in false_cases if r["method"] == "tfidf_typed_greedy")
    selective_false_taxonomy = Counter(str(r["taxonomy"]) for r in false_cases if r["method"] == "tfidf_selective_grounding_rerank")

    summary = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "NLP4LP orig, 331 fresh current-code queries",
        "metric_definitions": {
            "instantiation_ready": "Coverage >= 0.8 AND TypeMatch >= 0.8 under the predicted schema; no schema-correctness gate.",
            "strict_instantiation_ready": "SchemaCorrect AND Coverage >= 0.8 AND TypeMatch >= 0.8.",
        },
        "thresholds": {"coverage": COVERAGE_THRESHOLD, "type_match": TYPE_MATCH_THRESHOLD},
        "baseline": baseline_summary,
        "selective": selective_summary,
        "oracle": oracle_summary,
        "strict_transition": strict_transition,
        "ordinary_transition": ordinary_transition,
        "schema_transition": schema_transition,
        "false_ready_taxonomy_all": dict(false_taxonomy),
        "false_ready_taxonomy_baseline": dict(baseline_false_taxonomy),
        "false_ready_taxonomy_selective": dict(selective_false_taxonomy),
        "oracle_decomposition": oracle_decomp[0],
        "historical_compatibility": historical_rows,
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "README.md").write_text(
        "# Strict Instantiation Ready Diagnostic\n\n"
        "This directory is generated by `python tools/strict_instantiation_ready_diagnostic.py`.\n\n"
        "It evaluates a schema-correctness-gated readiness metric on fresh current-code NLP4LP per-query artifacts. "
        "Historical canonical result tables are not modified.\n\n"
        "Definition: `strict_instantiation_ready = schema_hit AND param_coverage >= 0.8 AND type_match >= 0.8`.\n\n"
        "Key files:\n"
        "- `method_summary.csv`: aggregate ordinary and strict readiness by method.\n"
        "- `per_query.csv`: row-level ordinary/strict/false-ready flags.\n"
        "- `false_ready_cases.csv`: ordinary-ready rows whose schema is wrong.\n"
        "- `threshold_diagnostic.csv`: 0.8/0.9/1.0 threshold sensitivity.\n"
        "- `oracle_decomposition.csv`: retrieval ceiling vs. strict grounding gap.\n",
        encoding="utf-8",
    )
    summary["outputs"] = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = generate(args.output_dir)
    print(json.dumps({
        "baseline_strict_ready": summary["baseline"]["strict_ready_count"],
        "selective_strict_ready": summary["selective"]["strict_ready_count"],
        "oracle_strict_ready": summary["oracle"]["strict_ready_count"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
