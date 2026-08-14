#!/usr/bin/env python3
"""Produce Stage-B artifacts for tfidf_selective_grounding_rerank."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.baselines import get_baseline
from tools import nlp4lp_downstream_utility as u


OUT_DIR = ROOT / "results" / "selective_grounding_rerank"
METHOD = "tfidf_selective_grounding_rerank"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ready(row: dict[str, str]) -> bool:
    return float(row["param_coverage"]) >= 0.8 and float(row["type_match"]) >= 0.8


def _exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail)


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def _run_downstream(baseline: str, out_dir: Path) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    ok = u.run_single_setting(
        variant="orig",
        baseline_arg=baseline,
        assignment_mode="typed",
        out_dir=out_dir,
    )
    elapsed = time.perf_counter() - start
    if not ok:
        raise RuntimeError(f"run_single_setting failed for {baseline}")
    with open(out_dir / f"nlp4lp_downstream_orig_{baseline}.json", encoding="utf-8") as f:
        return json.load(f)["aggregate"], elapsed


def _select(cands: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if mode == "baseline":
        return cands[0]
    if mode == "frozen":
        return max(cands, key=lambda c: (c["score_frozen"], c["retrieval_score"], -c["rank"], c["schema_id"]))
    if mode == "retrieval_only":
        return max(cands, key=lambda c: (c["normalized_tfidf"], c["retrieval_score"], -c["rank"], c["schema_id"]))
    if mode == "no_coverage":
        return max(cands, key=lambda c: (c["score_no_coverage"], c["retrieval_score"], -c["rank"], c["schema_id"]))
    if mode == "no_typematch":
        return max(cands, key=lambda c: (c["score_no_typematch"], c["retrieval_score"], -c["rank"], c["schema_id"]))
    raise ValueError(mode)


def _summary_for_selection(name: str, selected: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(selected)
    ready = {qid for qid, r in selected.items() if r["ready"]}
    base_ready = {qid for qid, r in baseline.items() if r["ready"]}
    schema = {qid for qid, r in selected.items() if r["schema_hit"]}
    base_schema = {qid for qid, r in baseline.items() if r["schema_hit"]}
    return {
        "ablation": name,
        "Schema_R1": len(schema) / n,
        "Coverage": sum(float(r["coverage"]) for r in selected.values()) / n,
        "TypeMatch": sum(float(r["type_match"]) for r in selected.values()) / n,
        "InstantiationReady": len(ready) / n,
        "ready_count": len(ready),
        "ready_gains": len(ready - base_ready),
        "ready_losses": len(base_ready - ready),
        "schema_recoveries": len(schema - base_schema),
        "schema_regressions": len(base_schema - schema),
    }


def run(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    if "NLP4LP_GOLD_CACHE" not in os.environ:
        cache = ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"
        if cache.exists():
            os.environ["NLP4LP_GOLD_CACHE"] = str(cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": METHOD,
        "trigger": "tfidf_top1_score - tfidf_top2_score <= 0.05",
        "margin_threshold": 0.05,
        "k": 5,
        "normalization": "min-max over top-k TF-IDF scores; all 1.0 if span is zero",
        "weights": {"normalized_tfidf": 0.50, "coverage": 0.25, "type_match": 0.25},
        "tie_break": ["higher consistency score", "higher raw TF-IDF score", "lower retrieval rank", "schema id"],
        "benchmark": "NLP4LP orig, 331 queries",
        "git_sha": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": "PYTHONHASHSEED=0 NLP4LP_GOLD_CACHE=... python3 tools/analyze_selective_grounding_rerank_stage_b.py",
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    baseline_agg, baseline_runtime = _run_downstream("tfidf", out_dir)
    candidate_agg, candidate_runtime = _run_downstream(METHOD, out_dir)
    baseline_rows = {
        r["query_id"]: r for r in _read_csv(out_dir / "nlp4lp_downstream_per_query_orig_tfidf.csv")
    }
    candidate_rows_raw = {
        r["query_id"]: r for r in _read_csv(out_dir / f"nlp4lp_downstream_per_query_orig_{METHOD}.csv")
    }

    eval_items = u._load_eval(ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl")
    gold_by_id = u._load_hf_gold(split="test")
    catalog, _ = u._load_catalog_as_problems(ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    tfidf = get_baseline("tfidf")
    tfidf.fit(catalog)

    candidate_diag_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    all_candidates: dict[str, list[dict[str, Any]]] = {}
    baseline_candidate: dict[str, dict[str, Any]] = {}
    selected_frozen: dict[str, dict[str, Any]] = {}
    trigger_count = 0
    extra_groundings = 0
    per_query_times: list[float] = []

    for ex in eval_items:
        qid = ex["query_id"]
        query = ex["query"]
        gold_id = ex["relevant_doc_id"]
        t0 = time.perf_counter()
        ranked = tfidf.rank(query, top_k=5)
        top1 = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else top1[1]
        margin = top1[1] - top2_score
        triggered = margin <= 0.05
        if triggered:
            trigger_count += 1
            extra_groundings += max(0, len(ranked) - 1)
        norm = u._normalize_retrieval_scores([s for _sid, s in ranked])
        cands: list[dict[str, Any]] = []
        for rank, ((sid, score), norm_score) in enumerate(zip(ranked, norm), 1):
            metrics = u._typed_greedy_schema_metrics(query, "orig", sid, gold_id, gold_by_id)
            row = {
                "problem_id": qid,
                "triggered": int(triggered),
                "margin": margin,
                "gold_schema": gold_id,
                "schema_id": sid,
                "rank": rank,
                "retrieval_score": score,
                "normalized_tfidf": norm_score,
                "coverage": metrics["coverage"],
                "type_match": metrics["type_match"],
                "ready": bool(metrics["ready"]),
                "schema_hit": sid == gold_id,
                "score_frozen": u._selective_grounding_consistency_score(norm_score, metrics["coverage"], metrics["type_match"]),
                "score_retrieval_only": norm_score,
                "score_no_coverage": (2 / 3) * norm_score + (1 / 3) * metrics["type_match"],
                "score_no_typematch": (2 / 3) * norm_score + (1 / 3) * metrics["coverage"],
                "n_expected_scalar": metrics["n_expected_scalar"],
                "n_filled": metrics["n_filled"],
                "key_overlap": metrics["key_overlap"],
            }
            cands.append(row)
            candidate_diag_rows.append(row)
        all_candidates[qid] = cands
        baseline_candidate[qid] = cands[0]
        selected_frozen[qid] = _select(cands, "frozen") if triggered else cands[0]
        per_query_times.append(time.perf_counter() - t0)

        base = baseline_rows[qid]
        cand = candidate_rows_raw[qid]
        if base["predicted_doc_id"] != cand["predicted_doc_id"]:
            old_correct = base["predicted_doc_id"] == gold_id
            new_correct = cand["predicted_doc_id"] == gold_id
            base_ready = _ready(base)
            cand_ready = _ready(cand)
            cls = "OTHER"
            if (not old_correct) and new_correct:
                cls = "TRUE_SCHEMA_RESCUE"
            elif (not old_correct) and (not new_correct) and cand_ready and not base_ready:
                cls = "WRONG_TO_WRONG_BUT_READINESS_GAIN"
            elif old_correct and not new_correct:
                cls = "CORRECT_TO_WRONG_REGRESSION"
            elif (not old_correct) and (not new_correct):
                cls = "WRONG_TO_WRONG_NO_MEANINGFUL_GAIN"
            selected = next((r for r in cands if r["schema_id"] == cand["predicted_doc_id"]), {})
            changed_rows.append({
                "problem_id": qid,
                "classification": cls,
                "gold_schema": gold_id,
                "old_schema": base["predicted_doc_id"],
                "new_schema": cand["predicted_doc_id"],
                "old_correct": int(old_correct),
                "new_correct": int(new_correct),
                "old_ready": int(base_ready),
                "new_ready": int(cand_ready),
                "margin": margin,
                "selected_score": selected.get("score_frozen", ""),
                "selected_coverage": selected.get("coverage", ""),
                "selected_type_match": selected.get("type_match", ""),
                "why": "frozen consistency score selected higher retrieval-grounding consistency under low TF-IDF margin",
            })
        if (not _ready(base)) and _ready(cand):
            new_correct = cand["predicted_doc_id"] == gold_id
            semantic = "SEMANTICALLY_BETTER" if new_correct else "INCORRECT_SCHEMA"
            semantic_rows.append({
                "problem_id": qid,
                "semantic_class": semantic,
                "gold_schema": gold_id,
                "old_schema": base["predicted_doc_id"],
                "new_schema": cand["predicted_doc_id"],
                "old_ready": 0,
                "new_ready": 1,
                "new_schema_correct": int(new_correct),
                "old_coverage": base["param_coverage"],
                "new_coverage": cand["param_coverage"],
                "old_type_match": base["type_match"],
                "new_type_match": cand["type_match"],
            })

    ablations = []
    selections = {
        "A0_baseline_typed_greedy": baseline_candidate,
        "A1_always_top5_frozen_score": {qid: _select(c, "frozen") for qid, c in all_candidates.items()},
        "A2_selective_top5_frozen_score": selected_frozen,
        "A3_selective_retrieval_only": {qid: (_select(c, "retrieval_only") if c[0]["triggered"] else c[0]) for qid, c in all_candidates.items()},
        "A4_selective_without_coverage": {qid: (_select(c, "no_coverage") if c[0]["triggered"] else c[0]) for qid, c in all_candidates.items()},
        "A5_selective_without_typematch": {qid: (_select(c, "no_typematch") if c[0]["triggered"] else c[0]) for qid, c in all_candidates.items()},
    }
    for name, selected in selections.items():
        ablations.append(_summary_for_selection(name, selected, baseline_candidate))

    ready_base = {qid for qid, r in baseline_rows.items() if _ready(r)}
    ready_cand = {qid for qid, r in candidate_rows_raw.items() if _ready(r)}
    schema_base = {qid for qid, r in baseline_rows.items() if r["predicted_doc_id"] == r["gold_doc_id"]}
    schema_cand = {qid for qid, r in candidate_rows_raw.items() if r["predicted_doc_id"] == r["gold_doc_id"]}
    transitions = {
        "readiness": {
            "both_ready": len(ready_base & ready_cand),
            "baseline_only": len(ready_base - ready_cand),
            "candidate_only": len(ready_cand - ready_base),
            "neither": len(set(baseline_rows) - (ready_base | ready_cand)),
            "candidate_only_ids": sorted(ready_cand - ready_base),
            "baseline_only_ids": sorted(ready_base - ready_cand),
            "mcnemar_p": _exact_mcnemar_p(len(ready_base - ready_cand), len(ready_cand - ready_base)),
        },
        "schema": {
            "both_correct": len(schema_base & schema_cand),
            "baseline_only": len(schema_base - schema_cand),
            "candidate_only": len(schema_cand - schema_base),
            "both_wrong": len(set(baseline_rows) - (schema_base | schema_cand)),
            "candidate_only_ids": sorted(schema_cand - schema_base),
            "baseline_only_ids": sorted(schema_base - schema_cand),
            "mcnemar_p": _exact_mcnemar_p(len(schema_base - schema_cand), len(schema_cand - schema_base)),
        },
    }
    metrics = {
        "baseline": baseline_agg,
        "candidate": candidate_agg,
        "absolute_pp_instantiation_ready": candidate_agg["instantiation_ready"] - baseline_agg["instantiation_ready"],
        "relative_instantiation_ready_change": (
            (candidate_agg["instantiation_ready"] / baseline_agg["instantiation_ready"]) - 1
        ),
        "baseline_ready_wilson95": _wilson(len(ready_base), len(baseline_rows)),
        "candidate_ready_wilson95": _wilson(len(ready_cand), len(candidate_rows_raw)),
        "transitions": transitions,
    }
    runtime = {
        "baseline_total_seconds": baseline_runtime,
        "candidate_total_seconds": candidate_runtime,
        "baseline_mean_ms_per_query": 1000 * baseline_runtime / len(baseline_rows),
        "candidate_mean_ms_per_query": 1000 * candidate_runtime / len(candidate_rows_raw),
        "diagnostic_median_ms_per_query": 1000 * statistics.median(per_query_times),
        "triggered_queries": trigger_count,
        "trigger_rate": trigger_count / len(eval_items),
        "extra_schema_grounding_calls": extra_groundings,
        "multiplicative_overhead": candidate_runtime / baseline_runtime if baseline_runtime else "",
        "absolute_ms_per_query_overhead": 1000 * (candidate_runtime - baseline_runtime) / len(eval_items),
    }

    family_rows = []
    groups: dict[str, list[str]] = {}
    for qid in baseline_rows:
        try:
            bucket = int(qid.rsplit("_", 1)[1]) % 5
        except Exception:
            bucket = 0
        groups.setdefault(f"id_mod5_{bucket}", []).append(qid)
    for family, ids in sorted(groups.items()):
        b_ready = sum(1 for qid in ids if qid in ready_base)
        c_ready = sum(1 for qid in ids if qid in ready_cand)
        family_rows.append({
            "family": family,
            "n": len(ids),
            "baseline_ready": b_ready,
            "candidate_ready": c_ready,
            "delta": c_ready - b_ready,
        })

    _write_csv(out_dir / "per_query.csv", list(candidate_rows_raw.values()), list(next(iter(candidate_rows_raw.values())).keys()))
    _write_csv(out_dir / "candidate_diagnostics.csv", candidate_diag_rows, [
        "problem_id", "triggered", "margin", "gold_schema", "schema_id", "rank",
        "retrieval_score", "normalized_tfidf", "coverage", "type_match", "ready",
        "schema_hit", "score_frozen", "score_retrieval_only", "score_no_coverage",
        "score_no_typematch", "n_expected_scalar", "n_filled", "key_overlap",
    ])
    _write_csv(out_dir / "changed_decisions.csv", changed_rows, [
        "problem_id", "classification", "gold_schema", "old_schema", "new_schema",
        "old_correct", "new_correct", "old_ready", "new_ready", "margin",
        "selected_score", "selected_coverage", "selected_type_match", "why",
    ])
    _write_csv(out_dir / "semantic_audit.csv", semantic_rows, [
        "problem_id", "semantic_class", "gold_schema", "old_schema", "new_schema",
        "old_ready", "new_ready", "new_schema_correct", "old_coverage",
        "new_coverage", "old_type_match", "new_type_match",
    ])
    _write_csv(out_dir / "ablations.csv", ablations, [
        "ablation", "Schema_R1", "Coverage", "TypeMatch", "InstantiationReady",
        "ready_count", "ready_gains", "ready_losses", "schema_recoveries",
        "schema_regressions",
    ])
    _write_csv(out_dir / "generalization_probe.csv", family_rows, ["family", "n", "baseline_ready", "candidate_ready", "delta"])
    for name, obj in (("metrics.json", metrics), ("runtime.json", runtime), ("transitions.json", transitions)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(
            "# Selective Grounding Rerank Stage-B Results\n\n"
            "Frozen configuration: margin <= 0.05, top-5 TF-IDF schemas, "
            "score = 0.50 * normalized_tfidf + 0.25 * coverage + 0.25 * type_match.\n\n"
            f"Baseline ready: {len(ready_base)}/331. Candidate ready: {len(ready_cand)}/331.\n"
            f"Triggered queries: {trigger_count}/331. Extra schema groundings: {extra_groundings}.\n"
        )
    return {"metrics": metrics, "runtime": runtime, "changed": changed_rows, "semantic": semantic_rows, "ablations": ablations}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
