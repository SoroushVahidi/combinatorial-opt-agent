#!/usr/bin/env python3
"""Stage-A diagnostic for selective top-k schema reranking.

This is a read-only diagnostic. It reuses the current TF-IDF retrieval and
typed-greedy grounding path, then evaluates oracle top-k ceilings and small
deterministic reranking rules. It does not modify production behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.baselines import get_baseline
from retrieval.baselines import _searchable_text
from tools import nlp4lp_downstream_utility as u


OUT_DIR = ROOT / "results" / "topk_schema_rerank_stage_a"
DEFAULT_GOLD_CACHE = ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"
MARGIN_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
K_VALUES = (1, 2, 3, 5, 10)


@dataclass(frozen=True)
class CandidateGrounding:
    query_id: str
    gold_schema: str
    schema_id: str
    rank: int
    retrieval_score: float
    retrieval_margin: float
    schema_hit: bool
    n_expected_scalar: int
    n_filled: int
    coverage: float
    type_match: float
    ready: bool
    key_overlap: float
    extracted_number_count: int
    unmatched_mention_count: int
    incompatible_assignment_count: int
    null_slot_count: int
    min_assignment_margin: float
    mean_assignment_margin: float
    lexical_overlap_query_schema: float


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1}


def lexical_overlap(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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


def assignment_margin(expected: str, candidates: list[u.NumTok]) -> float:
    if len(candidates) < 2:
        return math.inf
    ranked = sorted((choose_preference(expected, c) for c in candidates), reverse=True)
    top, second = ranked[0], ranked[1]
    return (top[0] - second[0]) * 1_000_000.0 + (top[1] - second[1])


def ground_candidate(
    query_id: str,
    query: str,
    gold_schema: str,
    schema_id: str,
    rank: int,
    retrieval_score: float,
    retrieval_margin: float,
    gold_by_id: dict[str, dict],
    id_to_text: dict[str, str],
) -> CandidateGrounding:
    gold = gold_by_id.get(gold_schema) or {}
    gold_params = gold.get("parameters") or {}
    pred = gold_by_id.get(schema_id) or {}
    pred_params = pred.get("parameters") or {}
    pred_info = pred.get("problem_info") or {}
    if isinstance(pred_info, dict) and isinstance(pred_info.get("parameters"), dict):
        expected_params = list(pred_info["parameters"].keys())
    elif isinstance(pred_params, dict):
        expected_params = list(pred_params.keys())
    else:
        expected_params = []

    gold_scalar_keys = {p for p, v in gold_params.items() if u._is_scalar(v)}
    pred_scalar_keys = {p for p in expected_params if u._is_scalar(gold_params.get(p))}
    # Match tools/nlp4lp_downstream_utility.py exactly: the production
    # evaluation intentionally passes through a set before greedy assignment.
    expected_scalar = list(pred_scalar_keys)
    n_expected = len(expected_scalar)
    key_overlap = (len(set(expected_scalar) & gold_scalar_keys) / len(gold_scalar_keys)) if gold_scalar_keys else 0.0

    candidates = list(u._extract_num_tokens(query, "orig"))
    extracted_count = len(candidates)
    n_filled = 0
    type_matches = 0
    margins: list[float] = []
    incompatible = 0
    for slot in expected_scalar:
        et = u._expected_type(slot)
        margins.append(assignment_margin(et, candidates))
        idx, tok = u._choose_token(et, candidates)
        if tok is None:
            continue
        if idx is not None and 0 <= idx < len(candidates):
            candidates.pop(idx)
        n_filled += 1
        if u._is_type_match(et, tok.kind):
            type_matches += 1
        else:
            incompatible += 1

    coverage = n_filled / max(1, n_expected) if n_expected else 0.0
    type_match = type_matches / max(1, n_filled) if n_filled else 0.0
    finite_margins = [m for m in margins if math.isfinite(m)]
    min_margin = min(finite_margins) if finite_margins else math.inf
    mean_margin = sum(finite_margins) / len(finite_margins) if finite_margins else math.inf
    return CandidateGrounding(
        query_id=query_id,
        gold_schema=gold_schema,
        schema_id=schema_id,
        rank=rank,
        retrieval_score=retrieval_score,
        retrieval_margin=retrieval_margin,
        schema_hit=schema_id == gold_schema,
        n_expected_scalar=n_expected,
        n_filled=n_filled,
        coverage=coverage,
        type_match=type_match,
        ready=coverage >= 0.8 and type_match >= 0.8,
        key_overlap=key_overlap,
        extracted_number_count=extracted_count,
        unmatched_mention_count=len(candidates),
        incompatible_assignment_count=incompatible,
        null_slot_count=max(0, n_expected - n_filled),
        min_assignment_margin=min_margin,
        mean_assignment_margin=mean_margin,
        lexical_overlap_query_schema=lexical_overlap(query, id_to_text.get(schema_id, "")),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _candidate_row(c: CandidateGrounding) -> dict[str, Any]:
    return {
        "problem_id": c.query_id,
        "gold_schema": c.gold_schema,
        "schema_id": c.schema_id,
        "rank": c.rank,
        "retrieval_score": c.retrieval_score,
        "retrieval_margin": c.retrieval_margin,
        "schema_hit": int(c.schema_hit),
        "n_expected_scalar": c.n_expected_scalar,
        "n_filled": c.n_filled,
        "coverage": c.coverage,
        "type_match": c.type_match,
        "ready": int(c.ready),
        "key_overlap": c.key_overlap,
        "extracted_number_count": c.extracted_number_count,
        "unmatched_mention_count": c.unmatched_mention_count,
        "incompatible_assignment_count": c.incompatible_assignment_count,
        "null_slot_count": c.null_slot_count,
        "min_assignment_margin": "" if math.isinf(c.min_assignment_margin) else c.min_assignment_margin,
        "mean_assignment_margin": "" if math.isinf(c.mean_assignment_margin) else c.mean_assignment_margin,
        "lexical_overlap_query_schema": c.lexical_overlap_query_schema,
    }


def summarize_selection(
    name: str,
    selected: dict[str, CandidateGrounding],
    baseline: dict[str, CandidateGrounding],
) -> dict[str, Any]:
    n = len(selected)
    ready = {qid for qid, c in selected.items() if c.ready}
    base_ready = {qid for qid, c in baseline.items() if c.ready}
    schema = {qid for qid, c in selected.items() if c.schema_hit}
    base_schema = {qid for qid, c in baseline.items() if c.schema_hit}
    return {
        "method": name,
        "schema_R1": len(schema) / n,
        "instantiation_ready": len(ready) / n,
        "ready_count": len(ready),
        "coverage": sum(c.coverage for c in selected.values()) / n,
        "type_match": sum(c.type_match for c in selected.values()) / n,
        "schema_recoveries": len(schema - base_schema),
        "schema_regressions": len(base_schema - schema),
        "ready_gains": len(ready - base_ready),
        "ready_losses": len(base_ready - ready),
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail)


def select_by_rule(cands: list[CandidateGrounding], rule: str) -> CandidateGrounding:
    if rule == "R0_tfidf_top1":
        return min(cands, key=lambda c: c.rank)
    if rule == "R1_max_coverage":
        return max(cands, key=lambda c: (c.coverage, c.retrieval_score, -c.rank))
    if rule == "R2_max_typematch":
        return max(cands, key=lambda c: (c.type_match, c.retrieval_score, c.coverage, -c.rank))
    if rule == "R3_ready_cov_type_tfidf":
        return max(cands, key=lambda c: (int(c.ready), c.coverage, c.type_match, c.retrieval_score, -c.rank))
    if rule == "R4_verified_cov_type_tfidf":
        return max(cands, key=lambda c: (int(c.ready), c.coverage, c.type_match, -c.null_slot_count, c.retrieval_score, -c.rank))
    if rule == "R5_small_consistency_score":
        scores = [c.retrieval_score for c in cands]
        lo, hi = min(scores), max(scores)
        span = hi - lo

        def score(c: CandidateGrounding) -> tuple[float, float, int]:
            retr = (c.retrieval_score - lo) / span if span > 1e-12 else 1.0
            consistency = 0.50 * retr + 0.25 * c.coverage + 0.25 * c.type_match
            return consistency, c.retrieval_score, -c.rank

        return max(cands, key=score)
    raise ValueError(f"unknown rule: {rule}")


def run_diagnostic(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    if "NLP4LP_GOLD_CACHE" not in os.environ and DEFAULT_GOLD_CACHE.exists():
        os.environ["NLP4LP_GOLD_CACHE"] = str(DEFAULT_GOLD_CACHE)

    eval_items = u._load_eval(ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl")
    gold_by_id = u._load_hf_gold(split="test")
    catalog, id_to_text = u._load_catalog_as_problems(ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    tfidf = get_baseline("tfidf")
    tfidf.fit(catalog)

    start_top1 = time.perf_counter()
    for ex in eval_items:
        tfidf.rank(ex["query"], top_k=1)
    top1_runtime = time.perf_counter() - start_top1

    start = time.perf_counter()
    per_query: dict[str, dict[str, Any]] = {}
    top10_groundings: dict[str, list[CandidateGrounding]] = {}
    schema_miss_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for ex in eval_items:
        qid = ex["query_id"]
        query = ex["query"]
        gold_schema = ex["relevant_doc_id"]
        ranked_full = tfidf.rank(query, top_k=len(catalog))
        ranked10 = ranked_full[:10]
        rank_map = {sid: i + 1 for i, (sid, _score) in enumerate(ranked_full)}
        top1_id, top1_score = ranked10[0]
        top2_score = ranked10[1][1] if len(ranked10) > 1 else top1_score
        margin = top1_score - top2_score
        cands: list[CandidateGrounding] = []
        for i, (sid, score) in enumerate(ranked10, 1):
            c = ground_candidate(
                query_id=qid,
                query=query,
                gold_schema=gold_schema,
                schema_id=sid,
                rank=i,
                retrieval_score=score,
                retrieval_margin=margin,
                gold_by_id=gold_by_id,
                id_to_text=id_to_text,
            )
            cands.append(c)
            candidate_rows.append(_candidate_row(c))
        top10_groundings[qid] = cands
        gold_rank = rank_map.get(gold_schema, len(catalog) + 1)
        top1 = cands[0]
        gold_ground = next((c for c in cands if c.schema_hit), None)
        per_query[qid] = {
            "query": query,
            "query_len": len(query.split()),
            "gold_schema": gold_schema,
            "top1_schema": top1_id,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "retrieval_margin": margin,
            "gold_rank": gold_rank,
            "baseline": top1,
            "gold_grounding_top10": gold_ground,
        }
        if top1_id != gold_schema:
            row = {
                "problem_id": qid,
                "gold_schema": gold_schema,
                "tfidf_top1_schema": top1_id,
                "topk_schemas": json.dumps([sid for sid, _ in ranked10]),
                "topk_scores": json.dumps([score for _sid, score in ranked10]),
                "top1_top2_margin": margin,
                "gold_rank": gold_rank,
                "gold_in_top2": int(gold_rank <= 2),
                "gold_in_top3": int(gold_rank <= 3),
                "gold_in_top5": int(gold_rank <= 5),
                "gold_in_top10": int(gold_rank <= 10),
                "query_length": len(query.split()),
                "lexical_overlap_gold": lexical_overlap(query, id_to_text.get(gold_schema, "")),
                "lexical_overlap_top1": lexical_overlap(query, id_to_text.get(top1_id, "")),
                "pred_ready": int(top1.ready),
                "pred_coverage": top1.coverage,
                "pred_type_match": top1.type_match,
                "gold_ready_if_top10": int(gold_ground.ready) if gold_ground else "",
                "gold_coverage_if_top10": gold_ground.coverage if gold_ground else "",
                "gold_type_match_if_top10": gold_ground.type_match if gold_ground else "",
            }
            schema_miss_rows.append(row)

    baseline = {qid: d["baseline"] for qid, d in per_query.items()}
    current_ready = sum(1 for c in baseline.values() if c.ready)
    current_schema = sum(1 for c in baseline.values() if c.schema_hit)
    elapsed_full_top10 = time.perf_counter() - start

    oracle_rows: list[dict[str, Any]] = []
    oracle_summary: dict[str, Any] = {}
    for k in K_VALUES:
        selected_gold: dict[str, CandidateGrounding] = {}
        selected_ready: dict[str, CandidateGrounding] = {}
        selected_gold_ready: dict[str, CandidateGrounding] = {}
        gold_in_k = 0
        true_rescues = 0
        wrong_ready = 0
        for qid, cands10 in top10_groundings.items():
            cands = cands10[:k]
            base = baseline[qid]
            gold = next((c for c in cands if c.schema_hit), None)
            if gold is not None:
                gold_in_k += 1
            selected_gold[qid] = gold if gold is not None else base
            ready_cands = [c for c in cands if c.ready]
            ready_choice = max(ready_cands, key=lambda c: (c.retrieval_score, -c.rank)) if ready_cands else base
            selected_ready[qid] = ready_choice
            selected_gold_ready[qid] = gold if (gold is not None and gold.ready) else base
            if (not base.ready) and gold is not None and gold.ready:
                true_rescues += 1
            if ready_choice.ready and not ready_choice.schema_hit:
                wrong_ready += 1
        for label, selected in (
            ("oracle_A_gold_schema", selected_gold),
            ("oracle_B_readiness", selected_ready),
            ("oracle_C_gold_ready", selected_gold_ready),
        ):
            s = summarize_selection(f"{label}_k{k}", selected, baseline)
            row = {
                **s,
                "k": k,
                "oracle": label,
                "gold_in_k": gold_in_k,
                "true_rescued_queries": true_rescues if label != "oracle_B_readiness" else "",
                "wrong_schema_ready_false_positives": wrong_ready if label == "oracle_B_readiness" else "",
                "projected_instantiation_ready": s["instantiation_ready"],
            }
            oracle_rows.append(row)
            oracle_summary[f"{label}_k{k}"] = row

    miss_class_rows: list[dict[str, Any]] = []
    miss_class_counts: dict[str, int] = {}
    for row in schema_miss_rows:
        qid = row["problem_id"]
        base = baseline[qid]
        gold_rank = int(row["gold_rank"])
        gold = per_query[qid]["gold_grounding_top10"]
        wrong_ready = base.ready and not base.schema_hit
        if gold_rank > 10:
            cls = "C_GOLD_NOT_IN_TOP10"
        elif gold is not None and gold.ready:
            cls = "A_RETRIEVAL_FIX_RESCUES_READY"
        elif gold is not None:
            cls = "B_RETRIEVAL_FIX_BUT_GROUNDING_STILL_FAILS"
        elif wrong_ready:
            cls = "D_WRONG_SCHEMA_READY_FALSE_POSITIVE"
        else:
            cls = "C_GOLD_NOT_IN_TOP10"
        miss_class_counts[cls] = miss_class_counts.get(cls, 0) + 1
        miss_class_rows.append({**row, "classification": cls})

    margin_rows: list[dict[str, Any]] = []
    miss_ids = {r["problem_id"] for r in schema_miss_rows}
    rescuable_ids_by_k = {
        k: {
            qid for qid, d in per_query.items()
            if (not baseline[qid].ready)
            and any(c.schema_hit and c.ready for c in top10_groundings[qid][:k])
        }
        for k in (2, 3, 5, 10)
    }
    ready_correct_ids = {qid for qid, c in baseline.items() if c.ready and c.schema_hit}
    for th in MARGIN_THRESHOLDS:
        triggered = {qid for qid, d in per_query.items() if d["retrieval_margin"] <= th}
        row = {
            "threshold": th,
            "triggered_queries": len(triggered),
            "trigger_rate": len(triggered) / len(per_query),
            "schema_miss_recall": len(triggered & miss_ids) / max(1, len(miss_ids)),
            "schema_misses_captured": len(triggered & miss_ids),
            "ready_correct_triggered": len(triggered & ready_correct_ids),
            "false_trigger_rate_ready_correct": len(triggered & ready_correct_ids) / max(1, len(ready_correct_ids)),
            "true_rescuable_k2_captured": len(triggered & rescuable_ids_by_k[2]),
            "true_rescuable_k3_captured": len(triggered & rescuable_ids_by_k[3]),
            "true_rescuable_k5_captured": len(triggered & rescuable_ids_by_k[5]),
            "true_rescuable_k10_captured": len(triggered & rescuable_ids_by_k[10]),
        }
        margin_rows.append(row)

    rules = (
        "R0_tfidf_top1",
        "R1_max_coverage",
        "R2_max_typematch",
        "R3_ready_cov_type_tfidf",
        "R4_verified_cov_type_tfidf",
        "R5_small_consistency_score",
    )
    reranker_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    method_selected: dict[str, dict[str, CandidateGrounding]] = {}

    for k in (2, 3, 5):
        for rule in rules:
            selected = {qid: select_by_rule(cands[:k], rule) for qid, cands in top10_groundings.items()}
            name = f"all_k{k}_{rule}"
            method_selected[name] = selected
            reranker_rows.append({**summarize_selection(name, selected, baseline), "k": k, "rule": rule, "threshold": "ALL", "reranked_queries": len(per_query)})

    for k in (2, 3, 5):
        for th in MARGIN_THRESHOLDS:
            triggered = {qid for qid, d in per_query.items() if d["retrieval_margin"] <= th}
            for rule in ("R3_ready_cov_type_tfidf", "R4_verified_cov_type_tfidf", "R5_small_consistency_score"):
                selected = {}
                for qid, cands in top10_groundings.items():
                    selected[qid] = select_by_rule(cands[:k], rule) if qid in triggered else baseline[qid]
                name = f"selective_k{k}_margin{th:g}_{rule}"
                method_selected[name] = selected
                reranker_rows.append({**summarize_selection(name, selected, baseline), "k": k, "rule": rule, "threshold": th, "reranked_queries": len(triggered)})

    for name, selected in method_selected.items():
        ready_both = ready_base_only = ready_candidate_only = ready_neither = 0
        schema_both = schema_base_only = schema_candidate_only = schema_neither = 0
        for qid, cand in selected.items():
            base = baseline[qid]
            if base.ready and cand.ready:
                ready_both += 1
            elif base.ready and not cand.ready:
                ready_base_only += 1
            elif cand.ready and not base.ready:
                ready_candidate_only += 1
            else:
                ready_neither += 1
            if base.schema_hit and cand.schema_hit:
                schema_both += 1
            elif base.schema_hit and not cand.schema_hit:
                schema_base_only += 1
            elif cand.schema_hit and not base.schema_hit:
                schema_candidate_only += 1
            else:
                schema_neither += 1
        transition_rows.append({
            "method": name,
            "ready_both": ready_both,
            "ready_baseline_only": ready_base_only,
            "ready_candidate_only": ready_candidate_only,
            "ready_neither": ready_neither,
            "ready_mcnemar_p": exact_mcnemar_p(ready_base_only, ready_candidate_only),
            "schema_both_correct": schema_both,
            "schema_baseline_only_correct": schema_base_only,
            "schema_candidate_only_correct": schema_candidate_only,
            "schema_both_wrong": schema_neither,
            "schema_mcnemar_p": exact_mcnemar_p(schema_base_only, schema_candidate_only),
        })

    best = max(
        reranker_rows,
        key=lambda r: (r["ready_count"], r["schema_R1"], -int(r["schema_regressions"]), -int(r["reranked_queries"])),
    )
    selective_rows = [r for r in reranker_rows if r["threshold"] != "ALL"]
    best_selective = max(
        selective_rows,
        key=lambda r: (r["ready_count"], r["schema_R1"], -int(r["schema_regressions"]), -int(r["reranked_queries"])),
    )
    best_no_schema_regression = max(
        (r for r in reranker_rows if int(r["schema_regressions"]) == 0),
        key=lambda r: (r["ready_count"], r["schema_R1"], -int(r["reranked_queries"])),
    )

    rank_distribution = {
        "rank_2": sum(1 for r in schema_miss_rows if int(r["gold_rank"]) == 2),
        "rank_3": sum(1 for r in schema_miss_rows if int(r["gold_rank"]) == 3),
        "rank_4_5": sum(1 for r in schema_miss_rows if 4 <= int(r["gold_rank"]) <= 5),
        "rank_6_10": sum(1 for r in schema_miss_rows if 6 <= int(r["gold_rank"]) <= 10),
        "rank_gt_10": sum(1 for r in schema_miss_rows if int(r["gold_rank"]) > 10),
    }
    true_rescues_k = {
        str(k): len(rescuable_ids_by_k[k])
        for k in (2, 3, 5, 10)
    }
    viable_rows: list[dict[str, Any]] = []
    best_viable: dict[str, Any] | None = None
    decision = "TOP2_NO_GO"
    max_true_rescues = max(true_rescues_k.values())
    if max_true_rescues >= 7:
        viable_rows = [
            r for r in reranker_rows
            if int(r["ready_count"]) >= 264 and int(r["schema_regressions"]) <= 1
        ]
        if viable_rows:
            best_viable = max(
                viable_rows,
                key=lambda r: (
                    r["threshold"] != "ALL",
                    -int(r["schema_regressions"]),
                    -int(r["reranked_queries"]),
                    r["ready_count"],
                    r["schema_R1"],
                ),
            )
            decision = "TOP2_GO"
        else:
            decision = "TOP2_WEAK_GO"

    summary = {
        "n_total_queries": len(per_query),
        "current_ready": current_ready,
        "current_instantiation_ready": current_ready / len(per_query),
        "current_schema_correct": current_schema,
        "current_schema_R1": current_schema / len(per_query),
        "schema_misses": len(schema_miss_rows),
        "rank_distribution": rank_distribution,
        "oracle_summary": oracle_summary,
        "true_rescues_by_k": true_rescues_k,
        "miss_class_counts": miss_class_counts,
        "best_reranker": best,
        "best_selective_reranker": best_selective,
        "best_no_schema_regression_reranker": best_no_schema_regression,
        "best_stage_b_viable_reranker": best_viable,
        "top1_runtime_seconds": top1_runtime,
        "full_top10_diagnostic_runtime_seconds": elapsed_full_top10,
        "estimated_top1_runtime_seconds": top1_runtime,
        "api_oracle": "NOT_USED",
        "decision": decision,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "schema_misses.csv", miss_class_rows, [
        "problem_id", "gold_schema", "tfidf_top1_schema", "topk_schemas", "topk_scores",
        "top1_top2_margin", "gold_rank", "gold_in_top2", "gold_in_top3",
        "gold_in_top5", "gold_in_top10", "query_length", "lexical_overlap_gold",
        "lexical_overlap_top1", "pred_ready", "pred_coverage", "pred_type_match",
        "gold_ready_if_top10", "gold_coverage_if_top10", "gold_type_match_if_top10",
        "classification",
    ])
    _write_csv(out_dir / "oracle_topk.csv", oracle_rows, [
        "k", "oracle", "method", "gold_in_k", "true_rescued_queries",
        "wrong_schema_ready_false_positives", "schema_R1", "instantiation_ready",
        "ready_count", "projected_instantiation_ready", "coverage", "type_match",
        "schema_recoveries", "schema_regressions", "ready_gains", "ready_losses",
    ])
    _write_csv(out_dir / "margin_analysis.csv", margin_rows, [
        "threshold", "triggered_queries", "trigger_rate", "schema_miss_recall",
        "schema_misses_captured", "ready_correct_triggered",
        "false_trigger_rate_ready_correct", "true_rescuable_k2_captured",
        "true_rescuable_k3_captured", "true_rescuable_k5_captured",
        "true_rescuable_k10_captured",
    ])
    _write_csv(out_dir / "reranker_results.csv", reranker_rows, [
        "method", "k", "rule", "threshold", "reranked_queries", "schema_R1",
        "instantiation_ready", "ready_count", "coverage", "type_match",
        "schema_recoveries", "schema_regressions", "ready_gains", "ready_losses",
    ])
    _write_csv(out_dir / "transitions.csv", transition_rows, [
        "method", "ready_both", "ready_baseline_only", "ready_candidate_only",
        "ready_neither", "ready_mcnemar_p", "schema_both_correct",
        "schema_baseline_only_correct", "schema_candidate_only_correct",
        "schema_both_wrong", "schema_mcnemar_p",
    ])
    _write_csv(out_dir / "candidate_groundings.csv", candidate_rows, [
        "problem_id", "gold_schema", "schema_id", "rank", "retrieval_score",
        "retrieval_margin", "schema_hit", "n_expected_scalar", "n_filled",
        "coverage", "type_match", "ready", "key_overlap", "extracted_number_count",
        "unmatched_mention_count", "incompatible_assignment_count", "null_slot_count",
        "min_assignment_margin", "mean_assignment_margin", "lexical_overlap_query_schema",
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
