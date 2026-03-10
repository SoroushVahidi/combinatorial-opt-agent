#!/usr/bin/env python3
"""Run our model (tfidf + global_consistency_grounding) on all benchmark cases.

Writes: artifacts/copilot_vs_model/our_model_outputs.jsonl

Usage:
    python artifacts/copilot_vs_model/run_our_model.py

Requirements:
    pip install scikit-learn rank_bm25
No GPU, no internet, no HF auth required.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

BENCH_FILE = ROOT / "artifacts" / "copilot_vs_model" / "benchmark_cases.jsonl"
OUT_FILE   = ROOT / "artifacts" / "copilot_vs_model" / "our_model_outputs.jsonl"
CATALOG    = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"

STOPWORDS = {
    'The','Each','All','Maximize','Minimize','Given','To','No','In','If','It',
    'On','An','Or','By','At','For','Is','Are','Let','We','Determine','Find',
    'Calculate','Compute','Sum','Also','Total','Note','Both','With','When',
    'They','This','That','These','Then','Thus','Any','One','Two',
}


def extract_clean_slots(text: str) -> list[str]:
    raw = re.findall(r'\b([A-Z][A-Za-z][A-Za-z0-9]*)\b', text)
    seen: set[str] = set()
    out = []
    for t in raw:
        if t not in STOPWORDS and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def infer_objective(schema_text: str, query: str) -> str:
    q_lo = query.lower()
    s_lo = schema_text.lower()
    if "maximize" in s_lo or "maximize" in q_lo:
        return "maximize"
    if "minimize" in s_lo or "minimize" in q_lo:
        return "minimize"
    return "unknown"


def run() -> None:
    from retrieval.baselines import get_baseline
    from tools.nlp4lp_downstream_utility import (
        _load_catalog_as_problems,
        _run_global_consistency_grounding,
    )

    # Load catalog
    catalog, id_to_text = _load_catalog_as_problems(CATALOG)
    retriever = get_baseline("tfidf")
    retriever.fit(catalog)

    # Load benchmark
    cases = [json.loads(l) for l in BENCH_FILE.open()]
    print(f"Running {len(cases)} benchmark cases …")

    results = []
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        query    = case["input_text"]
        case_id  = case["case_id"]
        gold_sid = case.get("gold_schema_id", "")

        # Stage 1 – schema retrieval
        ranked   = retriever.rank(query, top_k=3)
        pred_id  = ranked[0][0] if ranked else ""
        pred_scores = [(pid, round(sc, 4)) for pid, sc in ranked[:3]]
        schema_text = id_to_text.get(pred_id, "")
        schema_correct = int(pred_id == gold_sid)

        # Stage 2 – slot name extraction from predicted schema
        slots = extract_clean_slots(schema_text)

        # Stage 3 – numeric grounding (global_consistency_grounding)
        filled_values: dict = {}
        diagnostics: dict = {}
        if slots:
            try:
                filled_values, _, diagnostics = _run_global_consistency_grounding(
                    query, "orig", slots
                )
            except Exception as exc:
                diagnostics = {"error": str(exc)}

        # Collect output
        elapsed = time.perf_counter() - t0
        out = {
            "case_id": case_id,
            "input_text": query,
            "gold_schema_id": gold_sid,
            # retrieval
            "predicted_schema_id": pred_id,
            "retrieval_top3": pred_scores,
            "schema_correct": schema_correct,
            # schema
            "predicted_schema_text": schema_text,
            "predicted_objective_direction": infer_objective(schema_text, query),
            # grounding
            "predicted_slots": slots,
            "slot_value_assignments": {k: v for k, v in filled_values.items() if v is not None},
            # meta
            "method": "tfidf+global_consistency_grounding",
            "elapsed_sec": round(elapsed, 4),
            "grounding_diagnostics": {
                "n_top_assignments": len(diagnostics.get("top_assignments", [])),
                "top_assignment": (diagnostics.get("top_assignments") or [{}])[0],
            },
        }
        results.append(out)
        status = "✓" if schema_correct else "✗"
        n_filled = len(out["slot_value_assignments"])
        print(f"  [{i:02d}/{len(cases)}] {status} {case_id} | pred={pred_id} | filled={n_filled}/{len(slots)} slots")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_correct = sum(r["schema_correct"] for r in results)
    print(f"\nSchema retrieval accuracy: {n_correct}/{len(results)} = {n_correct/len(results):.1%}")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    run()
