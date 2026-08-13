"""Inference-independent OptMATH metrics."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def evaluate_results(results: Iterable[dict[str, Any]], tolerance: float = 0.05) -> dict[str, Any]:
    rows = list(results); n = len(rows)
    rate = lambda count: count / n if n else None
    generated = sum(r.get("generation", {}).get("status") == "COMPLETED" for r in rows)
    parsed = sum(bool((r.get("parsed") or {}).get("generated_code")) for r in rows)
    static = sum((r.get("static_validation") or {}).get("status") == "STATIC_VALID" for r in rows)
    executed = sum(r.get("execution_attempted", False) and (r.get("execution") or {}).get("status") == "COMPLETED" for r in rows)
    proxy: list[bool] = []
    for row in rows:
        gold, pred = row.get("gold_objective"), row.get("objective_value")
        if gold is None or pred is None or isinstance(gold, str): continue
        gold_r, pred_r = round(float(gold)), round(float(pred))
        proxy.append(abs(pred_r - gold_r) <= tolerance if gold_r == 0 else abs((pred_r - gold_r) / gold_r) <= tolerance)
    categories = Counter(r.get("error_category") or (r.get("execution") or {}).get("error_category") or "none" for r in rows)
    return {"n_attempted": n, "generation_success_rate": rate(generated), "parse_success_rate": rate(parsed), "static_valid_code_rate": rate(static), "execution_success_rate": rate(executed), "objective_proxy_evaluable": len(proxy), "objective_proxy_accuracy": sum(proxy) / len(proxy) if proxy else None, "semantic_accuracy": None, "semantic_accuracy_status": "NOT_EQUIVALENT_TO_SOLVER_VERIFIED_OPTIMATH_ACCURACY", "total_tokens": sum(int(r.get("generation", {}).get("token_counts", {}).get("total_tokens") or 0) for r in rows), "failure_categories": dict(categories)}
