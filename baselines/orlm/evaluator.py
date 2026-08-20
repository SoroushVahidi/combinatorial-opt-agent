"""Metrics for completed ORLM result records; no model execution required."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def evaluate_results(results: Iterable[dict[str, Any]], tolerance: float = 1e-6) -> dict[str, Any]:
    rows = list(results)
    n = len(rows)
    def rate(count: int) -> float | None:
        return count / n if n else None
    generation = sum(r.get("generation", {}).get("status") == "COMPLETED" for r in rows)
    parsed = sum(bool((r.get("parsed") or {}).get("coptpy_code")) for r in rows)
    static = sum((r.get("static_validation") or {}).get("status") == "STATIC_VALID" for r in rows)
    executed = sum(r.get("execution_attempted", False) and r.get("execution", {}).get("status") == "COMPLETED" for r in rows)
    proxy = []
    for r in rows:
        if r.get("gold_objective") is not None and r.get("objective_value") is not None:
            proxy.append(abs(float(r["gold_objective"]) - float(r["objective_value"])) <= tolerance)
    errors = Counter(r.get("error_category") or (r.get("execution") or {}).get("error_category") or "none" for r in rows)
    return {
        "n_attempted": n,
        "generation_success_rate": rate(generation),
        "parse_success_rate": rate(parsed),
        "static_valid_code_rate": rate(static),
        "execution_success_rate": rate(executed),
        "objective_value_proxy_evaluable": len(proxy),
        "objective_value_proxy_accuracy": sum(proxy) / len(proxy) if proxy else None,
        "mean_runtime_seconds": sum(float(r.get("generation", {}).get("runtime_seconds", 0.0)) for r in rows) / n if n else None,
        "total_tokens": sum(int(r.get("generation", {}).get("token_counts", {}).get("total_tokens") or 0) for r in rows),
        "failure_categories": dict(errors),
        "semantic_accuracy": None,
        "semantic_accuracy_status": "NOT_IMPLEMENTED_WITHOUT_GOLD_MODEL_EQUIVALENCE",
    }
