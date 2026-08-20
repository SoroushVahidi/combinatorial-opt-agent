"""Rescore stored results; objective equality remains a proxy, not semantics."""
from __future__ import annotations
from typing import Any, Iterable

def objective_proxy(objective: float | None, gold: Any, tolerance: float = 0.05) -> str:
    try:
        if objective is None or gold is None: return "NOT_EVALUABLE"
        g=float(gold); return "PASS" if abs(objective-g) <= tolerance*max(1.0,abs(g)) else "FAIL"
    except (TypeError, ValueError): return "NOT_EVALUABLE"

def summarize(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows=list(results); n=len(rows)
    def count(pred): return sum(bool(pred(r)) for r in rows)
    return {"n":n, "generation_completed":count(lambda r:r.get("generation",{}).get("status")=="COMPLETED"), "parse_success":count(lambda r:r.get("parsed",{}).get("status","").endswith("EXTRACTED")), "static_valid":count(lambda r:r.get("static_validation",{}).get("status")=="STATIC_VALID"), "execution_success":count(lambda r:r.get("execution",{}).get("status")=="COMPLETED"), "objective_proxy_is_not_semantic":True}
