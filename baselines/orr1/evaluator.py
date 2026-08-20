"""Official pass@k / mj@k scoring plus offline generation/parse/static metrics.

`compute_solving_accuracy` reproduces `eval/execute.py`'s scoring section
exactly (grouping by `problem_id`, `_within_tolerance`'s asymmetric handling
of "No Best Solution" and zero-gold objectives). It never substitutes this
repository's own `InstantiationReady`-style metric for OR-R1's official
accuracy definition.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from baselines.orr1.rollout import score_group


def evaluate_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    n = len(rows)

    def rate(count: int) -> float | None:
        return count / n if n else None

    generation = sum(r.get("failure_category") is None for r in rows)
    parsed = sum(bool((r.get("parsed") or {}).get("coptpy_code")) for r in rows)
    static_valid = sum((r.get("static_validation") or {}).get("status") == "STATIC_VALID" for r in rows)
    executed = sum(r.get("execution_attempted", False) and str((r.get("execution") or {}).get("status", "")).startswith("COMPLETED") for r in rows)
    return {
        "n_rollouts": n,
        "generation_success_rate": rate(generation),
        "parse_success_rate": rate(parsed),
        "static_valid_code_rate": rate(static_valid),
        "execution_attempted_rate": rate(executed),
        "mean_runtime_seconds": sum(float(r.get("runtime_seconds") or 0.0) for r in rows) / n if n else None,
    }


def compute_solving_accuracy(results: Iterable[dict[str, Any]], *, tolerance: float = 0.05) -> dict[str, Any]:
    """Group rollouts by `problem_id`, then apply official pass@k / mj@k.

    Every group in the input must share one gold answer, matching upstream's
    `assert len(set(gt_answers)) == 1`.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[row["problem_id"]].append(row)

    per_problem: dict[str, dict[str, Any]] = {}
    pass_hits = 0
    mj_hits = 0
    k_values: set[int] = set()
    for problem_id, rows in groups.items():
        gold_values = {row.get("gold_objective") for row in rows}
        if len(gold_values) != 1:
            raise ValueError(f"problem {problem_id!r} has inconsistent gold_objective across rollouts: {gold_values}")
        gold = next(iter(gold_values))
        pred_answers = [row.get("objective") if row.get("execution", {}).get("best_solution") is None else row["execution"]["best_solution"] for row in rows]
        score = score_group(pred_answers, gold, tolerance=tolerance)
        per_problem[problem_id] = score.to_dict()
        pass_hits += int(score.pass_at_k)
        mj_hits += int(score.mj_at_k)
        k_values.add(score.k)

    n = len(groups)
    k = next(iter(k_values)) if len(k_values) == 1 else None
    return {
        "n_problems": n,
        "rollout_group_size": k,
        "rollout_group_sizes_inconsistent": None if k is not None else sorted(k_values),
        f"pass@{k}" if k else "pass@k": pass_hits / n if n else None,
        f"mj@{k}" if k else "mj@k": mj_hits / n if n else None,
        "per_problem": per_problem,
    }
