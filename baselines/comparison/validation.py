"""Provenance validation, mock-evidence exclusion, and duplicate-run handling.

Nothing here silently repairs bad input: `validate_row` returns a list of
problems (empty = clean); callers decide whether to reject or warn.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from baselines.comparison.schema import CellState, UnifiedRow, is_state

_MOCK_MARKERS = ("MOCK", "PROXY_OR_MOCK", "PIPELINE_MOCK")


def is_mock_evidence(row: UnifiedRow) -> bool:
    """Conservative heuristic: treat as mock unless clearly real.

    Checked, in order: an explicit `MOCK`/`PROXY` marker in the checkpoint
    identifier (every baseline's mock pipeline in this repository names its
    backend this way -- e.g. `DEEPOR_PIPELINE_MOCK_OR_PROXY`,
    `ORR1_PIPELINE_MOCK_OR_PROXY`), then a missing git SHA (real runs in
    this repository always record one).
    """
    checkpoint = row.checkpoint_model
    if isinstance(checkpoint, str) and any(marker in checkpoint.upper() for marker in _MOCK_MARKERS):
        return True
    if row.local_git_sha in (CellState.UNKNOWN, None, ""):
        return True
    return False


def validate_row(row: UnifiedRow, *, known_problem_ids: set[str] | None = None) -> list[str]:
    problems: list[str] = []
    if not row.problem_id or row.problem_id == CellState.UNKNOWN:
        problems.append("missing_problem_id")
    elif known_problem_ids is not None and row.problem_id not in known_problem_ids:
        problems.append(f"problem_id_not_in_manifest:{row.problem_id}")
    if row.local_git_sha in (CellState.UNKNOWN, None, ""):
        problems.append("missing_git_sha")
    if is_state(row.checkpoint_model) and row.checkpoint_model not in (CellState.NOT_APPLICABLE,):
        problems.append(f"checkpoint_model_state:{row.checkpoint_model}")
    if row.system not in {"ours", "pamop", "orlm", "optmath", "generic", "deepor", "orr1"}:
        problems.append(f"unknown_system:{row.system}")
    if is_mock_evidence(row):
        problems.append("mock_evidence")
    return problems


@dataclass(frozen=True)
class RunKey:
    system: str
    method_variant: str
    checkpoint_revision: object


def detect_ambiguous_runs(rows: list[UnifiedRow]) -> dict[tuple[str, str], list[RunKey]]:
    """Group by (system, problem_id); flag groups spanning >1 distinct run identity.

    Returns only the ambiguous groups. Callers MUST NOT silently pick a
    "best" row from an ambiguous group -- report it and require explicit
    run selection (Phase 20: no best-of-runs cherry-picking).
    """
    by_key: dict[tuple[str, str], set[RunKey]] = defaultdict(set)
    for row in rows:
        by_key[(row.system, row.problem_id)].add(RunKey(row.system, row.method_variant, row.checkpoint_revision))
    return {k: sorted(v, key=lambda rk: (rk.method_variant, str(rk.checkpoint_revision))) for k, v in by_key.items() if len(v) > 1}


def select_rows(
    rows: list[UnifiedRow], *, allow_mock: bool = False, known_problem_ids: set[str] | None = None,
) -> tuple[list[UnifiedRow], dict[str, list[str]]]:
    """Filter to valid, non-ambiguous, (by default) non-mock rows.

    Returns `(accepted, rejected_reasons)`. Ambiguous (system, problem_id)
    groups are rejected entirely (not resolved) unless the caller has
    already deduplicated upstream.
    """
    ambiguous = detect_ambiguous_runs(rows)
    accepted: list[UnifiedRow] = []
    rejected: dict[str, list[str]] = {}
    for row in rows:
        key = (row.system, row.problem_id)
        reasons = validate_row(row, known_problem_ids=known_problem_ids)
        if not allow_mock and "mock_evidence" in reasons:
            pass  # kept in reasons, row rejected below
        elif allow_mock:
            reasons = [r for r in reasons if r != "mock_evidence"]
        if key in ambiguous:
            reasons.append("ambiguous_run_selection_required")
        if reasons:
            rejected[f"{row.system}:{row.problem_id}"] = reasons
        else:
            accepted.append(row)
    return accepted, rejected
