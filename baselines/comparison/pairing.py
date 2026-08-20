"""Pair two systems' UnifiedRows on `problem_id` for a chosen boolean metric.

Pairing requires both systems to have a MEASURED (non-CellState) boolean
value for the same problem_id under the same metric; anything else is
dropped from the pair set and counted, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from baselines.comparison.schema import UnifiedRow, is_measured
from baselines.comparison.statistics import TransitionTable, build_transition_table


@dataclass(frozen=True)
class PairingResult:
    metric_name: str
    system_a: str
    system_b: str
    paired_problem_ids: tuple[str, ...]
    table: TransitionTable
    unpaired_a_only: tuple[str, ...]  # problem_ids where A has a measurement but B does not
    unpaired_b_only: tuple[str, ...]
    unmeasured: tuple[str, ...]  # problem_ids present in both but the metric isn't measured for at least one

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name, "system_a": self.system_a, "system_b": self.system_b,
            "paired_problem_ids": list(self.paired_problem_ids), "table": self.table.to_dict(),
            "unpaired_a_only": list(self.unpaired_a_only), "unpaired_b_only": list(self.unpaired_b_only),
            "unmeasured": list(self.unmeasured),
        }


def pair_systems(
    rows_a: list[UnifiedRow], rows_b: list[UnifiedRow], *, metric: Callable[[UnifiedRow], object], metric_name: str,
) -> PairingResult:
    by_id_a = {r.problem_id: r for r in rows_a}
    by_id_b = {r.problem_id: r for r in rows_b}
    common_ids = sorted(set(by_id_a) & set(by_id_b))
    a_only_ids = sorted(set(by_id_a) - set(by_id_b))
    b_only_ids = sorted(set(by_id_b) - set(by_id_a))

    paired: list[str] = []
    unmeasured: list[str] = []
    outcomes: list[tuple[bool, bool]] = []
    for pid in common_ids:
        va, vb = metric(by_id_a[pid]), metric(by_id_b[pid])
        if isinstance(va, bool) and isinstance(vb, bool):
            paired.append(pid)
            outcomes.append((va, vb))
        else:
            unmeasured.append(pid)

    system_a = rows_a[0].system if rows_a else "unknown_a"
    system_b = rows_b[0].system if rows_b else "unknown_b"
    return PairingResult(
        metric_name, system_a, system_b, tuple(paired), build_transition_table(outcomes),
        tuple(a_only_ids), tuple(b_only_ids), tuple(unmeasured),
    )
