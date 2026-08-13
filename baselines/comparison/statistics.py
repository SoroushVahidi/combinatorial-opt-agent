"""Lightweight statistics for proportions and paired binary comparisons.

No SciPy dependency: the McNemar exact p-value uses the closed-form binomial
tail (`math.comb`), not a chi-square approximation, so it is exact for any
n including small samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WilsonInterval:
    point_estimate: float
    lower: float
    upper: float
    n: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> WilsonInterval:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence)
    if z is None:
        raise ValueError(f"unsupported confidence level: {confidence} (use 0.90, 0.95, or 0.99)")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return WilsonInterval(p, max(0.0, center - half), min(1.0, center + half), n, confidence)


@dataclass(frozen=True)
class TransitionTable:
    """2x2 table of paired binary outcomes between systems A and B."""

    both_succeed: int
    a_only: int
    b_only: int
    neither: int

    @property
    def n(self) -> int:
        return self.both_succeed + self.a_only + self.b_only + self.neither

    def to_dict(self) -> dict[str, object]:
        return {"both_succeed": self.both_succeed, "a_only": self.a_only, "b_only": self.b_only, "neither": self.neither, "n": self.n}


def build_transition_table(paired_outcomes: list[tuple[bool, bool]]) -> TransitionTable:
    both = sum(1 for a, b in paired_outcomes if a and b)
    a_only = sum(1 for a, b in paired_outcomes if a and not b)
    b_only = sum(1 for a, b in paired_outcomes if not a and b)
    neither = sum(1 for a, b in paired_outcomes if not a and not b)
    return TransitionTable(both, a_only, b_only, neither)


@dataclass(frozen=True)
class McNemarResult:
    a_only: int
    b_only: int
    discordant_n: int
    p_value: float | None
    note: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def mcnemar_exact(table: TransitionTable) -> McNemarResult:
    """Exact two-sided McNemar test via the binomial tail on discordant pairs.

    Returns `p_value=None` (never a fabricated number) whenever there are
    zero discordant pairs -- the test is undefined, not "not significant".
    """
    n = table.a_only + table.b_only
    if n == 0:
        return McNemarResult(table.a_only, table.b_only, 0, None, "zero discordant pairs: McNemar's test is undefined, not non-significant")
    k = min(table.a_only, table.b_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * tail)
    note = "exact binomial two-sided test on discordant pairs"
    if n < 10:
        note += "; n_discordant < 10, treat as a small-sample indication only"
    return McNemarResult(table.a_only, table.b_only, n, p, note)
