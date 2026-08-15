"""Metric taxonomy: NATIVE (system-specific) vs. SHARED (genuinely comparable).

Never combine these into one number. `NATIVE_METRICS` documents what each
system's own literature/evaluator reports; `SHARED_METRICS` documents only
the metrics this repository has verified can be computed identically across
multiple systems from a `UnifiedRow`. Adding a system to a shared metric's
`applicable_systems` list is a scientific claim -- do not do it without
checking the eligibility notes in `END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from baselines.comparison.schema import CellState, UnifiedRow, is_measured


@dataclass(frozen=True)
class NativeMetric:
    system: str
    name: str
    description: str
    source: str


NATIVE_METRICS: tuple[NativeMetric, ...] = (
    NativeMetric("ours", "schema_R1", "Schema Recall@1 against the fixed NLP4LP catalog.", "training/external/run_full_downstream_benchmark.py"),
    NativeMetric("ours", "param_coverage", "Fraction of expected scalar slots filled ('Coverage').", "same"),
    NativeMetric("ours", "type_match", "Type-correctness of filled scalar slots ('TypeMatch').", "same"),
    NativeMetric("ours", "instantiation_ready", "Coverage >= 0.8 AND TypeMatch >= 0.8 under the predicted schema; schema correctness is reported separately.", "same"),
    NativeMetric("ours", "strict_instantiation_ready", "Schema hit AND Coverage >= 0.8 AND TypeMatch >= 0.8; native schema-gated end-to-end readiness diagnostic.", "same"),
    NativeMetric("pamop", "execution_success_rate", "AMPL parse+solve succeeds (initial or after correction).", "results/pamop/*/summary.json"),
    NativeMetric("pamop", "objective_value_proxy", "Predicted objective within tolerance of gold, execution-successful rows only.", "same"),
    NativeMetric("pamop", "correction_rescue_count", "Rows that failed initially but succeeded after the G_exe/G_rev/G_comp/G_remod loop.", "same"),
    NativeMetric("orlm", "objective_value_proxy_accuracy", "Rounded relative-tolerance match vs. gold, execution-successful rows only.", "baselines/orlm/evaluator.py"),
    NativeMetric("optmath", "objective_proxy_accuracy", "Same tolerance family as ORLM; official OptMATH accuracy is not claimed.", "baselines/optmath/evaluator.py"),
    NativeMetric("deepor", "objective_value_pass_at_1", "Paper-specified pass@1 on objective value; not reproducible without an official checkpoint.", "docs/DEEPOR_PROVENANCE.md"),
    NativeMetric("orr1", "pass_at_k", "Official: any of k rollouts within 5% of gold (k=1 or 8).", "baselines/orr1/rollout.py"),
    NativeMetric("orr1", "mj_at_k", "Official: majority-voted rollout within 5% of gold.", "baselines/orr1/rollout.py"),
)


@dataclass(frozen=True)
class SharedMetric:
    name: str
    description: str
    applicable_systems: tuple[str, ...]
    compute: Callable[[list[UnifiedRow]], dict[str, object]]


def _rate(rows: list[UnifiedRow], predicate: Callable[[UnifiedRow], bool | None]) -> dict[str, object]:
    evaluable = [r for r in rows if predicate(r) is not None]
    if not evaluable:
        return {"n": 0, "rate": CellState.NOT_APPLICABLE}
    hits = sum(bool(predicate(r)) for r in evaluable)
    return {"n": len(evaluable), "rate": hits / len(evaluable)}


def parse_success_rate(rows: list[UnifiedRow]) -> dict[str, object]:
    return _rate(rows, lambda r: r.parse_success if isinstance(r.parse_success, bool) else None)


def executable_rate(rows: list[UnifiedRow]) -> dict[str, object]:
    """Fraction of rows whose generated artifact executed (any solver status)."""
    return _rate(rows, lambda r: r.execution_success if isinstance(r.execution_success, bool) else None)


def feasible_rate(rows: list[UnifiedRow]) -> dict[str, object]:
    return _rate(rows, lambda r: r.feasible if isinstance(r.feasible, bool) else None)


def objective_agreement_rate(rows: list[UnifiedRow]) -> dict[str, object]:
    """Objective-value proxy agreement -- explicitly a proxy, never 'semantic correctness'."""
    return _rate(rows, lambda r: r.objective_match if isinstance(r.objective_match, bool) else None)


SHARED_METRICS: tuple[SharedMetric, ...] = (
    SharedMetric("parse_success_rate", "Generated output yields a parseable code/model artifact.",
                 ("pamop", "orlm", "optmath", "generic", "deepor", "orr1"), parse_success_rate),
    SharedMetric("executable_rate", "The generated artifact executes against its target solver without error.",
                 ("pamop", "orlm", "optmath", "generic", "deepor", "orr1"), executable_rate),
    SharedMetric("feasible_rate", "Execution reports a feasible solution (not infeasible/unbounded/error).",
                 ("pamop", "orlm", "optmath", "generic", "deepor", "orr1"), feasible_rate),
    SharedMetric("objective_agreement_rate", "Predicted objective within a predeclared tolerance of gold -- a PROXY, not semantic correctness.",
                 ("pamop", "orlm", "optmath", "generic", "deepor", "orr1"), objective_agreement_rate),
)

# `ours` (tfidf_typed_greedy) is deliberately excluded from every SharedMetric
# above: it performs fixed-catalog scalar grounding, not full NL-to-model
# generation, so it has no "generated code", "execution", "feasible", or
# "objective" concept in the same sense the five full-formulation systems
# do. InstantiationReady and StrictInstantiationReady are the closest native
# analogues but are NOT the same claim as objective-value agreement -- see
# END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY.
#
# `generic` (a general-purpose API LLM with a zero-shot gurobipy prompt) IS
# eligible for parse/execution/feasible/objective agreement in principle --
# its output is real gurobipy code -- but it is NOT an optimization-trained
# baseline: it exists to bound how much of the task is attributable to a
# plain API model. Executable/objective cells are NOT_APPLICABLE until solver
# execution is actually run.

END_TO_END_OBJECTIVE_SUCCESS_ELIGIBILITY: dict[str, str] = {
    "ours": (
        "INELIGIBLE. tfidf_typed_greedy performs deterministic scalar grounding "
        "against a fixed catalog; it never generates an executable optimization "
        "instance or solves one, so steps 2 and 3 of END_TO_END_OBJECTIVE_SUCCESS "
        "('solving succeeds', 'objective agrees') do not apply. InstantiationReady "
        "is a predicted-schema grounding-readiness proxy, and "
        "StrictInstantiationReady is a schema-gated version of that proxy. "
        "Neither must be reported in the same column as objective-value agreement."
    ),
    "pamop": (
        "ELIGIBLE. Full AMPL formulation, solved locally with the same solver "
        "used elsewhere in this repository (HiGHS/Gurobi via AMPL), objective "
        "compared to gold with a predeclared tolerance. See "
        "results/pamop/fidelity_diagnostic_gpt5/."
    ),
    "orlm": "ELIGIBLE but BLOCKED on solver execution: 18/18 common-18 rows have valid generation/parse/static validation (official checkpoint), but coptpy is not installed, so no row has been executed and objective comparison is NOT_EVALUABLE.",
    "optmath": "ELIGIBLE. Official gurobipy execution + gold objective comparison is defined; common-18 inference is being run (see results/optmath/).",
    "generic": (
        "ELIGIBLE IN PRINCIPLE for parse/execution/feasible/objective-agreement: the zero-shot "
        "gurobipy output is real, runnable code. NOT an optimization-trained baseline; it bounds "
        "the general-purpose-LLM floor. Solver execution not yet attempted -- objective cells "
        "are NOT_APPLICABLE, never zero."
    ),
    "deepor": "BLOCKED. No official checkpoint or code exists to generate rows at all; the metric definition is eligible in principle (Pyomo execution + gold comparison) but UNAVAILABLE in practice.",
    "orr1": (
        "ELIGIBLE IN PRINCIPLE, WITH A MANDATORY CAVEAT. Official coptpy "
        "execution + gold objective comparison (pass@k/mj@k) is well-defined, "
        "but the official checkpoint requires training TGRPO transductively on "
        "the evaluation set itself (see docs/ORR1_PROVENANCE.md); any future row "
        "must record `transductive_training=True` and must not be silently "
        "compared to an inductively-evaluated system without that caveat "
        "surfaced in the same table."
    ),
}
