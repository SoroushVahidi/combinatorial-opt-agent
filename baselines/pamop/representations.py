"""The "structured representation" that sits at the partition tree's root node.

PaMOP section 3.2 ("Extracting Structured Representation"): before
partitioning, the problem is reduced to text spans for the objective
(``t_o``), constraints (``t_c``), parameters/variables (``t_v``), and a
concise global summary (``g``) -- produced in the paper by an LLM call
guided by a prompt the paper calls ``G_extr``.

This milestone does not implement any LLM call (see baselines/pamop/README.md
"Not implemented yet"). ``StructuredProblem`` is the paper's data structure;
``from_nlp4lp_record`` is a REPRODUCTION CHOICE bridge that builds one
directly from NLP4LP's own pre-existing structured fields (each NLP4LP
problem already ships machine-readable ``objective``, ``constraints``, and
``variables``/``parameters`` records -- see ``data.py``) instead of calling
an LLM to extract them. This is *not* a reproduction of PaMOP's own G_extr
step; it is a stand-in that lets the (separately paper-specified) partition-
tree construction in ``partition.py`` be built and tested without the LLM
stage. A future milestone can add a real ``G_extr``-driven builder that
produces the same ``StructuredProblem`` shape from raw free-text problem
descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VariableInfo:
    """One entry of t_v -- a decision variable or parameter description."""

    name: str
    description: str
    var_type: str | None = None  # e.g. "continuous", "integer" -- not used by partitioning


@dataclass(frozen=True)
class ConstraintInfo:
    """One entry of t_c."""

    index: int
    description: str
    # PaMOP: "we prompt the LLM to assign a vagueness score to each
    # constraint". That LLM call is out of scope for this milestone; the
    # field is stubbed so downstream code (a future leaf-modeling stage) has
    # somewhere to read it from once it exists.
    vagueness_score: float | None = None


@dataclass(frozen=True)
class StructuredProblem:
    """The root node's content: M's textual precursor (t_o, t_c, t_v, g)."""

    problem_id: str
    global_summary: str  # g
    objective_text: str  # t_o
    constraints: tuple[ConstraintInfo, ...]  # t_c
    variables: tuple[VariableInfo, ...]  # t_v
    source: str  # provenance tag: how this StructuredProblem was built

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError(f"problem {self.problem_id!r} has no constraints to partition")
        expected = tuple(range(len(self.constraints)))
        got = tuple(c.index for c in self.constraints)
        if got != expected:
            raise ValueError(
                f"problem {self.problem_id!r}: constraint indices must be "
                f"0..n-1 in order, got {got}"
            )


def from_nlp4lp_record(problem_id: str, record: dict[str, Any]) -> StructuredProblem:
    """Build a StructuredProblem from one NLP4LP ``problem_info``-shaped record.

    REPRODUCTION CHOICE (see module docstring): uses NLP4LP's own
    pre-existing ``objective``/``constraints``/``variables``/``parameters``
    fields directly, rather than an LLM extraction call. ``record`` is
    expected to look like the dataset's ``problem_info.json`` /
    ``train.jsonl`` row shape:

        {
          "parametrized_description": str,
          "objective": {"description": str, ...},
          "constraints": [{"description": str, ...}, ...],
          "variables": {name: {"description": str, "type": str, ...}, ...},
          "parameters": {name: {"description": str, ...}, ...},   # optional
        }
    """
    objective = record.get("objective") or {}
    objective_text = objective.get("description") or record.get("parametrized_description", "")

    raw_constraints = record.get("constraints") or []
    constraints = tuple(
        ConstraintInfo(index=i, description=c.get("description", ""))
        for i, c in enumerate(raw_constraints)
    )

    variables: list[VariableInfo] = []
    for name, info in (record.get("variables") or {}).items():
        variables.append(
            VariableInfo(
                name=name,
                description=(info or {}).get("description", ""),
                var_type=(info or {}).get("type"),
            )
        )
    for name, info in (record.get("parameters") or {}).items():
        variables.append(
            VariableInfo(name=name, description=(info or {}).get("description", ""), var_type="parameter")
        )

    global_summary = record.get("parametrized_description", "")

    return StructuredProblem(
        problem_id=problem_id,
        global_summary=global_summary,
        objective_text=objective_text,
        constraints=constraints,
        variables=tuple(variables),
        source="nlp4lp_native_fields",
    )


def synthetic_structured_problem(
    problem_id: str,
    *,
    global_summary: str,
    objective_text: str,
    constraint_texts: list[str],
    variables: list[tuple[str, str]],
) -> StructuredProblem:
    """Build a StructuredProblem from plain values, for tests/examples.

    Used to construct small, hand-written (non-gated) example problems for
    unit tests, so gated NLP4LP text never has to be committed.
    """
    return StructuredProblem(
        problem_id=problem_id,
        global_summary=global_summary,
        objective_text=objective_text,
        constraints=tuple(
            ConstraintInfo(index=i, description=t) for i, t in enumerate(constraint_texts)
        ),
        variables=tuple(VariableInfo(name=n, description=d) for n, d in variables),
        source="synthetic",
    )
