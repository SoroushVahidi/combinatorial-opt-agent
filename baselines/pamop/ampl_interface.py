"""Interface boundary for the (not yet implemented) AMPL rendering/
execution stage -- paper section 3.3's remaining pieces: turning a
``MergedModel`` into an actual ``.mod``/``.dat`` pair AMPL can run, then
calling Gurobi through AMPL to get a solution (eq. 5/6, error correction,
reverse translation -- none of that exists yet either).

This file defines ONLY the consumption contract, per this milestone's
explicit scope ("prepare only the interface boundary needed for the next
milestone... do not implement full AMPL rendering/execution yet"). No AMPL
or Gurobi call happens here. AMPL/`amplpy` were not installed for this
milestone (not needed -- see below).

What the next milestone must consume
-------------------------------------
``modeling.MergedModel`` already provides four AMPL-flavored text fields,
by construction of ``modeling_root_v1.txt``'s four labeled output sections
(PROVENANCE.md -- our own structuring choice, not the paper's):

  - ``parameters_text``: AMPL ``param`` declarations (may be empty if the
    problem has none as standalone parameters).
  - ``variables_text``: AMPL ``var`` declarations.
  - ``objective_text``: a single AMPL ``maximize``/``minimize`` statement.
  - ``constraints_text``: AMPL ``subject to`` statements, verbatim-merged
    from every leaf's ``G_mod`` output (``modeling.merge_bottom_up``).

None of these are validated as *syntactically correct* AMPL by this
milestone (no AMPL parser is available -- ``modeling.py``'s validation is
limited to structural heuristics: non-empty text, presence of `;`,
section-header parsing). A future renderer/executor must expect that a
concatenation of these four fields does not always produce a
syntactically valid ``.mod`` file, and must handle AMPL-reported syntax
errors as real, expected outcomes -- this is exactly the paper's own
"basic inspection" (regex syntax check) and "error solver" (§3.3) stages,
neither of which is implemented yet either.

Why AMPL was not installed this milestone
-------------------------------------------
Every check performed here (prompt design, output-section parsing,
symbol-reference heuristics) operates purely on text and needed no AMPL
runtime. Installing AMPL/`amplpy` only becomes strictly necessary once
something needs to actually *execute* a ``.mod`` file -- that is next
milestone's job, not this one's (task scope: "Do not install AMPL yet
unless implementation of the model representation literally cannot proceed
without it" -- it did not).
"""

from __future__ import annotations

from typing import Protocol

from .modeling import MergedModel


class AmplRenderer(Protocol):
    """Contract the next milestone's AMPL renderer/executor must implement.

    Not implemented in this milestone -- defined here only so
    ``modeling.MergedModel`` has a documented, stable consumer contract to
    build against.
    """

    def render(self, model: MergedModel) -> str:
        """Return a single ``.mod``-file-shaped AMPL model string built from
        ``model``'s four text fields. May raise on AMPL syntax errors once
        real validation exists -- not this milestone's job to define how."""
        ...

    def solve(self, rendered_model: str, data_file: str) -> object:
        """Invoke AMPL + a solver (paper: Gurobi) against ``rendered_model``
        and a companion data file, and return a solver result. Signature is
        illustrative only -- the real return type depends on which AMPL
        Python binding the next milestone chooses (``amplpy`` is the
        paper-faithful candidate; direct ``gurobipy`` is documented as an
        acceptable reconstruction-path deviation, see
        docs/PAMOP_REPRODUCTION_PLAN.md section 13.10)."""
        ...


def naive_concatenation_preview(model: MergedModel) -> str:
    """NOT a renderer -- a plain concatenation of the four text fields, for
    human inspection/debugging only (e.g. eyeballing a smoke-test result).
    Does not attempt to produce syntactically valid AMPL, does not handle
    section ordering rules AMPL itself requires, and must never be used as
    an actual render step. The real ``AmplRenderer.render`` implementation
    (next milestone) is expected to do meaningfully more than this.
    """
    parts = [model.parameters_text, model.variables_text, model.objective_text, model.constraints_text]
    return "\n\n".join(p for p in parts if p)
