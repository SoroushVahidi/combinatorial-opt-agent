"""Compatibility boundary for AMPL rendering/execution.

The concrete implementation now lives under ``baselines.pamop.ampl``:
``renderer.render_merged_model`` builds AMPL text from ``MergedModel``,
``validator.validate_ampl_model`` performs reconstructed static checks, and
``executor.AmplExecutor`` invokes AMPL/Gurobi. This module is kept as a
small compatibility surface for earlier milestone tests and callers.
"""

from __future__ import annotations

from typing import Protocol

from .ampl.executor import AmplExecutor
from .ampl.renderer import render_merged_model
from .modeling import MergedModel


class AmplRenderer(Protocol):
    """Protocol for objects that render and solve a PaMOP merged model."""

    def render(self, model: MergedModel) -> str:
        """Return a single ``.mod``-file-shaped AMPL model string."""
        ...

    def solve(self, rendered_model: str, data_file: str) -> object:
        """Invoke AMPL + a solver (paper: Gurobi) against rendered text."""
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


__all__ = ["AmplExecutor", "AmplRenderer", "naive_concatenation_preview", "render_merged_model"]
