"""Structured AMPL execution/correction types.

The PaMOP paper specifies AMPL as the generated modeling language and Gurobi
as the solver backend, but does not define a machine-readable result schema.
These dataclasses are reproduction scaffolding for recording execution and
correction traces without parsing raw logs downstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ErrorCategory(StrEnum):
    NONE = "none"
    MODEL_ERROR = "model_error"
    DATA_ERROR = "data_error"
    ENVIRONMENT_ERROR = "environment_error"


@dataclass(frozen=True)
class AmplDiagnostic:
    severity: DiagnosticSeverity
    code: str
    message: str
    category: ErrorCategory = ErrorCategory.MODEL_ERROR
    symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AmplRenderResult:
    model_text: str
    model_hash: str
    diagnostics: tuple[AmplDiagnostic, ...]

    @property
    def valid(self) -> bool:
        return not any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_text": self.model_text,
            "model_hash": self.model_hash,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "valid": self.valid,
        }


@dataclass(frozen=True)
class AmplExecutionResult:
    model_hash: str
    parse_success: bool
    model_load_success: bool
    solver_invocation_success: bool
    solver_status: str | None
    objective_value: float | None
    runtime_seconds: float
    diagnostics: tuple[AmplDiagnostic, ...] = field(default_factory=tuple)
    error_category: ErrorCategory = ErrorCategory.NONE
    stdout_tail: str = ""
    stderr_tail: str = ""
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return (
            self.parse_success
            and self.model_load_success
            and self.solver_invocation_success
            and self.error_category == ErrorCategory.NONE
            and (self.solver_status or "").lower() in {"solved", "optimal"}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_hash": self.model_hash,
            "parse_success": self.parse_success,
            "model_load_success": self.model_load_success,
            "solver_invocation_success": self.solver_invocation_success,
            "solver_status": self.solver_status,
            "objective_value": self.objective_value,
            "runtime_seconds": self.runtime_seconds,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "error_category": self.error_category,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "timed_out": self.timed_out,
            "success": self.success,
        }
