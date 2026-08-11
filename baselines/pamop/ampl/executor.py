"""AMPL execution wrapper (`G_exe` in this reproduction scaffold)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..llm.base import prompt_hash
from .types import AmplDiagnostic, AmplExecutionResult, DiagnosticSeverity, ErrorCategory
from .validator import objective_labels, validate_ampl_model

_SOLVE_RESULT_RE = re.compile(r"\bsolve_result\s*=\s*([A-Za-z_][A-Za-z0-9_]*)")
_OBJECTIVE_VALUE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$", re.MULTILINE)
_GUROBI_OBJECTIVE_RE = re.compile(r"objective\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)


class AmplExecutor:
    """Run AMPL through ``amplpy.modules run ampl``.

    The executor is intentionally subprocess-based so the default test
    environment does not need to import amplpy. Live runs can point
    ``python_executable`` at a user-local venv containing amplpy/modules.
    """

    def __init__(
        self,
        *,
        solver: str = "gurobi",
        python_executable: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.solver = solver
        self.python_executable = (
            python_executable
            or os.environ.get("PAMOP_AMPLPY_PYTHON")
            or sys.executable
        )
        self.timeout_seconds = timeout_seconds

    def execute(self, model_text: str) -> AmplExecutionResult:
        model_hash = prompt_hash(model_text)
        static_diagnostics = validate_ampl_model(model_text)
        static_errors = tuple(d for d in static_diagnostics if d.severity == DiagnosticSeverity.ERROR)
        if static_errors:
            category = _dominant_category(static_errors)
            return AmplExecutionResult(
                model_hash=model_hash,
                parse_success=False,
                model_load_success=False,
                solver_invocation_success=False,
                solver_status=None,
                objective_value=None,
                runtime_seconds=0.0,
                diagnostics=static_diagnostics,
                error_category=category,
            )

        objective = objective_labels(model_text)[0] if objective_labels(model_text) else None
        script = self._run_script(model_text, objective)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="pamop_ampl_") as tmpdir:
            run_path = Path(tmpdir) / "model.run"
            run_path.write_text(script, encoding="utf-8")
            cmd = [
                self.python_executable,
                "-m",
                "amplpy.modules",
                "run",
                "ampl",
                str(run_path),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                runtime = time.monotonic() - start
                return AmplExecutionResult(
                    model_hash=model_hash,
                    parse_success=True,
                    model_load_success=True,
                    solver_invocation_success=False,
                    solver_status=None,
                    objective_value=None,
                    runtime_seconds=runtime,
                    diagnostics=(
                        AmplDiagnostic(
                            DiagnosticSeverity.ERROR,
                            "ampl_timeout",
                            f"AMPL execution exceeded {self.timeout_seconds} seconds.",
                            ErrorCategory.ENVIRONMENT_ERROR,
                        ),
                    ),
                    error_category=ErrorCategory.ENVIRONMENT_ERROR,
                    stdout_tail=_tail(exc.stdout or ""),
                    stderr_tail=_tail(exc.stderr or ""),
                    timed_out=True,
                )

        runtime = time.monotonic() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + "\n" + stderr
        diagnostics = list(static_diagnostics)
        category = _classify_ampl_failure(combined, proc.returncode)
        solver_status = _parse_solve_result(stdout)
        objective_value = _parse_objective(stdout, objective)

        if proc.returncode != 0:
            diagnostics.append(
                AmplDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "ampl_process_failed",
                    _tail(combined) or f"AMPL exited with return code {proc.returncode}.",
                    category,
                )
            )
        elif solver_status and solver_status.lower() not in {"solved", "optimal"}:
            category = ErrorCategory.MODEL_ERROR
            diagnostics.append(
                AmplDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "solver_not_solved",
                    f"AMPL solve_result is {solver_status!r}.",
                    category,
                )
            )
        else:
            category = ErrorCategory.NONE

        parse_success = not any(d.code in {"syntax_error", "ampl_process_failed"} for d in diagnostics)
        model_load_success = parse_success and "not defined" not in combined.lower()
        solver_invocation_success = proc.returncode == 0 and solver_status is not None

        return AmplExecutionResult(
            model_hash=model_hash,
            parse_success=parse_success,
            model_load_success=model_load_success,
            solver_invocation_success=solver_invocation_success,
            solver_status=solver_status,
            objective_value=objective_value,
            runtime_seconds=runtime,
            diagnostics=tuple(diagnostics),
            error_category=category,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )

    def _run_script(self, model_text: str, objective: str | None) -> str:
        solver_options = ""
        if self.solver == "gurobi":
            solver_options = "option gurobi_options 'outlev=0';"
        elif self.solver == "highs":
            solver_options = "option highs_options 'outlev=0';"
        display_objective = f"display {objective};" if objective else ""
        return "\n".join(
            part
            for part in [
                f"option solver {self.solver};",
                solver_options,
                model_text,
                "solve;",
                "display solve_result;",
                display_objective,
            ]
            if part
        )


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:].strip()


def _parse_solve_result(stdout: str) -> str | None:
    match = _SOLVE_RESULT_RE.search(stdout)
    if match:
        return match.group(1)
    lowered = stdout.lower()
    if "optimal solution" in lowered:
        return "solved"
    if "infeasible" in lowered:
        return "infeasible"
    if "unbounded" in lowered:
        return "unbounded"
    return None


def _parse_objective(stdout: str, objective: str | None) -> float | None:
    if objective:
        for label, value in _OBJECTIVE_VALUE_RE.findall(stdout):
            if label == objective:
                return float(value)
    match = _GUROBI_OBJECTIVE_RE.search(stdout)
    if match:
        return float(match.group(1))
    return None


def _classify_ampl_failure(output: str, returncode: int) -> ErrorCategory:
    lowered = output.lower()
    if returncode == 0:
        return ErrorCategory.NONE
    env_markers = (
        "license",
        "no module named amplpy",
        "no such solver",
        "cannot execute",
        "permission denied",
        "not found",
        "gurobi error 10009",
    )
    if any(marker in lowered for marker in env_markers):
        return ErrorCategory.ENVIRONMENT_ERROR
    data_markers = ("cannot open data", "no such file or directory")
    if any(marker in lowered for marker in data_markers):
        return ErrorCategory.DATA_ERROR
    return ErrorCategory.MODEL_ERROR


def _dominant_category(diagnostics: tuple[AmplDiagnostic, ...]) -> ErrorCategory:
    for category in (ErrorCategory.ENVIRONMENT_ERROR, ErrorCategory.DATA_ERROR, ErrorCategory.MODEL_ERROR):
        if any(d.category == category for d in diagnostics):
            return category
    return ErrorCategory.NONE
