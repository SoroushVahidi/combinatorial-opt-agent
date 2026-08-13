"""Safe, opt-in subprocess harness for generated coptpy code.

This module never executes during ordinary validation. Callers must explicitly
request execution, and the harness requires a caller-provided temporary output
directory and never interpolates source into a shell command.
"""
from __future__ import annotations

import re
import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    attempted: bool
    status: str
    return_code: int | None
    stdout: str
    stderr: str
    objective_value: float | None
    error_category: str | None
    source_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


_OBJECTIVE_RE = re.compile(r"(?:best solution|objective|objval)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.I)


def coptpy_available() -> bool:
    """Detect installation without importing or invoking the solver."""
    return importlib.util.find_spec("coptpy") is not None


def execute_coptpy(code: str, *, timeout_seconds: int = 30, enabled: bool = False) -> ExecutionResult:
    if not enabled:
        return ExecutionResult(False, "DRY_RUN", None, "", "", None, "execution_disabled", None)
    if not coptpy_available():
        return ExecutionResult(False, "ENVIRONMENT_BLOCKED", None, "", "", None, "copt_unavailable", None)
    with tempfile.TemporaryDirectory(prefix="orlm_execute_") as directory:
        source = Path(directory) / "generated_model.py"
        source.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run([sys.executable, str(source)], cwd=directory, text=True, capture_output=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(True, "TIMEOUT", None, str(exc.stdout or ""), str(exc.stderr or ""), None, "execution_timeout", str(source))
        objective = None
        match = _OBJECTIVE_RE.search(completed.stdout)
        if match:
            objective = float(match.group(1))
        if completed.returncode != 0:
            category = "copt_api_failure" if "copt" in completed.stderr.lower() else "execution_failure"
            status = "FAILED"
        elif "infeasible" in completed.stdout.lower() or "infeasible" in completed.stderr.lower():
            category, status = "infeasible_model", "INFEASIBLE"
        elif "unbounded" in completed.stdout.lower() or "unbounded" in completed.stderr.lower():
            category, status = "unbounded_model", "UNBOUNDED"
        else:
            category, status = None, "COMPLETED"
        return ExecutionResult(True, status, completed.returncode, completed.stdout, completed.stderr, objective, category, str(source))
