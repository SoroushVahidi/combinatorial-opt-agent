"""Opt-in isolated gurobipy execution harness; dry-run is the default."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def gurobipy_available() -> bool:
    return importlib.util.find_spec("gurobipy") is not None


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
    conversion_mode: str = "disabled"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


_VALUE_RE = re.compile(r"(?:objval|objective|best solution)\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.I)


def execute_gurobi(code: str, *, timeout_seconds: int = 100, enabled: bool = False) -> ExecutionResult:
    if not enabled:
        return ExecutionResult(False, "DRY_RUN", None, "", "", None, "execution_disabled", None)
    if not gurobipy_available():
        return ExecutionResult(False, "ENVIRONMENT_BLOCKED", None, "", "", None, "gurobipy_unavailable", None)
    with tempfile.TemporaryDirectory(prefix="optmath_execute_") as directory:
        path = Path(directory) / "generated_model.py"
        path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run([sys.executable, str(path)], cwd=directory, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(True, "TIMEOUT", None, str(exc.stdout or ""), str(exc.stderr or ""), None, "execution_timeout", str(path))
        match = _VALUE_RE.search(completed.stdout)
        objective = float(match.group(1)) if match else None
        if completed.returncode != 0:
            category = "gurobi_api_failure" if "gurobi" in completed.stderr.lower() else "execution_failure"
            status = "FAILED"
        elif "infeasible" in (completed.stdout + completed.stderr).lower():
            category, status = "infeasible_model", "INFEASIBLE"
        elif "unbounded" in (completed.stdout + completed.stderr).lower():
            category, status = "unbounded_model", "UNBOUNDED"
        else:
            category, status = None, "COMPLETED"
        return ExecutionResult(True, status, completed.returncode, completed.stdout, completed.stderr, objective, category, str(path))
