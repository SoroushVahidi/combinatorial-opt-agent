"""Safe, opt-in subprocess harness for generated coptpy code.

Mirrors `02_grpo_train.py`'s `compile_script`/`run_code` (reward path) and
`eval/execute.py`'s `compile_script` (evaluation path): append
`ORR1_ADD_SCRIPT`, run in a subprocess with a timeout, and parse the fixed
"Just print the best solution: X" / "No Best Solution" markers -- never a
generic regex over stdout. This module never executes during ordinary
validation; callers must explicitly request execution.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baselines.orr1.config import ORR1_ADD_SCRIPT

_BEST_SOLUTION_MARKER = "Just print the best solution:"
_NO_SOLUTION_MARKER = "No Best Solution"


@dataclass(frozen=True)
class ExecutionResult:
    attempted: bool
    status: str  # DRY_RUN | ENVIRONMENT_BLOCKED | COMPLETED_WITH_SOLUTION | COMPLETED_NO_SOLUTION | TIMEOUT | FAILED
    return_code: int | None
    stdout: str
    stderr: str
    best_solution: str | None
    error_category: str | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def coptpy_available() -> bool:
    return importlib.util.find_spec("coptpy") is not None


def execute_coptpy(code: str, *, timeout_seconds: int = 600, enabled: bool = False) -> ExecutionResult:
    if not enabled:
        return ExecutionResult(False, "DRY_RUN", None, "", "", None, "execution_disabled")
    if not coptpy_available():
        return ExecutionResult(False, "ENVIRONMENT_BLOCKED", None, "", "", None, "coptpy_unavailable")
    script = code + ORR1_ADD_SCRIPT
    with tempfile.TemporaryDirectory(prefix="orr1_execute_") as directory:
        source = Path(directory) / "generated_model.py"
        source.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(source)], cwd=directory, text=True, capture_output=True,
                timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(True, "TIMEOUT", None, str(exc.stdout or ""), str(exc.stderr or ""), None, "execution_timeout")
        stdout = completed.stdout
        if completed.returncode != 0:
            return ExecutionResult(True, "FAILED", completed.returncode, stdout, completed.stderr, None, "execution_failure")
        pos = stdout.find(_BEST_SOLUTION_MARKER)
        if pos != -1:
            tail = stdout[pos:].replace(_BEST_SOLUTION_MARKER, "").strip()
            best = tail.split("\n", 1)[0]
            return ExecutionResult(True, "COMPLETED_WITH_SOLUTION", 0, stdout, completed.stderr, best, None)
        if _NO_SOLUTION_MARKER in stdout:
            return ExecutionResult(True, "COMPLETED_NO_SOLUTION", 0, stdout, completed.stderr, _NO_SOLUTION_MARKER, None)
        return ExecutionResult(True, "FAILED", 0, stdout, completed.stderr, None, "output_out_of_expectation")
