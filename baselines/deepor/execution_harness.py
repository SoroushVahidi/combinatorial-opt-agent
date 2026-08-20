"""Opt-in Pyomo subprocess harness; it is not used by lightweight tests."""
from __future__ import annotations
import os, shutil, subprocess, tempfile
from dataclasses import dataclass
from .config import DeepORConfig

@dataclass(frozen=True)
class ExecutionResult:
    attempted: bool; status: str; return_code: int | None = None; stdout: str = ""; stderr: str = ""; objective: float | None = None; error_category: str | None = None
    def to_dict(self): return self.__dict__.copy()

def check_environment(solver: str = "pyomo") -> dict[str, object]:
    return {"solver": solver, "pyomo_available": __import__("importlib.util").util.find_spec("pyomo") is not None, "executable": shutil.which("python")}

def execute(code: str, config: DeepORConfig, *, enabled: bool = False) -> ExecutionResult:
    if not enabled: return ExecutionResult(False, "DRY_RUN")
    with tempfile.TemporaryDirectory(prefix="deepor-") as directory:
        path=os.path.join(directory, "generated_model.py"); open(path, "w", encoding="utf-8").write(code)
        try:
            p=subprocess.run(["python", path], cwd=directory, capture_output=True, text=True, timeout=config.timeout_seconds, env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""})
            return ExecutionResult(True, "COMPLETED" if p.returncode == 0 else "FAILED", p.returncode, p.stdout[-10000:], p.stderr[-10000:], error_category=None if p.returncode == 0 else "execution_failure")
        except subprocess.TimeoutExpired as exc: return ExecutionResult(True, "TIMEOUT", None, str(exc.stdout or ""), str(exc.stderr or ""), error_category="execution_timeout")
