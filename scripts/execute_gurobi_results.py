"""Execute generated gurobipy code for OptMATH / generic-LLM result rows and record objective agreement.

Idempotent and evidence-preserving: rows whose `execution_attempted` is already
true are skipped. For each remaining row with a parsed `generated_code`, the
code is written to a temp file and executed in a subprocess using a caller-
supplied Python interpreter (must have gurobipy, e.g.
`/home/soroush/.venvs/gurobi/bin/python`). The objective value is read from
stdout; `objective_proxy_status` is set to PASS/FAIL using the same
round-then-relative-tolerance rule as `baselines.optmath.evaluator`.
Only `execution*`, `objective*`, and `semantic_evaluation_status` fields are
touched -- generation/parse/static evidence is never modified.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_VALUE_RE = re.compile(r"(?:objval|objective|best solution)\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.I)
_TIMEOUT_RE = r"Time limit reached|Timeout|timed out"
_MAX_STDOUT = 20000


def _objective_match(gold: Any, pred: Any, tolerance: float = 0.05) -> bool | None:
    """Same rule as baselines.optmath.evaluator.evaluate_results."""
    if gold is None or pred is None or isinstance(gold, str):
        return None
    gold_r, pred_r = round(float(gold)), round(float(pred))
    return abs(pred_r - gold_r) <= tolerance if gold_r == 0 else abs((pred_r - gold_r) / gold_r) <= tolerance


def _execute(code: str, interpreter: str, timeout_seconds: int, problem_id: str) -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="gurobi_execute_") as directory:
        path = Path(directory) / "generated_model.py"
        path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run([interpreter, str(path)], cwd=directory,
                                       capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            return {"status": "TIMEOUT", "return_code": None, "stdout": (exc.stdout or "")[:_MAX_STDOUT],
                    "stderr": (exc.stderr or "")[:_MAX_STDOUT], "objective_value": None,
                    "error_category": "execution_timeout", "source_path": str(path)}
        stdout = completed.stdout[:_MAX_STDOUT]
        stderr = completed.stderr[:_MAX_STDOUT]
        match = _VALUE_RE.search(stdout)
        objective = float(match.group(1)) if match else None
        combined = (stdout + stderr).lower()
        if completed.returncode != 0:
            category = "gurobi_api_failure" if "gurobi" in stderr.lower() else "execution_failure"
            status = "FAILED"
        elif "infeasible" in combined:
            category, status = "infeasible_model", "INFEASIBLE"
        elif "unbounded" in combined:
            category, status = "unbounded_model", "UNBOUNDED"
        else:
            category, status = None, "COMPLETED"
        return {"status": status, "return_code": completed.returncode, "stdout": stdout, "stderr": stderr,
                "objective_value": objective, "error_category": category, "source_path": str(path)}


def _run_file(path: Path, *, interpreter: str, timeout_seconds: int, git_sha: str) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    updated = 0
    out_lines: list[str] = []
    for line in lines:
        row = json.loads(line)
        if row.get("execution_attempted"):
            out_lines.append(line)
            continue
        code = ((row.get("parsed") or {}).get("generated_code") or "").strip()
        if not code:
            row["execution_attempted"] = True
            row["execution"] = {"status": "NO_CODE", "return_code": None, "stdout": "", "stderr": "",
                                "objective_value": None, "error_category": "no_generated_code", "source_path": None}
            row["objective_proxy_status"] = "NOT_EVALUABLE"
            out_lines.append(json.dumps(row))
            continue
        result = _execute(code, interpreter, timeout_seconds, str(row["problem_id"]))
        pred = result["objective_value"]
        gold = row.get("gold_objective")
        match = _objective_match(gold, pred)
        if result["status"] == "COMPLETED" and pred is None:
            result["status"] = "COMPLETED_NO_OBJECTIVE"
            result["error_category"] = "objective_not_parsed"
        row["execution_attempted"] = True
        row["execution"] = {
            "status": result["status"], "return_code": result["return_code"],
            "stdout": result["stdout"], "stderr": result["stderr"],
            "objective_value": result["objective_value"], "error_category": result["error_category"],
            "source_path": result["source_path"], "interpreter": interpreter,
            "timeout_seconds": timeout_seconds,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        row["objective_value"] = pred
        row["objective_proxy_status"] = "PASS" if match is True else ("FAIL" if match is False else "NOT_EVALUABLE")
        row["semantic_evaluation_status"] = "EXECUTION_OBJECTIVE_PROXY_ONLY"
        out_lines.append(json.dumps(row))
        updated += 1
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(out_lines), updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute generated gurobipy code for OptMATH/generic result rows.")
    parser.add_argument("--output", type=Path, required=True, help="results.jsonl to execute in place (idempotent).")
    parser.add_argument("--interpreter", default="/home/soroush/.venvs/gurobi/bin/python",
                        help="Python interpreter that has gurobipy.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-code execution timeout in seconds.")
    return parser


def _git_sha() -> str:
    try:
        import subprocess
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    args = build_parser().parse_args()
    git_sha = _git_sha()
    rows_seen, rows_updated = _run_file(args.output, interpreter=args.interpreter,
                                        timeout_seconds=args.timeout, git_sha=git_sha)
    print(json.dumps({"event": "execution_completed", "output": str(args.output), "rows_seen": rows_seen,
                      "rows_updated": rows_updated, "git_sha": git_sha}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
