"""Static-only safety and shape checks for generated coptpy source.

The official execution harness (`02_grpo_train.py`'s `run_code` and
`eval/execute.py`) appends a fixed suffix (`ORR1_ADD_SCRIPT`) that references
a variable literally named `model` (`model.status`, `model.objval`). Code
that solves correctly but assigns the COPT model to a different name would
still fail official scoring, so this validator checks for that literal name
in addition to the general coptpy shape checks shared with ORLM.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StaticValidationResult:
    status: str
    python_syntax_valid: bool
    coptpy_import_present: bool
    model_variable_present: bool
    objective_present: bool
    solve_call_present: bool
    constraint_signal_present: bool
    dangerous_operations: tuple[str, ...] = field(default_factory=tuple)
    unsupported_imports: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "python_syntax_valid": self.python_syntax_valid,
            "coptpy_import_present": self.coptpy_import_present,
            "model_variable_present": self.model_variable_present,
            "objective_present": self.objective_present,
            "solve_call_present": self.solve_call_present,
            "constraint_signal_present": self.constraint_signal_present,
            "dangerous_operations": list(self.dangerous_operations),
            "unsupported_imports": list(self.unsupported_imports),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def validate_code(code: str | None) -> StaticValidationResult:
    if not code or not code.strip():
        return StaticValidationResult("EMPTY", False, False, False, False, False, False, errors=("empty_code",))
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StaticValidationResult("SYNTAX_INVALID", False, False, False, False, False, False, errors=(str(exc),))

    imports: set[str] = set()
    unsupported: set[str] = set()
    dangerous: set[str] = set()
    assigned_names: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                if alias.name.split(".")[0] not in {"coptpy", "math", "numpy"}:
                    unsupported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            if node.module != "coptpy":
                unsupported.add(node.module or "<relative>")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr.lower())
                if node.func.attr.lower() in {"system", "popen", "run", "call", "check_output", "remove", "rmtree"}:
                    dangerous.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id.lower())
                if node.func.id in {"eval", "exec", "compile", "__import__"}:
                    dangerous.add(node.func.id)

    text = code.lower()
    has_import = "coptpy" in imports or "coptpy" in text
    has_model_var = "model" in assigned_names
    has_objective = "setobjective" in text or ".obj" in text
    has_solve = "solve" in calls or ".solve(" in text
    has_constraints = "addconstr" in text or "add_constraint" in text or "addconstraints" in text

    warnings: list[str] = []
    errors: list[str] = []
    if not has_import:
        errors.append("missing_coptpy_import")
    if not has_model_var:
        errors.append("missing_model_variable")  # ORR1_ADD_SCRIPT references `model.status`/`model.objval` verbatim.
    if not has_objective:
        errors.append("missing_objective_signal")
    if not has_solve:
        errors.append("missing_solve_call")
    if not has_constraints:
        warnings.append("no_constraint_signal")
    if unsupported:
        warnings.append("unsupported_imports_present")
    if dangerous:
        errors.append("dangerous_operation_present")

    status = "STATIC_VALID" if not errors else "STATIC_INVALID"
    return StaticValidationResult(
        status, True, has_import, has_model_var, has_objective, has_solve, has_constraints,
        tuple(sorted(dangerous)), tuple(sorted(unsupported)), tuple(warnings), tuple(errors),
    )
