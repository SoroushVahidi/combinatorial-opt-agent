"""Static-only safety and shape checks for generated coptpy source."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StaticValidationResult:
    status: str
    python_syntax_valid: bool
    coptpy_import_present: bool
    model_creation_present: bool
    objective_present: bool
    optimize_call_present: bool
    constraint_signal_present: bool
    suspicious_empty_model: bool
    dangerous_operations: tuple[str, ...] = field(default_factory=tuple)
    unsupported_imports: tuple[str, ...] = field(default_factory=tuple)
    possible_undefined_names: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "python_syntax_valid": self.python_syntax_valid,
            "coptpy_import_present": self.coptpy_import_present,
            "model_creation_present": self.model_creation_present,
            "objective_present": self.objective_present,
            "optimize_call_present": self.optimize_call_present,
            "constraint_signal_present": self.constraint_signal_present,
            "suspicious_empty_model": self.suspicious_empty_model,
            "dangerous_operations": list(self.dangerous_operations),
            "unsupported_imports": list(self.unsupported_imports),
            "possible_undefined_names": list(self.possible_undefined_names),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def validate_coptpy_code(code: str | None) -> StaticValidationResult:
    if not code or not code.strip():
        return StaticValidationResult("EMPTY", False, False, False, False, False, False, False, errors=("empty_code",))
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StaticValidationResult("PYTHON_SYNTAX_FAILURE", False, False, False, False, False, False, False, errors=(str(exc),))

    imports: set[str] = set()
    unsupported: set[str] = set()
    dangerous: set[str] = set()
    calls: set[str] = set()
    names: set[str] = set()
    assigned: set[str] = set()
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
                imported_names.add(alias.asname or alias.name.split(".")[0])
                if alias.name not in {"coptpy", "math", "numpy", "numpy as np"}:
                    unsupported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            imported_names.update(alias.asname or alias.name for alias in node.names)
            if node.module != "coptpy":
                unsupported.add(node.module or "<relative>")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr.lower())
                if node.func.attr.lower() in {"system", "popen", "run", "call", "check_output", "remove", "rmtree"}:
                    dangerous.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id.lower())
                if node.func.id in {"eval", "exec", "compile", "__import__"}:
                    dangerous.add(node.func.id)
        elif isinstance(node, ast.Name):
            names.add(node.id)
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                assigned.add(node.id)

    text = code.lower()
    has_import = "coptpy" in imports or "from coptpy" in text
    has_model = "envr" in text or "model(" in text or "create_model" in text
    has_objective = "setobjective" in text or "objective" in text
    has_optimize = "solve" in calls or "optimize" in calls
    has_constraints = "addconstr" in text or "add_constraint" in text or "addconstraints" in text
    builtins = {"True", "False", "None", "abs", "all", "any", "float", "int", "len", "max", "min", "print", "range", "str", "sum", "zip"}
    possible_undefined = tuple(sorted(names - assigned - imported_names - builtins))
    warnings: list[str] = []
    errors: list[str] = []
    if not has_import:
        errors.append("missing_coptpy_import")
    if not has_model:
        errors.append("missing_model_creation_signal")
    if not has_objective:
        errors.append("missing_objective_signal")
    if not has_optimize:
        errors.append("missing_optimize_or_solve_call")
    if not has_constraints:
        warnings.append("no_constraint_signal")
    if unsupported:
        warnings.append("unsupported_imports_present")
    if possible_undefined:
        warnings.append("possible_undefined_names_present")
    if dangerous:
        errors.append("dangerous_operation_present")
    suspicious = not has_constraints
    status = "STATIC_VALID" if not errors else "STATIC_INVALID"
    return StaticValidationResult(status, True, has_import, has_model, has_objective, has_optimize, has_constraints, suspicious, tuple(sorted(dangerous)), tuple(sorted(unsupported)), possible_undefined, tuple(warnings), tuple(errors))
