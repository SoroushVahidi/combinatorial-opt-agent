"""Static-only validation for generated gurobipy Python code."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StaticValidation:
    status: str
    syntax_valid: bool
    gurobi_import_present: bool
    model_creation_present: bool
    objective_present: bool
    constraints_present: bool
    optimize_call_present: bool
    dangerous_operations: tuple[str, ...] = field(default_factory=tuple)
    unsupported_imports: tuple[str, ...] = field(default_factory=tuple)
    possible_undefined_names: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "syntax_valid": self.syntax_valid, "gurobi_import_present": self.gurobi_import_present, "model_creation_present": self.model_creation_present, "objective_present": self.objective_present, "constraints_present": self.constraints_present, "optimize_call_present": self.optimize_call_present, "dangerous_operations": list(self.dangerous_operations), "unsupported_imports": list(self.unsupported_imports), "possible_undefined_names": list(self.possible_undefined_names), "warnings": list(self.warnings), "errors": list(self.errors)}


def validate_code(code: str | None) -> StaticValidation:
    if not code or not code.strip():
        return StaticValidation("EMPTY", False, False, False, False, False, False, errors=("empty_code",))
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StaticValidation("SYNTAX_INVALID", False, False, False, False, False, False, errors=(str(exc),))
    imports: set[str] = set(); imported: set[str] = set(); assigned: set[str] = set(); names: set[str] = set(); dangerous: set[str] = set(); unsupported: set[str] = set(); calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name); imported.add(alias.asname or alias.name.split(".")[0])
                if alias.name not in {"gurobipy", "math", "numpy"}:
                    unsupported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or ""); imported.update(alias.asname or alias.name for alias in node.names)
            if node.module != "gurobipy":
                unsupported.add(node.module or "<relative>")
        elif isinstance(node, ast.Name):
            names.add(node.id)
            if isinstance(node.ctx, (ast.Store, ast.Del)): assigned.add(node.id)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr.lower())
                if node.func.attr.lower() in {"system", "popen", "run", "call", "check_output", "remove", "rmtree"}: dangerous.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id.lower())
                if node.func.id in {"eval", "exec", "compile", "__import__"}: dangerous.add(node.func.id)
    text = code.lower()
    has_import = "gurobipy" in imports or "from gurobipy" in text
    has_model = "gp.model(" in text or "model =" in text or "model=" in text
    has_objective = "setobjective" in text or "objective" in text
    has_constraints = "addconstr" in text or "add_constr" in text
    has_optimize = "optimize" in calls
    builtins = {"True", "False", "None", "abs", "float", "int", "len", "max", "min", "print", "range", "str", "sum", "zip"}
    undefined = tuple(sorted(names - assigned - imported - builtins - {"GRB"}))
    errors: list[str] = []
    if not has_import: errors.append("missing_gurobipy_import")
    if not has_model: errors.append("missing_model_creation_signal")
    if not has_objective: errors.append("missing_objective_signal")
    if not has_constraints: errors.append("missing_constraint_signal")
    if not has_optimize: errors.append("missing_optimize_call")
    if dangerous: errors.append("dangerous_operation_present")
    warnings = ["unsupported_imports_present"] if unsupported else []
    if undefined: warnings.append("possible_undefined_names_present")
    return StaticValidation("STATIC_VALID" if not errors else "STATIC_INVALID", True, has_import, has_model, has_objective, has_constraints, has_optimize, tuple(sorted(dangerous)), tuple(sorted(unsupported)), undefined, tuple(warnings), tuple(errors))
