"""Static AMPL checks used before invoking AMPL.

These checks are intentionally conservative. The paper names a regex-style
"basic inspection" stage but does not publish the actual regexes; this file
therefore implements documented reproduction-choice diagnostics rather than
pretending to be a full AMPL parser.
"""

from __future__ import annotations

import re

from .types import AmplDiagnostic, DiagnosticSeverity

_DECL_RE = re.compile(r"\b(param|var)\s+([A-Za-z_][A-Za-z0-9_]*)\b([^;]*);")
_CONSTRAINT_RE = re.compile(
    r"\b(?:subject\s+to|s\.t\.)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:([^;]*);",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_OBJECTIVE_RE = re.compile(
    r"\b(maximize|minimize)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:([^;]*);",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_COMMENT_RE = re.compile(r"#.*?$", re.MULTILINE)

_AMPL_KEYWORDS = {
    "abs",
    "and",
    "binary",
    "by",
    "default",
    "diff",
    "else",
    "exp",
    "for",
    "if",
    "in",
    "integer",
    "log",
    "maximize",
    "min",
    "minimize",
    "max",
    "not",
    "or",
    "param",
    "prod",
    "set",
    "s",
    "sqrt",
    "subject",
    "sum",
    "then",
    "to",
    "var",
}


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _tokens_in_expressions(text: str) -> set[str]:
    stripped = _strip_comments(text)
    for regex in (_DECL_RE, _CONSTRAINT_RE, _OBJECTIVE_RE):
        stripped = regex.sub(" ", stripped)
    exprs = [m.group(3) for m in _DECL_RE.finditer(text)]
    exprs.extend(m.group(2) for m in _CONSTRAINT_RE.finditer(text))
    exprs.extend(m.group(3) for m in _OBJECTIVE_RE.finditer(text))
    return set(_IDENT_RE.findall("\n".join(exprs)))


def declared_symbols(text: str) -> dict[str, str]:
    return {name: kind.lower() for kind, name, _rest in _DECL_RE.findall(text)}


def objective_labels(text: str) -> tuple[str, ...]:
    return tuple(label for _sense, label, _expr in _OBJECTIVE_RE.findall(text))


def validate_ampl_model(text: str) -> tuple[AmplDiagnostic, ...]:
    diagnostics: list[AmplDiagnostic] = []
    if not text.strip():
        return (
            AmplDiagnostic(
                DiagnosticSeverity.ERROR,
                "empty_model",
                "AMPL model text is empty.",
            ),
        )

    declarations: dict[str, list[str]] = {}
    for kind, name, _rest in _DECL_RE.findall(text):
        declarations.setdefault(name, []).append(kind.lower())
    for name, kinds in sorted(declarations.items()):
        if len(kinds) > 1:
            diagnostics.append(
                AmplDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "duplicate_symbol",
                    f"Symbol {name!r} is declared multiple times as {kinds}.",
                    symbol=name,
                )
            )

    var_names = {name for name, kinds in declarations.items() if "var" in kinds}
    if not var_names:
        diagnostics.append(
            AmplDiagnostic(
                DiagnosticSeverity.ERROR,
                "missing_variable",
                "AMPL model has no variable declaration.",
            )
        )

    objectives = objective_labels(text)
    if not objectives:
        diagnostics.append(
            AmplDiagnostic(
                DiagnosticSeverity.ERROR,
                "missing_objective",
                "AMPL model has no maximize/minimize objective.",
            )
        )
    elif len(objectives) > 1:
        diagnostics.append(
            AmplDiagnostic(
                DiagnosticSeverity.ERROR,
                "multiple_objectives",
                f"AMPL model has multiple objectives: {list(objectives)}.",
            )
        )

    constraint_labels: dict[str, int] = {}
    for label, _expr in _CONSTRAINT_RE.findall(text):
        constraint_labels[label] = constraint_labels.get(label, 0) + 1
    if not constraint_labels:
        diagnostics.append(
            AmplDiagnostic(
                DiagnosticSeverity.ERROR,
                "missing_constraint",
                "AMPL model has no subject-to constraint.",
            )
        )
    for label, count in sorted(constraint_labels.items()):
        if count > 1:
            diagnostics.append(
                AmplDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "duplicate_constraint_label",
                    f"Constraint label {label!r} appears {count} times.",
                    symbol=label,
                )
            )
    for label, expr in _CONSTRAINT_RE.findall(text):
        if _looks_malformed_expression(expr):
            diagnostics.append(
                AmplDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "malformed_constraint_expression",
                    f"Constraint {label!r} has a malformed expression.",
                    symbol=label,
                )
            )
    for _sense, label, expr in _OBJECTIVE_RE.findall(text):
        if _looks_malformed_expression(expr):
            diagnostics.append(
                AmplDiagnostic(
                    DiagnosticSeverity.ERROR,
                    "malformed_objective_expression",
                    f"Objective {label!r} has a malformed expression.",
                    symbol=label,
                )
            )

    declared = set(declarations)
    labels = set(constraint_labels) | set(objectives)
    unresolved = _tokens_in_expressions(text) - declared - labels - _AMPL_KEYWORDS
    for token in sorted(unresolved):
        diagnostics.append(
            AmplDiagnostic(
                DiagnosticSeverity.ERROR,
                "unresolved_symbol",
                f"Expression references undeclared symbol {token!r}.",
                symbol=token,
            )
        )

    statement_count = text.count(";")
    starts = len(_DECL_RE.findall(text)) + len(_CONSTRAINT_RE.findall(text)) + len(_OBJECTIVE_RE.findall(text))
    if statement_count > starts:
        diagnostics.append(
            AmplDiagnostic(
                DiagnosticSeverity.WARNING,
                "unparsed_statement",
                "Model contains semicolon-terminated text not recognized by the lightweight validator.",
            )
        )

    return tuple(diagnostics)


def _looks_malformed_expression(expr: str) -> bool:
    stripped = expr.strip()
    if not stripped:
        return True
    return stripped.endswith(("+", "-", "*", "/", "<=", ">=", "=", "<", ">"))
