"""LaTeX and Markdown formatters for unified schema problem entries.

Provides utilities to render a problem entry (conforming to the unified
schema defined in schema/problem_template.json) as LaTeX or GitHub-flavoured
Markdown with math.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def formulation_to_latex(problem: dict[str, Any]) -> str:
    """Convert a unified schema problem dict to a LaTeX formulation string.

    Parameters
    ----------
    problem:
        A problem entry conforming to the unified schema.

    Returns
    -------
    str
        A LaTeX string suitable for inclusion in a document.
    """
    lines: list[str] = []
    name = problem.get("problem_name", "Unknown Problem")
    lines.append(f"\\section*{{{name}}}")

    formulation = problem.get("formulation", {})

    # Sets
    sets = formulation.get("sets", [])
    if sets:
        lines.append("\\subsection*{Sets}")
        lines.append("\\begin{itemize}")
        for s in sets:
            lines.append(
                f"  \\item ${s.get('symbol', '')}$: {s.get('description', '')}"
            )
        lines.append("\\end{itemize}")

    # Parameters
    params = formulation.get("parameters", [])
    if params:
        lines.append("\\subsection*{Parameters}")
        lines.append("\\begin{itemize}")
        for p in params:
            lines.append(
                f"  \\item ${p.get('symbol', '')}$: {p.get('description', '')} "
                f"({p.get('type', '')})"
            )
        lines.append("\\end{itemize}")

    # Decision Variables
    dvars = formulation.get("decision_variables", [])
    if dvars:
        lines.append("\\subsection*{Decision Variables}")
        lines.append("\\begin{itemize}")
        for v in dvars:
            lines.append(
                f"  \\item ${v.get('symbol', '')}$: {v.get('description', '')} "
                f"({v.get('type', '')})"
            )
        lines.append("\\end{itemize}")

    # Objective
    obj = formulation.get("objective", {})
    sense = obj.get("sense", "minimize").capitalize()
    expr = obj.get("expression_latex", "")
    desc = obj.get("description", "")
    if expr:
        lines.append("\\subsection*{Objective}")
        lines.append(f"\\textbf{{{sense}}}")
        lines.append("\\begin{equation*}")
        lines.append(f"  {expr}")
        lines.append("\\end{equation*}")
        if desc:
            lines.append(desc)

    # Constraints
    constraints = formulation.get("constraints_ilp", [])
    if constraints:
        lines.append("\\subsection*{Constraints}")
        lines.append("\\begin{align*}")
        for c in constraints:
            cname = c.get("name", "")
            cexpr = c.get("expression_latex", "")
            cdesc = c.get("description", "")
            comment = f"\\quad \\text{{\\small {cdesc}}}" if cdesc else ""
            label = f"\\tag{{{cname}}}" if cname else ""
            lines.append(f"  {cexpr} {comment} {label} \\\\")
        lines.append("\\end{align*}")

    # LP Relaxation note
    lp_relax = formulation.get("constraints_lp_relaxation", "")
    if lp_relax:
        lines.append("\\subsection*{LP Relaxation}")
        lines.append(lp_relax)

    return "\n".join(lines)


def formulation_to_markdown(problem: dict[str, Any]) -> str:
    """Convert a unified schema problem dict to GitHub-flavoured Markdown.

    Parameters
    ----------
    problem:
        A problem entry conforming to the unified schema.

    Returns
    -------
    str
        Markdown string with inline and display math using ``$$`` delimiters.
    """
    lines: list[str] = []
    name = problem.get("problem_name", "Unknown Problem")
    lines.append(f"## {name}")
    lines.append("")

    nl_descs = problem.get("natural_language_descriptions", [])
    if nl_descs:
        lines.append(nl_descs[0])
        lines.append("")

    formulation = problem.get("formulation", {})

    # Sets
    sets = formulation.get("sets", [])
    if sets:
        lines.append("### Sets")
        lines.append("")
        for s in sets:
            lines.append(
                f"- $${s.get('symbol', '')}$$: {s.get('description', '')}"
            )
        lines.append("")

    # Parameters
    params = formulation.get("parameters", [])
    if params:
        lines.append("### Parameters")
        lines.append("")
        for p in params:
            lines.append(
                f"- $${p.get('symbol', '')}$$: {p.get('description', '')} "
                f"(*{p.get('type', '')}*)"
            )
        lines.append("")

    # Decision Variables
    dvars = formulation.get("decision_variables", [])
    if dvars:
        lines.append("### Decision Variables")
        lines.append("")
        for v in dvars:
            lines.append(
                f"- $${v.get('symbol', '')}$$: {v.get('description', '')} "
                f"(*{v.get('type', '')}*)"
            )
        lines.append("")

    # Objective
    obj = formulation.get("objective", {})
    expr = obj.get("expression_latex", "")
    sense = obj.get("sense", "minimize")
    if expr:
        lines.append("### Objective")
        lines.append("")
        lines.append(f"**{sense.capitalize()}**")
        lines.append("")
        lines.append(f"$$\n{expr}\n$$")
        lines.append("")

    # Constraints
    constraints = formulation.get("constraints_ilp", [])
    if constraints:
        lines.append("### Constraints")
        lines.append("")
        for c in constraints:
            cname = c.get("name", "")
            cexpr = c.get("expression_latex", "")
            cdesc = c.get("description", "")
            header = f"**{cname}**" if cname else ""
            desc_text = f" — {cdesc}" if cdesc else ""
            lines.append(f"{header}{desc_text}")
            lines.append("")
            lines.append(f"$$\n{cexpr}\n$$")
            lines.append("")

    # LP Relaxation
    lp_relax = formulation.get("constraints_lp_relaxation", "")
    if lp_relax:
        lines.append("### LP Relaxation")
        lines.append("")
        lines.append(lp_relax)
        lines.append("")

    # Complexity
    complexity = problem.get("complexity_class", "")
    if complexity:
        lines.append(f"**Complexity:** {complexity}")
        lines.append("")

    return "\n".join(lines)
