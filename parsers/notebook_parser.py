"""Parser for Jupyter notebook (.ipynb) files.

Extracts markdown cells, code cells, LaTeX formulations, and natural
language descriptions from a Jupyter notebook.
"""

import logging
import re
from pathlib import Path
from typing import Any

import nbformat

logger = logging.getLogger(__name__)

# Regex patterns for LaTeX extraction
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_ALIGN_RE = re.compile(
    r"\\begin\{align\*?\}(.+?)\\end\{align\*?\}", re.DOTALL
)

# Keywords that suggest a code cell contains optimization logic
_OPT_KEYWORDS = {
    "addVar", "addConstr", "setObjective",
    "Var", "Objective", "Constraint",
    "addVars", "addConstrs",
    "model.addVar", "model.addConstr",
}


def _extract_latex(text: str) -> list[str]:
    """Extract LaTeX math blocks from a markdown string.

    Parameters
    ----------
    text:
        Markdown text possibly containing ``$$...$$`` or
        ``\\begin{align}...\\end{align}`` blocks.

    Returns
    -------
    list[str]
        Extracted LaTeX strings (stripped).
    """
    results: list[str] = []
    for match in _DISPLAY_MATH_RE.finditer(text):
        content = match.group(1).strip()
        if content:
            results.append(content)
    for match in _ALIGN_RE.finditer(text):
        content = match.group(0).strip()
        if content:
            results.append(content)
    return results


def _is_opt_code(source: str) -> bool:
    """Return True if a code cell appears to contain optimization code."""
    return any(kw in source for kw in _OPT_KEYWORDS)


_MIN_PARAGRAPH_LENGTH = 40


def _extract_nl_descriptions(markdown_cells: list[str]) -> list[str]:
    """Extract sentence-level NL descriptions from markdown cells.

    Strips LaTeX blocks and short/heading lines, returning substantive text.
    """
    descriptions: list[str] = []
    for cell in markdown_cells:
        # Remove display math blocks
        cleaned = _DISPLAY_MATH_RE.sub("", cell)
        cleaned = _ALIGN_RE.sub("", cleaned)
        # Remove markdown headings
        cleaned = re.sub(r"^#+\s+", "", cleaned, flags=re.MULTILINE)
        # Split into paragraphs and keep non-trivial ones
        for para in cleaned.split("\n\n"):
            para = para.strip()
            if len(para) > _MIN_PARAGRAPH_LENGTH:
                descriptions.append(para)
    return descriptions


def parse_jupyter_notebook(path: Path) -> dict[str, Any]:
    """Parse a Jupyter notebook and extract structured content.

    Parameters
    ----------
    path:
        Path to the ``.ipynb`` file.

    Returns
    -------
    dict with keys:
        - ``markdown_cells`` (list[str]): raw markdown cell sources.
        - ``code_cells`` (list[str]): optimization-related code cell sources.
        - ``latex_formulations`` (list[str]): extracted LaTeX blocks.
        - ``nl_descriptions`` (list[str]): natural language description paragraphs.
    """
    try:
        notebook = nbformat.read(str(path), as_version=4)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read notebook %s: %s", path, exc)
        return {
            "markdown_cells": [],
            "code_cells": [],
            "latex_formulations": [],
            "nl_descriptions": [],
        }

    markdown_cells: list[str] = []
    code_cells: list[str] = []
    latex_formulations: list[str] = []

    for cell in notebook.cells:
        source = cell.get("source", "")
        if cell.cell_type == "markdown":
            markdown_cells.append(source)
            latex_formulations.extend(_extract_latex(source))
        elif cell.cell_type == "code":
            if _is_opt_code(source):
                code_cells.append(source)

    nl_descriptions = _extract_nl_descriptions(markdown_cells)

    return {
        "markdown_cells": markdown_cells,
        "code_cells": code_cells,
        "latex_formulations": latex_formulations,
        "nl_descriptions": nl_descriptions,
    }
