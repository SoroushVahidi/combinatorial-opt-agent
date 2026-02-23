"""Parser for reStructuredText (.rst) documentation files.

Extracts titles, descriptions, math formulations, and variable definitions
from RST files, particularly those used by Gurobi OptiMods.
"""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Matches a ``.. math::`` directive (indented content block follows)
_MATH_DIRECTIVE_RE = re.compile(
    r"\.\. math::\s*\n((?:[ \t]+.*\n|\n)*)",
    re.MULTILINE,
)

# Matches a ``.. dropdown::`` directive title line
_DROPDOWN_RE = re.compile(
    r"\.\. dropdown::\s*(.+)\n((?:[ \t]+.*\n|\n)*)",
    re.MULTILINE,
)

# RST section underline characters (in priority order)
_UNDERLINE_CHARS = "=-~^\"'`#*+<>"


def _extract_title(text: str) -> str:
    """Extract the first RST section title from text."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            i + 1 < len(lines)
            and stripped
            and len(lines[i + 1].strip()) >= len(stripped)
            and lines[i + 1].strip()
            and all(c == lines[i + 1].strip()[0] for c in lines[i + 1].strip())
            and lines[i + 1].strip()[0] in _UNDERLINE_CHARS
        ):
            return stripped
    return ""


def _extract_description(text: str) -> str:
    """Extract introductory prose (first substantial paragraph after the title)."""
    # Skip the title underline block
    lines = text.splitlines()
    in_title = True
    description_lines: list[str] = []
    blank_count = 0

    for line in lines:
        stripped = line.strip()
        # Skip RST directives
        if stripped.startswith(".."):
            break
        if in_title:
            if stripped and all(c in _UNDERLINE_CHARS for c in stripped) and stripped:
                in_title = False
            continue
        if not stripped:
            blank_count += 1
            if description_lines and blank_count > 1:
                break
        else:
            blank_count = 0
            description_lines.append(stripped)

    return " ".join(description_lines).strip()


def _extract_math_blocks(text: str) -> list[str]:
    """Extract all ``.. math::`` directive content blocks."""
    results: list[str] = []
    for match in _MATH_DIRECTIVE_RE.finditer(text):
        block = match.group(1)
        # Dedent and clean
        cleaned_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if cleaned_lines:
            results.append("\n".join(cleaned_lines))
    return results


def _extract_dropdown_sections(text: str) -> list[dict[str, str]]:
    """Extract ``.. dropdown::`` sections (title + content)."""
    sections: list[dict[str, str]] = []
    for match in _DROPDOWN_RE.finditer(text):
        title = match.group(1).strip()
        content_lines = [
            ln.strip() for ln in match.group(2).splitlines() if ln.strip()
        ]
        sections.append({"title": title, "content": " ".join(content_lines)})
    return sections


def parse_rst_documentation(path: Path) -> dict[str, Any]:
    """Parse a reStructuredText file and extract structured content.

    Parameters
    ----------
    path:
        Path to the ``.rst`` file.

    Returns
    -------
    dict with keys:
        - ``title`` (str): section title.
        - ``description`` (str): introductory description text.
        - ``math_formulations`` (list[str]): LaTeX math blocks.
        - ``variable_definitions`` (list[dict]): dropdown section content.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to read RST file %s: %s", path, exc)
        return {
            "title": "",
            "description": "",
            "math_formulations": [],
            "variable_definitions": [],
        }

    title = _extract_title(text)
    description = _extract_description(text)
    math_blocks = _extract_math_blocks(text)
    dropdowns = _extract_dropdown_sections(text)

    return {
        "title": title,
        "description": description,
        "math_formulations": math_blocks,
        "variable_definitions": dropdowns,
    }
