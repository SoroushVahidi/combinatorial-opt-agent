"""
Catalog enrichment: detect incomplete problems and fill missing formulation data.

Workflow:
  1. Scan the catalog for entries that are missing formulation fields
     (variables, objective, or constraints).
  2. For each incomplete problem, attempt to fetch the missing formulation from
     a public web source derived from the problem's source/id/description.
  3. Return the enriched entries so they can be merged back into the catalog
     (e.g. via ``build_extended_catalog.py --enrich``).

Supported sources (currently):
  * ``gurobi_modeling_examples`` — fetches the Jupyter notebook from GitHub raw
    content (``https://raw.githubusercontent.com/Gurobi/modeling-examples/``)
    and parses the structured "Model Formulation" section.

Other sources gracefully return None and are skipped.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def find_incomplete_problems(catalog: list[dict]) -> list[dict]:
    """Return catalog entries that have a missing or incomplete formulation.

    A problem is considered incomplete if its ``formulation`` dict is missing,
    or if any of ``variables`` (non-empty list), ``objective``, or
    ``constraints`` is absent from the formulation.

    Args:
        catalog: list of problem dicts loaded from the JSON catalog.

    Returns:
        Subset of *catalog* entries that have missing formulation fields.
    """
    incomplete = []
    for problem in catalog:
        form = problem.get("formulation") or {}
        has_vars = bool(form.get("variables") or [])
        has_obj = bool(form.get("objective"))
        has_constraints = "constraints" in form  # can be empty list
        if not (has_vars and has_obj and has_constraints):
            incomplete.append(problem)
    return incomplete


# ---------------------------------------------------------------------------
# Gurobi modeling-examples notebook fetcher / parser
# ---------------------------------------------------------------------------

# The Gurobi modeling-examples repository stores each problem in a subfolder
# whose name matches the last part of the problem id after the ``gurobi_ex_``
# prefix.  Most notebooks follow the pattern:
#   https://raw.githubusercontent.com/Gurobi/modeling-examples/master/{slug}/{slug}.ipynb
# A few use a different notebook filename — we try several candidates.
_GITHUB_RAW = "https://raw.githubusercontent.com/Gurobi/modeling-examples/master"


def _notebook_slug(problem_id: str) -> str:
    """Convert ``gurobi_ex_XXX`` → ``XXX`` (the GitHub subfolder name)."""
    return problem_id.replace("gurobi_ex_", "", 1)


def _candidate_notebook_urls(slug: str) -> list[str]:
    """Return candidate GitHub raw URLs to try for *slug*."""
    # Primary: same name for folder and notebook
    candidates = [
        f"{_GITHUB_RAW}/{slug}/{slug}.ipynb",
    ]
    # Some notebooks use a shortened name without specific suffixes
    for suffix in ("_game", "_planning", "_scheduling", "_optimization", "_analysis"):
        if slug.endswith(suffix):
            short = slug[: -len(suffix)]
            candidates.append(f"{_GITHUB_RAW}/{slug}/{short}.ipynb")
    return candidates


def _fetch_url(url: str, timeout: int = 8) -> bytes | None:
    """Fetch *url*; return raw bytes or None on any error."""
    req = urllib.request.Request(url, headers={"User-Agent": "catalog-enrichment/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None


def _parse_notebook_formulation(notebook_json: dict, problem_name: str) -> dict | None:
    """Parse a Gurobi modeling-examples Jupyter notebook for formulation data.

    Looks for a markdown cell that contains at least one of the structured
    subsections ``### Decision Variables``, ``### Objective``, or
    ``### Constraints`` (all within a ``## Model Formulation`` section).

    Returns a partial formulation dict with any of the three keys
    (``variables``, ``objective``, ``constraints``) that could be extracted,
    or *None* if nothing useful was found.
    """
    cells = notebook_json.get("cells", [])
    # Collect all markdown text
    sections: list[str] = []
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            sections.append("".join(cell.get("source", [])))

    # Find the formulation section
    formulation_text = ""
    for sec in sections:
        if "## Model Formulation" in sec or "### Decision Variables" in sec:
            formulation_text += "\n" + sec

    if not formulation_text:
        return None

    result: dict[str, Any] = {}

    # --- Decision Variables ---
    var_block = _extract_subsection(formulation_text, "Decision Variables")
    if var_block:
        variables = _parse_variables(var_block)
        if variables:
            result["variables"] = variables

    # --- Objective ---
    obj_block = _extract_subsection(formulation_text, "Objective")
    if obj_block:
        obj = _parse_objective(obj_block)
        if obj:
            result["objective"] = obj

    # --- Constraints ---
    constr_block = _extract_subsection(formulation_text, "Constraints")
    if constr_block:
        constraints = _parse_constraints(constr_block)
        result["constraints"] = constraints

    return result if result else None


def _extract_subsection(text: str, section_name: str) -> str:
    """Return the text of a ``### <section_name>`` block until the next ``###``."""
    pattern = re.compile(
        r"###\s+" + re.escape(section_name) + r"[^\n]*\n(.*?)(?=###|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _parse_variables(block: str) -> list[dict]:
    """Parse decision-variable lines from a subsection block.

    Recognises lines like:
      ``$select_{j} \\in \\{0, 1\\}$: description text``
      ``$0 \\leq assign_{i,j} \\leq 1$: description text``
    """
    variables: list[dict] = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match lines that start with a LaTeX variable definition
        m = re.match(r"\$([^$]+)\$[:\s]*(.*)", line)
        if not m:
            # Remove leading list markers and try again
            cleaned = re.sub(r"^[-*]\s*", "", line)
            m = re.match(r"\$([^$]+)\$[:\s]*(.*)", cleaned)
        if m:
            var_expr = m.group(1).strip()
            description = m.group(2).strip()

            # Handle bound notation: ``0 \leq x \leq 1``  or  ``a \leq x``
            # Extract the variable (non-numeric token between \leq / \geq tokens)
            bound_m = re.match(
                r"[0-9.]+\s*(?:\\leq|\\geq|\\le|\\ge)\s*(.+?)"
                r"(?:\s*(?:\\leq|\\geq|\\le|\\ge).+)?$",
                var_expr,
            )
            if bound_m:
                var_expr = bound_m.group(1).strip()

            # Extract symbol (everything before \in or \leq or \geq)
            sym_m = re.match(r"([^\\]+?)(?:\s*\\in|\s*\\leq|\s*\\geq|\s*\\ge|\s*\\le|$)", var_expr)
            symbol = sym_m.group(1).strip() if sym_m else var_expr
            # Domain: the whole var_expr
            domain = var_expr
            variables.append(
                {"symbol": symbol, "description": description, "domain": domain}
            )
    return variables


def _parse_objective(block: str) -> dict | None:
    """Parse objective sense and expression from a subsection block."""
    text = block.lower()
    # Detect sense
    if "\\text{min}" in text or "minimize" in text or "\\min" in text or "min " in text:
        sense = "minimize"
    elif "\\text{max}" in text or "maximize" in text or "\\max" in text or "max " in text:
        sense = "maximize"
    else:
        return None

    # Extract LaTeX expression if present (between \begin{equation} and \end{equation})
    eq_m = re.search(r"\\begin\{equation\}(.*?)\\end\{equation\}", block, re.DOTALL)
    if eq_m:
        expression = eq_m.group(1).strip()
    else:
        # Fallback: take the first inline LaTeX
        inline_m = re.search(r"\$(.*?)\$", block)
        if inline_m:
            expression = inline_m.group(1).strip()
        else:
            # Use the whole block text (cleaned)
            expression = " ".join(block.split())[:200]

    return {"sense": sense, "expression": expression}


def _parse_constraints(block: str) -> list[dict]:
    """Parse constraint entries from a subsection block.

    Each constraint is a bullet ``- **Name**. description ...`` followed by
    an optional equation block.
    """
    constraints: list[dict] = []
    # Split on bullet points that follow a named constraint
    parts = re.split(r"\n-\s+\*\*", block)
    for part in parts:
        if not part.strip():
            continue
        # Format: "Name**. description\n\nequation..."
        nm = re.match(r"([^*]+)\*+[.:]?\s*(.*)", part, re.DOTALL)
        if not nm:
            continue
        constr_name = nm.group(1).strip()
        rest = nm.group(2).strip()
        # Extract expression: look for \begin{equation}...\end{equation}
        eq_m = re.search(r"\\begin\{equation\}(.*?)\\end\{equation\}", rest, re.DOTALL)
        if eq_m:
            expression = eq_m.group(1).strip()
        else:
            # Fallback: first line
            expression = rest.split("\n")[0].strip() or constr_name
        description = constr_name + (": " + rest.split("\n")[0].strip() if rest else "")
        constraints.append({"expression": expression, "description": description})
    return constraints


# ---------------------------------------------------------------------------
# Top-level per-problem enrichment dispatcher
# ---------------------------------------------------------------------------

def fetch_formulation_from_web(problem: dict, timeout: int = 8) -> dict | None:
    """Attempt to fetch missing formulation data for *problem* from the web.

    Currently handles:
    * ``source == "gurobi_modeling_examples"`` — fetches the Jupyter notebook
      from GitHub raw content and parses its "Model Formulation" section.

    Returns a (possibly partial) formulation dict on success, or *None* if the
    source is unsupported, the network is unavailable, or parsing yields nothing
    useful.  Never raises.
    """
    try:
        source = problem.get("source", "")
        pid = problem.get("id", "")
        if source == "gurobi_modeling_examples" or pid.startswith("gurobi_ex_"):
            return _fetch_gurobi_formulation(pid, problem.get("name", ""), timeout)
    except Exception:
        pass
    return None


def _fetch_gurobi_formulation(problem_id: str, problem_name: str, timeout: int) -> dict | None:
    """Fetch and parse a Gurobi modeling-examples notebook from GitHub."""
    slug = _notebook_slug(problem_id)
    for url in _candidate_notebook_urls(slug):
        raw = _fetch_url(url, timeout=timeout)
        if raw is None:
            continue
        try:
            nb = json.loads(raw)
        except json.JSONDecodeError:
            continue
        formulation = _parse_notebook_formulation(nb, problem_name)
        if formulation:
            return formulation
    return None


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def enrich_catalog(
    catalog: list[dict],
    verbose: bool = False,
    timeout: int = 8,
) -> list[dict]:
    """Enrich incomplete catalog entries by fetching missing formulation data.

    Scans *catalog* for incomplete problems, attempts to fetch missing
    formulation fields from the web, and returns a list of **enriched**
    problem dicts (deep-copied from the originals, never mutating the input).

    Only problems whose formulation was actually improved are included in the
    return value.  The caller can merge these into ``custom_problems.json``
    (which ``build_extended_catalog.py`` then picks up).

    Args:
        catalog: list of problem dicts.
        verbose: print progress messages to stdout.
        timeout: per-request network timeout in seconds.

    Returns:
        List of enriched problem dicts (may be empty if nothing was improved).
    """
    incomplete = find_incomplete_problems(catalog)
    if verbose:
        print(f"Found {len(incomplete)} incomplete problems out of {len(catalog)} total.")

    enriched: list[dict] = []
    for problem in incomplete:
        pid = problem.get("id", "")
        if verbose:
            print(f"  Enriching {pid!r} ...", end=" ", flush=True)

        fetched = fetch_formulation_from_web(problem, timeout=timeout)
        if fetched is None:
            if verbose:
                print("no data found.")
            continue

        # Merge fetched data into a copy of the existing problem
        updated = deepcopy(problem)
        existing_form = updated.setdefault("formulation", {})

        improved = False
        for field in ("variables", "objective", "constraints"):
            if fetched.get(field) is not None and field not in existing_form:
                existing_form[field] = fetched[field]
                improved = True

        if improved:
            if verbose:
                fields = [f for f in ("variables", "objective", "constraints") if fetched.get(f) is not None]
                print(f"filled in: {fields}")
            enriched.append(updated)
        else:
            if verbose:
                print("nothing new (all fields already present or empty fetch).")

    if verbose:
        print(f"Enriched {len(enriched)} problems.")
    return enriched
