"""Parser for MPS (Mathematical Programming System) format files.

Supports standard MPS format with sections:
NAME, ROWS, COLUMNS, RHS, BOUNDS, RANGES, ENDATA.
"""

import gzip
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_mps_file(path: Path) -> dict[str, Any]:
    """Parse a standard MPS format file and return structured metadata.

    MPS Format Sections
    -------------------
    NAME       — instance name
    ROWS       — constraints; type N (objective), L (≤), G (≥), E (=)
    COLUMNS    — non-zero coefficients; integer variables bracketed by
                 ``'MARKER'  'INTORG'`` / ``'MARKER'  'INTEND'`` pairs
    RHS        — right-hand side values
    BOUNDS     — variable bounds (LO, UP, FX, FR, MI, BV, LI, UI)
    RANGES     — optional range constraints
    ENDATA     — end of data marker

    Parameters
    ----------
    path:
        Path to the MPS file (plain ``.mps`` or gzip-compressed ``.mps.gz``).

    Returns
    -------
    dict with keys:
        - ``name`` (str): instance name.
        - ``num_variables`` (int): total number of structural variables.
        - ``num_constraints`` (int): number of constraints (excluding objective).
        - ``num_binary_vars`` (int): number of binary variables.
        - ``num_integer_vars`` (int): number of general integer variables.
        - ``constraint_types`` (dict): counts of L/G/E constraint types.
        - ``objective_coefficients`` (str): name of the objective row.
    """
    opener = gzip.open if str(path).endswith(".gz") else open

    name = path.stem.replace(".mps", "")
    rows: dict[str, str] = {}           # row_name -> type (N/L/G/E)
    columns: dict[str, list[str]] = {}  # var_name -> rows it appears in
    current_section = ""
    int_marker_active = False
    integer_vars: set[str] = set()
    binary_vars: set[str] = set()

    try:
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip()
                if not line or line.startswith("$"):
                    continue

                # Section headers are not indented
                if not line.startswith(" ") and not line.startswith("\t"):
                    token = line.split()[0].upper()
                    if token in {
                        "NAME", "ROWS", "COLUMNS", "RHS",
                        "BOUNDS", "RANGES", "ENDATA",
                    }:
                        current_section = token
                        parts = line.split()
                        if token == "NAME" and len(parts) > 1:
                            name = parts[1]
                    continue

                parts = line.split()
                if not parts:
                    continue

                if current_section == "ROWS":
                    row_type = parts[0].upper()
                    row_name = parts[1] if len(parts) > 1 else ""
                    rows[row_name] = row_type

                elif current_section == "COLUMNS":
                    if len(parts) >= 2 and parts[1] == "'MARKER'":
                        int_marker_active = len(parts) > 2 and "'INTORG'" in parts[2]
                        continue
                    var_name = parts[0]
                    if var_name not in columns:
                        columns[var_name] = []
                    # Each column entry has: var row1 val1 [row2 val2]
                    for i in range(1, len(parts) - 1, 2):
                        columns[var_name].append(parts[i])
                    if int_marker_active:
                        integer_vars.add(var_name)

                elif current_section == "BOUNDS":
                    bound_type = parts[0].upper()
                    var_name = parts[2] if len(parts) > 2 else ""
                    if bound_type == "BV":
                        binary_vars.add(var_name)
                    elif bound_type in ("LI", "UI"):
                        integer_vars.add(var_name)

    except (OSError, UnicodeDecodeError) as exc:
        logger.error("Error reading MPS file %s: %s", path, exc)
        return {}

    objective_row = next(
        (r for r, t in rows.items() if t == "N"), ""
    )
    constraint_rows = {r: t for r, t in rows.items() if t != "N"}

    return {
        "name": name,
        "num_variables": len(columns),
        "num_constraints": len(constraint_rows),
        "num_binary_vars": len(binary_vars),
        "num_integer_vars": len(integer_vars - binary_vars),
        "constraint_types": {
            "leq": sum(1 for t in constraint_rows.values() if t == "L"),
            "geq": sum(1 for t in constraint_rows.values() if t == "G"),
            "eq": sum(1 for t in constraint_rows.values() if t == "E"),
        },
        "objective_coefficients": objective_row,
    }
