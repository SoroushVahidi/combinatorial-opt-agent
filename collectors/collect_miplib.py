"""Phase 2 scaffold for MIPLIB 2017 collection.

MIPLIB 2017 is a benchmark library of Mixed-Integer Programming instances.
When network access is available, this script will:
1. Download instance metadata from https://miplib.zib.de/
2. Download .mps.gz instance files
3. Parse each instance using parse_mps_file()
4. Convert to the unified schema
5. Save to data/processed/miplib.json

Usage:
    python collectors/collect_miplib.py
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIPLIB_BASE_URL = "https://miplib.zib.de"
MIPLIB_DOWNLOAD_URL = "https://miplib.zib.de/downloads/benchmark.zip"
OUTPUT_FILE = Path("data/processed/miplib.json")
RAW_DIR = Path("data/raw/miplib")

# Known MIPLIB tags used for categorisation
MIPLIB_TAGS = [
    "benchmark",
    "easy",
    "hard",
    "open",
    "infeasible",
    "binary",
    "general_integer",
    "continuous",
    "feasibility",
    "decomposable",
]


def parse_mps_file(path: Path) -> dict[str, Any]:
    """Parse a standard MPS format file and return structured metadata.

    MPS Format Sections
    -------------------
    NAME       — instance name
    ROWS       — constraints; type N (objective), L (≤), G (≥), E (=)
    COLUMNS    — non-zero coefficients (variable × row)
    RHS        — right-hand side values
    BOUNDS     — variable bounds (LO, UP, FX, FR, MI, BV, LI, UI)
    RANGES     — optional range constraints
    ENDATA     — end marker

    Parameters
    ----------
    path:
        Path to the .mps file (plain text or .mps.gz).

    Returns
    -------
    dict
        Parsed instance metadata including variable/constraint counts.
    """
    opener = gzip.open if str(path).endswith(".gz") else open

    name = path.stem.replace(".mps", "")
    rows: dict[str, str] = {}          # row_name -> type (N/L/G/E)
    columns: dict[str, list[str]] = {} # var_name -> list of rows it appears in
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

                # Section headers start at column 0
                if not line.startswith(" "):
                    current_section = line.split()[0].upper()
                    continue

                parts = line.split()
                if not parts:
                    continue

                if current_section == "NAME":
                    name = parts[0] if parts else name

                elif current_section == "ROWS":
                    row_type = parts[0]
                    row_name = parts[1] if len(parts) > 1 else ""
                    rows[row_name] = row_type

                elif current_section == "COLUMNS":
                    if "'MARKER'" in line:
                        int_marker_active = "'INTORG'" in line
                        continue
                    var_name = parts[0]
                    if var_name not in columns:
                        columns[var_name] = []
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
        logger.error("Error reading %s: %s", path, exc)
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


def main() -> None:
    """Print status; full collection downloads instances from MIPLIB."""
    logger.info(
        "MIPLIB 2017 collector is a Phase 2 scaffold. "
        "Implement network download logic and call parse_mps_file() to enable collection."
    )
    logger.info("MIPLIB base URL: %s", MIPLIB_BASE_URL)
    logger.info("Known tags: %s", MIPLIB_TAGS)


if __name__ == "__main__":
    main()
