"""Phase 2 scaffold for GAMS Model Library collection.

GAMS Model Library collection requires a GAMS installation.
See the project README for setup instructions.

When GAMS is available, this script will:
1. Connect to https://www.gams.com/latest/gamslib_ml/libhtml/
2. Download all available .gms model files
3. Parse each model to extract formulation structure
4. Convert to the unified schema
5. Save to data/processed/gamslib.json

Usage:
    python collectors/collect_gamslib.py
"""

import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_FILE = Path("data/processed/gamslib.json")

# Known GAMS Model Library problem families and representative models
GAMSLIB_CATEGORIES: dict[str, dict[str, Any]] = {
    "transportation": {
        "description": "Transportation and network flow models",
        "representative_models": ["trnsport", "alan"],
        "formulation_type": "LP/MIP",
    },
    "assignment": {
        "description": "Assignment and matching problems",
        "representative_models": ["assign", "bid"],
        "formulation_type": "MIP",
    },
    "knapsack": {
        "description": "Knapsack variants",
        "representative_models": ["knpsk1", "knpsk2"],
        "formulation_type": "MIP",
    },
    "scheduling": {
        "description": "Job scheduling and sequencing problems",
        "representative_models": ["flowshp1", "flowshp2", "jobshop"],
        "formulation_type": "MIP",
    },
    "location": {
        "description": "Facility location problems",
        "representative_models": ["location", "ufl"],
        "formulation_type": "MIP",
    },
    "tsp": {
        "description": "Traveling salesman and routing problems",
        "representative_models": ["tsp3", "tsp4"],
        "formulation_type": "MIP",
    },
    "portfolio": {
        "description": "Portfolio optimization models",
        "representative_models": ["markowitz"],
        "formulation_type": "QP",
    },
    "nonlinear": {
        "description": "Nonlinear programming models",
        "representative_models": ["ramsey", "chenery"],
        "formulation_type": "NLP",
    },
    "network_flow": {
        "description": "Min-cost flow and network design",
        "representative_models": ["mcsched", "netflow"],
        "formulation_type": "LP/MIP",
    },
    "set_covering": {
        "description": "Set covering and partitioning",
        "representative_models": ["setcov"],
        "formulation_type": "MIP",
    },
}


def parse_gms_file(path: Path) -> dict[str, Any]:
    """Parse a GAMS .gms model file and extract formulation structure.

    A .gms file is structured in GAMS Data Exchange (GDX) format with
    sections such as:
      SETS        — index sets
      PARAMETERS  — numeric data
      VARIABLES   — decision variables
      EQUATIONS   — constraints and objective
      MODEL       — model declaration
      SOLVE       — solve statement

    Parameters
    ----------
    path:
        Path to the .gms file.

    Returns
    -------
    dict
        Extracted model structure. Currently a stub awaiting GAMS installation.
    """
    # Stub implementation — requires GAMS to be installed for full parsing.
    logger.warning(
        "parse_gms_file is a stub. Install GAMS and implement full parsing for %s",
        path,
    )
    return {
        "name": path.stem,
        "sets": [],
        "parameters": [],
        "variables": [],
        "equations": [],
        "source": str(path),
    }


def main() -> None:
    """Print setup instructions; full collection requires GAMS installation."""
    print(
        "GAMSLib collection requires GAMS installation. See README for setup."
    )
    logger.info(
        "GAMS Model Library collector is a Phase 2 scaffold. "
        "Install GAMS and implement parse_gms_file() to enable collection."
    )
    logger.info("Known categories: %s", list(GAMSLIB_CATEGORIES.keys()))


if __name__ == "__main__":
    main()
