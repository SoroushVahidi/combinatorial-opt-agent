"""Phase 2 scaffold for OR-Library collection.

OR-Library (Beasley's OR-Library) hosts benchmark problem instances for
numerous combinatorial optimization problem families.
When network access is available, this script will:
1. Download problem family index from http://people.brunel.ac.uk/~mastjjb/jeb/info.html
2. Download instance data files for each family
3. Parse instance formats
4. Convert to the unified schema
5. Save to data/processed/or_library.json

Usage:
    python collectors/collect_or_library.py
"""

import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OR_LIBRARY_BASE_URL = "http://people.brunel.ac.uk/~mastjjb/jeb"
OUTPUT_FILE = Path("data/processed/or_library.json")
RAW_DIR = Path("data/raw/or_library")

# Full catalogue of OR-Library problem families
OR_LIBRARY_FAMILIES: dict[str, dict[str, Any]] = {
    "set_covering": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/scpinfo.html",
        "description": (
            "Set covering problem: find minimum-cost subset of columns that "
            "covers all rows of a 0/1 matrix."
        ),
        "formulation_type": "MIP",
        "num_instances": 80,
    },
    "set_packing": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/sppinfo.html",
        "description": (
            "Set packing problem: find maximum-weight subset of columns such "
            "that no row is covered more than once."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "set_partitioning": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/sppinfo.html",
        "description": (
            "Set partitioning problem: find minimum-cost subset of columns "
            "that covers every row exactly once."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "knapsack_01": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/knapsackinfo.html",
        "description": (
            "0-1 knapsack problem: select items to maximise value subject to "
            "a weight capacity constraint."
        ),
        "formulation_type": "MIP",
        "num_instances": 2890,
    },
    "multidimensional_knapsack": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/mknapinfo.html",
        "description": (
            "Multidimensional (m-constraint) 0-1 knapsack problem."
        ),
        "formulation_type": "MIP",
        "num_instances": 270,
    },
    "uncapacitated_warehouse_location": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/capinfo.html",
        "description": (
            "Uncapacitated facility location (warehouse location) problem: "
            "open facilities and assign customers to minimise total cost."
        ),
        "formulation_type": "MIP",
        "num_instances": 71,
    },
    "capacitated_warehouse_location": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/capinfo.html",
        "description": (
            "Capacitated facility location problem with capacity constraints "
            "on each open facility."
        ),
        "formulation_type": "MIP",
        "num_instances": 71,
    },
    "p_median": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/pmedinfo.html",
        "description": (
            "p-median problem: locate p facilities to minimise total "
            "weighted distance from customers."
        ),
        "formulation_type": "MIP",
        "num_instances": 40,
    },
    "graph_coloring": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/colourinfo.html",
        "description": (
            "Graph colouring problem: assign colours to vertices such that "
            "no adjacent vertices share a colour, minimising colours used."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "max_clique": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/clqinfo.html",
        "description": (
            "Maximum clique problem: find the largest complete subgraph."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "max_cut": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/maxcutinfo.html",
        "description": (
            "Maximum cut problem: partition graph vertices to maximise the "
            "number (or weight) of edges between the two parts."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "steiner_tree": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/steininfo.html",
        "description": (
            "Steiner tree problem: find minimum-cost tree spanning a given "
            "subset of terminal vertices."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "job_shop_scheduling": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/jobshopinfo.html",
        "description": (
            "Job shop scheduling: schedule jobs on machines to minimise "
            "makespan, with precedence constraints."
        ),
        "formulation_type": "MIP",
        "num_instances": 82,
    },
    "flow_shop_scheduling": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/flowshopinfo.html",
        "description": (
            "Flow shop scheduling: schedule jobs through a fixed sequence "
            "of machines to minimise makespan."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "single_machine_scheduling": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/wtjinfo.html",
        "description": (
            "Single machine scheduling to minimise total weighted tardiness."
        ),
        "formulation_type": "MIP",
        "num_instances": 125,
    },
    "assignment": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/assigninfo.html",
        "description": (
            "Linear assignment problem: assign n agents to n tasks to "
            "minimise total assignment cost."
        ),
        "formulation_type": "LP",
        "num_instances": None,
    },
    "generalized_assignment": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/gapinfo.html",
        "description": (
            "Generalised assignment problem: assign tasks to agents with "
            "per-agent capacity constraints to minimise cost."
        ),
        "formulation_type": "MIP",
        "num_instances": 900,
    },
    "bin_packing": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/binpackinfo.html",
        "description": (
            "Bin packing problem: pack items of various sizes into the "
            "fewest fixed-capacity bins."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "quadratic_assignment": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/qapinfo.html",
        "description": (
            "Quadratic assignment problem: assign facilities to locations "
            "to minimise total flow-distance cost."
        ),
        "formulation_type": "MIP/NLP",
        "num_instances": None,
    },
    "vehicle_routing": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/vrpinfo.html",
        "description": (
            "Capacitated vehicle routing problem: route vehicles from a "
            "depot to customers to minimise total distance."
        ),
        "formulation_type": "MIP",
        "num_instances": None,
    },
    "min_cost_flow": {
        "url": f"{OR_LIBRARY_BASE_URL}/orlib/mincostinfo.html",
        "description": (
            "Minimum cost flow problem on a directed network with arc "
            "capacities and costs."
        ),
        "formulation_type": "LP",
        "num_instances": None,
    },
}


def main() -> None:
    """Print status; full collection downloads instances from OR-Library."""
    logger.info(
        "OR-Library collector is a Phase 2 scaffold. "
        "Implement network download logic to enable collection."
    )
    logger.info(
        "Known problem families (%d): %s",
        len(OR_LIBRARY_FAMILIES),
        list(OR_LIBRARY_FAMILIES.keys()),
    )


if __name__ == "__main__":
    main()
