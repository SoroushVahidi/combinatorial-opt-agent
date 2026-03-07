#!/usr/bin/env python3
"""
Build gams_examples_manifest.csv and gams_examples_inventory.json from
data_private/gams_models/raw/gamspy-examples. Classifies by likely_family (heuristic).
Does not modify license or delete anything.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Repo root relative to script
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data_private" / "gams_models" / "raw" / "gamspy-examples"
MANIFESTS = REPO_ROOT / "data_private" / "gams_models" / "manifests"
EXAMPLES = REPO_ROOT / "data_private" / "gams_models" / "examples"

# Heuristic: dir name or base name -> likely_family (see docs for scheme)
FAMILY_MAP = {
    "trnsport": "transportation_flow",
    "TransportationOn-Off": "transportation_flow",
    "traffic": "transportation_flow",
    "partssupply": "transportation_flow",
    "trnspwl": "transportation_flow",
    "blend": "production_blending",
    "food": "production_blending",
    "prodmix": "production_blending",
    "millco": "production_blending",
    "nurses": "scheduling_assignment",
    "flowshop": "scheduling_assignment",
    "carseq": "scheduling_assignment",
    "rcpsp": "scheduling_assignment",
    "sgolfer": "scheduling_assignment",
    "paintshop": "scheduling_assignment",
    "knapsack": "packing_knapsack",
    "cpack": "packing_knapsack",
    "cutstock": "packing_knapsack",
    "EmergencyCentreAllocation": "covering_facility_location",
    "PMU": "covering_facility_location",
    "PMU-cost": "covering_facility_location",
    "PMU-OBI": "covering_facility_location",
    "ms_cflp": "covering_facility_location",
    "radar_placement": "covering_facility_location",
    "Immunization": "covering_facility_location",
    "tsp": "network_graph_path",
    "tsp4": "network_graph_path",
    "disneyland_itinerary": "network_graph_path",
    "whouse": "inventory_supply_capacity",
    "reservoir": "inventory_supply_capacity",
    "clsp": "inventory_supply_capacity",
    "SimpleLP": "generic_lp",
    "linear": "generic_lp",
    "BoundaryLP": "generic_lp",
    "SimpleMIP": "generic_mip",
    "robustlp": "generic_lp",
    "qp6": "generic_qp",
    "OPF2bus": "network_energy",
    "OPF5bus": "network_energy",
    "acopf": "network_energy",
    "MultiperiodACOPF24bus": "network_energy",
    "DED": "network_energy",
    "DED-PB": "network_energy",
    "DEDESSwind": "network_energy",
    "RampSenDED": "network_energy",
    "EnvironmentalED": "network_energy",
    "EDsensitivity": "network_energy",
    "MOED": "network_energy",
    "EnergyHub": "network_energy",
    "TEP": "network_energy",
    "WaterEnergy": "network_energy",
    "Sharpe": "allocation_budgeting_investment",
    "CVaR": "allocation_budgeting_investment",
    "MAD": "allocation_budgeting_investment",
    "Regret": "allocation_budgeting_investment",
    "BondIndex": "allocation_budgeting_investment",
    "InternationalMeanVar": "allocation_budgeting_investment",
    "MeanVarMip": "allocation_budgeting_investment",
    "DedicationMip": "allocation_budgeting_investment",
    "DedicationNoBorrow": "allocation_budgeting_investment",
    "StochDedicationBL": "allocation_budgeting_investment",
    "SelectiveHedging": "allocation_budgeting_investment",
    "PutCall": "allocation_budgeting_investment",
    "ThreeStageSPDA": "allocation_budgeting_investment",
    "pickstock": "allocation_budgeting_investment",
    "recipes": "production_blending",
    "rowing_optimization": "scheduling_assignment",
    "mpc": "generic_control",
    "transport": "transportation_flow",
    "gapmin": "scheduling_assignment",
}


def likely_family(name: str, path: Path) -> str:
    """Coarse family from folder/base name."""
    base = path.parent.name if path.is_file() else path.name
    key = base if base in FAMILY_MAP else name
    return FAMILY_MAP.get(key, FAMILY_MAP.get(base, "unknown"))


def likely_usage(row: dict) -> str:
    """Comma-separated likely usage tags."""
    usages = []
    ext = (row.get("extension") or "").lower()
    fam = (row.get("likely_family") or "unknown").lower()
    size = int(row.get("size_bytes") or 0)
    if ext == "py" and size < 20_000:
        usages.append("schema_acceptance")
        usages.append("family_classification")
    if ext == "py":
        usages.append("optimization_role_extraction")
    if "transportation" in fam or "blend" in fam or "knapsack" in fam or "nurses" in fam:
        usages.append("external_eval")
    if ext == "ipynb" and size < 100_000:
        usages.append("weak_supervision")
    if not usages:
        usages.append("catalog_only")
    return ",".join(usages)


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    base_rel = RAW.relative_to(REPO_ROOT) if RAW.is_dir() else None

    if not RAW.is_dir():
        print("Raw gamspy-examples not found at", RAW)
        return

    # Models: .py under models/
    for py in sorted((RAW / "models").rglob("*.py")):
        rel = py.relative_to(RAW)
        name = py.stem
        size = py.stat().st_size
        fam = likely_family(name, py)
        original_path = str(Path("data_private") / "gams_models" / "raw" / "gamspy-examples" / rel)
        copied_path = ""
        if size <= 3500 and name in ("trnsport", "blend", "SimpleLP", "SimpleMIP", "knapsack"):
            copied_path = str(Path("data_private") / "gams_models" / "examples" / py.name)
        row = {
            "name": name,
            "original_path": original_path,
            "copied_path": copied_path,
            "extension": "py",
            "size_bytes": size,
            "source_group": "gamspy-examples-models",
            "likely_family": fam,
            "likely_usage": "",  # set below
            "notes": "",
        }
        row["likely_usage"] = likely_usage(row)
        rows.append(row)

    # Notebooks: .ipynb under notebooks/
    for nb in sorted((RAW / "notebooks").rglob("*.ipynb")):
        rel = nb.relative_to(RAW)
        name = nb.stem
        size = nb.stat().st_size
        fam = likely_family(name, nb)
        original_path = str(Path("data_private") / "gams_models" / "raw" / "gamspy-examples" / rel)
        row = {
            "name": f"{nb.parent.name}_{name}" if nb.parent != RAW / "notebooks" else name,
            "original_path": original_path,
            "copied_path": "",
            "extension": "ipynb",
            "size_bytes": size,
            "source_group": "gamspy-examples-notebooks",
            "likely_family": fam,
            "likely_usage": likely_usage({"extension": "ipynb", "size_bytes": size, "likely_family": fam}),
            "notes": "catalog only (notebook)",
        }
        rows.append(row)

    # CSV
    fieldnames = [
        "name", "original_path", "copied_path", "extension", "size_bytes",
        "source_group", "likely_family", "likely_usage", "notes",
    ]
    csv_path = MANIFESTS / "gams_examples_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("Wrote", csv_path, "rows:", len(rows))

    # JSON inventory
    inv = {
        "source": "data_private/gams_models/raw/gamspy-examples",
        "family_scheme": "heuristic from path/name; see GAMSPY_LOCAL_EXAMPLES_COLLECTION.md",
        "count": len(rows),
        "by_extension": {},
        "by_family": {},
        "entries": rows,
    }
    for r in rows:
        ext = r["extension"]
        inv["by_extension"][ext] = inv["by_extension"].get(ext, 0) + 1
        fam = r["likely_family"]
        inv["by_family"][fam] = inv["by_family"].get(fam, 0) + 1
    json_path = MANIFESTS / "gams_examples_inventory.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)
    print("Wrote", json_path)


if __name__ == "__main__":
    main()
