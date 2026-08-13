"""Common NLP4LP manifest loading and drift verification.

`baselines/comparison/manifests/nlp4lp_common_18.json` is the authoritative
record. `verify_baseline_manifests` checks each of the four lightweight
baseline packages' own manifest files against it and reports drift --
it never silently overwrites a baseline's manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_MANIFEST_PATH = Path(__file__).resolve().parent / "manifests" / "nlp4lp_common_18.json"

_BASELINE_MANIFEST_PATHS = {
    "orlm": _REPO_ROOT / "baselines/orlm/manifests/nlp4lp_common_manifest.json",
    "optmath": _REPO_ROOT / "baselines/optmath/manifests/nlp4lp_common_manifest.json",
    "deepor": _REPO_ROOT / "baselines/deepor/manifests/nlp4lp_common_manifest.json",
    "orr1": _REPO_ROOT / "baselines/orr1/manifests/nlp4lp_common_manifest.json",
}


def load_common_manifest(path: Path | str = COMMON_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_baseline_manifests(common: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Returns {baseline: [drift descriptions]}; empty list means no drift."""
    common = common or load_common_manifest()
    problems: dict[str, list[str]] = {}
    for name, path in _BASELINE_MANIFEST_PATHS.items():
        issues: list[str] = []
        if not path.exists():
            issues.append("manifest_file_missing")
            problems[name] = issues
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("pilot_ids") != common["pilot_ids"]:
            issues.append(f"pilot_ids_drift: {data.get('pilot_ids')} != {common['pilot_ids']}")
        if data.get("future_evaluation_ids") != common["future_evaluation_ids"]:
            issues.append(f"future_evaluation_ids_drift: {data.get('future_evaluation_ids')} != {common['future_evaluation_ids']}")
        problems[name] = issues
    return problems


def pamop_empirical_manifest_note(common: dict[str, Any] | None = None) -> str:
    """Explains the known, real divergence between PaMOP's executed evidence and the shared pilot_ids convention.

    Do not "fix" this by relabeling PaMOP's IDs or the shared pilot_ids --
    both are real, already-committed selections; the divergence itself is
    the fact to report (see PHASE 7 / PHASE 9 of the comparison-harness task).
    """
    common = common or load_common_manifest()
    divergence = common["known_manifest_divergence"]
    pamop_ids = divergence["pamop_empirical_pilot_ids"]
    return (
        f"The shared pilot_ids convention {common['pilot_ids']} (used by the "
        "ORLM/OptMATH/DeepOR/OR-R1 lightweight manifests) is NOT identical to "
        f"the 6 problem IDs PaMOP's gpt-5.4 fidelity diagnostic actually executed "
        f"({pamop_ids}). Overlap: "
        f"{sorted(set(common['pilot_ids']) & set(pamop_ids))} "
        f"(4 of 6). Both ID sets are subsets of the same 18-instance "
        f"future_evaluation_ids superset {common['future_evaluation_ids']}, so "
        "they remain comparable at the 18-instance level, but any 6-instance "
        "pilot-vs-pilot comparison must state which 6 it means."
    )
