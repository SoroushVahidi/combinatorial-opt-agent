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
    """Explains the (now resolved) divergence between PaMOP's executed evidence and the shared pilot_ids convention.

    PaMOP's original 6-problem diagnostic used a different 6 than the shared
    `pilot_ids`; the scaled extension (2026-08-15) executed the remaining 12,
    so PaMOP's evidence now covers the full 18-instance
    `future_evaluation_ids` set. A 6-instance pilot-vs-pilot comparison must
    still state which 6 it means.
    """
    common = common or load_common_manifest()
    divergence = common["known_manifest_divergence"]
    pamop_ids = divergence["pamop_empirical_pilot_ids"]
    return (
        f"PaMOP's original gpt-5.4 diagnostic executed 6 IDs "
        f"({pamop_ids}), which were NOT the shared pilot_ids convention "
        f"{common['pilot_ids']}. That divergence is RESOLVED at the 18-instance "
        "level: the 2026-08-15 scaled extension executed the remaining 12 IDs, "
        "so PaMOP's empirical evidence now covers the full "
        f"`future_evaluation_ids` set {common['future_evaluation_ids']}. "
        "Any 6-instance pilot-vs-pilot comparison must still state which 6 it means."
    )
