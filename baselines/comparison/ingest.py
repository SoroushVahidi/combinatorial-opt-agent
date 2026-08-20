"""Explicit, known-location ingestion for each system's real result files.

No filesystem crawling: every path below is a fixed, explicit location.
Systems with zero real result files on disk (ORLM, OptMATH, DeepOR, OR-R1 as
of 2026-08-13) simply produce an empty row list -- callers must not invent
rows for them.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from baselines.comparison.adapters import adapt_generic, adapt_optmath, adapt_orlm, adapt_ours, adapt_pamop
from baselines.comparison.manifests import load_common_manifest
from baselines.comparison.schema import UnifiedRow

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _current_git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()

PAMOP_FIDELITY_DIAGNOSTIC_DIR = _REPO_ROOT / "results/pamop/fidelity_diagnostic_gpt5"

ORLM_COMMON18_DIR = _REPO_ROOT / "results/orlm/common18_official_checkpoint"
ORLM_COMMON18_RESULTS_PATH = ORLM_COMMON18_DIR / "results.jsonl"

GENERIC_COMMON18_DIR = _REPO_ROOT / "results/generic_llm/common18_official"
GENERIC_COMMON18_RESULTS_PATH = GENERIC_COMMON18_DIR / "results.jsonl"

OPTMATH_COMMON18_DIR = _REPO_ROOT / "results/optmath/common18_official_checkpoint"
OPTMATH_COMMON18_RESULTS_PATH = OPTMATH_COMMON18_DIR / "results.jsonl"

KNOWN_RESULT_LOCATIONS: dict[str, list[str]] = {
    "ours": ["generated fresh via tools.nlp4lp_downstream_utility (CPU-only, ~1s); see ingest_ours()"],
    "pamop": [str(PAMOP_FIDELITY_DIAGNOSTIC_DIR / "per_problem.csv"), str(PAMOP_FIDELITY_DIAGNOSTIC_DIR / "run_metadata.json")],
    "orlm": [str(ORLM_COMMON18_RESULTS_PATH)],
    "optmath": [str(OPTMATH_COMMON18_RESULTS_PATH)],
    "generic": [str(GENERIC_COMMON18_RESULTS_PATH)],
    "deepor": [],
    "orr1": [],
}


def _doc_id_for_problem_id(problem_id: int) -> str:
    """PaMOP's stable convention, verified against results/pamop/*/selected_ids.json: doc_id index = problem_id - 1."""
    return f"nlp4lp_test_{problem_id - 1}"


def ingest_ours(*, manifest: dict | None = None, save_subset_to: Path | None = None) -> list[UnifiedRow]:
    """Runs the CPU-only, deterministic typed-greedy benchmark fresh and filters to the common manifest.

    This intentionally does not reuse any committed per-query CSV, since the
    only ones on disk predate 49 grounding-fix commits (see
    docs/BASELINE_STALENESS_AUDIT_2026-08-12.md) and would silently reintroduce
    the staleness this repository has already flagged as a defect.
    """
    manifest = manifest or load_common_manifest()
    doc_ids = {_doc_id_for_problem_id(pid) for pid in manifest["future_evaluation_ids"]}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="comparison_ours_") as tmp:
        subprocess.run(
            [sys.executable, "-m", "tools.nlp4lp_downstream_utility", "--variant", "orig", "--baseline", "tfidf",
             "--assignment-mode", "typed", "--output-dir", tmp, "--skip-gemini-preflight"],
            cwd=_REPO_ROOT, check=True, capture_output=True, text=True,
            env={"NLP4LP_GOLD_CACHE": "results/eswa_revision/00_env/nlp4lp_gold_cache.json", "PYTHONHASHSEED": "0", "PATH": __import__("os").environ.get("PATH", "")},
        )
        per_query_path = Path(tmp) / "nlp4lp_downstream_per_query_orig_tfidf.csv"
        with per_query_path.open(encoding="utf-8") as fh:
            all_rows = list(csv.DictReader(fh))
    subset_rows = [r for r in all_rows if r["query_id"] in doc_ids]
    if save_subset_to is not None:
        save_subset_to.parent.mkdir(parents=True, exist_ok=True)
        with save_subset_to.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(subset_rows[0].keys()) if subset_rows else [])
            writer.writeheader()
            writer.writerows(subset_rows)
    git_sha = _current_git_sha()
    unified = [adapt_ours(r) for r in subset_rows]
    for row in unified:
        row.local_git_sha = git_sha
    return unified


def ingest_pamop() -> list[UnifiedRow]:
    per_problem_path = PAMOP_FIDELITY_DIAGNOSTIC_DIR / "per_problem.csv"
    run_metadata_path = PAMOP_FIDELITY_DIAGNOSTIC_DIR / "run_metadata.json"
    if not per_problem_path.exists():
        return []
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8")) if run_metadata_path.exists() else {}
    with per_problem_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [adapt_pamop(r, run_metadata=run_metadata) for r in rows]


def ingest_orlm() -> list[UnifiedRow]:
    """Reads the official-checkpoint ORLM common-18 JSONL (fixed location)."""
    return _ingest_common18_jsonl(ORLM_COMMON18_RESULTS_PATH, adapt_orlm)


def ingest_optmath() -> list[UnifiedRow]:
    """Reads the official-checkpoint OptMATH common-18 JSONL (fixed location).

    Guarded: a partially-written checkpoint (fewer rows than the manifest's
    future_evaluation_ids) is treated as 'not yet complete' and yields no
    rows, so a running job can never leak a partial run into the report.
    """
    return _ingest_common18_jsonl(OPTMATH_COMMON18_RESULTS_PATH, adapt_optmath)


def _ingest_common18_jsonl(path: Path, adapt, *, expect_ids: set[str] | None = None) -> list[UnifiedRow]:
    if not path.exists():
        return []
    manifest_ids = expect_ids if expect_ids is not None else {
        str(pid) for pid in load_common_manifest()["future_evaluation_ids"]
    }
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if len(rows) < len(manifest_ids):
        return []  # checkpoint still being written (partial run) -- never report it
    return [adapt(r) for r in rows]


def ingest_generic() -> list[UnifiedRow]:
    """Reads the generic-LLM (gpt-5.4) common-18 JSONL (fixed location)."""
    return _ingest_common18_jsonl(GENERIC_COMMON18_RESULTS_PATH, adapt_generic)


def ingest_all(*, systems: list[str] | None = None, save_ours_subset_to: Path | None = None) -> dict[str, list[UnifiedRow]]:
    systems = systems or ["ours", "pamop", "orlm", "optmath", "generic", "deepor", "orr1"]
    out: dict[str, list[UnifiedRow]] = {}
    if "ours" in systems:
        out["ours"] = ingest_ours(save_subset_to=save_ours_subset_to)
    if "pamop" in systems:
        out["pamop"] = ingest_pamop()
    if "orlm" in systems:
        out["orlm"] = ingest_orlm()
    if "optmath" in systems:
        out["optmath"] = ingest_optmath()
    if "generic" in systems:
        out["generic"] = ingest_generic()
    for name in ("deepor", "orr1"):
        if name in systems:
            out[name] = []  # No real result files exist on disk yet (see availability.py).
    return out
