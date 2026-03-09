"""
check_training_env.py — Lightweight training environment diagnostic for NLP4LP learning stage.

Run before submitting a real training job to verify that the Python environment,
key packages, CUDA visibility, and input data/config artifacts are all present.

Usage:
    python src/learning/check_training_env.py

Exit codes:
    0  — all required checks passed (torch, transformers present).
    1  — one or more required checks failed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(label: str, value: str, ok: bool = True) -> bool:
    status = "✓" if ok else "✗"
    print(f"  [{status}] {label}: {value}")
    return ok


def _import_pkg(name: str) -> tuple[bool, str]:
    """Try to import *name* and return (success, version_or_error)."""
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Check groups
# ---------------------------------------------------------------------------

def check_python() -> None:
    print("\n=== Python environment ===")
    _check("sys.executable", sys.executable)
    _check("python version", sys.version.split()[0])
    _check("working directory", os.getcwd())


def check_packages() -> dict[str, bool]:
    """Return a dict of required-package → availability."""
    print("\n=== Required packages ===")
    results: dict[str, bool] = {}

    for pkg in ("torch", "transformers", "sentence_transformers", "datasets", "accelerate"):
        ok, info = _import_pkg(pkg)
        _check(pkg, info, ok)
        results[pkg] = ok

    return results


def check_cuda(torch_available: bool) -> None:
    print("\n=== CUDA / GPU ===")
    if not torch_available:
        _check("torch.cuda.is_available()", "skipped (torch not importable)", ok=False)
        _check("GPU count", "skipped", ok=False)
        return
    import torch  # noqa: PLC0415
    cuda_ok = torch.cuda.is_available()
    _check("torch.cuda.is_available()", str(cuda_ok), ok=cuda_ok)
    gpu_count = torch.cuda.device_count()
    _check("GPU count", str(gpu_count), ok=(gpu_count >= 0))
    if cuda_ok:
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
            _check(f"  GPU {i}", f"{name}  ({vram} GiB VRAM)")


def check_artifacts(repo_root: Path) -> dict[str, bool]:
    """Check that key input directories and config files exist."""
    print("\n=== Input artifacts ===")
    paths_to_check = {
        "artifacts/learning_ranker_data/nlp4lp": repo_root / "artifacts" / "learning_ranker_data" / "nlp4lp",
        "artifacts/learning_aux_data/nl4opt":    repo_root / "artifacts" / "learning_aux_data" / "nl4opt",
        "configs/learning/experiment_matrix_stage3.json":
            repo_root / "configs" / "learning" / "experiment_matrix_stage3.json",
        # Existing training data (retrieval side)
        "data/processed/nlp4lp_eval_orig.jsonl":
            repo_root / "data" / "processed" / "nlp4lp_eval_orig.jsonl",
    }
    results: dict[str, bool] = {}
    for label, path in paths_to_check.items():
        exists = path.exists()
        _check(label, "found" if exists else "MISSING", ok=exists)
        results[label] = exists
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk up from this file to find the project root (contains 'requirements.txt')."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()


def main() -> int:
    print("=" * 60)
    print("combinatorial-opt-agent: training environment check")
    print("=" * 60)

    repo_root = _find_repo_root()
    print(f"\nDetected repo root: {repo_root}")

    check_python()
    pkg_results = check_packages()
    check_cuda(torch_available=pkg_results.get("torch", False))
    artifact_results = check_artifacts(repo_root)

    # --- Summary ---
    print("\n=== Summary ===")
    required_pkg_ok = pkg_results.get("torch", False) and pkg_results.get("transformers", False)
    print(f"  Required packages (torch, transformers): {'✓ OK' if required_pkg_ok else '✗ MISSING'}")

    missing_artifacts = [k for k, v in artifact_results.items() if not v]
    if missing_artifacts:
        print("  Missing artifacts (create before training):")
        for m in missing_artifacts:
            print(f"    - {m}")
    else:
        print("  All expected artifacts present: ✓")

    overall_ok = required_pkg_ok
    print(f"\nOverall required-package check: {'PASSED ✓' if overall_ok else 'FAILED ✗'}")

    if not overall_ok:
        print(
            "\nAction needed: activate a conda env or venv that has torch and transformers.\n"
            "See docs/WULVER_RESOURCE_INVENTORY.md for the established environment pattern."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
