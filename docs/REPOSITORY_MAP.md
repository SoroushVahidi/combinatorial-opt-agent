# Repository Map

For the full annotated directory tree, see [`REPO_STRUCTURE.md`](REPO_STRUCTURE.md)
(kept as the single detailed map to avoid two files drifting out of sync).
For current scientific status, start at [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md).

## Quick classification

| Directory | Status | Notes |
|---|---|---|
| `manuscript/` | ACTIVE | KAIS manuscript source (`main.tex`, authoritative) |
| `tools/`, `retrieval/`, `formulation/`, `src/` | ACTIVE | Core pipeline source |
| `baselines/pamop/` | ACTIVE | PaMOP reproduction, in progress |
| `tests/` | ACTIVE | Pytest suite |
| `results/paper/` | ACTIVE (with one known-stale file) | Camera-ready tables/figures; see `PROJECT_STATUS.md` §3 |
| `results/eswa_revision/` | ACTIVE (evidence base) / historical naming | Contains the corrected canonical downstream tables despite the "eswa" directory name |
| `results/pamop/` | ACTIVE | PaMOP pilot + forensics results |
| `docs/` (root-level files) | ACTIVE | Current status, reproduction, known issues |
| `docs/archive/`, `docs/archive_internal_status/`, `docs/provenance/`, `docs/eswa_revision/`, `analysis/archive/` | ARCHIVE | Provenance only, not authoritative — each has its own README |
| `app.py`, `demo/`, `feedback_server.py`, `analyze_feedback.py`, `deploy_to_hf.py`, `telemetry.py` | ACTIVE but out-of-scope | Demo/UX, not the evaluated pipeline |
| `outputs/`, `logs/`, `cache/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` | GENERATED | Reproducible; mostly gitignored |
| `data/` | ACTIVE | Dataset registry + small catalogs; large gated data not stored here |
| `batch/`, `jobs/` | ACTIVE | Slurm / HPC job definitions |

See `REPO_STRUCTURE.md` for the file-level tree and the ★/⚠/✦ authority markers.
