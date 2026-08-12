# Repository Map

For the full annotated directory tree, see [`REPO_STRUCTURE.md`](REPO_STRUCTURE.md)
(kept as the single detailed map to avoid two files drifting out of sync).
For current scientific status, start at [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md).

## Quick classification

| Directory | Status | Notes |
|---|---|---|
| `manuscript/` | ACTIVE | KAIS manuscript source (`main.tex`, authoritative) |
| `tools/`, `retrieval/`, `formulation/`, `src/` | ACTIVE | Core pipeline source |
| `baselines/pamop/` | ACTIVE | PaMOP reproduction, pilot executed, fidelity gate resolved |
| `baselines/orlm/` | ACTIVE (scaffold only) | ORLM interfaces, no model call — see `baselines/orlm/README.md` |
| `tests/` | ACTIVE | Pytest suite |
| `results/paper/` | ACTIVE (with camera-ready numbers now flagged stale relative to current code) | Camera-ready tables/figures; see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` |
| `results/eswa_revision/` | ACTIVE (evidence base) / historical naming, **`postfix_main_metrics.csv` flagged STALE_RELATIVE_TO_CURRENT_CODE** | Contains the canonical downstream tables the manuscript was built from; see the staleness audit |
| `results/pamop/` | ACTIVE | PaMOP pilot + forensics + fidelity diagnostic results |
| `results/baseline_staleness_audit_2026-08-12/`, `results/max_weight_matching_validation/` | ACTIVE | Fresh, current-code method comparison + mechanism/error analysis (Phase 4) |
| `scripts/analysis/`, `scripts/pamop_fidelity_diagnostic.py` | ACTIVE | Phase 4 analysis scripts (MWM mechanism/error analysis, PaMOP fidelity diagnostic) |
| `docs/` (root-level files) | ACTIVE | Current status, reproduction, known issues |
| `docs/archive/`, `docs/archive_internal_status/`, `docs/provenance/`, `docs/eswa_revision/`, `analysis/archive/` | ARCHIVE | Provenance only, not authoritative — each has its own README |
| `app.py`, `demo/`, `feedback_server.py`, `analyze_feedback.py`, `deploy_to_hf.py`, `telemetry.py` | ACTIVE but out-of-scope | Demo/UX, not the evaluated pipeline |
| `outputs/`, `logs/`, `cache/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` | GENERATED | Reproducible; mostly gitignored |
| `data/` | ACTIVE | Dataset registry + small catalogs; large gated data not stored here |
| `batch/`, `jobs/` | ACTIVE | Slurm / HPC job definitions |

See `REPO_STRUCTURE.md` for the file-level tree and the ★/⚠/✦ authority markers.
