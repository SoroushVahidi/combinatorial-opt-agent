# Contributing

This is a companion research repository for the EAAI manuscript on retrieval-assisted
instantiation of natural-language optimization problems. Contributions should respect
the experimental integrity of the published results.

## Running the test suite

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/
```

Most tests are self-contained and CPU-only. Tests tagged `requires_network` are skipped
unless a HuggingFace token is available.

## Validating the paper artifacts

```bash
python scripts/paper/run_repo_validation.py
```

This checks that all required tables, figures, analysis reports, and source-of-truth
documents are present and structurally sound.

## Source-of-truth locations

| What | Where |
|------|-------|
| Paper framing, authoritative claims | `docs/EAAI_SOURCE_OF_TRUTH.md` |
| Camera-ready tables | `results/paper/eaai_camera_ready_tables/` |
| Camera-ready figures | `results/paper/eaai_camera_ready_figures/` |
| Experiment reports | `analysis/eaai_*_report.md` |

## Protected paths — do not overwrite casually

- `results/paper/` — camera-ready artifacts; regenerate only via `tools/build_eaai_camera_ready_figures.py`
- `analysis/eaai_*` — experiment reports; regenerate only by re-running the canonical experiment scripts
- `docs/EAAI_SOURCE_OF_TRUTH.md` — manuscript authority; edit only when the manuscript itself changes

## Adding a new grounding method

1. Add extraction / scoring logic in `tools/nlp4lp_downstream_utility.py` under the
   appropriate section (see the table of contents in the module docstring).
2. Add a method key to the `--method` choices in `main()` and a dispatch branch in
   `run_setting()`.
3. Add unit tests in `tests/` following the naming pattern `test_<method_name>.py`.
4. Run `python -m pytest tests/ -q` to confirm all tests pass.

## Adding a new retrieval baseline

Add the baseline class in `retrieval/baselines.py` (inheriting from `BaseRetriever`)
and register it in `retrieval/search.py`. Add tests in `tests/test_baselines.py`.

## Code style

- Python 3.10+, type hints encouraged
- No hard-coded absolute paths; use `ROOT = Path(__file__).resolve().parent.parent`
- Keep secrets out of source files; use `.env` (gitignored) or environment variables
