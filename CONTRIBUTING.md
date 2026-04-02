# Contributing

This is a companion research repository for the EAAI manuscript on retrieval-assisted
instantiation of natural-language optimization problems.  Contributions should respect
the experimental integrity of the published results.

---

## Development setup

```bash
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

No special environment variables are needed for most contributions.  Downstream
benchmark scripts additionally require `HF_TOKEN` for access to the gated
`udell-lab/NLP4LP` dataset — see [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md) for setup.

---

## Running the test suite

```bash
python -m pytest tests/ -q
```

All tests are CPU-only and self-contained.  Tests tagged `requires_network` are skipped
when `HF_TOKEN` is absent.  The full suite (~1 400 tests) should pass without any
external services.

---

## Validating paper artifacts

```bash
python scripts/paper/run_repo_validation.py
```

This checks that all required tables, figures, analysis reports, and canonical
documents are present and structurally sound.  Run this before submitting a PR that
touches `results/paper/` or `analysis/eaai_*`.

---

## Protected paths — do not overwrite casually

- `results/paper/` — camera-ready artifacts; regenerate only via
  `tools/build_eaai_camera_ready_figures.py`
- `analysis/eaai_*` — experiment reports; regenerate only by re-running the canonical
  experiment scripts (see [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md))
- `docs/EAAI_SOURCE_OF_TRUTH.md` — manuscript authority; edit only when the manuscript
  itself changes

---

## Adding a new grounding method

1. Add extraction / scoring logic in `tools/nlp4lp_downstream_utility.py` under the
   appropriate section (see the module docstring table of contents).
2. Add a method key to the `--method` choices in `main()` and a dispatch branch in
   `run_setting()`.
3. Add unit tests in `tests/` following the naming pattern `test_<method_name>.py`.
4. Run `python -m pytest tests/ -q` to confirm all tests pass.

---

## Adding a new retrieval baseline

Add the baseline class in `retrieval/baselines.py` (inheriting from `BaseRetriever`)
and register it in `retrieval/search.py`.  Add tests in `tests/test_baselines.py`.

---

## Code style

- Python 3.10+; type hints encouraged
- No hard-coded absolute paths; use `ROOT = Path(__file__).resolve().parent.parent`
- Keep secrets out of source files; use `.env` (gitignored) or environment variables
- Do not silently change benchmark numbers or rewrite docs to claim stronger results
  than the artifacts support

---

## Source-of-truth locations

| What | Where |
|------|-------|
| Paper framing, authoritative claims | `docs/EAAI_SOURCE_OF_TRUTH.md` |
| Canonical metrics + provenance | `docs/RESULTS_PROVENANCE.md` |
| Camera-ready tables | `results/paper/eaai_camera_ready_tables/` |
| Camera-ready figures | `results/paper/eaai_camera_ready_figures/` |
| Experiment reports | `analysis/eaai_*_report.md` |
