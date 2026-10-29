# Testing and dependencies

Maintenance checks must not acquire data or generate scientific results. Run from
the repository root in an isolated environment. Python 3.10+ is the source baseline;
CI uses 3.12. `requirements-dev.txt` contains small test dependencies. The broader
`requirements.txt` is optional for maintenance and is not a historical lockfile.

## Minimal offline commands

```bash
python scripts/check_repository_syntax.py
python scripts/check_docs_integrity.py
python -m pytest -q tests/test_cardinal_downloader.py tests/test_documentation_integrity.py tests/test_offline_policy.py tests/test_repo_hygiene.py tests/test_consistency_benchmark.py
```

Syntax checks parse tracked Python and run `bash -n` on shell scripts without
executing research. Documentation checks inspect tracked Markdown destinations
and conflict markers, not untracked private files. External endpoints and fragment
anchors require separate review. The checker is intentionally a lightweight parser.

## Broader tests and explicit capabilities

`python -m pytest -q` selects `tests/`; adapters can be checked separately with
`python -m pytest -q baselines/ --import-mode=importlib`. Missing optional libraries
may prevent collection or individual tests. Such failures are not successful tests.
NumPy, SciPy, scikit-learn, rank-bm25 and requests are used by portions of the broader
suite. App tests additionally depend on Gradio and its compatible stack. Dense
retrieval uses sentence-transformers/torch; API adapters may need their provider SDKs.
Consult imports and the relevant component README before installing optional stacks.
Do not infer original experiment versions from a newly installed environment.

| Capability | Marker / CLI opt-in | Requirements and scope |
|---|---|---|
| External HTTP or dataset access | `requires_network` / `--run-network` | Deliberate access to authorized endpoints; no reachability probe during collection |
| Model loading/inference | `requires_models` / `--run-models` | Optional model stack and weights; download tests also need `--run-network` |
| Live provider calls | `requires_providers` / `--run-providers` | Authorized credentials and budget, plus network opt-in; mocked provider tests stay offline |
| Scientific solver execution | `requires_solvers` / `--run-solvers` | Appropriate backend/license; no solver experiments in maintenance CI |
| Research datasets | `requires_research_data` / `--run-research-data` | Reviewed dataset access/provenance; downloads also require network opt-in |
| Other costly integrations | `requires_integration` / `--run-integration` | Component-specific environment and explicit review |

Existing model-download and external-fetch tests are marked. New tests that need
these capabilities must declare all applicable markers. The opt-ins are test
policy, not an operating-system sandbox: do not run arbitrary research subprocesses
under the assumption that Python socket blocking controls them. Ordinary pytest
blocks Python Internet socket connections from configuration onward and sets HF
and Transformers offline mode. Unix-domain sockets remain available. The legacy
`network_available` fixture now means network was requested, not that an endpoint
has been proved reachable. Tokens alone never enable tests.

## Workflow boundaries

`maintenance.yml` runs only the minimal commands above, with read-only repository
permissions and no research secrets. Installing pytest contacts the package index;
the tests themselves do not contact providers or acquire research data.

Collection and NLP4LP benchmark workflows remain available through manual dispatch
with an explicit confirmation input. They never run on ordinary maintenance pushes,
never push generated results, and no longer upload raw logs or processed datasets.
Their ephemeral outputs are not preservation archives; use authorized research
infrastructure when outputs need to be retained and reviewed. The manual HF access
diagnostic performs authenticated access checks only. Historical execution scripts
remain unchanged. No workflow was dispatched during this maintenance task.

These trigger/permission changes follow the official
[GitHub workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).

## Known historical regression failures

Two existing dataset-backed aggregate tests fail when explicitly enabled with
`PYTHONHASHSEED=0`: `test_production_candidate_regression` and
`test_full_diagnostic_preserves_current_baseline`. Query 3 reproduced the same
failures on the Query 2 baseline. The observed readiness values differ from the
old expectations; the tests and scientific implementation were not numerically
rewritten to hide this discrepancy. Both tests now also require
`--run-research-data`. This capability marker is not a claim that they pass.
See the [Query 3 validation report](repository_cleanup_q3_20261029/07_VALIDATION_RESULTS.md).
