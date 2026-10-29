# Natural-language combinatorial optimization research

Research software for fixed-catalog schema retrieval and assisted scalar grounding
of natural-language optimization problems. The implementation separates retrieval,
parameter extraction and assignment, evaluation, and experimental baselines. This
is a research framework; historical experiment scripts are retained for provenance.

## Manuscript and public evidence

**Retrieval-Assisted Scalar Grounding for Natural-Language Optimization**, Soroush
Vahidi (2026), is available as a **public pre-submission manuscript** targeting the
International Journal of Data Science and Analytics. It is not an accepted or
journal-published article. No DOI has been assigned here.

- [Versioned manuscript, supplement and exact editable sources](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0)
- [Immutable reproducibility evidence v2.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0)
- [Exact Hugging Face evidence revision](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0)
- [Manuscript index, including historical journal packages](manuscript/MANUSCRIPT_README.md)
- [Claim-to-artifact and reanalysis guide](docs/JDSA_REPRODUCIBILITY.md)

The published evidence supports saved-record and statistical reanalysis. It does
not reproduce every original experiment or establish independent semantic ground
truth. Retrieval accuracy differs from grounding accuracy; structural readiness
is not complete scalar agreement, and numerical agreement is not semantic fidelity.
Whole-procedure comparisons and controlled decoder comparisons have different
interpretations. We do not claim autonomous formulation, unseen-schema
generalization, or model-ranking superiority.

## Repository map

| Location | Purpose |
|---|---|
| [retrieval/](retrieval/) | Catalog retrieval, lexical/dense baselines and search |
| [tools/](tools/) | Grounding, diagnostics and saved-record analysis utilities |
| [src/](src/) | Dataset adapters and implementation modules |
| [baselines/](baselines/) | External-method adapters and their tests |
| [training/](training/), [batch/](batch/) | Research and cluster execution infrastructure; explicit runs only |
| [scripts/](scripts/) | Data staging, validation and maintenance utilities |
| [tests/](tests/), [configs/](configs/) | Tests and experiment configuration |
| [data/](data/) | Catalogs, derived records and local data locations; access restrictions apply |
| [results/](results/README.md) | Preserved historical outputs and provenance navigation |
| [manuscript/](manuscript/MANUSCRIPT_README.md) | Previous manuscript packages; final JDSA version is the release above |
| [docs/](docs/README.md) | Current navigation, historical audits and research documentation |

Scientific directories and historical records have not been relocated. EAAI,
ESWA, KAIS, DKE and SNCS materials describe earlier work, not newly maintained
systems. The [previous repository overview](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/2ed093db6c31b905d533693c8f6fe8ef70fe3833/README.md)
remains available with its historical results. Its six-row SNCS metric table
must not be interpreted as six unconditional proportions: Exact20 has conditional
denominators; consult the current evidence definitions.

## Installation and minimal offline validation

Use Python 3.10 or later; maintenance CI uses Python 3.12. Create an isolated
environment. The small maintenance suite requires only pytest and its development
helpers; it does not require models, datasets, API keys or solvers.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/check_repository_syntax.py
python scripts/check_docs_integrity.py
python -m pytest -q tests/test_cardinal_downloader.py tests/test_documentation_integrity.py tests/test_offline_policy.py tests/test_repo_hygiene.py tests/test_consistency_benchmark.py
```

Dependency installation itself requires a package index or a prepared local wheel
cache. The validation commands run offline. The broader `requirements.txt` is a
research convenience list, not an authenticated historical environment lockfile.
Install only the optional capabilities needed for your work. See
[testing and dependency guidance](docs/TESTING.md) for the broader suite and
explicit network, model, provider, solver and dataset opt-ins.

## Availability, provenance and licensing

[MIT](LICENSE) covers repository-owned code subject to its terms. It does not
relicense third-party datasets, source-derived records, external code or model
outputs. Some already-tracked NLP4LP-derived material still requires a
redistribution review; public Git history is not proof of permission. Gated
NLP4LP access and licensing-uncertain OptMATH material must not be republished
through a new raw-data release. See [access and licensing](docs/DATA_ACCESS_AND_LICENSING.md).

Original prompts, provider responses, local M5 archives, validity ledgers and
unreleased research are not interchangeable with the sanitized public package.
Missing historical observations, runtime settings and source-level checks remain
limitations. Do not regenerate frozen results as routine maintenance.

## Citation and contributions

Cite: Soroush Vahidi (2026). *Retrieval-Assisted Scalar Grounding for
Natural-Language Optimization*. Public pre-submission manuscript, version 1.0.0.
[Permanent manuscript release](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0). Include the evidence version when
using its saved records. Do not cite this as a JDSA publication.

See [CONTRIBUTING.md](CONTRIBUTING.md) for preservation and review expectations.
Contact: [sv96@njit.edu](mailto:sv96@njit.edu).
