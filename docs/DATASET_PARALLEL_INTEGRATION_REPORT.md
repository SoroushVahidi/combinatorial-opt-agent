# Dataset Parallel Integration Report

Date: 2026-04-02 (UTC)

This report captures additive dataset-layer work for the parallel branch focused on normalization + expanded schema catalog readiness.

## Execution evidence (retrieval attempts)

The following commands were run in this environment:
- `python scripts/get_mamo.py`
- `python scripts/get_structuredor.py`
- `python scripts/get_cardinal_nl4opt.py`
- `python scripts/get_industryor.py`

Observed network blocker pattern: `CONNECT tunnel failed, response 403` for most GitHub/raw requests.

## Target dataset status

| Dataset | Retrieval | Normalization | Adapter registration | Benchmark runner compatibility | Expanded schema catalog inclusion | Exact blocker |
|---|---|---|---|---|---|---|
| MAMO | ❌ Failed in this env | ✅ Adapter implemented | ✅ Registered as `mamo` | ✅ Capability-safe | ✅ Included (source-only fallback when local data absent) | Raw URLs returned `URLError Tunnel connection failed: 403 Forbidden`; clone fallback failed with `fatal: unable to access ... CONNECT tunnel failed, response 403`. |
| StructuredOR | ❌ Failed in this env | ✅ Adapter implemented | ✅ Registered as `structuredor` | ✅ Capability-safe | ✅ Included (source-only fallback when local data absent) | `git clone` failed: `fatal: unable to access 'https://github.com/CardinalOperations/StructuredOR/': CONNECT tunnel failed, response 403`. |
| CardinalOperations/NL4OPT | ⚠️ Partial (test path only) | ✅ Adapter implemented | ✅ Registered as `cardinal_nl4opt` | ✅ Capability-safe | ✅ Included | `train` and `dev` split fetches failed with `URLError Tunnel connection failed: 403 Forbidden`; only `test` URL returned content (not treated as full success). |
| IndustryOR | ❌ Failed in this env | ✅ Adapter implemented | ✅ Registered as `industryor` | ✅ Capability-safe | ✅ Included (source-only fallback when local data absent) | `git clone` failed: `fatal: unable to access 'https://github.com/CardinalOperations/IndustryOR/': CONNECT tunnel failed, response 403`. |

## Additional readiness (future normalization)

These remained prepared/aligned in the matrix and catalog pipeline:
- OptiMUS (data-dependent normalized adapter)
- Gurobi Modeling Examples (catalog-only)
- Gurobi OptiMods (catalog-only)
- GAMS Model Library (catalog/source-only)
- MIPLIB 2017 (catalog/source-only)
- OR-Library (catalog-only)
- Pyomo Examples (catalog-only)

## Transparency notes
- No raw large dataset payloads were committed.
- All fetch scripts emit `provenance.json` with warnings/errors under ignored `data/external/<dataset>/`.
- Failures are preserved as explicit blocker outputs and non-zero exits.
