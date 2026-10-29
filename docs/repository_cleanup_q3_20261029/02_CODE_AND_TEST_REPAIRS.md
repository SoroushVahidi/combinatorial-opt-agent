# Code and test repairs

## Cardinal raw-snapshot downloader

The baseline combined two incompatible implementations: an unmatched brace at the
end of the first `main`, a duplicate tail using undefined `SPLITS`/`_fetch`/`_parse_jsonl`,
incompatible provenance shapes, and conflicting success semantics. Deleting one
brace would not repair it. The full script and Cardinal adapter were inspected.

The coherent implementation retains the first implementation's raw JSONL staging
contract: train/dev/test files under ignored `data/external/cardinal_nl4opt/`,
standard-library HTTP, explicit CLI invocation, nonzero status for missing/invalid
splits. Upstream fields remain raw; no inferred labels or schema claims are added.
`--output-dir` supports isolated destinations. Existing valid files are reused;
existing provenance is preserved unless explicit `--force` is supplied. Rows must
be nonempty objects; blank lines are ignored for counts. Validation precedes atomic
replacement, and failed downloads leave original split bytes intact. Current
reports include per-split source URLs, counts, status, SHA256 and errors. Force may
refresh provenance even on partial failure; inspect the errors and exit status.

Fourteen synthetic fixture tests cover raw-byte preservation, complete success,
empty/malformed/non-object/non-UTF8 payloads, HTTP/URL/timeout failures, no-network
reuse, invalid-existing-file refusal and provenance preservation. No dataset was
downloaded and no existing dataset overwritten. Live upstream paths/schema and
end-to-end adapter coverage remain unverified; this is not a new dataset validation.

## Behavior-preserving warning repair

`tools/run_eaai_final_solver_attempt.py` uses a raw string for the existing
backslash-containing forbidden token. The entire Python AST is identical before
and after. Replacing that token with a different pattern would change behavior;
this repair deliberately does not do so. No solver was executed.

## Offline tests and syntax checks

Root pytest policy replaces unconditional Hugging Face/GitHub reachability probes.
Network, model, provider, solver, research-data and integration capabilities are
opt-in. A subprocess fixture proves collection-time socket blocking and marker
skipping. Catalog-enrichment failure testing now uses a deterministic URL error.
Model fetch tests and live dataset tests declare their capabilities explicitly.
Two historical aggregate regressions additionally declare research-data dependence;
their existing failures remain documented, not converted into passes.

A tracked-file syntax checker parses Python with warnings treated as errors and
checks bash scripts, including SLURM scripts, without executing them. Documentation
checker fixtures cover tracked targets, encoded paths, images, reference definitions,
external URLs, code examples, private untracked files and conflict markers.

See [validation results](07_VALIDATION_RESULTS.md) for the baseline-confirmed failures.
