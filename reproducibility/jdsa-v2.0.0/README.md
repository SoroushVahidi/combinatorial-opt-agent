# JDSA evidence v2.0.0 — frozen-record reanalysis

This release supports *Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study*. It publishes a curated, permission-scoped subset of the local M5 evidence, with the final D1 author decisions and final JDSA claim map. It preserves positive, negative, incomplete and summary-only findings.

**This is saved-record reanalysis, not full experimental reproduction.** No new model inference, solver experiment or historical observation was generated. The original M5 candidate/private archive remains local. The historical Hugging Face v1.2 data remain distinct and unchanged at revision `24a047fa5d661be6cfcf75980ac8f62e6c942243`.

## Verify a standalone download

Extract the versioned GitHub ZIP or download only this versioned Hugging Face release directory. With Python3.12:

```sh
python3 -m venv /tmp/jdsa-reanalysis-env
/tmp/jdsa-reanalysis-env/bin/python -m pip install -r requirements-validation.txt
/tmp/jdsa-reanalysis-env/bin/python -I verify.py --output /tmp/jdsa-verification.json
```

The verifier uses only this directory, Python's standard library and pinned NumPy2.3.5. It performs no network, API, model or solver calls. The JSON output contains PASS/FAIL, individual checks, reaggregated results, bootstrap settings and limits. Exit0 means the supported checks passed; exit1 means failure. Keep outputs outside the immutable directory. Extra files, modified inputs and missing artifacts fail integrity verification. Do not run Python in a way that creates bytecode within the release.

The standalone package needs no credentials or upstream dataset downloads. The optional `upstream_required/` original evaluators require a separately authorized complete source layout and raw local records; they are provided as provenance, not as runnable standalone entry points. See `UPSTREAM_ACCESS.md`.

## What can be checked

- 97 frozen metric inputs: ten core arms, twelve retrieval cells,72 historical campaign inputs and four optional LLM configurations (some overlap between these families).
- 16 paired primary contrasts:327 frozen groups, B10000, NumPy seed42, exact group-swap p values and Holm adjustment.
- 24 factorial cells on49 queries/277 slots,61 saved primary/development checkpoints, historical B2000 Python-random bootstrap intervals and240 one-factor exact tests.
- 217 relationship targets across three services and three preserved recovery stages: validity/label ledgers, request denominators, pairwise agreement, kappa and16 source clusters.
- Common18 numerical/classification ledgers; PaMOP precision remains unresolved.
- Candidate ceilings, six pilot arms,49-query distinct retention, reference-only count sensitivity, G3 inventory counts and24 D1 fixture records with12 author decisions.

The machine manifest `claims.json` covers9 main tables,1 main figure,1 algorithm,26 supplementary tables,183 numeral-containing source passages and47 registered investigations. `AUDIT.md` describes exactly what each supports. Historical aggregate-only results are preserved without claiming they were independently recomputed from missing observations.

## Contents and integrity

`release_manifest.json` lists every payload path, size and SHA256. `CHECKSUMS.sha256` also authenticates the manifest. `provenance/build_sources.json` links retained/projection files to original hashes, versions and transformations. `provenance/m5_inventory.json` preserves all2954 original inventory entries; omission from this release does not authorize deleting original evidence. `claims.json` is a source-to-result map, not a certificate of semantic correctness.

Software is MIT; author-owned released evidence is CC BY-NC-SA4.0, consistent with the existing dataset. Third-party rights are not relicensed. See `LICENSE_DATA.md` and `PUBLICATION_REVIEW.md`. Cite the manuscript and versioned evidence release; do not cite this subset as an unrestricted redistribution of NLP4LP, NL4Opt or OptMATH.
