# Phase 4 artifact consistency — read-only audit

The public repository and three referenced dataset metadata records were inspected on 29 October 2026. No dataset/model files were uploaded, deleted or changed. No upstream corpus was downloaded. GitHub was public, with main as its default branch; the complete remote tree of Phase 3 commit `49827d274cae995005681fb03a299a5363583861` was checked before preparing Phase 4.

| Statement | Verified public state / disposition |
|---|---|
| Historical public v1.2 | The [Hugging Face card](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics) labels itself v1.2, text-free derived metrics. API revision remains `24a047fa5d661be6cfcf75980ac8f62e6c942243`, modified 24 August 2026, CC BY-NC-SA 4.0. Existing statement correct. |
| NLP4LP access | Public metadata identifies gated access and CC BY-NC 4.0 at revision `f0f2abb5d4fa01812ab2c818dd350f6c34041d15`. No restricted records redistributed. |
| OptMATH-Train | Public card metadata still reports unknown license at `2c1fbcb448da6c8b3fb267f73e20c83a7c5bf63f`. Existing unresolved-license statement retained; no new release. |
| Local M5 candidate | Remote Phase 3 tree lacks `reproducibility/m5/`, `verify_llm217_m5.py` and the corrected per-judgment release ledger. The paper correctly says not uploaded. Local archived evidence used for checks is not asserted public. |
| Core and extraction records | Remote tree contains the historical final-method metrics and external extraction records. These support their stated aggregation checks, not recovery of missing original scalar mappings. |
| Common-18 corrections | Public historical logs/fields are distinct from the local corrected M3 ledger. `COMMON18_CORRECTED_CASES.csv` was absent from the checked public tree; the paper does not promise that local bundle is public. |
| Current review package | Updated the main Code availability paragraph from Phase 3 to Phase 4. The supplementary Phase 2 provenance sentence is an accurate historical description, not a claim that M5 was released. |

`PUBLIC_AVAILABILITY_CHECK.json` contains read-only metadata and public file listings; `GITHUB_AVAILABILITY_CHECK.json` records the scoped complete-tree checks. Historical HF fields can predate later corrections and must not override the authenticated correction summaries. Document publication is not release of M5. The source ZIP reproduces the article and supplement, not missing predictions, precision, prompts or environments.

No objectively inaccurate scientific availability assertion was found beyond the review-package phase designation. Missing local-only records remain a reproducibility/access limitation. The final delivery verifies the Phase 4 remote files and public PDF/ZIP checksums; it does not alter the historical Hugging Face release.
