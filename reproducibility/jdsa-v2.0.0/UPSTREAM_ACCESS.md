# Upstream access and original reproduction boundaries

The default offline verifier needs no upstream access. To inspect source-dependent correctness, obtain data directly under the applicable upstream terms and preserve its original revision and license. Do not assume author access transfers to downstream users.

| Source | Pinned source / access | Released here | Not released here |
|---|---|---|---|
| NLP4LP | https://huggingface.co/datasets/udell-lab/NLP4LP ; gated research access, CC BY-NC4.0 metadata; accepted gate terms must be checked by each user | Original study metric counts, IDs, grouping labels, numerical score/choice projections | Problem texts, candidate context tokens/values, source witnesses and full raw cache |
| NL4Opt | https://github.com/nl4opt/nl4opt-competition/tree/49f1e0d66b7fdcd33305a7f281c2a7c13f5620ea ; MIT notice | Problem/item IDs and original study judgment/validity labels | Raw corpus, submitted prompts, private/provider response prose |
| OptMATH | https://huggingface.co/datasets/shushulei/OptMATH-Train ; source license not confirmed | Aggregate research counts and common18 outcome metadata | Raw training/source text, generated program artifacts and weights |
| Hosted providers | Gemini via Vertex AI; DeepSeek via io.net; Cohere; Azure GPT-5.4 | Model identifiers where saved and numeric/class labels as research observations | Credentials, account details, raw response prose, unconfirmed account-specific redistribution rights |

Upstream access alone does not recover missing historical observations. Do not rerun current models and label their outputs as historical. Provider model names without snapshot/serving metadata are not complete environment specifications.

The original historical code is preserved by Git commit `d3071ba31eaf7babdf1e7fe234d8527191ed93de`. Source records in `provenance/investigation_sources.json` pin earlier tracked/untracked worktree versions; untracked file hashes authenticate bytes, not a nonexistent commit. `provenance/providers.json` distinguishes known settings and unknown fields. `requirements-validation.txt` locks only this reanalysis environment, not the historical inference stack.

Optional local source validators are preserved in `upstream_required/`. They assume the original repository/raw-evidence directory structure and cannot be made equivalent to the public projections by inventing missing files. Original checks passed locally before publication:5120 relationship checks,350 common18 checks/108 ledger rows/72 saved generated cases, and the ten core arms. These receipts do not make the excluded records public.
