# Data access, provenance and redistribution

The repository's [MIT license](../LICENSE) does not automatically cover third-party
datasets, external code, model outputs or source-derived records. Each upstream
source retains its own access and licensing conditions. Access is not a grant of
redistribution rights. This is a provenance inventory, not legal clearance.

Use the sanitized, immutable [public evidence package](JDSA_REPRODUCIBILITY.md).
Do not republish additional gated NLP4LP data, licensing-uncertain OptMATH records,
provider conversations, credentials, raw responses or excluded M5 archives.

## Unresolved tracked material

The [Query 2 preservation inventory](repository_cleanup_q2_20261029/03_PRESERVED_FILES.md)
identifies existing source-derived material requiring review. This includes the
331-record `results/eswa_revision/00_env/nlp4lp_gold_cache.json`, NLP4LP evaluation
variants/catalogs, the PaMOP subset, combined problem catalogs and collected examples.
The cache contains reference parameters, problem information, source programs and
solutions. Therefore a blanket statement that no source-derived NLP4LP records
exist in Git would be inaccurate. Historical availability alone does not resolve
redistribution rights. These files are preserved byte-for-byte in Query 3, with no
new raw-data publication or archive expansion. Removal requires separate approval,
a verified preservation plan and review of source terms and retained copies.

Empty evaluation placeholders, old cleanup scripts and demo helpers also remain
preserved pending investigation. The Cardinal downloader was repaired with synthetic
offline fixtures; this neither acquires Cardinal data nor clears its redistribution.

## Private and historical evidence

Untracked and ignored material in existing worktrees remains untouched. Original
model predictions, prompts, author annotations, validity ledgers, environment records
and unsuccessful experiments must be retained even if absent from the final paper.
Public summaries cannot replace these originals. Download or rerun instructions
in historical documentation do not authorize a new acquisition or publication.

Manual research workflows no longer automatically upload raw datasets or push
results. The public release tags and exact Hugging Face revision are unchanged.
