# KAIS Source of Truth

**Created:** 2026-07-23
**Status:** Authoritative for the current submission target — supersedes
[`docs/EAAI_SOURCE_OF_TRUTH.md`](EAAI_SOURCE_OF_TRUTH.md) and
[`docs/CURRENT_PAPER_SOURCE_OF_TRUTH.md`](archive/CURRENT_PAPER_SOURCE_OF_TRUTH.md) for
**venue and manuscript-file** purposes only. The underlying experimental facts documented
in those files (benchmark numbers, subset sizes, blockers) are unchanged and remain valid;
only the target journal, manuscript source location, and front-matter/formatting details
changed.

## What changed

The manuscript is now targeted at **Knowledge and Information Systems (KAIS)**, Springer
Nature, not *Engineering Applications of Artificial Intelligence* (EAAI). The actual
LaTeX manuscript source (previously absent from this repository — only companion code,
results, and documentation existed here) was added at the repository root as
`Retrieval_Assisted_Instantiation_of_Natural_Language_Optimization_Problems.zip` on
2026-07-23, then unpacked, audited, and adapted for KAIS in
[`manuscript/`](../manuscript/). **`manuscript/main.tex` is now the authoritative
manuscript source**; the zip is kept as-is for provenance.

See [`manuscript/MANUSCRIPT_README.md`](../manuscript/MANUSCRIPT_README.md) for the full
list of changes made during the KAIS adaptation pass (bibliography style, Declarations
section, new tables/figures added from already-committed camera-ready artifacts, bib
cleanup, a naming-bug fix, etc.) and for open items that still require manual action
(peer-review blinding policy confirmation, official Springer template conversion, PDF
compile verification).

## Benchmark facts (unchanged from the EAAI-era documents; still authoritative)

Everything in [`docs/EAAI_SOURCE_OF_TRUTH.md`](EAAI_SOURCE_OF_TRUTH.md) under "Benchmark
and Evaluation Story," "Key Metrics," and the authoritative-file table remains accurate
and should still be used as the canonical source for repository-side benchmark numbers.
Only that document's venue-specific framing (target journal name, double-anonymization
requirement assumed for EAAI) is superseded — see `manuscript/MANUSCRIPT_README.md` for
the KAIS-specific status of those items (in particular, KAIS's peer-review blinding
policy could not be confirmed from public Springer Nature pages and needs manual
verification before submission).
