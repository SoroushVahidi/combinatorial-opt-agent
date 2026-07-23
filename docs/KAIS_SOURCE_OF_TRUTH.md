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
cleanup, a naming-bug fix, etc.).

**2026-07-23 (final preparation pass):** a second, deeper pass converted the manuscript to
the official Springer Nature LaTeX template (`sn-jnl.cls`, downloaded directly from
Springer Nature's own resource host), confirmed KAIS's peer-review policy is
single-anonymous (so author identity stays in the manuscript), fixed a real
`InstantiationReady` metric-definition/implementation mismatch, added three verified
recent references, standardized table formatting, regenerated two figures with a
duplicated/mismatched baked-in title, and added a dedicated Limitations subsection. Full
details in [`manuscript/KAIS_FINAL_PREPARATION_REPORT.md`](../manuscript/KAIS_FINAL_PREPARATION_REPORT.md).
The manuscript now compiles cleanly to a 36-page PDF (`manuscript/main.pdf`).

## Benchmark facts (unchanged from the EAAI-era documents; still authoritative)

Everything in [`docs/EAAI_SOURCE_OF_TRUTH.md`](EAAI_SOURCE_OF_TRUTH.md) under "Benchmark
and Evaluation Story," "Key Metrics," and the authoritative-file table remains accurate
and should still be used as the canonical source for repository-side benchmark numbers.
Only that document's venue-specific framing (target journal name, double-anonymization
requirement assumed for EAAI) is superseded — see `manuscript/MANUSCRIPT_README.md` for
the KAIS-specific status of those items (in particular, KAIS's peer-review blinding
policy could not be confirmed from public Springer Nature pages and needs manual
verification before submission).
