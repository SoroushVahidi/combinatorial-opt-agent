# Redundancy-removal report

The combined narrative proxy falls from **15,282 to 12,391 words: an 18.92% reduction**. Main prose falls from 7,621 to 4,713 (38.16%); supplementary prose changes from 7,661 to 7,678 (+0.22%). Measuring both documents avoids treating relocation alone as deletion. There was no page-count target.

`measure_narrative.py` recursively expands each document's inputs and excludes tables/figures/algorithms and captions, displayed/inline mathematics, frontmatter except the Abstract, declarations, bibliography, comments and non-prose TeX commands. It counts alphabetic words including hyphenated words. This is a reproducible narrative-length proxy, not a journal word count or an exact measure of semantically duplicate sentences. [NARRATIVE_COUNTS.json](NARRATIVE_COUNTS.json) contains the before/after result.

| Repetition audited | Consolidated explanation | Remaining shorter uses |
|---|---|---|
| Reference-derived parameter assistance | Main §3.2 | Methods/limitations refer to the setup; S4.4 specifies differing input boundaries |
| Benchmark exposure and known-catalog scope | Main §3.2, §6.3 | RQ1 and Conclusion retain only the scope needed to interpret retrieval |
| Readiness versus correctness | Definitions §3.4; RQ2 §5.2 | Abstract/contributions/conclusion state the contribution without repeating metric derivations |
| Historical reconstruction and availability | S4.3–S4.5; main §6.3–6.4 and declarations | Result-specific absent records retained at the relevant result |
| Semantic agreement versus accuracy | Main §6.2; Supplement S7/S10.2 | Detailed counts/protocols remain once in the supplement |
| Limits of formulation comparisons | Supplement S5 | Main §6.2 gives a concise scope statement |
| Wide system landscape | Supplement S1/S11 | Main related work centers structured grounding, Text2Opt-Bench and endpoint distinctions |
| Statistical procedure descriptions | Main §4.4; supplemental procedure/history references | The original distinct families and historical settings are retained rather than collapsed |

No weak or negative outcome was removed to improve appearance. All original table numerical cells, 37 numbered objects and 47 investigation destinations are retained. Legitimate local qualifications remain where omitting them would make a result misleading. The public-versus-local availability edits change descriptions of access, not the measurements.

Recompute without modifying source:

```bash
python docs/jdsa_stage2_20261029/measure_narrative.py
```
