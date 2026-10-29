# Reproducibility preservation

MANUSCRIPT_RELEASE_PRESERVED: YES.
EVIDENCE_RELEASE_PRESERVED: YES.
HUGGING_FACE_PRESERVED: YES.
SCIENTIFIC_DATA_PRESERVED: YES.
UNPUBLISHED_WORK_PRESERVED: YES (within the explicitly recorded observation scope).

Public release metadata, tag targets and all ten assets were independently fetched
before and after implementation. IDs, sizes, SHA256 digests, release descriptions
and pinned commits are identical. Both release pages and the exact Hugging Face
tree return HTTP 200. The HF manifest is byte-identical to the GitHub evidence
archive manifest. See [release receipt](RELEASE_PRESERVATION.json).

- Manuscript: `jdsa-manuscript-v1.0.0` → `00adeca9d3d761da7f8ee2e5dcee7fc8d535f2bc`.
- Report history: `0db80afe6122b1b20fd28d4f3a440001e645cdf1` remains retained.
- Evidence: `jdsa-evidence-v2.0.0` → `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`.
- HF: `f94012872211ab6a828147bb138fbe8ab695b982`, release `jdsa-evidence-v2.0.0`.

All 172 evidence archive entries authenticated: 170 payloads plus the manifest and
checksum file. The frozen verifier and its inputs are unchanged. The 9,737-check
numerical reanalysis was not rerun in this repository-maintenance query; integrity
verification must not be described as a new scientific-verifier execution.

Compared with Query 2, 670 existing scientific-directory files are byte-identical.
Only the manuscript navigation Markdown and one historical results-report link
changed within those directories. No CSV/JSON scientific records, manuscript TeX,
PDFs, style files, figures or source archives changed. Four tracked source ZIPs and
the public final manuscript source ZIP pass CRC checks and retain their hashes.
See [integrity receipt](INTEGRITY_RESULTS.json).

The source-derived NLP4LP/PaMOP/catalog material listed in Query 2 still requires
redistribution review. Existing source field metadata supports the concern; public
availability does not supply missing permission. No extra restricted content is
republished, and no unresolved file is deleted. The Cardinal code issue is repaired;
the other 18 investigation files remain unchanged. Local M2–M5 originals, validity
ledgers and ignored scientific records are preserved. Third-party rights and
historical evidence limitations remain explicit in the new public navigation.
