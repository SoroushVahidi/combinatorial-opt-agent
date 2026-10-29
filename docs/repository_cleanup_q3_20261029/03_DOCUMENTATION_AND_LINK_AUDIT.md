# Documentation and link audit

The historical ESWA report now links to `../../../docs/archive/EXPERIMENT_AUDIT.md`;
the displayed path was corrected too. No numerical text in that report changed.

Root and manuscript indexes now point to the [permanent manuscript release](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0),
with descriptive PDF, supplementary PDF and exact source-ZIP links. The manuscript
is explicitly pre-submission, not accepted or published by JDSA. No DOI is invented.
The older checked-in JDSA package remains unchanged and is identified as historical.

The [public reproducibility index](../JDSA_REPRODUCIBILITY.md) uses the immutable
GitHub evidence tag and exact Hugging Face revision. Its claim table is derived
from authenticated claims/manifest paths, retains evidence limitations, and links
to all 47 investigation groups. Historical locations are distinguished from the
final manuscript crosswalk. No missing observations or runtime settings are inferred.

The README, contributing guide and documentation index replace stale EAAI/SNCS
current-authority framing. Historical status/reviewer/structure documents carry a
scope notice while retaining their original content. The previous README is linked
at Query 2's exact commit to preserve historical results without treating its
six-row table as six unconditional metrics.

## Validation boundaries

`python scripts/check_docs_integrity.py` checks repository-relative inline/image
links and reference-definition destinations across tracked Markdown. It ignores
fenced/inline code and does not inspect untracked or ignored private files. External
URLs and fragment anchors are not validated by this lightweight parser. The previous
checker only scanned selected canonical documents and also traversed local files.

All newly introduced public artifact URLs were verified via release metadata,
independent asset downloads, pinned Git objects and the exact HF manifest/tree.
Public release pages returned HTTP 200. No inaccessible authenticated URL was
classified as incorrect. Historical external websites were not exhaustively
re-audited; this query does not claim that every historical external link is live.

Workflow documentation follows [official GitHub syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).
Research workflows remain manual; confirmation inputs, read-only repository
permissions and disabled checkout credential persistence prevent automatic pushes.
Raw logs and processed data are no longer uploaded. Maintenance CI does not dispatch
research workflows, acquire data, run models or execute scientific solvers.
