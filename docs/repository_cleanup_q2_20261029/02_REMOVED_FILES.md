# Removed files

Baseline: `d3071ba31eaf7babdf1e7fe234d8527191ed93de` (verified current `origin/main`). Cleanup branch: `cleanup/repository-q2-safe-files-20261029`.


All six exact paths were tracked, present and unchanged since the audit. All are recoverable from Git at the baseline commit using `git show <baseline>:<exact-path>`; no history rewrite or Git garbage collection was performed.

| Removed path | Bytes | Reason and retained content |
|---|---:|---|
| `data/processed/.gitkeep` | 0 | Empty placeholder; parent retains nonempty tracked content. |
| `notebooks/.gitkeep` | 0 | Empty placeholder; parent retains nonempty tracked content. |
| `outputs/consistency_benchmark/.gitkeep` | 0 | Empty placeholder; parent retains nonempty tracked content. |
| `validation/.gitkeep` | 0 | Empty placeholder; parent retains nonempty tracked content. |
| `manuscript/sncs/main.pdf` | 351282 | Byte-identical to the retained descriptive SNCS PDF; no unique result or provenance. |
| `manuscript/sncs/submission_package/main.pdf` | 351282 | Byte-identical to the retained descriptive SNCS PDF; no unique result or provenance. |

Retained PDF: [Retrieval-Assisted-Instantiation_SN-Computer-Science_Manuscript.pdf](../../manuscript/sncs/Retrieval-Assisted-Instantiation_SN-Computer-Science_Manuscript.pdf). Its 351,282 bytes have SHA-256 `98d12f1c39d248fc90e5c02736196678313168e8d06072770513e4a989035ff1`, identical to each removed PDF. All four placeholders have the empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Reference and dependency checks

Tracked-text searches found no references to any of the four placeholder paths. Full SNCS PDF paths occur in submission metadata and the historical Stage 4 audit; the upload manifest also names the relative `main.pdf`. All three now contain a dated working-link relocation notice. The upload-order instruction forbidding `main.pdf` inside the source ZIP remains valid and unchanged. Historical prose below the notices is deliberately retained; it is not a current missing-file instruction.

No test, script or pipeline reference to either removed SNCS path was found. These are compiled deliverables, not TeX build inputs. The SNCS source ZIP and all original source/figure files are unchanged. The different KAIS `manuscript/submission_package/main.pdf` is not a deletion candidate and remains. No release tag/tree or uploaded asset is modified by this branch-local deletion.

Reference commands: `git grep -n -I -E` for the six exact paths across tracked text; `rg -n "main\.pdf" manuscript/sncs` restricted to Markdown/scripts; pipeline/test searches for SNCS and placeholder names. Source/ZIP/PDF hash checks and tests are recorded in the validation report.
