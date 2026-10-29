# Phase 4 reference correction report

Checked 29 October 2026. All **48 records** in the editable Phase 3 bibliography were examined, including four uncited records. The corrected documents use **41 distinct numbered references**; three additional scientifically relevant preprints remain explicitly identified and linked in the text. Historical BibTeX entries are retained, so this is not deletion of prior work.

## Corrected publication records

1. **MIPLIB-NL.** [Official PMLR record](https://proceedings.mlr.press/v306/li26dz.html): Zhong Li, Hongliang Lu, Tao Wei, Yuxuan Chen, Wenyu Liu, Yuan Lan, Fan Zhang and Zaiwen Wen; *Constructing Industrial-Scale Optimization Modeling Benchmark*; Proceedings of the 43rd International Conference on Machine Learning, PMLR **306**, **68993–69036**, **2026**. Converted to `@inproceedings`. No proceedings DOI is supplied by that record, so the arXiv DOI is not presented as a conference DOI. Its benchmark-scale claim remains supported by the official abstract.
2. **AlphaOPT.** The [official KDD list](https://kdd2026.kdd.org/papers/) contains the paper and links directly to [ACM DOI 10.1145/3770855.3817974](https://doi.org/10.1145/3770855.3817974). [ACM-deposited metadata](https://api.crossref.org/works/10.1145/3770855.3817974) authenticates *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2*, **2389–2400**, **2026**, published 8 August. All 15 authors match the starting record. Converted to a proceedings entry; Table S1's year changes from 2025 preprint vintage to 2026 publication. Direct ACM HTML access was restricted; the official conference DOI and publisher deposit jointly establish publication, not merely acceptance. The experience-library description does not imply model-weight training.
3. **Models Can Model, But Can't Bind.** The [official COLM paper](https://proceedings.colm.cc/paper_files/paper/2026/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html) appears in volume 3, **2026**. The [volume](https://proceedings.colm.cc/paper_files/paper/2026) explicitly calls itself pre-proceedings pending DOI assignment, while its abstracts/PDFs are final. Converted to an official conference entry with that qualification, no invented pages/DOI; author name now includes Alexander M. Berenbeim. The final PDF supports oracle externalization, trained binding specialists and the unsuccessful untrained two-stage case (Appendix B.1). Related Work now says COLM paper rather than author preprint.
4. Added verified page ranges to four ACL 2026 entries and Chain-of-Experts, using their official metadata. No new experimental comparison was introduced.

## Unpublished work and the JDSA rule

The [current official instructions](https://link.springer.com/journal/41060/submission-guidelines) restrict the reference list to published or accepted work: “Personal communications and unpublished works should only be mentioned in the text.” Accordingly:

- **ReMatch:** [arXiv record](https://arxiv.org/abs/2403.01567), latest v2, supports retrieval-enhanced schema matching. No newer primary publication/acceptance was authenticated in the targeted publisher/proceedings search.
- **OptimAI:** [arXiv record](https://arxiv.org/abs/2504.16918), latest v3, supports the formulation/planner/code-revision description. The [OpenReview submission PDF](https://openreview.net/pdf?id=JtgZkVdAIP) is labelled under review; the forum/API access challenge prevents authenticating a decision. An under-review PDF is not acceptance. The checked ICLR proceedings index did not establish publication.
- **Execution-guided decoding:** the [Microsoft Research publication page](https://www.microsoft.com/en-us/research/publication/robust-text-to-sql-generation-with-execution-guided-decoding/) identifies the [original work](https://arxiv.org/abs/1807.03100) as arXiv, and supports partial-execution rejection of faulty SQL programs. No different later paper was substituted.

Each is now discussed by author/method name, explicitly marked unpublished preprint, and directly linked in the prose. OptimAI's supplementary table row remains, with a note referring to the main discussion. Their BibTeX records remain uncited in editable source and do not appear in either rendered reference list. Publication status remains unresolved beyond these authenticated records; absence from a search does not establish that no publication exists anywhere.

## Metadata conflicts and access limits

The original BM25 article PDF states **3(4):333–389 (2009)** and DOI 10.1561/1500000019. A current migrated publisher/deposit record instead reports volume 4 and different pagination. The [published PDF masthead](https://www.ccs.neu.edu/home/vip/teach/IRcourse/IR_surveys/robertson_foundations.pdf) supports retaining the existing citation; a metadata migration does not justify overwriting the primary document.

For SIRL, the official NeurIPS page gives 106027–106069 while the DOI deposit gives 117446–117488. The starting entry had no pages; none are fabricated or selected arbitrarily. Venue volume 38, conference year 2025 and DOI remain. The online publication metadata dates to 2026. NL4Opt similarly retains its official publisher citation year 2022 despite the 2023 PDF/publication date.

Some direct publisher requests were restricted, and a supplementary Crossref refresh encountered rate limiting. `REFERENCE_AUDIT_METADATA.json` records actual HTTP statuses/errors and the starting bibliography; it is not a claim that every full text was newly retrieved. For established records, retained Phase 3 verification was compared with accessible primary records. Corrections are separately recorded in `CORRECTED_PRIMARY_METADATA.json`; those records and the delivered bibliography supersede old fields in the input audit. No DOI, acceptance or publication was inferred from an unavailable endpoint.

## Full bibliography disposition

| Key | Primary record inspected | Phase 4 disposition |
|---|---|---|
| `affolter2019nli` | [Source](https://doi.org/10.1007/s00778-019-00567-8) | Retained; primary metadata/claim checked against existing verification |
| `ahmaditeshnizi2024optimus_icml` | [Source](https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html) | Retained; primary metadata/claim checked against existing verification |
| `ahmed2025opt2code` | [Source](https://www.sciencedirect.com/science/article/pii/S2949719125000615) | Retained; primary metadata/claim checked against existing verification |
| `almasian2023cqe` | [Source](https://aclanthology.org/2023.emnlp-main.793/) | Retained; primary metadata/claim checked against existing verification |
| `astorga2025autoformulation` | [Source](https://proceedings.mlr.press/v267/astorga25a.html) | Retained; primary metadata/claim checked against existing verification |
| `atzeni2026semantic` | [Source](https://doi.org/10.1016/j.datak.2025.102494) | Retained; primary metadata/claim checked against existing verification; uncited archival entry, not printed |
| `biemans2025explainableoptimization` | [Source](https://link.springer.com/chapter/10.1007/978-3-032-08327-2_3) | Retained; primary metadata/claim checked against existing verification; uncited archival entry, not printed; fresh machine metadata limited, no unsupported change |
| `chen2024m3embedding` | [Source](https://aclanthology.org/2024.findings-acl.137/) | Retained; primary metadata/claim checked against existing verification |
| `chen2025solver` | [Source](https://proceedings.neurips.cc/paper_files/paper/2025/hash/98555b92e5dedbdad5921049281995bc-Abstract-Conference.html) | Retained authenticated venue/year/DOI; inconsistent pagination omitted |
| `deerwester1990lsa` | [Source](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9) | Retained; primary metadata/claim checked against existing verification; fresh machine metadata limited, no unsupported change |
| `deng2021strug` | [Source](https://aclanthology.org/2021.naacl-main.105/) | Retained; primary metadata/claim checked against existing verification |
| `ding2026orr1` | [Source](https://ojs.aaai.org/index.php/AAAI/article/view/36983) | Retained; primary metadata/claim checked against existing verification |
| `gao2026models` | [Source](https://arxiv.org/abs/2605.21751v2) | Corrected to official COLM 2026 volume 3 pre-proceedings; Alexander M. Berenbeim; DOI pending |
| `holm1979followup` | [Source](https://www.jstor.org/stable/4615733) | Retained; primary metadata/claim checked against existing verification; fresh machine metadata limited, no unsupported change |
| `huang2017beamfollowup` | [Source](https://aclanthology.org/D17-1227/) | Retained; primary metadata/claim checked against existing verification |
| `huang2025orlm` | [Source](https://doi.org/10.1287/opre.2024.1233) | Retained; primary metadata/claim checked against existing verification |
| `jiang2025llmopt` | [Source](https://proceedings.iclr.cc/paper_files/paper/2025/file/fbe6dd68b0cf2b1d43b458d2b8ca31b0-Paper-Conference.pdf) | Retained; primary metadata/claim checked against existing verification; fresh machine metadata limited, no unsupported change |
| `kadioglu2024ner4opt` | [Source](https://doi.org/10.1007/s10601-024-09376-5) | Retained; primary metadata/claim checked against existing verification; fresh machine metadata limited, no unsupported change |
| `kong2025alphaopt` | [Source](https://arxiv.org/abs/2510.18428) | Corrected to ACM KDD 2026 V.2:2389–2400; all 15 authors verified; DOI 10.1145/3770855.3817974 |
| `kuhn1955assignment` | [Source](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109) | Retained; primary metadata/claim checked against existing verification; fresh machine metadata limited, no unsupported change |
| `lhoest2021datasets` | [Source](https://aclanthology.org/2021.emnlp-demo.21/) | Retained; primary metadata/claim checked against existing verification |
| `li2026solver` | [Source](https://aclanthology.org/2026.findings-acl.102/) | Added official ACL pages 2138–2153 |
| `lu2025optmath` | [Source](https://proceedings.mlr.press/v267/lu25o.html) | Retained; primary metadata/claim checked against existing verification |
| `miplibnl2026` | [Source](https://arxiv.org/abs/2602.10450) | Corrected to ICML 2026 / PMLR 306:68993–69036; all eight authors verified |
| `nlp4lp_dataset` | [Source](https://huggingface.co/datasets/udell-lab/NLP4LP) | Retained; primary metadata/claim checked against existing verification; fresh machine metadata limited, no unsupported change |
| `pan2025pamop` | [Source](https://www.ijcai.org/proceedings/2025/296) | Retained; primary metadata/claim checked against existing verification |
| `pedregosa2011sklearn` | [Source](https://jmlr.org/papers/v12/pedregosa11a.html) | Retained; primary metadata/claim checked against existing verification |
| `powell2025scheduling_explanations` | [Source](https://doi.org/10.1007/s10844-025-00940-w) | Retained; primary metadata/claim checked against existing verification; uncited archival entry, not printed |
| `punyakanok2004ilp` | [Source](https://aclanthology.org/C04-1197/) | Retained; primary metadata/claim checked against existing verification |
| `ramamonjison2022autoformulation` | [Source](https://aclanthology.org/2022.emnlp-industry.4/) | Retained; primary metadata/claim checked against existing verification |
| `ramamonjison2023nl4opt` | [Source](https://proceedings.mlr.press/v220/ramamonjison23a.html) | Retained; primary metadata/claim checked against existing verification |
| `rizzi2025conceptual` | [Source](https://doi.org/10.1016/j.datak.2025.102452) | Retained; primary metadata/claim checked against existing verification; uncited archival entry, not printed; fresh machine metadata limited, no unsupported change |
| `robertson2009bm25` | [Source](https://doi.org/10.1561/1500000019) | Retained original published PDF pagination; conflicting migrated/deposited metadata disclosed |
| `roy2017udg` | [Source](https://ojs.aaai.org/index.php/AAAI/article/view/10959) | Retained; primary metadata/claim checked against existing verification |
| `scholak2021picard` | [Source](https://aclanthology.org/2021.emnlp-main.779/) | Retained; primary metadata/claim checked against existing verification |
| `sheetrit2024rematch` | [Source](https://arxiv.org/abs/2403.01567) | Preprint status only authenticated; retained discussion/link in text, not printed bibliography |
| `thind2025optimai` | [Source](https://arxiv.org/abs/2504.16918) | Preprint and under-review submission found; no authenticated acceptance; retained in text only |
| `wang2018execution` | [Source](https://arxiv.org/abs/1807.03100) | Microsoft Research identifies arXiv; retained in text only |
| `wang2020ratsql` | [Source](https://aclanthology.org/2020.acl-main.677/) | Retained; primary metadata/claim checked against existing verification |
| `xiao2024chainofexperts` | [Source](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d45ee77826332c100a1e15f7765b99ff-Abstract-Conference.html) | Added official ICLR pages 48519–48537 |
| `xiao2026deepor` | [Source](https://ojs.aaai.org/index.php/AAAI/article/view/40699) | Retained; primary metadata/claim checked against existing verification |
| `xie2026escaping` | [Source](https://aclanthology.org/2026.findings-acl.690/) | Added official ACL pages 14100–14116 |
| `yang2026orthought` | [Source](https://doi.org/10.1016/j.ait.2026.100059) | Retained; primary metadata/claim checked against existing verification; fresh machine metadata limited, no unsupported change |
| `yu2018spider` | [Source](https://aclanthology.org/D18-1425/) | Retained; primary metadata/claim checked against existing verification |
| `yu2026uncertainty` | [Source](https://aclanthology.org/2026.acl-long.1975/) | Added official ACL pages 42658–42669 |
| `zhai2025equivamap` | [Source](https://proceedings.mlr.press/v267/zhai25a.html) | Retained; primary metadata/claim checked against existing verification |
| `zhang2024optllm` | [Source](https://aclanthology.org/2024.naacl-industry.42/) | Retained; primary metadata/claim checked against existing verification |
| `zhang2026dual` | [Source](https://aclanthology.org/2026.acl-long.266/) | Added official ACL pages 5895–5908 |

## Claim support and citation form

The closest-work distinctions remain those documented in the [Phase 3 reference audit](../jdsa_revision_phase3_20261029/REFERENCE_VERIFICATION.md): generated formulations versus supplied slots; execution versus binding/semantic correctness; trained versus untrained interventions. Changed publication records support the same scientific claims, with Gao checked against the final conference PDF. No published accuracy is imported as a local baseline. No claim that an unexecuted method was experimentally defeated is introduced. Every citation command contains at most four references; numbering is regenerated by BibTeX. References and cross-document citations resolve in clean builds.
