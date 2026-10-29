# Reference-verification report

All 48 BibTeX entries and every in-text citation were inspected on 29 October 2026. There are 41 distinct cited keys across the two PDFs (25 main references and 23 supplementary references, with overlap). Seven uncited historical entries remain in the editable bibliography but are not printed. No cited key was added or removed.

Primary publisher/proceedings pages, publisher-deposited Crossref metadata, official PDFs, author-hosted manuscripts and the dataset repository were used. Response receipts/metadata are in [REFERENCE_PRIMARY_METADATA.json](REFERENCE_PRIMARY_METADATA.json), targeted PDF checks in [REFERENCE_PDF_CHECKS.json](REFERENCE_PDF_CHECKS.json), and the full link audit in [URL_CHECKS.json](URL_CHECKS.json). HTTP 200 alone is not a bibliographic verification.

## Corrections

- Ramamonjison 2022: use the published byline “Rindra Ramamonjison” and “Timothy Yu”; remove unsupported extra initials.
- Ner4Opt: restore Kadıoğlu diacritics from publisher metadata.
- StruG: use ACL BibTeX family/given-name fields for Ahmed Hassan Awadallah; retain every name component.
- StruG, CQE and the 2004 semantic-role paper: expand abbreviated proceedings titles to verified titles.
- SIRL: add official NeurIPS page range 106027–106069; retain the conference year 2025 despite a later web-publication date.
- Remove redundant publisher URL fields when a full DOI link already prints. No DOI, acceptance claim, date or missing page range was invented.

## Entry-by-entry audit

| BibTeX key | Review status / source | Finding |
|---|---|---|
| `robertson2009bm25` | [VERIFIED WITH METADATA CONFLICT — publisher PDF](https://www.nowpublishers.com/article/DownloadEBook/INR-019) | Publisher-indexed PDF masthead confirms 3(4), 333–389 (2009). Crossref returns conflicting 4(1–2), 1–174; retain the paper masthead, not registry error. |
| `deerwester1990lsa` | [VERIFIED — primary/deposited metadata](https://api.crossref.org/works/10.1002%2F%28SICI%291097-4571%28199009%2941%3A6%3C391%3A%3AAID-ASI1%3E3.0.CO%3B2-9) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `pedregosa2011sklearn` | [VERIFIED — primary/deposited metadata](https://jmlr.org/papers/v12/pedregosa11a.html) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `lhoest2021datasets` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2021.emnlp-demo.21/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `nlp4lp_dataset` | [PARTIAL — repository/title/access verified](https://huggingface.co/datasets/udell-lab/NLP4LP) | Repository confirms gated access and CC BY-NC 4.0; personal creator attribution and release-year description retained from the OptiMUS baseline, not independently authenticated by the gated card. |
| `ramamonjison2022autoformulation` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2022.emnlp-industry.4/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `ramamonjison2023nl4opt` | [VERIFIED WITH YEAR DISCREPANCY — PMLR](https://proceedings.mlr.press/v220/ramamonjison23a.html) | Publisher citation export uses 2022; PDF masthead and HTML publication metadata use 2023. Retain existing 2022 citation with explicit masthead-2023 note, rather than silently choose one. |
| `ahmaditeshnizi2024optimus_icml` | [VERIFIED — primary/deposited metadata](https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `zhang2024optllm` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2024.naacl-industry.42/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `kadioglu2024ner4opt` | [VERIFIED — primary/deposited metadata](https://api.crossref.org/works/10.1007%2Fs10601-024-09376-5) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `ahmed2025opt2code` | [VERIFIED — primary/deposited metadata](https://api.crossref.org/works/10.1016%2Fj.nlp.2025.100185) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `biemans2025explainableoptimization` | [VERIFIED — primary/deposited metadata](https://link.springer.com/chapter/10.1007/978-3-032-08327-2_3) | Publisher chapter citation year 2026 retained; first-online 12 October 2025 distinguished. Uncited historical entry. |
| `powell2025scheduling_explanations` | [VERIFIED — primary/deposited metadata](https://doi.org/10.1007/s10844-025-00940-w) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `xiao2024chainofexperts` | [VERIFIED — primary/deposited metadata](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d45ee77826332c100a1e15f7765b99ff-Abstract-Conference.html) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `jiang2025llmopt` | [VERIFIED — official PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/fbe6dd68b0cf2b1d43b458d2b8ca31b0-Paper-Conference.pdf) | Title, full author list and ICLR 2025 verified on paper first page; no proceedings page range invented. |
| `pan2025pamop` | [VERIFIED — primary/deposited metadata](https://www.ijcai.org/proceedings/2025/296) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `huang2025orlm` | [VERIFIED — primary/deposited metadata](https://api.crossref.org/works/10.1287%2Fopre.2024.1233) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `lu2025optmath` | [VERIFIED — primary/deposited metadata](https://proceedings.mlr.press/v267/lu25o.html) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `xiao2026deepor` | [VERIFIED — publisher web record](https://ojs.aaai.org/index.php/AAAI/article/view/40699) | AAAI record confirms authors, 2026, 40(40), 34052–34060 and DOI; shell request failure did not imply a bad reference. |
| `affolter2019nli` | [VERIFIED — primary/deposited metadata](https://link.springer.com/article/10.1007/s00778-019-00567-8) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `yu2018spider` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/D18-1425/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `sheetrit2024rematch` | [VERIFIED — primary/deposited metadata](https://arxiv.org/abs/2403.01567) | arXiv identity/byline checked; explicitly unpublished preprint in prose, not in printed bibliography. No journal acceptance inferred. |
| `astorga2025autoformulation` | [VERIFIED — primary/deposited metadata](https://proceedings.mlr.press/v267/astorga25a.html) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `ding2026orr1` | [VERIFIED — primary/deposited metadata](https://api.crossref.org/works/10.1609%2Faaai.v40i1.36983) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `rizzi2025conceptual` | [PARTIAL — publisher search record](https://www.sciencedirect.com/science/article/pii/S0169023X25000473) | Publisher-indexed title, 159, September 2025, article 102452 and DOI match. Direct page blocked; full author metadata not freshly verified. Uncited historical entry. |
| `atzeni2026semantic` | [VERIFIED — primary/deposited metadata](https://api.crossref.org/works/10.1016%2Fj.datak.2025.102494) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `miplibnl2026` | [VERIFIED — primary/deposited metadata](https://proceedings.mlr.press/v306/li26dz.html) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `zhai2025equivamap` | [VERIFIED — primary/deposited metadata](https://proceedings.mlr.press/v267/zhai25a.html) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `chen2025solver` | [VERIFIED — primary/deposited metadata](https://proceedings.neurips.cc/paper_files/paper/2025/hash/98555b92e5dedbdad5921049281995bc-Abstract-Conference.html) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `gao2026models` | [VERIFIED — primary/deposited metadata](https://proceedings.colm.cc/paper_files/paper/2026/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html) | Official COLM 2026 pre-proceedings confirms title/byline and status; no DOI assigned. Author manuscript v2 also supports the modeling/binding and untrained-pipeline qualifications. |
| `chen2024m3embedding` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2024.findings-acl.137/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `yu2026uncertainty` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2026.acl-long.1975/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `xie2026escaping` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2026.findings-acl.690/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `li2026solver` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2026.findings-acl.102/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `zhang2026dual` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2026.acl-long.266/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `thind2025optimai` | [VERIFIED — primary/deposited metadata](https://arxiv.org/abs/2504.16918) | arXiv identity/byline checked; explicitly unpublished preprint in prose, not in printed bibliography. No journal acceptance inferred. |
| `kong2025alphaopt` | [VERIFIED — primary/deposited metadata](https://api.crossref.org/works/10.1145%2F3770855.3817974) | ACM-deposited Crossref metadata confirms 2026, page range and all 15 authors; direct ACM page blocked. Key name is historical, not publication year. |
| `yang2026orthought` | [PARTIAL — publisher search record](https://www.sciencedirect.com/science/article/pii/S3050860626000141) | Publisher-indexed title, volume 6, 2026, article 100059 and DOI match. Direct page blocked; full author metadata not freshly verified. |
| `holm1979followup` | [VERIFIED — paper scan; publisher access limited](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf) | Primary-paper scan supports Holm, title, 1979, volume 6 and 65–70; JSTOR is challenged. Issue 2 retained without a fresh independently readable publisher issue page. |
| `huang2017beamfollowup` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/D17-1227/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `deng2021strug` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2021.naacl-main.105/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `roy2017udg` | [VERIFIED — publisher record and author PDF](https://ojs.aaai.org/index.php/AAAI/article/view/10959) | Roy/Roth, title, AAAI 31(1), 2017 and DOI confirmed; author-hosted PDF checked. No unverified page range added. |
| `punyakanok2004ilp` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/C04-1197/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `almasian2023cqe` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2023.emnlp-main.793/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `wang2018execution` | [VERIFIED — primary/deposited metadata](https://arxiv.org/abs/1807.03100) | arXiv identity/byline checked; explicitly unpublished preprint in prose, not in printed bibliography. No journal acceptance inferred. |
| `wang2020ratsql` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2020.acl-main.677/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `scholak2021picard` | [VERIFIED — primary/deposited metadata](https://aclanthology.org/2021.emnlp-main.779/) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |
| `kuhn1955assignment` | [VERIFIED — primary/deposited metadata](https://api.crossref.org/works/10.1002%2Fnav.3800020109) | Title, author sequence, venue/year and supplied volume/pages/DOI inspected; retained except corrections above. |

## Citation and literature scope

The 41 citation keys resolve, every printed reference is cited, and every citation remains attached to the appropriate topic. The manuscript attributes column/value grounding to StruG, schema linking to RAT-SQL, quantity extraction to CQE, classical assignment to Kuhn and formulation-equivalence checks to EquivaMap. Text2Opt-Bench is explicitly prior work: its oracle BIND and trained-binding interventions are not equated with an untrained two-stage pipeline. The [author manuscript, Section B.1](https://arxiv.org/html/2605.21751v2) confirms that qualification. The task-level system table remains a protocol comparison without performance rankings.

Unpublished OptimAI, ReMatch and execution-guided decoding remain identified in prose and omitted from numbered lists. No new claim of “first” discovery, state of the art, acceptance or provider superiority was introduced.

Access failures, remaining partial verifications and metadata conflicts are retained above. Inaccessible sites do not demonstrate an incorrect reference. The current author should confirm unresolved creator/contact and publication-status details before submission; the audit does not silently fabricate replacements.
