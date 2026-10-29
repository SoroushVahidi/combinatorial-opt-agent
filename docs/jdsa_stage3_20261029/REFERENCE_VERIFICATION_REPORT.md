# Reference-verification report

All 48 master bibliography entries and all in-text citations were inspected. The final documents cite 41 distinct keys (25 main entries and 23 supplement entries, with overlap); seven unused master entries are preserved. Three of those seven are unpublished preprints discussed explicitly in prose rather than added to the numbered bibliography. No missing author, DOI or page metadata was invented.

Evidence: [HTTP/metadata ledger](REFERENCE_METADATA.json), [primary-page follow-up checks](REFERENCE_PRIMARY_PAGE_CHECKS.json), [field-level corrections](REFERENCE_CORRECTIONS.json), [citation-number crosswalk](CITATION_NUMBER_CROSSWALK.json) and [URL checks](URL_CHECKS.json). The `fields` entries in the metadata ledger are the pre-correction bibliography snapshot; final field values are in the Stage 3 `.bib` file and the correction ledger. Automated HTTP restrictions remain recorded even when a browser-accessible primary page supplied verification. Metadata verification concerns the fields exposed by that source; it does not independently validate a paper's results.

## Verified corrections and retained ambiguities

- Restored complete Datasets authors, publisher name forms for the 2022 autoformulation paper, StruG author surname parsing and Kadıoğlu diacritics; expanded verified proceedings titles.
- Added official LLMOPT pages 101580–101606, SIRL pages 106027–106069, and scikit-learn issue/article identifier 85 with its JMLR source.
- Removed eight redundant printed publisher URL fields while retaining full DOI links and all original URLs in the correction ledger. Other valid URLs are retained.
- NL4Opt retains 2022 because that is the publisher's recommended citation/BibTeX; the 2023 page/PDF date conflict remains explicitly documented. It is not silently resolved by guessing.
- Ner4Opt retains the publisher's combined issue 3–4; SIRL retains conference year 2025 despite 2026 online metadata.
- Text2Opt-Bench is described as the official COLM 2026 pre-proceedings item; a nonexistent DOI is not supplied. OptimAI, ReMatch and execution-guided SQL remain identified as unpublished preprints.

## Entry-by-entry inventory

| Key | Use | Verification obtained / boundary | Primary or registry source |
|---|---|---|---|
| `robertson2009bm25` | Cited | NOT independently reverified in full: access restricted/rate-limited or page lacks inspectable structured metadata. Existing entry retained; no assertion that it is incorrect. | [Source](https://doi.org/10.1561/1500000019) |
| `deerwester1990lsa` | Cited | NOT independently reverified in full: access restricted/rate-limited or page lacks inspectable structured metadata. Existing entry retained; no assertion that it is incorrect. | [Source](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9) |
| `pedregosa2011sklearn` | Cited | Bibliographic metadata retrieved from primary. | [Source](https://jmlr.org/papers/v12/pedregosa11a.html) |
| `lhoest2021datasets` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2021.emnlp-demo.21.bib) |
| `nlp4lp_dataset` | Cited | Dataset landing page accessible; dataset reference retained. Gated source access is not a bibliographic failure. | [Source](https://huggingface.co/datasets/udell-lab/NLP4LP) |
| `ramamonjison2022autoformulation` | Cited | Bibliographic metadata retrieved from crossref, primary, publisher_bibtex. | [Source](https://aclanthology.org/2022.emnlp-industry.4.bib) |
| `ramamonjison2023nl4opt` | Cited | Primary-page inspection; Recommended publisher BibTeX/citation says 2022; HTML metadata/PDF masthead show 2023. Existing ambiguity note retained. | [Source](https://proceedings.mlr.press/v220/ramamonjison23a.html) |
| `ahmaditeshnizi2024optimus_icml` | Cited | Bibliographic metadata retrieved from primary. | [Source](https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html) |
| `zhang2024optllm` | Cited | Bibliographic metadata retrieved from crossref, primary, publisher_bibtex. | [Source](https://aclanthology.org/2024.naacl-industry.42.bib) |
| `kadioglu2024ner4opt` | Cited | Primary-page inspection; Combined issue is 3–4 despite abbreviated article metadata showing issue 3; retained 3–4. | [Source](https://link.springer.com/journal/10601/volumes-and-issues/29-3) |
| `ahmed2025opt2code` | Cited | NOT independently reverified in full: access restricted/rate-limited or page lacks inspectable structured metadata. Existing entry retained; no assertion that it is incorrect. | [Source](https://www.sciencedirect.com/science/article/pii/S2949719125000615) |
| `biemans2025explainableoptimization` | Uncited master / text-only context | Bibliographic metadata retrieved from crossref, primary. | [Source](https://link.springer.com/chapter/10.1007/978-3-032-08327-2_3) |
| `powell2025scheduling_explanations` | Uncited master / text-only context | Bibliographic metadata retrieved from crossref, primary. | [Source](https://doi.org/10.1007/s10844-025-00940-w) |
| `xiao2024chainofexperts` | Cited | Bibliographic metadata retrieved from primary. | [Source](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d45ee77826332c100a1e15f7765b99ff-Abstract-Conference.html) |
| `jiang2025llmopt` | Cited | Bibliographic metadata retrieved from publisher_metadata. | [Source](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fbe6dd68b0cf2b1d43b458d2b8ca31b0-Abstract-Conference.html) |
| `pan2025pamop` | Cited | Bibliographic metadata retrieved from crossref, primary. | [Source](https://www.ijcai.org/proceedings/2025/296) |
| `huang2025orlm` | Cited | Primary-page inspection; Publisher article: title, authors, 2025, 73(6), 2986–3009 and DOI. | [Source](https://pubsonline.informs.org/doi/10.1287/opre.2024.1233) |
| `lu2025optmath` | Cited | Bibliographic metadata retrieved from primary. | [Source](https://proceedings.mlr.press/v267/lu25o.html) |
| `xiao2026deepor` | Cited | Bibliographic metadata retrieved from crossref. | [Source](https://api.crossref.org/works/10.1609%2Faaai.v40i40.40699) |
| `affolter2019nli` | Cited | Bibliographic metadata retrieved from primary. | [Source](https://doi.org/10.1007/s00778-019-00567-8) |
| `yu2018spider` | Cited | Bibliographic metadata retrieved from crossref, primary, publisher_bibtex. | [Source](https://aclanthology.org/D18-1425.bib) |
| `sheetrit2024rematch` | Uncited master / text-only context | Bibliographic metadata retrieved from primary. | [Source](https://arxiv.org/abs/2403.01567) |
| `astorga2025autoformulation` | Cited | Bibliographic metadata retrieved from primary. | [Source](https://proceedings.mlr.press/v267/astorga25a.html) |
| `ding2026orr1` | Cited | Primary-page inspection; Official AAAI article: title, authors, proceedings year and DOI. | [Source](https://ojs.aaai.org/index.php/AAAI/article/view/36983) |
| `rizzi2025conceptual` | Uncited master / text-only context | Bibliographic metadata retrieved from crossref. | [Source](https://api.crossref.org/works/10.1016%2Fj.datak.2025.102452) |
| `atzeni2026semantic` | Uncited master / text-only context | NOT independently reverified in full: access restricted/rate-limited or page lacks inspectable structured metadata. Existing entry retained; no assertion that it is incorrect. | [Source](https://doi.org/10.1016/j.datak.2025.102494) |
| `miplibnl2026` | Cited | Bibliographic metadata retrieved from primary. | [Source](https://proceedings.mlr.press/v306/li26dz.html) |
| `zhai2025equivamap` | Cited | Bibliographic metadata retrieved from primary. | [Source](https://proceedings.mlr.press/v267/zhai25a.html) |
| `chen2025solver` | Cited | Primary-page inspection; Official NeurIPS 38/2025 proceedings, 106027–106069; online metadata date 2026 is not substituted for conference year. | [Source](https://proceedings.neurips.cc/paper_files/paper/2025/hash/98555b92e5dedbdad5921049281995bc-Abstract-Conference.html) |
| `gao2026models` | Cited | Primary-page inspection; Official COLM 2026 pre-proceedings; no DOI invented. | [Source](https://proceedings.colm.cc/paper_files/paper/2026/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html) |
| `chen2024m3embedding` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2024.findings-acl.137.bib) |
| `yu2026uncertainty` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2026.acl-long.1975.bib) |
| `xie2026escaping` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2026.findings-acl.690.bib) |
| `li2026solver` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2026.findings-acl.102.bib) |
| `zhang2026dual` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2026.acl-long.266.bib) |
| `thind2025optimai` | Uncited master / text-only context | Bibliographic metadata retrieved from primary. | [Source](https://arxiv.org/abs/2504.16918) |
| `kong2025alphaopt` | Cited | NOT independently reverified in full: access restricted/rate-limited or page lacks inspectable structured metadata. Existing entry retained; no assertion that it is incorrect. | [Source](https://dl.acm.org/doi/10.1145/3770855.3817974) |
| `yang2026orthought` | Cited | NOT independently reverified in full: access restricted/rate-limited or page lacks inspectable structured metadata. Existing entry retained; no assertion that it is incorrect. | [Source](https://doi.org/10.1016/j.ait.2026.100059) |
| `holm1979followup` | Cited | NOT independently reverified in full: access restricted/rate-limited or page lacks inspectable structured metadata. Existing entry retained; no assertion that it is incorrect. | [Source](https://www.jstor.org/stable/4615733) |
| `huang2017beamfollowup` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/D17-1227.bib) |
| `deng2021strug` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2021.naacl-main.105.bib) |
| `roy2017udg` | Cited | Primary-page inspection; Official AAAI article: title, authors, 2017, 31(1) and DOI; no unverified page range added. | [Source](https://ojs.aaai.org/index.php/AAAI/article/view/10959) |
| `punyakanok2004ilp` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/C04-1197.bib) |
| `almasian2023cqe` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2023.emnlp-main.793.bib) |
| `wang2018execution` | Uncited master / text-only context | Bibliographic metadata retrieved from primary. | [Source](https://arxiv.org/abs/1807.03100) |
| `wang2020ratsql` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2020.acl-main.677.bib) |
| `scholak2021picard` | Cited | Bibliographic metadata retrieved from primary, publisher_bibtex. | [Source](https://aclanthology.org/2021.emnlp-main.779.bib) |
| `kuhn1955assignment` | Cited | Primary-page inspection; Publisher article: title, author, March 1955, 2(1–2), 83–97 and DOI. | [Source](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109) |

Coverage categories: incomplete: 7, metadata: 32, dataset_page: 1, primary_page: 8. These categories summarize this audit, not a binary assertion of bibliographic truth.

## External-access limitations

The now publishers BM25 page, Wiley LSA route, ScienceDirect OPT2CODE route and ACM AlphaOPT route restricted this environment. Some other publisher landing pages were accessible without usable metadata; JSTOR returned a script-based Holm landing page. Their prior bibliographic fields were preserved. Canonical DOI/publisher links are not marked broken merely because a fetch failed. No new title/venue/year correction was based on search-result inference alone.

The URL ledger covers reader-facing URLs in the manuscripts plus guideline/primary-source additions. Provider API base URLs are classified as API endpoints and were not invoked. There were no confirmed dead (404/410) reader links; the new branch URL is checked after publication. See the publication receipt for the final public-source access result.
