# Primary-source positioning and journal checks

Checked on 2026-10-29. The bibliography's original 41 cited keys are retained across main/supplement. This report records sources consulted to qualify the narrative; it does not claim an exhaustive priority search.

| Primary source | Checked implication for this revision |
|---|---|
| [JDSA submission guidelines](https://link.springer.com/journal/41060/submission-guidelines) | Regular Papers report original research; Springer two-column format, 150–250-word abstract, 4–6 keywords, decimal headings and numbered references; flat LaTeX source organization |
| [Text2Opt-Bench, COLM 2026 pre-proceedings](https://proceedings.colm.cc/paper_files/paper/2026/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html), [author manuscript v2](https://arxiv.org/html/2605.21751v2) | Explicit modeling/binding distinction, scale/intervention analysis and oracle-versus-trained-specialist qualifications; this paper cannot claim first discovery of binding difficulty |
| [StruG](https://aclanthology.org/2021.naacl-main.105/) | Column grounding, value grounding and column–value mapping are established structured-grounding objectives |
| [RAT-SQL](https://aclanthology.org/2020.acl-main.677/) | Relation-aware schema encoding/linking precedes this diagnostic use of fixed-catalog retrieval |
| [EquivaMap](https://proceedings.mlr.press/v267/zhai25a.html) | Formulation equivalence through variable mappings is a different validation endpoint from stored-reference scalar agreement |
| [Spider](https://aclanthology.org/D18-1425/) | Cross-domain text-to-SQL context is relevant but does not supply interchangeable grounding scores |
| [Beam-search errors](https://aclanthology.org/D17-1227/) | Exact search and a correct selected output are distinct; cited as methodological context rather than a result guarantee here |

The main uses `pdflatex,iicol,sn-basic,Numbered`, has a 220-word abstract and six keywords. Supplementary wide tables retain the baseline one-column layout. Source files and figure assets are flat in the revised directory. No journal acceptance or complete submission-compliance certification is implied.

[URL_CHECKS.json](URL_CHECKS.json) records 64 active manuscript/bibliography URLs and response status. Some publisher endpoints return 401/403/406/418/420/429 or network failures; a chat API returns 405 for a non-POST probe. These are not silently marked as successful or dead links. New branch URLs are rechecked after publication. All internal LaTeX references and citations resolve.
