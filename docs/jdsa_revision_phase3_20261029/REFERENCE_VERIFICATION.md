# Citation and reference verification — 29 October 2026

Scope: all 44 citation keys used by the main manuscript and supplementary sources. Primary proceedings, publisher pages, author preprints and publisher-deposited Crossref records were compared with the bibliography. The accompanying JSON records titles, author lists, dates, venues and available DOI metadata; metadata verification is not a claim that every full paper was re-read. Claim support was examined most closely for the changed analytical Related Work and the named closest methods. No published accuracy was imported as a local measurement.

`REFERENCE_METADATA.json` records the bibliographic inputs and primary-source metadata collected during the audit; the corrected deliverable is `manuscript/jdsa-phase3/references.bib`. The corrections below supersede any starting-entry fields retained in that audit snapshot.

## Corrections and bibliographic boundaries

- The rendered bibliography exposed a `sn-basic` behavior that removes periods from journal names. arXiv identifiers were moved out of the journal field into the standard `eprint` field, preserving their decimal separators without modifying the vendor style.

- Gao et al. is now an `@misc` arXiv preprint with its arXiv DOI. COLM 2026 acceptance remains explicitly author-reported; no invented proceedings/pages. The v2 full text distinguishes oracle externalization, a failed untrained pipeline and successful trained specialists. Related Work now represents all three, without importing their numerical results.
- Chain-of-Experts uses **Yuan Wang**, matching its official ICLR PDF and metadata. This does not change **Yuan Jessica Wang** in the separate DeepOR record. StruG retains **Ahmed Hassan Awadallah**, as printed in the actual ACL PDF despite abbreviated website metadata. OptiMUS retains the author's printed AhmadiTeshnizi capitalization.
- NL4Opt uses the publisher's 2022 citation year; the PDF/metadata publication date is 2023. SIRL is the NeurIPS 2025 proceedings contribution, although deposited publication metadata dates its publication to 2026. Neither date difference creates a new experimental result.
- MIPLIB-NL remains a preprint citation. Its author record reports ICML 2026 acceptance; no separate proceedings record was authenticated here. OptimAI, AlphaOPT, ReMatch and execution-guided decoding remain identified as preprints where that is the verified cited source. This is not a claim that no subsequent version can exist.
- PMLR/ICLR entries without an authenticated DOI retain their primary proceedings URLs. No DOI was guessed. References use numbered square brackets, with no citation command exceeding four entries.
- Holm's JSTOR endpoint returned an access challenge during the fresh check. The established 1979 citation is retained and this access limitation remains explicit. Restricted publisher access for OPT2CODE, ORLM, BM25, LSA and Kuhn was supplemented by publisher-deposited Crossref metadata; metadata alone does not authenticate local implementations.
- JDSA asks for cited published/accepted work. The author/editor should confirm treatment of the explicitly identified preprints before submission; no acceptance was inferred from availability on arXiv.

## Claim support and comparability

| Work | Primary source and supported comparison | Boundary retained |
|---|---|---|
| OptiMUS | [ICML proceedings](https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html) and linked PDF: formulation, solver code, debugging; NLP4LP provenance | Our catalog is constructed from its benchmark; schema lookup is not complete formulation |
| NL4Opt | [PMLR](https://proceedings.mlr.press/v220/ramamonjison23a.html): entity recognition followed by meaning representation | Does not provide accepted labels for our selected 217 relationships |
| OptMATH | [ICML](https://proceedings.mlr.press/v267/lu25o.html): bidirectional data synthesis and model training | Our external extraction check is component-level only |
| PaMOP | [IJCAI paper](https://www.ijcai.org/proceedings/2025/0296.pdf): tree partitioning, constraint modeling, correction | Local GPT-5.4/AMPL reconstruction is not the original reported benchmark |
| ORLM | [Publisher DOI](https://doi.org/10.1287/opre.2024.1233) and deposited record: customizable modeling-model training framework | Missing local execution is not evidence of poor published-method performance |
| DeepOR | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40699): intermediate reasoning, supervised training and solver-feedback reinforcement learning | Not executed as a new comparator here |
| Models Can Model, But Can't Bind | [Author full text, v2](https://arxiv.org/html/2605.21751v2), Sections 5.1–5.2 and Appendix B.1 | Oracle, trained and untrained interventions remain distinct; no claim of priority for binding difficulty |
| EquivaMap | [ICML](https://proceedings.mlr.press/v267/zhai25a.html): decision-variable mappings and feasibility/optimality preservation | Execution/objective proximity in our archives does not certify equivalence |
| StruG, RAT-SQL | [StruG PDF](https://aclanthology.org/2021.naacl-main.105.pdf), [RAT-SQL](https://aclanthology.org/2020.acl-main.677/) | Schema/value linking precedents, not executed local baselines |
| PICARD, execution-guided decoding | [PICARD](https://aclanthology.org/2021.emnlp-main.779/), [author preprint](https://arxiv.org/abs/1807.03100) | Syntax/execution constraints are narrower than source semantics |
| CQE, UDG, ILP semantic roles | [CQE](https://aclanthology.org/2023.emnlp-main.793/), [UDG](https://ojs.aaai.org/index.php/AAAI/article/view/10959), [ILP inference](https://aclanthology.org/C04-1197/) | Quantity/unit representations and constrained inference motivate distinctions; do not validate our heuristic |
| Other generation/memory/training systems | Primary records in the inventory below | Retained as task-level context, without invented baseline measurements or claims of experimental defeat |

## Complete citation inventory

| Key | Title | Primary link | Verification scope/status |
|---|---|---|---|
| `affolter2019nli` | A Comparative Survey of Recent Natural Language Interfaces for Databases | [2019](https://link.springer.com/10.1007/s00778-019-00567-8) | Publisher-deposited Crossref metadata checked; publisher access restricted or metadata absent |
| `ahmaditeshnizi2024optimus_icml` | OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models | [2024](https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html) | Primary proceedings/publisher/preprint record checked |
| `ahmed2025opt2code` | OPT2CODE: A Retrieval-Augmented Framework for Solving Linear Programming Problems | [2025](https://www.sciencedirect.com/science/article/pii/S2949719125000615) | Publisher-deposited Crossref metadata checked; publisher access restricted or metadata absent |
| `almasian2023cqe` | CQE: A Comprehensive Quantity Extractor | [2023](https://aclanthology.org/2023.emnlp-main.793/) | Primary proceedings/publisher/preprint record checked |
| `astorga2025autoformulation` | Autoformulation of Mathematical Optimization Models Using LLMs | [2025](https://proceedings.mlr.press/v267/astorga25a.html) | Primary proceedings/publisher/preprint record checked |
| `chen2024m3embedding` | M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | [2024](https://aclanthology.org/2024.findings-acl.137/) | Primary proceedings/publisher/preprint record checked |
| `chen2025solver` | Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling | [2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/98555b92e5dedbdad5921049281995bc-Abstract-Conference.html) | Primary proceedings/publisher/preprint record checked |
| `deerwester1990lsa` | Indexing by Latent Semantic Analysis | [1990](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9) | Publisher-deposited Crossref metadata checked; publisher access restricted or metadata absent |
| `deng2021strug` | Structure-Grounded Pretraining for Text-to-SQL | [2021](https://aclanthology.org/2021.naacl-main.105/) | Primary proceedings/publisher/preprint record checked |
| `ding2026orr1` | OR-R1: Automating Modeling and Solving of Operations Research Optimization Problem via Test-Time Reinforcement Learning | [2026](https://ojs.aaai.org/index.php/AAAI/article/view/36983) | Primary proceedings/publisher/preprint record checked |
| `gao2026models` | Models Can Model, But Can't Bind: Structured Grounding in Text-to-Optimization | [2026](https://arxiv.org/abs/2605.21751v2) | Author preprint v2; COLM acceptance author-reported, official proceedings not independently authenticated |
| `holm1979followup` | A Simple Sequentially Rejective Multiple Test Procedure | [1979](https://www.jstor.org/stable/4615733) | JSTOR access challenge; retained established citation, fresh full primary-record check incomplete |
| `huang2017beamfollowup` | When to Finish? Optimal Beam Search for Neural Text Generation (modulo beam size) | [2017](https://aclanthology.org/D17-1227/) | Primary proceedings/publisher/preprint record checked |
| `huang2025orlm` | ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling | [2025](https://pubsonline.informs.org/doi/10.1287/opre.2024.1233) | Publisher-deposited Crossref metadata checked; publisher access restricted or metadata absent |
| `jiang2025llmopt` | LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch | [2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/fbe6dd68b0cf2b1d43b458d2b8ca31b0-Paper-Conference.pdf) | Official ICLR PDF title page and abstract checked |
| `kadioglu2024ner4opt` | Ner4Opt: Named Entity Recognition for Optimization Modelling from Natural Language | [2024](https://link.springer.com/10.1007/s10601-024-09376-5) | Publisher-deposited Crossref metadata checked; publisher access restricted or metadata absent |
| `kong2025alphaopt` | AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library | [2025](https://arxiv.org/abs/2510.18428) | Primary proceedings/publisher/preprint record checked |
| `kuhn1955assignment` | The Hungarian Method for the Assignment Problem | [1955](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109) | Publisher-deposited Crossref metadata checked; publisher access restricted or metadata absent |
| `lhoest2021datasets` | Datasets: A Community Library for Natural Language Processing | [2021](https://aclanthology.org/2021.emnlp-demo.21/) | Primary proceedings/publisher/preprint record checked |
| `li2026solver` | Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design | [2026](https://aclanthology.org/2026.findings-acl.102/) | Primary proceedings/publisher/preprint record checked |
| `lu2025optmath` | OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling | [2025](https://proceedings.mlr.press/v267/lu25o.html) | Primary proceedings/publisher/preprint record checked |
| `miplibnl2026` | Constructing Industrial-Scale Optimization Modeling Benchmark | [2026](https://arxiv.org/abs/2602.10450) | Author preprint; ICML 2026 acceptance author-reported, official proceedings not independently authenticated |
| `nlp4lp_dataset` | NLP4LP: A Dataset for Natural Language to Linear Programming and Mixed-Integer Linear Programming | [2024](https://huggingface.co/datasets/udell-lab/NLP4LP) | Owner dataset card/API and OptiMUS provenance checked; descriptive dataset title |
| `pan2025pamop` | Guiding Large Language Models in Modeling Optimization Problems via Question Partitioning | [2025](https://ijcai.org/proceedings/2025/296) | Primary proceedings/publisher/preprint record checked |
| `pedregosa2011sklearn` | Scikit-learn: Machine Learning in Python | [2011](https://jmlr.org/papers/v12/pedregosa11a.html) | Primary proceedings/publisher/preprint record checked |
| `punyakanok2004ilp` | Semantic Role Labeling via Integer Linear Programming Inference | [2004](https://aclanthology.org/C04-1197/) | Primary proceedings/publisher/preprint record checked |
| `ramamonjison2022autoformulation` | Augmenting Operations Research with Auto-Formulation of Optimization Models From Problem Descriptions | [2022](https://aclanthology.org/2022.emnlp-industry.4/) | Primary proceedings/publisher/preprint record checked |
| `ramamonjison2023nl4opt` | NL4Opt Competition: Formulating Optimization Problems Based on Their Natural Language Descriptions | [2022](https://proceedings.mlr.press/v220/ramamonjison23a.html) | Primary proceedings/publisher/preprint record checked |
| `robertson2009bm25` | The Probabilistic Relevance Framework: BM25 and Beyond | [2009](https://www.emerald.com/ftinr/article/4/1-2/1/1326508/The-Probabilistic-Relevance-Framework-BM25-and) | Publisher-deposited Crossref metadata checked; publisher access restricted or metadata absent |
| `roy2017udg` | Unit Dependency Graph and Its Application to Arithmetic Word Problem Solving | [2017](https://ojs.aaai.org/index.php/AAAI/article/view/10959) | Primary proceedings/publisher/preprint record checked |
| `scholak2021picard` | PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models | [2021](https://aclanthology.org/2021.emnlp-main.779/) | Primary proceedings/publisher/preprint record checked |
| `sheetrit2024rematch` | ReMatch: Retrieval Enhanced Schema Matching with LLMs | [2024](https://arxiv.org/abs/2403.01567) | Primary proceedings/publisher/preprint record checked |
| `thind2025optimai` | OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents | [2025](https://arxiv.org/abs/2504.16918) | Primary proceedings/publisher/preprint record checked |
| `wang2018execution` | Robust Text-to-SQL Generation with Execution-Guided Decoding | [2018](https://arxiv.org/abs/1807.03100) | Primary proceedings/publisher/preprint record checked |
| `wang2020ratsql` | RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers | [2020](https://aclanthology.org/2020.acl-main.677/) | Primary proceedings/publisher/preprint record checked |
| `xiao2024chainofexperts` | Chain-of-Experts: When LLMs Meet Complex Operations Research Problems | [2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d45ee77826332c100a1e15f7765b99ff-Abstract-Conference.html) | Primary proceedings/publisher/preprint record checked |
| `xiao2026deepor` | DeepOR: A Deep Reasoning Foundation Model for Optimization Modeling | [2026](https://ojs.aaai.org/index.php/AAAI/article/view/40699) | Primary proceedings/publisher/preprint record checked |
| `xie2026escaping` | Escaping the Sisyphus Dilemma: Experience Replay for Robust Text-to-Optimization Modeling | [2026](https://aclanthology.org/2026.findings-acl.690/) | Primary proceedings/publisher/preprint record checked |
| `yang2026orthought` | ORThought: Benchmarking and Automating Logistics Optimization Modeling via Structured LLM Reasoning | [2026](https://linkinghub.elsevier.com/retrieve/pii/S3050860626000141) | Publisher-deposited Crossref metadata checked; publisher access restricted or metadata absent |
| `yu2018spider` | Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task | [2018](https://aclanthology.org/D18-1425/) | Primary proceedings/publisher/preprint record checked |
| `yu2026uncertainty` | Uncertainty-Aware Test-Time Search for Optimization Problem Solving | [2026](https://aclanthology.org/2026.acl-long.1975/) | Primary proceedings/publisher/preprint record checked |
| `zhai2025equivamap` | EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations | [2025](https://proceedings.mlr.press/v267/zhai25a.html) | Primary proceedings/publisher/preprint record checked |
| `zhang2024optllm` | Solving General Natural-Language-Description Optimization Problems with Large Language Models | [2024](https://aclanthology.org/2024.naacl-industry.42/) | Primary proceedings/publisher/preprint record checked |
| `zhang2026dual` | Dual-Cluster Memory Agent: Resolving Multi-Paradigm Ambiguity in Optimization Problem Solving | [2026](https://aclanthology.org/2026.acl-long.266/) | Primary proceedings/publisher/preprint record checked |
