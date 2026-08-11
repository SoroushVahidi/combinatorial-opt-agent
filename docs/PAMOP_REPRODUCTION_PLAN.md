# PaMOP Reproduction Plan — Investigation & Feasibility Report

**Status:** Investigation/planning only. No PaMOP code has been implemented as part of
this pass. No manuscript file and no committed benchmark result was modified.

**Paper under investigation:**
Xiaotian Pan, Junhao Fang, Feng Wu, Sijia Zhang, Yi-Xiang Hu, Shaoang Li, Xiang-Yang Li.
*"Guiding Large Language Models in Modeling Optimization Problems via Question
Partitioning."* IJCAI 2025, Main Track, pp. 2657–2665.
DOI: [10.24963/ijcai.2025/296](https://doi.org/10.24963/ijcai.2025/296).
Affiliation (all authors): School of Computer Science and Technology, University of
Science and Technology of China (USTC).

**Investigation date:** 2026-08-11
**Investigated by:** automated repository assistant, on request, against
`https://github.com/SoroushVahidi/combinatorial-opt-agent` (local clone:
`~/combinatorial-opt-agent`, branch `main`).

---

## 1. Does official public PaMOP code exist?

**Answer: NO** (high confidence, not merely "not found").

Evidence for a *negative* finding, not just an inconclusive search:

- The IJCAI proceedings page (<https://www.ijcai.org/proceedings/2025/296>) exposes only
  a PDF and BibTeX link — no code/project/supplementary link field is populated.
- The full paper PDF was downloaded and converted to text
  (`pdftotext -layout`) and searched for `github|available|http|code is|repository|open-source|release`.
  The **only** hits are an unrelated sentence ("open-source and closed-source" base
  models tested) and the Gurobi documentation URL in the bibliography. **There is no
  code/data availability statement anywhere in the paper.**
- `BeinuoYang/Awesome-LLM4Opt`, a actively maintained GitHub list that explicitly tags
  entries with `[[official code]]` when a released implementation exists (confirmed by
  inspecting adjacent 2025/07–2025/08 entries that *do* carry that tag, e.g. ORMind,
  "Automated Optimization Modeling"), lists the PaMOP paper **without** a code tag —
  i.e., a domain-specific curator already checked and found none.
- `gh search repos`, `gh search code` for `PaMOP`, `question-partitioning`, and related
  terms returned no relevant hits — only unrelated projects (a Go PAM auth module, an
  options-pricing class literally named `PamOP`, an unrelated person's GitHub handle
  `pamop`, LeetCode partitioning exercises).
- No arXiv preprint of this paper could be located (searched by exact title and by full
  author list). The IJCAI camera-ready appears to be the only public version.
- No OpenReview page, appendix PDF, slide deck, poster, or project webpage was found.
- The USTC personal page for the corresponding author
  (`staff.ustc.edu.cn/~xiangyangli/publication.html`) is presently unreachable
  (`ECONNREFUSED`) from this workstation's network path — likely a China-mainland
  hosting/network-path issue rather than evidence the page doesn't exist. This is the
  one avenue not fully exhausted; see §12 next steps.
- No Hugging Face Space/model and no GitLab project were found under "PaMOP" or
  paraphrases of the title.
- Author GitHub usernames were guessed from the paper's email prefixes
  (`px259637522`, `kd1198497461`) and plausible name variants — all return 404.

### 1a. URLs investigated (all checked, none is official PaMOP code)

| URL | What it is | Official? | Notes |
|---|---|---|---|
| https://www.ijcai.org/proceedings/2025/296 | IJCAI abstract page | N/A | No code link present |
| https://www.ijcai.org/proceedings/2025/0296.pdf | Full paper PDF | — | Primary source used in §2–§6 below |
| https://beinuoyang.github.io/Awesome-LLM4Opt/ | Curated LLM4Opt paper list | No | Lists PaMOP with **no** code tag (contrast with other entries that have one) |
| https://github.com/ai4co/awesome-fm4co | Curated FM4CO list | No | Surfaced only via search ranking; does not itself carry a PaMOP code link |
| http://staff.ustc.edu.cn/~xiangyangli/publication.html | Corresponding author's USTC page | Unverified | Connection refused from this network; not confirmed either way |
| https://github.com/teshnizi/OptiMUS | OptiMUS (different paper, used as baseline **by** PaMOP) | Official **for OptiMUS**, not PaMOP | Reference-only per task instructions, §4 below |
| https://huggingface.co/datasets/udell-lab/NLP4LP | NLP4LP dataset (gated) | Official dataset, not PaMOP code | Used by PaMOP as its benchmark |
| github.com search: `PaMOP`, `question-partitioning`, `pamop` | GitHub-wide search | No | All hits unrelated (see above) |
| gitlab.com, huggingface.co/spaces search | — | No | No relevant hits |

**Conclusion:** treat "no official code" as the working assumption for planning
purposes, with one unresolved lead (USTC author page, network-blocked) noted for
follow-up before committing engineering time to a from-scratch build.

---

## 2. Paper-derived algorithm specification

Extracted directly from the IJCAI camera-ready PDF (`pdftotext -layout`, cross-checked
against the rendered figures). Quotes are verbatim; classification: **A** = explicitly
stated, **B** = inferable with high confidence, **C** = unspecified, a reproduction
choice is required.

### 2.1 Problem representation (A)

A model is `M = (m_p, m_v, m_o, m_c)`: `m_p` = constants/parameters, `m_v` = decision
variables `x`, `m_o` = objective `f(x)`, `m_c` = constraints `g_i(x) ≤ 0`.

### 2.2 Pipeline overview (A)

1. **Structured extraction** → root node of a partition tree.
2. **Partition tree construction** (root: independent-set separation; deeper layers:
   constraint clustering) → leaf nodes = sub-problems.
3. **Leaf-node modeling** with self-augmented prompts → per-leaf constraint formulas
   `m_{c,i}`.
4. **Bottom-up merge** of leaf formulas into a complete AMPL model `M`.
5. **Correction loop**: basic inspection (syntax/regex + parameter-vs-data check) →
   solver execution + LLM self-debug ("error solver") → reverse-translation semantic
   consistency check for persistent logical errors.
6. **Validation**: run the AMPL model + data through Gurobi; success = "optimal solution
   that meets the problem's requirements."

### 2.3 Structured extraction (A, mechanism; C, exact prompt text)

Before partitioning, an LLM call guided by a prompt referred to as `G_extr` produces:
- `t_o` — objective description text
- `t_c` — constraint description text
- `t_v` — parameter/variable description text
- `g` — a concise whole-problem summary (explicitly motivated as avoiding "modeling
  bias from missing the global context")
- a **vagueness score** per constraint, stored on the root node.

The actual prompt string for `G_extr` is **not reproduced** in the paper (C).

### 2.4 Partition tree construction (A mechanism; C exact parameters)

Two distinct partitioning methods, applied at different tree depths:

**(a) Independent-set separation — root node only (A mechanism).**
Build a bipartite graph `G = (V ∪ C, E)` where `V` = constants/variables (from `t_v`),
`C` = constraints (from `t_c`), and an edge `(v, c) ∈ E` denotes a *high-confidence
association*, defined by keyword-extraction-and-matching between constraint text and
variable descriptions (explicitly credited as "inspired by Gasse et al. [2019]", the
GNN-for-MILP paper — note this is a *conceptual* citation, not a claim that PaMOP uses a
GNN; PaMOP's own method is classical graph search). A graph-search algorithm separates
independent connected subgraphs; each becomes a child of the root.
- **C:** which graph-search algorithm (e.g., connected components vs. something more
  elaborate), and the exact keyword-matching confidence threshold, are not given.

**(b) Constraint clustering — subsequent layers (A mechanism, formula given; C several
constants).**
Distance between two constraints:
```
d_{i,j} = 1/(s_{i,j} + ε) − 1/(1 + ε)
```
where `s_{i,j}` is a similarity combining three signals with a **weighted average that
"applies different [weights] to different layers of the partition tree"**:
1. Adjacency/contextual prioritization — "constraints are typically extracted in order,
   so adjacent ones are likely similar," so adjacent pairs get boosted similarity.
2. Keyword similarity — count of shared top-k TF-IDF keywords between two constraints.
3. Vector similarity — cosine similarity of GloVe embeddings ("trained on Wikipedia
   2014" — Pennington et al. 2014).

"Noise points" from clustering are **kept as potentially relevant constraints, not
discarded.** Partitioning stops per node when its constraint set is "small" or "highly
similar" — no numeric threshold given.

- **C — unspecified constants/choices needed to actually run this:**
  the value of `ε`; the per-layer weight schedule for the three similarity signals;
  `k` in "top-k TF-IDF keywords"; which GloVe dimensionality/vocab size (50/100/200/300d,
  6B/42B/840B tokens — Pennington et al. released several); which clustering algorithm
  consumes the pairwise distance matrix (paper describes only the *distance metric*, not
  the clustering rule — e.g., hierarchical agglomerative, DBSCAN, or a custom greedy
  merge are all consistent with "noise points kept, not discarded" phrasing); the
  numeric stopping thresholds for "small number of constraints" / "highly similar
  ones."

### 2.5 Leaf-node modeling — "self-augmented prompts" (A formula; C prompt text)

```
m_{c,i} = G_mod( g, t_v, {t_{c,j} : j ∈ cons_i} )
```
i.e. each leaf's prompt is *augmented* with the global summary `g` and the full variable
list `t_v` (not just that leaf's local constraints) — this augmentation is what "self-
augmented" refers to. For **vague** constraints specifically, the prompt is further
augmented with information from the node's **parent and sibling** nodes. The paper
states CoT-style prompting (Wei et al. 2022) was used as inspiration, plus an unspecified
"set of principles for the task of modeling optimization problems... to ensure the
stability of outputs" (C — principles not enumerated).

**Output language: AMPL**, not raw Python/solver-API code (A, explicit, with stated
rationale: "LLMs treat mathematical formulas, modeling languages, and programming
languages as different 'languages'... we directly generate code in the modeling
language instead of formulas"). This is an important divergence from OptiMUS-style
Python-code generation and from this repository's own pipeline.

### 2.6 Merge (A)

Leaf formulas are merged bottom-up; since variables/parameters were already described
globally at extraction time, the paper claims "minimal conflict between formulas modeled
at different nodes," and the root node's objective is completed at this stage:
`M = (m_p, m_v, m_o, m_c) = G_mod(g, t_v, t_o, m_c)`.

### 2.7 Correction procedure (A mechanism; C thresholds/prompt text)

Three layers, in order:

1. **Basic Inspection** — regex-based syntax check + parameter-vs-data-file
   verification (Figure 2 box "[Y]/[N] Regex").
2. **Error-solver loop** ("Gexe") — following Zhang et al. 2023 (*Self-Edit*), the model
   + data are run through the solver; solver errors are embedded back into a correction
   prompt, iterated until valid:
   `solution(ans, e) = Solve(M, data)`; `M = G_exe(M, e)`.
3. **Reverse translation** — for logical (not syntactic) inconsistencies that survive
   step 2: extract each constraint's intended-meaning annotation from the partition-tree
   analysis; hide the annotation for the constraint under test to get a "partial black
   box" model `m'_c`; one LLM infers the constraint's natural-language meaning from the
   parameter set (`G_rev`); a second LLM, acting as "language expert," judges binary
   semantic consistency `μ ∈ {0,1}` between the inferred text and the original (`G_comp`);
   inconsistent constraints are sent back with the original problem+model for
   regeneration (`G_remod`):
   ```
   t'_c = G_rev(m_p, m_v, m'_c)
   μ = G_comp(t_c, t'_c),  μ ∈ {0,1}
   M = G_remod(T, m_c, μ)
   ```

- **C:** exact prompt text for all of `G_exe`, `G_rev`, `G_comp`, `G_remod`; the regex
  patterns used for basic inspection; whether reverse translation runs on *every*
  constraint or only ones flagged as vague at extraction time (Fig. 2 suggests the
  latter but this is not stated in prose).

### 2.8 Stopping criteria and hyperparameters (A, both explicit)

- **Temperature = 0.2** ("controls randomness of the model's output").
- **Maximum number of failed iterations = 5** (default). Not stated whether this budget
  is global (across the whole correction loop) or per-node/per-constraint (C).
- Success/exit condition: "When the solver successfully solves the problem without any
  error, we consider the problem resolved and exit the system."
- No `top_p`, `max_tokens`, or other decoding parameters are reported (C).

### 2.9 Model / runtime (A, with a gap noted)

- **LLM: "GPT-4"** — no snapshot/version string given anywhere in the paper (e.g.
  `gpt-4-0613` vs `gpt-4-turbo` vs `gpt-4o`) (**C** — see §9).
- **Modeling language: AMPL** (Fourer et al. 1987), chosen specifically because it
  separates model from data files.
- **Solver: Gurobi**, invoked by AMPL, to obtain the objective value (A, explicit).
- Additional cross-model ablation (Table 4) also tests **GPT-3.5-turbo** and
  **Llama-3.3-70b**, with PaMOP achieving 24.3%/71.2% and 54.7%/83.2%
  accuracy/execution-rate respectively — i.e. GPT-4 is the headline configuration but
  not the only one tested.

### 2.10 Evaluation metric definitions (A, exact wording)

- **Accuracy** = "success rate" — a problem counts as solved when its generated
  AMPL+data, run through Gurobi, "yields an optimal solution that meets the problem's
  requirements." The paper does **not** state the exact correctness check (e.g. numeric
  tolerance on the objective value vs. an exact match vs. comparison against a reference
  optimum) — **C**.
- **Execution Rate** ("code executability rate" in the abstract, "Execution Rate" in the
  body/tables — same metric, two names) = proportion of generated model files that can
  be *executed* by the solver at all, independent of whether the resulting solution is
  correct.
- **Compile Error (CE) rate** = fraction of generated programs that fail to compile
  (e.g. missing parameters).
- **Runtime Error (RE) rate** = fraction of generated programs that compile but error
  during execution (e.g. infeasible/unbounded models, often attributed to missing
  constraints).

### 2.11 NLP4LP subset used (A count; C exact overlap with the public HF dataset)

"The NLP4LP dataset [AhmadiTeshnizi et al., 2024] is collected from optimization
textbooks and manuals... In total, it contains **54 LP problems and 13 MILP problems**"
— i.e. **67 problems total**, evaluated at the *problem* level (not a query/sub-question
level). This 67-problem count is **substantially smaller** than the ≈269-problem NLP4LP
v0.3 catalog referenced by later OptiMUS papers, and does not by itself indicate which
NLP4LP release/split the authors pulled from the gated `udell-lab/NLP4LP` HF repo, or
whether the split maps cleanly onto that repo's current file layout (**C** — needs
direct verification once the dataset is inspected locally; see §8).

### 2.12 Additional real-world evaluation (A, but not publicly reproducible)

Table 2 uses **custom, unreleased, real-world problem instances** (storage, scheduling,
placement A/B, mining, HR allocation) — e.g. "Storage" has 3,795 parameters, 124
decision variables, 162 constraints. These are **not part of any public dataset**, are
not described in enough detail to reconstruct (no full problem text given, only the
scale numbers), and there is no indication they will ever be released. **This part of
the paper's evaluation is not reproducible by any group without the authors'
cooperation** — it should be scoped out of any reproduction claim entirely, not
approximated.

### 2.13 Baselines / results tables (A — reported for context, not to be treated as our
own results)

| Method | Accuracy | Exec. Rate | CE rate | RE rate |
|---|---|---|---|---|
| Default | 25.3% | 48.3% | 40.2% | 11.5% |
| Chain-of-Thought | 28.8% | 51.5% | 38.7% | 9.8% |
| Progressive Hint | 33.5% | 52.3% | 34.6% | 13.1% |
| Tree-of-Thought | 36.4% | 54.4% | 35.1% | 10.5% |
| Reflexion | 40.3% | 69.7% | 19.1% | 11.2% |
| OptiMUS | 56.7% | 78.4% | 11.8% | 9.8% |
| **PaMOP (paper)** | **62.3%** | **86.8%** | **7.3%** | **5.9%** |

Ablation (Table 3, GPT-4): Prompt Only 25.3%/48.3%; + Partition 48.5%/63.4%;
Full (+ Correction) 62.3%/86.8%.

### 2.14 Limitations acknowledged by the authors (A)

"Our method has not been specifically designed to optimize algorithms for these complex
constraints; instead, it primarily relies on the inherent logical reasoning capabilities
of LLMs" — i.e. the authors themselves note the method is bounded by base-model
reasoning quality on high-complexity real-world constraint interactions, not by the
partitioning/correction scaffolding.

---

## 3. Supplementary material search — result

No appendix, supplementary PDF, arXiv preprint, OpenReview page, poster, slide deck, or
project webpage was located (see §1 for the specific queries and pages checked). The
IJCAI camera-ready PDF is the **only** available primary source. One lead — the
corresponding author's USTC personal page — could not be reached from this network and
remains open (§12).

---

## 4. OptiMUS cross-check (reference only, per task scope)

`teshnizi/OptiMUS` (MIT license, actively maintained, last push 2025-11-04) was
inspected **only** for NLP4LP/eval conventions common to the ecosystem, not as a PaMOP
substitute:

- **NLP4LP license, per OptiMUS's own README:** "CC BY NC 4.0 (allowing only
  non-commercial use)... models trained using the dataset should not be used outside of
  research purposes." The HF dataset card itself currently states **CC BY-NC-SA 4.0**
  (adds a share-alike clause) — a minor discrepancy between the OptiMUS README and the
  live HF card; treat the HF card as authoritative since it is the operative license
  text.
- **Error-correction convention** (`execute_code.py`): subprocess-executes generated
  Python, on failure embeds the stderr into a fixed debug prompt template, and retries
  up to `max_tries` (default 3 in their code) — structurally the same "solver-error into
  correction-prompt" idea PaMOP describes for its own `G_exe`, but OptiMUS operates on
  **Python code**, not AMPL, and its default model in `main.py` is now `gpt-4o` (not
  bare `gpt-4`), confirming that even the OptiMUS team has already migrated off raw
  GPT-4 in their own reference implementation.
- **This repo's existing `src/data_adapters/optimus.py`** already encodes the
  NLP4LP-via-OptiMUS JSONL convention (`train/validation/test.jsonl` under
  `data/external/optimus/`) and is the natural place to extend/mirror conventions from,
  **not** a place to plug PaMOP logic into.

No OptiMUS behavior is substituted for any undocumented PaMOP behavior in the design
below; OptiMUS was used purely to corroborate the NLP4LP license text and to confirm
that "solver-error-into-prompt" self-debugging is a standard, independently-supported
NLP4LP-ecosystem convention rather than something PaMOP-specific we'd otherwise have to
invent from nothing.

---

## 5. Comparison compatibility with our current pipeline

Our repository's paper-core pipeline (per `docs/EAAI_SOURCE_OF_TRUTH.md`,
`README.md`) evaluates **schema retrieval + deterministic scalar grounding** on the
NLP4LP `orig` variant, **331 test queries**, with metrics `Schema R@1`, `Coverage`,
`TypeMatch`, and `InstantiationReady_i = 1[Coverage_i ≥ 0.8 ∧ TypeMatch_i ≥ 0.8]`
(exact formula confirmed in `tools/run_strict_instantiation_ready.py`). This is a
**non-generative, largely deterministic** pipeline (TF-IDF retrieval + typed-greedy
grounding) evaluated at **query granularity**.

PaMOP evaluates **full NL → generative AMPL model → Gurobi solve** correctness and
executability, at **problem granularity** (67 problems, not 331 queries), using a
**stochastic, multi-stage LLM pipeline** (temperature 0.2, several chained LLM calls per
problem).

**These are not the same measurement and must never be presented in the same table as
if they were.** Specifically:

| | Our pipeline | PaMOP |
|---|---|---|
| Unit of evaluation | NL query (331, `orig`) | Full problem (67: 54 LP + 13 MILP) |
| Task | Schema retrieval + scalar slot grounding | Full NL → executable optimization model |
| Output | Structured slot values against a fixed schema catalog | AMPL model + data file |
| Correctness check | Coverage/TypeMatch thresholds on scalar values | Gurobi returns an optimal solution meeting requirements |
| Generative? | No (deterministic retrieval/grounding) | Yes (LLM-generated model, 5-retry correction loop) |
| Determinism | Fully deterministic given fixed catalog | Stochastic (temperature 0.2, multi-call) |

### What can and cannot be reported

- **CAN** cite PaMOP's own published NLP4LP numbers (62.3% accuracy / 86.8% execution
  rate) as **literature context**, explicitly labeled as *the authors' own reported
  numbers, not independently verified*, since no code exists to rerun them (§1).
- **CANNOT** place PaMOP's 62.3%/86.8% next to our `InstantiationReady`/`Coverage`
  numbers in one table as if they measured the same thing — different units,
  different tasks, different denominators.
- **CAN**, if we build the local reproduction (§6), report our **own measured**
  Accuracy/Execution-Rate/CE-rate/RE-rate from **our own run** — but this must be
  labeled as a local reimplementation result, never as "PaMOP's number," because our
  prompts/model/clustering choices necessarily differ from the undisclosed original
  (§2.4, §2.5, §2.7 all have Category-C gaps).
- **CAN** derive a genuinely apples-to-apples comparison point: run our local PaMOP
  reproduction, **parse its generated AMPL model+data back into scalar parameter
  values**, and score those values with **our own** `Coverage`/`TypeMatch`/
  `InstantiationReady` metric implementation. This lets us ask "how good is PaMOP's
  *parameter grounding* by our metric" on the *same* NLP4LP queries our main pipeline
  is scored on — a legitimate, same-metric, cross-method comparison, distinct from
  PaMOP's own accuracy/execution-rate metric.
- **CAN** report runtime and API cost per problem/query for both pipelines — these are
  comparable regardless of task-definition differences (though PaMOP's cost will be
  orders of magnitude higher, being multi-call LLM-generative vs. ours being
  retrieval+deterministic-grounding).
- **CAN** report determinism/reproducibility as a qualitative + quantitative axis: our
  pipeline is exactly reproducible; PaMOP is not (stochastic decoding, no seed control
  mentioned in the paper — another Category-C gap), so a reproduction should report
  variance across ≥3 seeded/temperature-fixed reruns if this axis is included.
- **Dataset-overlap caveat:** before any of the above, we must confirm whether our
  331-query `orig` variant and PaMOP's 67-problem subset draw from **overlapping or
  disjoint** underlying NLP4LP problem instances (§2.11, §8) — if the 67-problem subset
  cannot be identified inside the current `udell-lab/NLP4LP` HF snapshot, PaMOP's
  numbers may reference an **earlier, smaller NLP4LP release** that no longer exists in
  its original form, which would itself be worth stating plainly in any writeup.

---

## 6. Proposed faithful local reproduction architecture

No implementation was performed this pass (per task instructions: investigate/plan
only). Proposed layout, to be built **only after** the open items in §12 are resolved
or explicitly accepted as reconstruction choices:

```
baselines/pamop/
├── README.md                      # per-detail EXACT vs RECONSTRUCTED table (see below)
├── config/
│   ├── pamop_paper_faithful.yaml  # Config A — closest-possible reproduction (§7)
│   └── pamop_current_model.yaml   # Config B — modern rerun, clearly labeled
├── prompts/
│   ├── extraction_prompt.txt          # G_extr  (RECONSTRUCTED — text not in paper)
│   ├── vagueness_scoring_prompt.txt   # RECONSTRUCTED
│   ├── leaf_modeling_prompt.txt       # G_mod   (RECONSTRUCTED, eq. (3) structure EXACT)
│   ├── debug_prompt.txt               # G_exe   (RECONSTRUCTED; Self-Edit-style, per §2.7)
│   ├── reverse_translation_prompt.txt # G_rev   (RECONSTRUCTED)
│   ├── consistency_check_prompt.txt   # G_comp  (RECONSTRUCTED)
│   └── remodel_prompt.txt             # G_remod (RECONSTRUCTED)
├── partitioning/
│   ├── structured_extraction.py       # builds root node: t_o, t_c, t_v, g, vagueness scores
│   ├── bipartite_independent_sets.py  # root-level independent-set separation (§2.4a)
│   ├── constraint_clustering.py       # distance/similarity + clustering (§2.4b)
│   └── tree.py                        # PartitionTree / Node dataclasses
├── modeling/
│   ├── leaf_modeler.py                # G_mod per leaf → AMPL fragment (eq. 3)
│   └── merge.py                       # bottom-up merge → full AMPL model (eq. 4)
├── correction/
│   ├── basic_inspection.py            # regex syntax + parameter-vs-data check
│   ├── solver_debug_loop.py           # G_exe: AMPL/solver execution + self-debug (eq. 5)
│   └── reverse_translation.py         # G_rev / G_comp / G_remod loop (eq. 6)
├── validation/
│   └── ampl_solver_runner.py          # invokes AMPL+solver, parses objective/solution
├── runner/
│   └── run_pamop_nlp4lp.py            # orchestrates the full pipeline over an NLP4LP split
├── metrics/
│   └── pamop_metrics.py               # Accuracy / ExecutionRate / CERate / RERate (§2.10, exact defs)
└── results_schema.json                # per-instance record: id, tree, generated AMPL, solver log, metrics
```

Design notes:

- Mirrors this repo's existing conventions: YAML config (cf. `configs/llm_baselines.yaml`),
  a `DatasetAdapter`-shaped entry point so it can reuse `src/data_adapters/optimus.py`'s
  NLP4LP-loading conventions rather than reinventing dataset I/O, and a `tools/`-style
  standalone runner script pattern (cf. `tools/llm_baselines.py`).
- **Every** Category-C item from §2 must appear as a named, documented, overridable
  field in `config/*.yaml` — never hard-coded silently. Minimum required fields:
  `epsilon` (distance formula), `similarity_weights_by_layer`, `tfidf_top_k`,
  `glove_variant` (dimension + corpus), `clustering_algorithm`, `leaf_stop_min_constraints`,
  `leaf_stop_similarity_threshold`, `max_failed_iterations_scope` (`global` vs
  `per_node`), `solver_debug_max_tries`, `decoding.max_tokens`, `decoding.top_p`,
  `correctness_check` (`exact_objective_match` vs `tolerance`, with a numeric
  `tolerance` field), `reverse_translation_scope` (`all_constraints` vs `vague_only`).
- `baselines/pamop/README.md` must contain a literal table with three columns —
  **paper section**, **status (EXACT / RECONSTRUCTED / CONFIGURABLE-UNKNOWN)**,
  **our choice + justification** — populated from §2 of this document, kept in sync as
  choices are made.
- AMPL + solver dependency is a hard external requirement not currently present on this
  workstation (`amplpy` not installed, no AMPL binary on `PATH`, no `gurobipy`) — see §9.

---

## 7. API / model requirements

### 7.1 What the paper specifies

"GPT-4" with temperature 0.2 — **no snapshot identifier**. This is Category C: OpenAI
has shipped many models under the "GPT-4" umbrella (`gpt-4-0314`, `gpt-4-0613`,
`gpt-4-32k`, `gpt-4-turbo`, `gpt-4o`, `gpt-4.1`, `gpt-4.5`), and the paper's 62.3%/86.8%
numbers are tied to whichever one the authors actually called — which we cannot recover.

### 7.2 Current availability (checked live against OpenAI's deprecations page,
2026-08-11)

| Model | Status | Notes |
|---|---|---|
| `gpt-4-0613` (base "GPT-4") | **Scheduled shutdown 2026-10-23** | Still callable today, but not for long; this is the closest thing to "the model the paper most plausibly meant" for a 2025 IJCAI submission |
| `gpt-4-32k` | Already shut down (2025-06-06) | Not usable |
| `gpt-4-turbo` | **Scheduled shutdown 2026-10-23** | Alternative plausible candidate for "GPT-4" as of paper-writing time |
| `gpt-4o` | Available | OpenAI's own recommended successor for most retired GPT-4 variants |
| `gpt-4.1` | Available | Newer GPT-4-class model |

**Conclusion:** an exact reproduction using the *literal* model the authors used is
**not verifiable** (they didn't say which one) and is **only available for a few more
weeks** even under a best guess (`gpt-4-0613`/`gpt-4-turbo`, shutdown 2026-10-23). This
repo's `.env.example` already documents `OPENAI_API_KEY` as an optional provider, so API
access itself is not the blocker — model availability and identity are.

### 7.3 Two required configurations (per task instructions)

- **`pamop_paper_faithful`** — `model: gpt-4-0613` (best-guess closest surviving
  snapshot to "GPT-4" circa the paper's writing), `temperature: 0.2`,
  `max_failed_iterations: 5`, all other Category-C fields set to a documented
  best-effort default (§6) and flagged in the README as reconstructed, not verified.
  **Time-sensitive**: must be exercised before 2026-10-23 or this configuration becomes
  impossible to run at all without falling back further (e.g. to `gpt-4-turbo`, also
  shutting down the same date, or an even older archived snapshot if OpenAI still allows
  fine-grained selection).
- **`pamop_current_model`** — `model: gpt-4o` (or `gpt-4.1`), explicitly labeled in all
  outputs/tables as *"PaMOP reimplementation, modern rerun — not the paper's own
  numbers, not directly comparable to Table 1 of the IJCAI paper."*

Neither configuration should ever be silently substituted for the other, and neither
output should be captioned as reproducing "the PaMOP paper's numbers" — only as "our
local PaMOP-architecture reimplementation, run under configuration X."

---

## 8. NLP4LP data access

- **HF_TOKEN is set** in this workstation's environment, and a cached HF token file
  exists at `~/.cache/huggingface/token`. Token *value* was not read, printed, or
  logged anywhere in this investigation.
- This repository has a **prior, committed, successful** access-verification record:
  `results/eswa_revision/00_env/hf_access_check_runtime.md` (dated 2026-03-10, GitHub
  Actions run) confirms `udell-lab/NLP4LP` was reachable with the configured token.
- **License, from the live HF dataset card:** **CC BY-NC-SA 4.0** — non-commercial
  research use only, share-alike. (OptiMUS's own README states the slightly different
  "CC BY NC 4.0" without the share-alike clause — treat the HF card, being the operative
  license text users click through, as authoritative; flag the discrepancy rather than
  silently picking one.) This is compatible with a research reproduction baseline but
  **not** with any commercial redistribution of PaMOP-generated artifacts derived from
  it.
- **Open item, not yet resolved:** whether PaMOP's self-reported "54 LP + 13 MILP = 67
  problems" (§2.11) is identifiable as a specific named split/subfolder inside the
  current `udell-lab/NLP4LP` HF snapshot, or whether it corresponds to an earlier,
  smaller NLP4LP release (the OptiMUS lineage grew from an initial ~67-problem set in
  the 2023/2024 OptiMUS v1 paper up to ~269 problems by OptiMUS v0.3) that predates the
  dataset now hosted on HF. This must be checked by actually pulling the dataset and
  counting LP/MILP-tagged problems before any "same subset" comparison claim is made
  (§5).

---

## 9. Reproduction risk assessment

| Component | Paper spec. completeness | Reproduction confidence | Uncertainty | Proposed resolution |
|---|---|---|---|---|
| Problem representation / pipeline stages | Complete | High | None | Implement as documented (§2.1–2.2) |
| Structured extraction (`G_extr`) mechanism | Complete (what it produces) | High | Exact prompt text missing | Reconstruct prompt from stated outputs; document as RECONSTRUCTED |
| Independent-set separation (root) | Mechanism complete, algorithm choice open | Moderate | Graph-search algorithm, confidence threshold | Use connected-components as a documented, simplest-faithful choice; make configurable |
| Constraint clustering (deeper layers) | Formula given, several constants missing | Moderate | `ε`, layer weights, `k`, GloVe variant, clustering algorithm, stop thresholds | Expose all as config; pick literature-standard defaults (GloVe 6B.300d, k=10, hierarchical clustering) and document as best-effort |
| Leaf-node modeling (`G_mod`) | Formula (eq. 3) exact; prompt text missing | Moderate–High | Exact prompt wording, "principles" list | Reconstruct from CoT + eq. (3) inputs; document as RECONSTRUCTED |
| Merge (bottom-up) | Complete | High | None | Implement directly |
| Correction: basic inspection | Mechanism named, regex not given | Moderate | Exact regex patterns | Write reasonable AMPL-syntax regex checks; document as RECONSTRUCTED |
| Correction: solver-debug loop (`G_exe`) | Mechanism + equation given, prompt missing | Moderate–High | Prompt text, iteration-budget scope (global vs per-node) | Use OptiMUS-style debug-prompt template as a documented structural analog (not claimed identical); default to global budget=5 per paper's literal wording |
| Correction: reverse translation | Mechanism + equations given, prompt/scope missing | Moderate | Exact prompts for `G_rev`/`G_comp`/`G_remod`; whether it runs on all constraints or vague-only | Reconstruct; default to vague-only per Fig. 2 implication, configurable |
| Stopping criteria / hyperparameters | Temperature and max-iterations explicit | High (for these two) | `top_p`, `max_tokens`, correctness-check tolerance | Use provider defaults for unspecified decoding params; require exact-objective-match unless configured otherwise; document |
| LLM identity | "GPT-4" only, no snapshot | Low | Exact snapshot unrecoverable; base model itself sunsetting 2026-10-23 | Ship two configs (§7.3); never claim exact reproduction |
| Modeling language / solver | AMPL + Gurobi, explicit | High (spec) / Low (local availability) | Neither AMPL nor Gurobi is installed on this workstation | Acquire AMPL (Community Edition) + Gurobi (academic) licenses, or document a HiGHS/CBC-via-`amplpy` deviation explicitly, consistent with how this repo already documents its own SciPy-HiGHS deviation from Gurobi |
| NLP4LP subset identity | Problem count given (67), exact split unclear | Moderate | Whether 67-problem set is recoverable from current HF snapshot | Pull dataset, enumerate LP/MILP-tagged problems, attempt to match counts before claiming subset alignment |
| Real-world Table 2 evaluation | Scale numbers only, no reproducible text | Not reproducible | Problem text never released | Explicitly exclude from any reproduction claim |
| Metric definitions (Accuracy/ExecRate/CE/RE) | Explicit prose definitions | High | Objective-match tolerance unspecified | Implement as defined; default to exact match with a configurable tolerance, document choice |

**Overall fidelity classification: MODERATE.**

Rationale: the *architecture* (tree partitioning → self-augmented leaf modeling →
three-layer correction) is specified with enough structural and mathematical detail
(explicit equations for the core operations, explicit hyperparameters for temperature
and retry budget, explicit metric definitions) to build a genuinely faithful skeleton —
this pushes fidelity above LOW. But the complete absence of prompt text, several
numeric constants in the clustering step, the unresolved LLM snapshot identity, and the
unresolved NLP4LP-subset-identity question mean no implementation can currently claim to
reproduce the paper's *published numbers* — this caps fidelity below HIGH. A from-scratch
build under this plan should always be described as **"a faithful architectural
reimplementation of PaMOP, not a reproduction of its reported results."**

---

## 10. Estimated compute / API cost

Rough order-of-magnitude, for planning only (no run has been performed):

- Per problem, PaMOP's pipeline issues **at minimum** 1 extraction call + 1
  vagueness-scoring call + N leaf-modeling calls (N = number of leaf nodes, likely
  2–6 for NLP4LP-scale problems) + up to 5 correction-loop calls (debug/reverse-
  translation) per the stated `max_failed_iterations=5` budget — call it **8–15 GPT-4-
  class calls per problem** in the typical case, more on hard instances.
- Over the 67-problem NLP4LP subset: roughly **550–1,000 GPT-4-class calls** for one
  full `pamop_paper_faithful` run. At GPT-4-class ($/1K token) pricing with problem
  descriptions + partial-model context in each call (context grows through the
  correction loop), a single full run is a **low-tens-of-dollars** order of magnitude,
  not a major cost item — but should be budgeted and logged per the repo's existing
  `telemetry.py`/cost-tracking conventions before committing to multi-seed reruns for
  the determinism analysis proposed in §5.
- `pamop_current_model` (gpt-4o/gpt-4.1) will typically be **cheaper per call** than
  legacy GPT-4 pricing, partially offsetting the cost of running both configurations.
- If the determinism/variance analysis in §5 is pursued (≥3 reruns per configuration to
  characterize stochastic variance), multiply the above by the number of reruns.
- AMPL Community Edition and Gurobi academic licenses are free for this kind of
  non-commercial research use but are **not currently installed** and must be acquired
  before any run (§9); this is a setup cost, not a per-run API cost.

---

## 11. Reproduction-fidelity assessment (summary)

**MODERATE.** See §9 for the component-by-component table and rationale. The core
tree-partitioning-and-correction *architecture* can be reconstructed with confidence;
the exact *published numbers* (62.3% / 86.8%) cannot be reproduced or verified, only
approximated by a clearly-labeled reimplementation, because of (a) no released prompts,
(b) no disclosed GPT-4 snapshot, (c) several undisclosed clustering constants, and (d)
an unconfirmed mapping between the paper's 67-problem subset and the currently-hosted
NLP4LP HF dataset.

---

## 12. Exact next implementation steps

1. **Resolve the one open code-search lead**: retry
   `staff.ustc.edu.cn/~xiangyangli/publication.html` from a different network path (the
   `ECONNREFUSED` seen here may be workstation/network-specific, not a dead host), and
   check the personal pages of the other six authors if locatable via USTC directory or
   Google Scholar, specifically looking for a code/data availability note not present in
   the camera-ready PDF. If this turns up nothing, the "no official code" conclusion in
   §1 should be treated as final.
2. **Pull `udell-lab/NLP4LP` locally** (token already available, §8) and enumerate its
   LP/MILP-tagged problems to determine whether PaMOP's 67-problem subset (54 LP + 13
   MILP) is identifiable inside the current HF snapshot, and whether it overlaps with
   the 331-query `orig` variant this repo already evaluates on. This gates every
   claim in §5 about "the same compatible subset."
3. **Acquire AMPL (Community Edition) + a solver** — Gurobi (academic license) to match
   the paper exactly, or explicitly adopt a HiGHS/CBC-via-`amplpy` deviation consistent
   with this repo's already-documented Gurobi→SciPy-HiGHS substitution pattern for its
   own paper-core results. Decide and document before writing any solver-execution code.
4. **Draft `baselines/pamop/README.md`'s exact-vs-reconstructed table** from §2/§9 of
   this document as the first concrete file, before any Python module — this operationalizes
   "every unspecified detail must be configurable and documented" as an artifact
   reviewers can check against the paper directly.
5. **Implement `baselines/pamop/` per §6**, gated behind the two configs in §7.3, with
   `pamop_paper_faithful` prioritized given the `gpt-4-0613`/`gpt-4-turbo` shutdown date
   of 2026-10-23 — if this reproduction is wanted at all under something resembling the
   original model, the window to do so is closing.
6. **Only after (1)–(5)**: run both configurations on the confirmed-overlapping NLP4LP
   subset from step 2, report our own measured Accuracy/ExecutionRate/CE/RE numbers
   (never captioned as "PaMOP's numbers"), and separately compute the
   parameter-grounding cross-metric described in §5 (parse PaMOP-generated AMPL+data
   back into scalar values, score with our own `Coverage`/`TypeMatch`/
   `InstantiationReady`) as the one genuinely apples-to-apples comparison point against
   our existing paper-core pipeline.
7. Do **not** implement a speculative approximation of PaMOP for numeric comparison
   purposes before steps 1–4 are complete, per the original task's explicit instruction.
