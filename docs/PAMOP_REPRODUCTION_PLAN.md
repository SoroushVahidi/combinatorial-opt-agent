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

**Update, 2026-08-11 (same day, follow-up pass):** environment verification (HF
NLP4LP access, Gurobi) and NLP4LP subset-alignment analysis added — see
[§13 NLP4LP Subset Alignment and Environment Verification](#13-nlp4lp-subset-alignment-and-environment-verification).
Both original blockers (HF access, Gurobi) are **resolved**; the exact-subset
question is now precisely characterized (not fully resolved — see §13.5).

**Update, 2026-08-11 (same day, second follow-up):** Milestone 1 implemented
— non-LLM question-partitioning scaffold under `baselines/pamop/`, 35 passing
unit tests, a live smoke run against `pamop_possible_269`, and a
suspicious-overlap check against our own manuscript (nothing concerning
found) — see
[§14 Implementation Status: Milestone 1](#14-implementation-status-milestone-1-non-llm-partitioning-scaffold).

**Update, 2026-08-11 (same day, third follow-up):** Milestone 2 implemented
— LLM-based structured-extraction stage (`G_extr`), a provider-agnostic LLM
interface (OpenAI/Gemini/Cohere/Fireworks/CloudRift), a fixed NLP4LP loader
gap, and a tiny live smoke test. Found that this workstation's "OpenAI"
credentials are actually a CloudRift key and its Gemini credentials are
empty strings despite both env vars being present — documented, not hidden.
See
[§15 Implementation Status: Milestone 2](#15-implementation-status-milestone-2-llm-extraction-stage).

**Update, 2026-08-11 (same day, fourth follow-up):** Milestone 3 implemented
— `G_mod` (eq. 3), the bottom-up merge (eq. 4), and a working Azure OpenAI
provider (`gpt-4.1-mini`, confirmed live, now the primary paper-faithful
path). Full pipeline (`G_extr` → partition → `G_mod` → merge) verified live
on one real NLP4LP problem in this pass, no AMPL yet. See
[§16 Implementation Status: Milestone 3](#16-implementation-status-milestone-3-self-augmented-modeling-and-merge).

**Update, 2026-08-11 (same day, fifth follow-up):** Final core execution
milestone implemented — AMPL rendering/execution, `G_exe`, reconstructed
`G_rev`/`G_comp`/`G_remod`, and the bounded correction loop. AMPL/Gurobi
works through a user-local `amplpy` install, and a one-problem live NLP4LP
smoke test solved after one correction iteration. See
[§17 Implementation Status: Milestone 4](#17-implementation-status-milestone-4-ampl-execution-and-correction-loop).

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
- **Resolved (was "open item"), see §13:** PaMOP's "54 LP + 13 MILP = 67 problems"
  (§2.11) is now identified precisely as the NLP4LP release dated **2024-05-13** (named
  and dated on the dataset's own version-history page, and matching OptiMUS v2's exact
  wording verbatim). It predates, and is not a labeled split of, the current
  `udell-lab/NLP4LP` HF snapshot (361 problems as of this investigation). §13 works out
  exactly which of our 331 catalog entries could vs. could not contain it.

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
| Modeling language / solver | AMPL + Gurobi, explicit | High (spec) / **Gurobi: High (verified working, §13.2, no longer a blocker) / AMPL: Low (still not installed, §13.6)** | Whether AMPL itself is essential to the *method* (yes, per §13.6) vs. replaceable for solving only | Install AMPL Community Edition (free, sufficient for NLP4LP-scale problems), or document a direct-`gurobipy` deviation explicitly for the leaf/merge generation target, consistent with how this repo already documents its own SciPy-HiGHS deviation from Gurobi |
| NLP4LP subset identity | Problem count given (67), exact split unclear | **Now precisely characterized, §13** — Moderate | *Which* 67 of the 269 structurally-possible catalog entries; original 2024-05-13 archived release is behind two interactive-only walls (reCAPTCHA, OpenReview bot-challenge) | Use the 269-entry `POSSIBLE_MATCH` block in `data/baselines/pamop/nlp4lp_pamop_subset.jsonl` as an evidence-bounded proxy (protocol **C**, §13.7); attempt manual archived-snapshot retrieval to upgrade to exact IDs |
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

1. **Resolve the open leads, now three of them**: (a) retry
   `staff.ustc.edu.cn/~xiangyangli/publication.html` from a different network path (the
   `ECONNREFUSED` seen here may be workstation/network-specific, not a dead host); (b)
   manually (interactively, solving the human-verification challenge) retrieve the
   archived 2024-05-13 "67 instances" NLP4LP release from
   `nlp4lp.vercel.app` (client-rendered, reCAPTCHA-gated — not fetchable by this
   automated investigation, §13.5); (c) manually retrieve the OpenReview supplementary
   material at `openreview.net/forum?id=HobyL1B9CZ` (bot-challenge-gated, same
   limitation). (b) or (c) succeeding would let us exactly identify PaMOP's 67 problems
   by content-matching against our catalog, upgrading §13's "POSSIBLE_MATCH" rows to
   real matches. If all three turn up nothing, the "no official code" conclusion in §1
   stands, and the manifest in §13 (269 possible / 62 no-match, 0 exact) becomes the
   permanent ceiling on subset-alignment precision.
2. ~~Pull `udell-lab/NLP4LP` locally and enumerate its LP/MILP-tagged problems~~ —
   **done, see §13.** Result: PaMOP's 67-problem subset traces to a specific named,
   dated NLP4LP release (2024-05-13, 54 LP + 13 MILP, per OptiMUS v2) that is **not**
   recoverable by ID from the current HF snapshot; 269 of our 331 catalog entries are
   structurally eligible to contain it (existed pre-publication), the other 62 are
   provably excluded (added 2026-02, after PaMOP was published). Exact membership
   within the 269 remains unresolved pending manual retrieval of the archived release
   (§13.5, §13.9 step 1).
3. **Acquire AMPL (Community Edition)** — Gurobi is **already installed and verified**
   (§13.2; not a blocker), but AMPL/`amplpy` itself is still absent (§13.6). Install it,
   or explicitly adopt the direct-`gurobipy` deviation described in §13.6, consistent
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
6. **Only after (1)–(5)**: run both configurations on the 269-entry
   `POSSIBLE_MATCH` block from `data/baselines/pamop/nlp4lp_pamop_subset.jsonl`
   (catalog indices 0–268 / HF ids 1–269 — the best-available, evidence-bounded proxy
   for PaMOP's actual 67, per protocol classification **C** in §13.7) *and* on the full
   331-query canonical benchmark, report our own measured Accuracy/ExecutionRate/CE/RE
   numbers on both (never captioned as "PaMOP's numbers"), and separately compute the
   parameter-grounding cross-metric described in §5 (parse PaMOP-generated AMPL+data
   back into scalar values, score with our own `Coverage`/`TypeMatch`/
   `InstantiationReady`) as the one genuinely apples-to-apples comparison point against
   our existing paper-core pipeline.
7. Do **not** implement a speculative approximation of PaMOP for numeric comparison
   purposes before steps 1, 3, and 4 are complete, per the original task's explicit
   instruction.

---

## 13. NLP4LP Subset Alignment and Environment Verification

**Follow-up investigation, 2026-08-11.** Data-alignment and environment-verification
pass only — no PaMOP code implemented, no manuscript file touched, no existing
benchmark result changed. This section is additive to §§1–12 above; where it changes an
earlier conclusion, the original section was edited in place with a pointer back here.

### 13.1 Machine / repository state

- Hostname: `al-khwarizmi` (same local workstation as the original investigation).
- Repository: `~/combinatorial-opt-agent`, branch `main`, working tree clean before this
  pass began, up to date with `origin/main` (`git fetch` confirmed no upstream changes).
- No reset, discard, or destructive git operation was used.

### 13.2 Hugging Face NLP4LP access — PASS

- **Local HF dataset cache checked first:** no `NLP4LP` entry existed under
  `~/.cache/huggingface/{datasets,hub}` — nothing was cached, so a fresh metadata pull
  was necessary (metadata/listing only, not a bulk file download).
- **Authentication:** the workstation's existing `HF_TOKEN` (already configured, not
  touched, never printed) successfully authenticated against `HfApi().dataset_info(...)`
  for `udell-lab/NLP4LP`. **No token value was read, echoed, or logged anywhere in this
  investigation.**
- **Current snapshot (`sha af17e517...`, last modified 2026-04-20):** 2,127 files → **361
  distinct problem directories** (`data/1/` … `data/361/`, gaps filled by
  `-infeasible`/`-unsolved` suffixed variants), plus a root-level aggregated
  `train.jsonl` (361 rows, one per problem) with fields
  `id, type, description, parametrized_description, keywords, parameters, problem_info,
  optimus_code, solution`.
- **License (from the live HF dataset card):** **CC BY-NC-SA 4.0**, non-commercial
  research use, gated behind a click-through contact-info agreement — consistent with
  §8's original finding, now directly re-confirmed via the API rather than a page
  fetch.
- **Difficulty/type labels:** `metadata.json`'s `type` field is a *difficulty* label
  (`easy` / `hard` / `case study`), **not** an LP/MILP label — LP vs. MILP is only
  recoverable from a `# Problem type: LP|MIP` comment embedded in each problem's
  `optimus-code.py` / `optimus_code` field. Across the current 361-problem snapshot:
  LP = 134, MIP = 169, no-comment/`UNKNOWN` = 58 (mostly `hard`/`case study` items).

### 13.3 Gurobi verification — PASS

Gurobi was **not** on the default `python3` (a `modal-venv`), which is why a naive
`import gurobipy` looked like "not installed." A dedicated environment was found and
used instead: `~/.venvs/gurobi` (Python 3.12, `gurobipy==13.0.2`), plus a license file
at `~/gurobi.lic` (permissions `600`, contents never read or printed).

| Check | Result |
|---|---|
| Installed | **YES** (`~/.venvs/gurobi`) |
| Python import works | **YES** |
| License works | **YES** — WLS Academic license, ID printed by Gurobi itself at solve time (non-commercial use only); no license file contents were read or echoed by this investigation |
| Trivial solve works | **YES** — `max x+y s.t. x+y<=5, x,y in [0,10]` solved to `Status=OPTIMAL`, `ObjVal=5.0` |
| Version | **13.0.2** |

**Gurobi is removed from the PaMOP blocker list** (§13.9). It was never actually
missing — the original report's "neither Gurobi nor amplpy importable" check (§9 of the
original pass) had simply run against the wrong Python environment.

### 13.4 What PaMOP's "67 problems" precisely means — resolved

PaMOP's Table/§4.3 text ("54 LP problems and 13 MILP problems") is a **verbatim
citation**, not an independent count. Cross-checking the actual OptiMUS v2 paper text
(`arXiv:2402.10172`, downloaded and converted with `pdftotext`, since PaMOP cites
"[AhmadiTeshnizi et al., 2024]" for NLP4LP) finds the identical phrase:

> "...create NLP4LP (Natural Language Processing for Linear Programming), a benchmark
> consisting of **54 LP and 13 MILP problems (67 instances in total)**." — OptiMUS v2,
> §"NLP4LP" (arXiv:2402.10172, lines 399–403 of the extracted text)

This is corroborated independently by the dataset's own (now largely dead, JS-rendered)
version-history page, `nlp4lp.vercel.app`, whose captured text describes three
historical releases:

| Date | Total instances | Note |
|---|---|---|
| 2023-09 | 40 | initial release |
| 2023-10 | 51 | +11 from a Linear and Convex Optimization textbook |
| **2024-05-13** | **67** | +16 more (page text also mentions ComplexOR/NL4OPT components alongside this release — unverified secondary detail, see caveat below) |

**Conclusion:** PaMOP's "67 problems" = **the NLP4LP dataset exactly as it existed on
2024-05-13** (54 LP + 13 MILP), the version OptiMUS v2 itself introduced and evaluated
on. It is a **named, dated, external** release — not a split label inside the dataset
repository, and not something PaMOP's own authors selected or filtered themselves.
PaMOP simply reused the dataset as OptiMUS v2 had already defined and sized it. There is
no train/test split distinction at this size (67 was the *entire* dataset at that
point); "test" as a split name only appears later, once the dataset grew past its
original single-block form.

*Caveat:* the `nlp4lp.vercel.app` page is a client-rendered SPA behind reCAPTCHA;
WebFetch could only recover its rendered text, not its underlying download links or raw
markup (confirmed via direct `curl`, which returned only a reCAPTCHA loader with no
usable content). The "plus ComplexOR and NL4OPT components" phrase attached to the
2024-05-13 row in the fetched summary could not be independently verified against raw
HTML and is flagged here as a lower-confidence secondary detail — it does not affect the
54/13/67 figure itself, which is independently confirmed by the OptiMUS v2 paper text
verbatim.

### 13.5 Attempted retrieval of the original 67-problem release — blocked, not abandoned

Two possible sources for the exact archived 2024-05-13 file set were identified and
attempted:

1. **`nlp4lp.vercel.app`** — confirmed to be a Next.js SPA; `curl` against it returns only
   `recaptcha/api.js` and an Amazon textbook link, no dataset content or API endpoint in
   the static HTML. Downloading requires a real browser and passing a reCAPTCHA
   challenge — **not achievable by this automated, non-interactive investigation.**
2. **OpenReview supplementary material**, `openreview.net/forum?id=HobyL1B9CZ`
   (referenced by OptiMUS's own `optimus-v0.2` branch README as a second, alternate
   distribution channel for the original NLP4LP + ComplexOR data) — confirmed to return
   an OpenReview bot-challenge page (`ChallengeRequiredError`, HTTP 403) from both
   `curl` and the OpenReview API directly. Also **not achievable non-interactively.**

Both are **legitimate, likely-live leads for a human to pursue manually** (retry with an
authenticated interactive browser session), not dead ends — they are listed as the
first next step in the updated §12.

### 13.6 Our current 331-query pipeline, traced exactly

The claim in the original report (§5) that our 331 "queries" might represent multiple
*query variants* per underlying problem was **incorrect and is corrected here**: they
are one-query-per-problem, same granularity as PaMOP. Specifically:

- Canonical catalog: `data/catalogs/nlp4lp_catalog.jsonl`, 335 rows total = **331 rows
  with `meta.split == "test"`** (`doc_id` = `nlp4lp_test_0` … `nlp4lp_test_330`,
  sequential `meta.index` 0–330) **+ 4 rows with `meta.split == "handcrafted"`**
  (`product_mix_lp`, `diet_lp`, `investment_lp`, `transportation_lp` — public-source,
  non-NLP4LP entries added post-manuscript, per `docs/RESULTS_PROVENANCE.md`). The
  manuscript's "331-entry catalog" = exactly this `test` block.
- Each `test` entry's `text` field is the problem's **parametrized description**
  (named-placeholder form, e.g. `"Maximize the sum of ProfitPerDollarCondos..."`), not
  the raw natural-language description — confirmed by direct comparison against the HF
  `problem_info.json:parametrized_description` field for several ids (below).
- **Exact index-to-HF-id mapping** (established by direct text comparison at 8 points
  spanning the full range — indices 0, 1, 50, 100, 150, 200, 268, 269, 270, 300, 320,
  329, 330 — all consistent with a single two-piece arithmetic rule, no exceptions
  found):

  ```
  catalog index i,   0 <= i <= 268   ->  HF problem id (i + 1)     [ids   1 .. 269]
  catalog index i, 269 <= i <= 330   ->  HF problem id (i + 24)    [ids 293 .. 354]
  ```

  i.e. our 331-query catalog = HF ids **{1..269} ∪ {293..354}**, explicitly **excluding**
  HF ids 270–292 (a 23-problem "dev" split) and 355–361 (a 7-problem "case study" set).
  This reproduces 269 + 62 = 331 exactly and was verified, not merely arithmetically
  inferred: catalog index 268 and HF id 269 both start "A dessert shop produces..."; index
  269 and HF id 293 both start "A daycare center has children..."; index 100/HF id 101
  ("crab cakes and lobster rolls"); index 200/HF id 201 ("laminate planks"); etc.
- **Timing, from the HF repo's own commit history** (`api.list_repo_commits`, 53 commits
  total): ids 1–269 existed continuously from the dataset's initial HF upload
  (2024-11-02) onward — i.e. throughout PaMOP's entire research-and-publication window.
  Ids 270–292 ("Add dev split with 23 instances") and 293–354 ("Add 62 new test
  instances... with full problem formulations") were both added **2026-02-12**; ids
  355–361 ("Add case study instances") were added **2026-02-27** — all six months **after**
  PaMOP's IJCAI 2025 publication (Aug 2025). **These 62+23+7 = 92 newer problems cannot,
  under any interpretation, be part of PaMOP's evaluation set.**
- Implication: **269 of our 331 catalog entries (indices 0–268) are structurally
  old enough to possibly contain PaMOP's 67; the other 62 (indices 269–330) provably
  cannot.**

### 13.7 Mapping PaMOP's 67 to our data — manifest and match counts

A reproducible script, `tools/pamop_nlp4lp_alignment.py`, was written to generate the
mapping deterministically (not by hand), producing
`data/baselines/pamop/nlp4lp_pamop_subset.jsonl` (331 rows, one per catalog `test`
entry; **no gated NLP4LP text included** — only ids, indices, and structural provenance
fields per row: `pamop_problem_identifier`, `current_nlp4lp_catalog_doc_id`,
`current_nlp4lp_catalog_index`, `current_nlp4lp_hf_problem_id`, `lp_or_milp`,
`text_match_status`, `schema_match_status`, `optimus_code_available`,
`mapping_confidence`, `evidence`, `notes`).

Because §13.5's two retrieval attempts were both blocked, **no content-level matching
against PaMOP's actual 67 problems was possible** — `lp_or_milp` and
`pamop_problem_identifier` are left `null`/unresolved throughout, and every row's
`text_match_status`/`schema_match_status` is `NOT_ATTEMPTED` (there is nothing to match
*against* yet). What the manifest **does** establish, per the task's required
classification scheme:

| Classification | Count | Meaning |
|---|---|---|
| **EXACT MATCH** | **0** | No archived 2024-05-13 snapshot was retrievable to confirm any single problem's identity |
| **HIGH-CONFIDENCE MATCH** | **0** | Same reason |
| **POSSIBLE MATCH** | **269** | Catalog indices 0–268 (HF ids 1–269) — old enough to possibly be one of PaMOP's 67; membership not further narrowed |
| **NO MATCH** | **62** | Catalog indices 269–330 (HF ids 293–354) — added 2026-02, six months after PaMOP was published; provably not PaMOP's problems |
| **UNRESOLVED** | *(not a row count — an open question)* | Exactly *which* 67 of the 269 possible entries are PaMOP's; requires either the archived 2024-05-13 file set (§13.5) or a direct research-question to the authors |

### 13.8 Relationship between PaMOP's 67 and our 331 — explicit answers

- **Are PaMOP's 67 a subset of our 331 test queries?** Almost certainly **partially**,
  and possibly **entirely**, but not confirmed exactly. All 67 must be drawn from HF ids
  ≤ 354 that existed pre-publication — i.e. from within the 269-id block ids 1–269 (our
  indices 0–268) — since that block has existed unchanged since before PaMOP's dataset
  even reached 269 total problems (it grew from the original 40→51→67→…→269 sequence
  entirely before the Nov 2024 HF upload). We cannot rule out that a small number of the
  original 67 were later renamed, deduplicated, or dropped somewhere in that growth
  process (some churn is visible in the `-infeasible`/`-unsolved` suffix pattern at ids
  like 3, 28, 51, 58 etc.), so "possibly all 67, likely most" is the honest statement,
  not a guaranteed 67-for-67 subset.
- **Does each PaMOP problem correspond to multiple query variants?** **No** — both
  benchmarks are one-query-per-problem; this corrects the original report's speculative
  claim to the contrary (§13.6).
- **Is there a one-to-many or many-to-one mapping?** No — it is a straightforward
  problem-to-problem correspondence where one exists at all.
- **Are the 67 from a different NLP4LP version?** **Yes, precisely characterized**: the
  named 2024-05-13 release, a strict, much smaller predecessor of the current 361-problem
  snapshot (§13.4).
- **Are any problems missing from the currently accessible snapshot?** Cannot be ruled
  out — see the churn caveat above — but the dataset's growth pattern (versions only ever
  add, historically) makes wholesale removal unlikely; partial renumbering/relabeling
  (e.g. `id` → `id-infeasible`) is the more probable form of "missing," and such
  relabeled entries are still present in the snapshot, just not under their original bare
  numeric id.
- **Quantified:** 0 exact, 0 high-confidence, 269 possible, 62 no-match, "which 67 of the
  269" unresolved.

### 13.9 Fair-comparison protocol classification

Per the task's A/B/C/D scheme: **C — only partial overlap** (a bounded superset of the
true overlap is known; the exact overlap is not).

Following the task's instruction for classification C, **both** protocols are defined,
using `data/baselines/pamop/nlp4lp_pamop_subset.jsonl`:

1. **Matched-subset comparison (proxy, not exact):** run/evaluate on the 269
   `POSSIBLE_MATCH` rows only (catalog indices 0–268). This is a **superset** of PaMOP's
   actual 67 — any comparison run on it should be captioned "PaMOP-era-eligible subset
   (n=269), a superset bound on PaMOP's actual 67-problem evaluation set, not the exact
   set" — never captioned as "the same 67 problems."
2. **Our full canonical 331-query evaluation** — unchanged, continues to be the
   manuscript's authoritative benchmark, unaffected by any of this PaMOP-alignment work.

Numeric results from (1) must **never** be presented as directly reproducing or
matching PaMOP's published Table 1 numbers (62.3% / 86.8%) — per §5's existing rule,
different task, different metric, and now also confirmed: different (larger,
superset) problem count.

### 13.10 Is AMPL actually required?

Re-reading §2.5/§2.9/§2.7 of this report's own paper extraction with this specific
question in mind:

- AMPL is **not** merely PaMOP's execution/validation environment — it is the
  **explicit generation target for the LLM itself**, adopted for a stated
  methodological reason: *"LLMs treat mathematical formulas, modeling languages, and
  programming languages as different 'languages'... we directly generate code in the
  modeling language instead of formulas."* The leaf-node modeling step (`G_mod`, eq. 3)
  and the merge step (eq. 4) both target AMPL syntax directly, and the paper's
  Compile-Error-rate / Runtime-Error-rate metrics are specifically about AMPL-vs-Gurobi
  compile/runtime failures.
- Because of that, swapping AMPL for direct `gurobipy` Python generation is **not** a
  transparent, fidelity-neutral substitution — it changes what the LLM is asked to
  produce (AMPL model statements vs. Python API calls), which plausibly changes the
  error taxonomy the CE-rate/RE-rate metrics are measuring, even if the two
  representations are mathematically equivalent once correct.
- Solving itself has no such constraint: once a valid model exists, Gurobi can solve an
  AMPL-formulated model (via `amplpy` + the AMPL executable driving Gurobi, matching the
  paper) or a `gurobipy`-native model with identical mathematical content, with identical
  final objective values.

**AMPL required: YES for a paper-faithful (`pamop_paper_faithful`) reproduction** — it is
a deliberate part of the method, not an incidental tooling choice, and dropping it would
compromise comparability of the CE-rate/RE-rate metrics specifically (not necessarily
final Accuracy, which only depends on the objective value once a model is valid).
**NO — replaceable — for a `pamop_current_model` variant**, where direct `gurobipy`
generation is an acceptable, clearly-labeled deviation if `amplpy`/AMPL licensing proves
inconvenient, at the cost of CE-rate/RE-rate not being strictly comparable to the
paper's own breakdown (Accuracy and Execution-Rate remain comparable in spirit, since
those depend on final solve success, not on which language got there).

AMPL was **not installed** as part of this investigation, per the task's explicit
instruction not to install/reconfigure software beyond what's needed to answer the
question — the above is an assessment, not an implementation.

### 13.11 Updated blocker list

| Item | Classification | Status |
|---|---|---|
| Hugging Face NLP4LP access | ~~Blocker~~ | **RESOLVED** — auth confirmed working (§13.2) |
| Gurobi | ~~Blocker~~ | **RESOLVED** — installed, licensed, verified with a trivial solve (§13.3) |
| AMPL / `amplpy` | **BLOCKER** (for `pamop_paper_faithful` only) | Still not installed; required for methodological fidelity per §13.10; not required for `pamop_current_model` |
| Exact identity of PaMOP's 67 problems | **NON-BLOCKING UNCERTAINTY** | Bounded to 269 possible candidates (§13.7); does not block starting implementation, only blocks claiming an exact-subset comparison until §13.5's leads are pursued manually |
| Missing prompt templates (`G_extr`, `G_mod`, `G_exe`, `G_rev`, `G_comp`, `G_remod`) | **IMPLEMENTATION CHOICE** | Must be reconstructed and documented per §6/§9; not blocking, just requires disclosure |
| Exact GPT-4 snapshot | **IMPLEMENTATION CHOICE** (time-boxed) | Two configs already specified (§7.3); `gpt-4-0613`/`gpt-4-turbo` shut down 2026-10-23 |
| Clustering constants (`ε`, layer weights, TF-IDF `k`, GloVe variant, clustering algorithm, stop thresholds) | **IMPLEMENTATION CHOICE** | Must be configurable per §6; document defaults |
| Correctness-check tolerance | **IMPLEMENTATION CHOICE** | Default to exact match, configurable |
| Correction-loop iteration scope (global vs. per-node) | **IMPLEMENTATION CHOICE** | Default to global per paper's literal wording (§2.8) |

### 13.12 Can implementation now begin with scientifically defensible fidelity?

**YES — MODERATE confidence.**

Not HIGH, because the exact 67-problem evaluation set is still unresolved (§13.7) and
several prompt-level and clustering-constant details remain reconstructions rather than
verified originals (unchanged from §9's original assessment). Not NO, because every
remaining item is now either resolved (HF access, Gurobi) or classified as a
documentable implementation choice / non-blocking uncertainty — nothing on the list
requires information that is fundamentally unobtainable to *start* building, only to
claim exact numeric reproduction.

**Recommended first implementation milestone:** build
`baselines/pamop/README.md` (the exact-vs-reconstructed table, §6/§9/§13.11 consolidated)
and `baselines/pamop/config/*.yaml` skeletons first — zero solver/LLM dependency, pure
documentation-as-code — followed immediately by `partitioning/structured_extraction.py`
and `partitioning/constraint_clustering.py` against the **269-entry `POSSIBLE_MATCH`
block**, since that is the only part of the pipeline that has no AMPL/Gurobi dependency
at all and can be fully unit-tested today without installing anything further. AMPL
acquisition (§13.10) can proceed in parallel and gates only the later
`modeling/`/`correction/`/`validation/` stages.

---

## 14. Implementation Status: Milestone 1, Non-LLM Partitioning Scaffold

**Follow-up implementation pass, 2026-08-11 (same day, second follow-up).**
Builds the first concrete milestone identified in §13.12. No manuscript file
touched, no existing benchmark result changed, no LLM/AMPL/solver call
implemented.

### 14.1 What was built

`baselines/pamop/` — an independent, unofficial reproduction scaffold:

| File | Status | Content |
|---|---|---|
| `README.md` | Done | Disclosure, citation, published-numbers-are-not-our-numbers rule, dataset-scope rule, reconstructed-choices table |
| `config.py` | Done | Typed config schema; `PamopConfig.require()` raises `UnspecifiedPaperDetailError` rather than silently defaulting |
| `configs/paper_faithful.yaml` | Done | Paper-stated values only (`temperature=0.2`, `max_correction_iterations=5`, `generation_target=AMPL`, `solver_backend=gurobi_via_ampl`); everything else `null` |
| `configs/reconstructed_default.yaml` | Done | Every field filled, each non-paper-specified value tagged `# REPRODUCTION CHOICE` with a one-line reason |
| `representations.py` | Done | `StructuredProblem`/`ConstraintInfo`/`VariableInfo` (the paper's t_o/t_c/t_v/g root-node content); built from NLP4LP's own structured fields, not an LLM call |
| `data.py` | Done | NLP4LP loader restricted to `SUBSET_POSSIBLE_269` (ids 1–269 only); `assert_not_post_pamop()` guard; no `"pamop_67"` subset exists anywhere |
| `partition.py` | Done | Independent-set separation (root) + constraint clustering (deeper layers), eq. (2) distance formula exact, `PartitionTree`/`PartitionNode` with `to_dict`/`from_dict`/`to_json` |
| `run_partitioning.py` | Done | Diagnostics-only CLI; never writes raw problem text |
| `tests/test_config.py`, `tests/test_data.py`, `tests/test_partition.py` | Done | 35 tests, synthetic data only, no gated text committed |

### 14.2 PaMOP stages now implemented

Only **paper section 3.2, "Construction of the Partition Tree"**, minus its
own LLM-based structured-extraction sub-step:

- Independent-set separation at the root (bipartite constraint–variable
  graph, keyword-match-confidence edges, connected components).
- Constraint clustering at deeper layers (eq. 2 distance formula, exact;
  three-signal combined similarity — adjacency, TF-IDF keyword overlap,
  embedding cosine — with configurable per-layer weights).
- Recursive leaf-stop condition ("small number of constraints or highly
  similar ones," both thresholds configurable).

**Not implemented:** structured extraction via LLM (`G_extr`), vagueness
scoring, self-augmented leaf modeling (`G_mod`), AMPL generation, the full
correction loop (`G_exe`/`G_rev`/`G_comp`/`G_remod`), solver execution, and
all four published evaluation metrics (Accuracy/Execution-Rate/CE-rate/
RE-rate) — see `baselines/pamop/README.md` "Not implemented yet".

### 14.3 Paper-faithful vs. reconstructed, by equation/component

| Component | Fidelity |
|---|---|
| Eq. (2) distance formula (`d_ij = 1/(s_ij+eps) - 1/(1+eps)`) | **Exact**, paper-specified |
| Independent-set separation mechanism (bipartite graph, keyword-confidence edges) | **Exact mechanism**, paper-specified |
| Clustering distance metric inputs (adjacency, TF-IDF keyword overlap, embedding cosine) | **Exact three signals**, paper-specified |
| Independent-set graph-search algorithm (which one) | Reconstruction choice: connected components |
| Clustering algorithm consuming the distance matrix | Reconstruction choice: agglomerative, average linkage |
| Vector-similarity source | Reconstruction choice: TF-IDF-cosine fallback (paper: GloVe, unavailable locally) |
| `epsilon`, `tfidf_top_k`, per-layer weights, leaf-stop thresholds, bipartite edge threshold | Reconstruction choices, all in `reconstructed_default.yaml`, all `null` (and load-time-erroring) in `paper_faithful.yaml` |
| Structured-representation source (t_o/t_c/t_v/g) | Reconstruction choice: NLP4LP's own fields, not `G_extr` |

### 14.4 Unresolved parameters (unchanged from §9/§13, not affected by this milestone)

Exact PaMOP 67-problem membership; all LLM prompt templates
(`G_extr`/`G_mod`/`G_exe`/`G_rev`/`G_comp`/`G_remod`); exact GPT-4 snapshot;
GloVe variant; correctness-check tolerance; correction-loop iteration scope.
None of these block *this* milestone (which touches none of the LLM/AMPL/
solver stages) — they block the *next* one.

### 14.5 Validation results

- **Unit tests:** `python -m pytest baselines/pamop/tests -v` → **35 passed**,
  0 failed, 0 skipped, ~0.8s. Covers: deterministic partitioning (identical
  output across repeated builds), valid parent/child relationships, no
  cycles + full reachability from root, every constraint assigned to exactly
  one leaf, independent-set separation actually splitting disjoint synthetic
  blocks, trivial single-leaf collapse, non-decreasing depth, JSON
  serialization round-trip, no raw constraint text leaking into the
  serialized tree, config loading (both configs), `paper_faithful.yaml`
  correctly raising `UnspecifiedPaperDetailError`, the `pamop_possible_269`
  id range, `PostPamopIdError` on ids ≥ 270, and — explicitly — that
  `"pamop_67"` is rejected as an unknown subset name.
- **Smoke run** (`run_partitioning.py`, live HF-authenticated NLP4LP access,
  `reconstructed_default.yaml`, `--subset pamop_possible_269`):
  - n=12: 12/12 succeeded, 0 failures, 0 determinism mismatches, avg tree
    depth 0.33, avg node count 2.08, avg leaf count 1.75.
  - n=40: 39/40 succeeded (3.9s wall time), 1 failure
    (`RemoteEntryNotFoundError`), 0 determinism mismatches, avg tree depth
    0.26, avg node count 1.79, avg leaf count 1.54.
  - The one failure is a **known, documented loader gap**, not an algorithm
    bug: a handful of ids in 1–269 exist on HF only under a suffixed path
    (e.g. `3-infeasible`, `28-unsolved`) that `data.py` does not yet resolve
    — recorded in `baselines/pamop/README.md` "Known limitation."
  - Shallow average tree depth reflects real NLP4LP data (most "easy"
    problems have few enough constraints to satisfy the leaf-stop condition
    immediately) — this is honest output, not a tuning artifact.
  - No raw NLP4LP problem text was written to any committed file; smoke-run
    JSON output was written only to a local scratch path, never git-added.

### 14.6 Suspiciously-close prior-work overlap check — **nothing concerning found**

Checked the PaMOP paper against this repository's own manuscript
(`manuscript/main.tex`) and pipeline design for the specific overlap axes
requested: retrieval-assisted schema selection, typed numeric grounding,
schema-conditioned scalar instantiation, structural verification,
retrieval-vs-grounding bottleneck analysis, evaluation metrics, and
distinctive terminology/pipeline decomposition.

- **Mechanism-level comparison:** our manuscript's own Related Work section
  (`manuscript/main.tex`, §"Related Work") already explicitly groups and
  distinguishes itself from the *family* PaMOP belongs to — LLM-generative,
  end-to-end optimization modeling (it names OptiMUS, OptLLM,
  Chain-of-Experts, and LLMOPT by name in this role) — stating our method
  instead does "a narrower intermediate task: retrieval-assisted schema
  grounding followed by scalar parameter instantiation," fully
  deterministic, no LLM component in the benchmarked pipeline. PaMOP itself
  is not cited there (plausibly missed during Related Work drafting, not
  evidence of anything), but it is mechanically a member of exactly the
  generative-family group already being contrasted against, not an
  individually-borrowed idea.
- **"Structured representation" vs. "schema retrieval":** superficially
  similar phrasing, mechanically opposite: PaMOP *generates* a fresh
  structured representation per problem via an LLM call (`G_extr`), with no
  fixed catalog and no retrieval step; our pipeline *retrieves* from a fixed
  335-entry schema catalog via TF-IDF/BM25/LSA, with no generative step at
  all in the benchmarked path. Not the same idea wearing different words.
- **"Basic inspection" vs. structural verification:** PaMOP's regex-based
  syntax/parameter check before solving and our `formulation/verify.py`
  structural LP-consistency check are both instances of the generic,
  independently-obvious idea "sanity-check before an expensive solve" — a
  common pattern across this entire subfield (OptiMUS does the same via
  `execute_and_debug`'s pre-solve checks), not a distinctive borrowed
  mechanism.
- **Retrieval-vs-grounding bottleneck analysis:** this specific analytical
  contribution (isolating whether schema retrieval or scalar grounding is
  the dominant bottleneck, with an oracle-schema ablation) is unique to our
  manuscript; PaMOP's ablation studies a different axis entirely (module
  contribution to solver-outcome accuracy: prompt-only vs. +partition vs.
  +correction). No overlap.
- **Evaluation metrics:** disjoint families by construction — PaMOP's
  Accuracy/Execution-Rate/CE-rate/RE-rate are solver-outcome metrics; ours
  (Schema R@1/Coverage/TypeMatch/InstantiationReady) are retrieval-and-
  grounding metrics. No shared definitions or naming.
- **Chronology:** PaMOP was published at IJCAI 2025 (main track, Aug 2025).
  Our manuscript's current KAIS-targeted form postdates that (retargeted
  from EAAI to KAIS in July 2026, per `docs/KAIS_SOURCE_OF_TRUTH.md`), so if
  anything the chronological question runs the other direction (could our
  work have been influenced by PaMOP, not the reverse) — and given the
  mechanism differences documented above, the answer is that both papers
  independently target the same public benchmark (NLP4LP) and the same
  general problem domain (NL-to-optimization), which is expected and
  unremarkable in this crowded, active subfield (also shared by OptiMUS,
  Chain-of-Experts, LLMOPT, OPT2CODE, Ner4Opt, and others our own Related
  Work already cites) — not evidence of idea appropriation in either
  direction.

**Conclusion: no concerning overlap found.** The only overlap is at the
generic shared-benchmark/shared-problem-domain level, already disclosed and
contextualized in our own manuscript's Related Work section. No accusation,
implicit or explicit, is being made in either direction.

### 14.7 Files added/changed (this pass)

Added: `baselines/__init__.py`, `baselines/pamop/__init__.py`,
`baselines/pamop/README.md`, `baselines/pamop/config.py`,
`baselines/pamop/representations.py`, `baselines/pamop/data.py`,
`baselines/pamop/partition.py`, `baselines/pamop/run_partitioning.py`,
`baselines/pamop/configs/paper_faithful.yaml`,
`baselines/pamop/configs/reconstructed_default.yaml`,
`baselines/pamop/tests/__init__.py`, `baselines/pamop/tests/test_config.py`,
`baselines/pamop/tests/test_data.py`, `baselines/pamop/tests/test_partition.py`.
Changed: `docs/PAMOP_REPRODUCTION_PLAN.md` (this section).
No other file in the repository was touched.

### 14.8 Exact next milestone

Per §13.12's recommendation, now that the partitioning scaffold exists: the
**LLM-based structured-extraction stage (`G_extr`)** that turns raw NLP4LP
free text into a `StructuredProblem` the way the paper actually describes
(rather than this milestone's NLP4LP-native-fields bridge), gated behind an
explicit model-selection decision between `pamop_paper_faithful`
(`gpt-4-0613`, time-boxed — shuts down 2026-10-23, §7.2) and
`pamop_current_model` (`gpt-4o`, already the `reconstructed_default.yaml`
placeholder). This should ship with its own prompt file(s) under a new
`baselines/pamop/prompts/` directory, each explicitly marked
reconstructed (no original prompt text exists to copy, §1/§2.3), and
its own test suite using mocked/recorded LLM responses so tests stay
network- and API-key-free. AMPL/`amplpy` acquisition (§13.10) and the manual
archived-dataset retrieval (§13.5, `nlp4lp.vercel.app` / OpenReview) can
proceed in parallel and are not blocking for this next step either.

---

## 15. Implementation Status: Milestone 2, LLM Extraction Stage

**Follow-up implementation pass, 2026-08-11 (third same-day follow-up).**
Adds the LLM-based structured-extraction stage (`G_extr`) and a
provider-agnostic LLM interface on top of Milestone 1's partitioning
scaffold. No manuscript file touched, no existing benchmark result changed,
AMPL generation and solver execution still not implemented.

### 15.1 LLM/API environment status (verified, not assumed)

This required more care than expected — two real, non-obvious findings:

| Provider | Env var(s) checked | Status | Finding |
|---|---|---|---|
| OpenAI | `OPENAI_API_KEY` | **Present but not usable for OpenAI** | The 64-char value is byte-identical to `CLOUDRIFT_API_KEY` (confirmed via SHA-256 comparison, values never printed) and has a `rift_...` prefix — it is a CloudRift-issued key aliased into the OpenAI env-var names, presumably so other OpenAI-SDK-based tooling in this environment transparently targets CloudRift. `OPENAI_BASE_URL` is likewise set to `https://inference.cloudrift.ai/v1`. **There is no genuine OpenAI credential on this workstation.** A live call to the real `api.openai.com` endpoint with this key correctly returns HTTP 401 (verified, §15.5). |
| Gemini | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | **Present as names, not usable** | Both variables exist in the environment but `GOOGLE_API_KEY`'s value is an **empty string** (`len() == 0`) and `GEMINI_API_KEY` is unset. `get_env_token()` correctly treats an empty string as "not configured" (§15.5's regression test covers exactly this). |
| Cohere | `COHERE_API_KEY`, `CO_API_KEY` | **Working** | 40-char key, verified with a real live call (§15.5). |
| Fireworks AI | `FIREWORKS_API_KEY` | **Present, not live-tested** | 25-char key looks well-formed; not exercised this milestone (task scope called for "a tiny adapter smoke test," already satisfied by Cohere + CloudRift) — verify before relying on `configs/providers/fireworks_current.yaml`. |
| CloudRift AI | `CLOUDRIFT_API_KEY`, `CLOUDRIFT_BASE_URL`, `CLOUDRIFT_MODEL` | **Working** | 64-char key, verified with a real live call (§15.5); this is the *same* key aliased as `OPENAI_API_KEY` above. |
| Hugging Face (`udell-lab/NLP4LP`) | — | **Working** (unchanged from §13) | Not a blocker. |
| Gurobi | — | **Working** (unchanged from §13) | Not a blocker. |

Practical consequence: **2 of 5 target providers (OpenAI, Gemini) do not
currently have usable credentials on this workstation**, despite both
having been described as "already configured." Neither is a code bug —
`get_env_token` and the per-provider `ProviderAuthError` checks correctly
detect and report both conditions rather than silently proceeding or
crashing unhelpfully. This is reported here rather than worked around
because §7's original guidance ("do not silently choose a snapshot," "do
not ask for keys unless a real auth failure occurs") implies the same
standard for reporting what actually happened.

### 15.2 Provider-agnostic LLM interface

`baselines/pamop/llm/`:

| File | Role |
|---|---|
| `types.py` | `ModelConfig`, `LLMResponse` (text, provider, model, timestamp, temperature/top_p/max_tokens, prompt/completion/total tokens, latency, retry count, prompt hash, finish reason — no field ever holds a secret), `ProviderAuthError`, `ProviderCallError` |
| `base.py` | `LLMProvider` ABC — subclasses implement only `_call`; `generate()` handles timing, retry-with-backoff, and building the common `LLMResponse`. `ProviderAuthError` is deliberately never retried (§15.6 notes one gap here). `prompt_hash()`, `get_env_token()` shared helpers |
| `openai_provider.py` | Always passes an explicit `base_url="https://api.openai.com/v1"` — **deliberately ignores** the ambient `OPENAI_BASE_URL`, precisely because of the CloudRift-aliasing finding above |
| `gemini_provider.py` | Checks `GEMINI_API_KEY` then `GOOGLE_API_KEY` (google-genai `Client`, Developer API) |
| `cohere_provider.py` | Checks `COHERE_API_KEY` then `CO_API_KEY` (Cohere `ClientV2.chat`) |
| `fireworks_provider.py`, `cloudrift_provider.py`, `_openai_compatible.py` | Both target OpenAI-compatible REST endpoints via the `openai` SDK (no separate `fireworks-ai` package required); each provider's own base URL/key is explicit and never cross-wired with `openai_provider.py`'s |
| `registry.py` | `get_provider(name)` / `list_providers()`, lazy per-provider SDK import |

### 15.3 Prompt recovery — targeted re-check, still nothing found

Per task scope, this was a targeted re-check, not a repeat of §1's full
search: one new web search for `G_extr`-specific terminology, and one retry
of the previously-unreachable USTC corresponding-author page
(`staff.ustc.edu.cn/~xiangyangli/publication.html`) — still
`ECONNREFUSED` from this network. No new information. §1's "no official
code" conclusion stands; `baselines/pamop/prompts/extraction_v1.txt` is a
full reconstruction, explicitly marked as such both in-file and in
`prompts/PROVENANCE.md`, which separates the paper-specified requirements
(four output fields `t_o`/`t_c`/`t_v`/`g` and a per-constraint vagueness
score, section 3.2) from the reconstructed wording (everything else,
including the vagueness-score numeric scale, which the paper never states —
this template asks for `[0, 1]`, a reproduction choice).

### 15.4 `G_extr` implementation

`baselines/pamop/extraction.py`:

- `validate_extraction(raw)` — strict schema validation (required fields,
  non-empty strings, `vagueness_score ∈ [0,1]`, variable name is a valid
  identifier with no duplicates, `type` in the allowed enum or absent).
  Rejects and reports a specific reason; **never repairs or fills in
  content** — PaMOP's own repair mechanisms are the separate, later,
  not-yet-implemented correction loop (section 3.3), not part of `G_extr`.
- `extract_structured_problem(problem_id, raw_text, provider, config)` —
  renders the prompt, calls the provider, parses JSON (tolerating a stray
  ```` ```json ```` fence, not tolerating actually-malformed JSON), validates,
  and retries (asking again, not patching) up to
  `config.llm.extraction_max_retries` times on failure. Returns an
  `ExtractionResult` bundling the `StructuredProblem`, the raw
  `LLMResponse`, the `PromptTemplate` used, and the attempt count.
- `representations.from_llm_extraction()` reshapes validated output into the
  same `StructuredProblem` Milestone 1's `partition.build_partition_tree`
  already consumes — verified wired end-to-end, live (§15.5) and in tests
  (`test_extraction.py::test_extraction_wires_into_partition_tree`).

### 15.5 Tiny live API smoke test

Per task instructions: attempted the paper-faithful OpenAI path first, then
fell back to a provider with a verified-working credential, on 3 NLP4LP
problems (ids 1, 2, 5) from the `pamop_possible_269` subset. No raw problem
text appears in this document, any committed file, or the smoke script's
output — only counts, token usage, hashes, and timing.

**Step 1 — `paper_faithful` config, OpenAI, `gpt-4-0613`:** failed as
predicted by §15.1 — `ProviderCallError` after retries, underlying cause an
HTTP 401 from `api.openai.com` (the CloudRift-aliased key does not
authenticate against the real OpenAI API). This is the expected, correctly
surfaced outcome given no genuine OpenAI credential exists here — not a
code defect.

**Step 2 — `cohere_current` config (explicitly NOT paper-faithful),
`command-r7b-12-2024`:**

| Problem id | Status | Attempts | Constraints extracted | Variables extracted | Latency (s) | Prompt tokens | Completion tokens |
|---|---|---|---|---|---|---|---|
| 1 | success | 1 | 4 | 7 | 5.52 | 536 | 456 |
| 2 | success | 1 | 2 | 5 | 4.14 | 536 | 300 |
| 5 | success | 1 | 4 | 8 | 6.70 | 550 | 498 |

All three succeeded on the **first** attempt (no retries needed — the model
produced schema-valid JSON immediately every time), `finish_reason:
COMPLETE` for all three. Total for the 3-problem smoke test: ~2,626 tokens
across both directions. **Cost:** not tracked via a pricing API this
milestone; at Cohere's published per-token rates for a `command-r7b`-class
model this is on the order of a fraction of a US cent for the whole
3-problem test — not a meaningful cost signal either way, consistent with
"tiny." No `sbatch` job was needed (each call completed in single-digit
seconds).

### 15.6 A real gap found during the smoke test, documented rather than hidden

The OpenAI 401 in step 1 was retried twice before `generate()` gave up
(3 attempts total, ~unnoticeable extra latency here since it fails fast).
`LLMProvider.generate()` only skips retries for `ProviderAuthError`, which
is raised when a key is **absent**; a key that is *present but rejected by
the endpoint* (this case) surfaces as the SDK's own exception type
(`openai.AuthenticationError`) inside the generic `except Exception` retry
branch instead. This is not a correctness bug — the failure is still
reported clearly as `ProviderCallError` with an accurate retry count — but
it is a minor missed optimization (retrying a definite auth rejection is
pointless) worth fixing in a future pass by mapping each provider SDK's own
auth-error type to `ProviderAuthError` at the call site, not just at client
construction.

### 15.7 LLM-dependency comparison: PaMOP vs. our main pipeline

For future manuscript-revision use (manuscript itself **not** touched this
pass):

- **PaMOP requires an LLM at inference/modeling time**, for every stage this
  reproduction has touched so far and every stage still to come: structured
  extraction (`G_extr`, this milestone), self-augmented leaf modeling
  (`G_mod`, not yet implemented), and the correction loop (`G_exe`/`G_rev`/
  `G_comp`/`G_remod`, not yet implemented). Every one of those calls is a
  network round-trip to a third-party API, with the accuracy/cost/latency
  characteristics that implies (§15.5's ~5-second, few-hundred-token calls
  are representative of a *single* extraction call for a *small* NLP4LP
  problem — the full pipeline issues many such calls per problem, per §10 of
  this document).
- **Our main retrieval-assisted instantiation pipeline does not require an
  external generative LLM/API at inference time.** Its benchmarked path
  (schema retrieval via TF-IDF/BM25/LSA, deterministic scalar-slot
  grounding, structural verification) runs entirely locally and
  deterministically, as already documented in `manuscript/main.tex`'s
  Problem Scope section ("a fully deterministic pipeline ... with no
  learned or large-language-model components").
- This gives our pipeline several potential, genuinely real advantages
  worth naming precisely — **without overselling them as proof of general
  superiority**, and without ever describing our pipeline as using "no
  AI" (it is a deterministic, rule-based *system*, not an absence of any
  computational technique; TF-IDF/BM25/LSA are themselves standard
  information-retrieval methods, and calling a system that uses them
  "no AI" would be an inaccurate simplification, not a modest one):
  - **Reproducibility**: identical input always produces identical output;
    PaMOP's temperature-0.2 sampling and multi-stage LLM pipeline do not
    have this property without extra seeding/variance-control work (§5 of
    this document already flags this as an open axis to measure if
    pursued).
  - **Cost**: no per-query API spend; PaMOP's cost is nonzero and scales
    with both problem count and problem complexity (§10).
  - **Model-version stability**: nothing in our benchmarked path depends on
    a vendor's model catalog remaining available — directly relevant given
    §7.2's finding that PaMOP's own base model (plain "GPT-4") is
    mid-deprecation as of this investigation.
  - **Offline/restricted-deployment feasibility**: a fully local pipeline
    can run without external network access or a third-party API
    dependency at all; PaMOP's architecture cannot, by construction, at any
    of its LLM-touching stages.
  - **Privacy**: no problem text leaves the local environment in our
    benchmarked path; every PaMOP stage that touches an LLM sends the
    problem text (or a derived structured form of it) to a third-party API.
  - The other side of this comparison is equally real and already stated in
    our own manuscript's Problem Scope: our pipeline's task is narrower
    (schema retrieval + scalar grounding, not full model generation), so
    this is a difference in **what each system is trying to do**, not
    simply a win on every axis — a narrower, non-generative task is
    *easier* to make deterministic and local than PaMOP's full NL-to-model
    generation task is.
- **This section is written for a future manuscript revision to draw on if
  useful — the manuscript itself was not modified in this pass.**

### 15.8 Suspicious prior-work overlap check — repeated for this milestone, **nothing concerning found**

Extended the same audit (§14.6) to this milestone's new surface area
specifically: LLM-based structured extraction, JSON-schema validation of
LLM output, vagueness scoring, and the provider-abstraction/config-driven
paper-faithful-vs-reconstructed design pattern itself.

- **Structured extraction with JSON-schema validation**: PaMOP's `G_extr`
  produces a *fresh* structured representation per problem via an LLM call,
  validated here against a JSON schema we wrote (not the paper's, which
  doesn't exist — §15.3). Our manuscript's schema-retrieval stage does the
  structurally opposite thing: it *matches* an incoming query against a
  **fixed, pre-existing** catalog of 335 schemas; it never generates a new
  schema from scratch and has no analogous "extraction validation" concept
  at all. No mechanism overlap.
- **Vagueness scoring**: unique to PaMOP among everything checked; nothing
  resembling a per-constraint ambiguity score exists anywhere in our
  manuscript or pipeline (our closest concept, `TypeMatch`/`Coverage`, scores
  *slot-assignment* quality against a schema, not *linguistic ambiguity* of
  a constraint description — a different axis entirely).
- **Provider-abstraction pattern** (`baselines/pamop/llm/`, this milestone's
  own architecture): this is *our own* design choice for building the
  reproduction, not something drawn from the PaMOP paper (which describes
  no such abstraction — it names only "GPT-4" as its runtime). Not a
  candidate for overlap analysis in either direction; noted here only for
  completeness since the task asked to repeat the audit "while reading
  PaMOP deeply" during this milestone.
- **Chronology**: unchanged from §14.6 — PaMOP (Aug 2025) predates this
  repository's current manuscript work; no direction of influence is
  plausible beyond both papers targeting the same public benchmark, already
  disclosed in our manuscript's Related Work.

**Conclusion: no concerning overlap found**, consistent with §14.6. No
accusation, implicit or explicit, is made in either direction.

### 15.9 Configuration changes

- `config.py`: `LlmConfig` gained `provider` (which of the 5 registered
  providers) and `extraction_max_retries` (G_extr-specific retry budget,
  distinct from `max_correction_iterations`, the paper's later
  solver-debug-loop budget — paper doesn't specify either for extraction).
- `configs/paper_faithful.yaml`: `llm.provider: openai` added as a
  **high-confidence inference** (paper never says "OpenAI" but names only
  OpenAI model families as its base models); `llm.model` changed from
  `null` to `gpt-4-0613`, explicitly commented as a **reproduction choice**
  (closest surviving plain-"GPT-4" snapshot, not a paper-stated fact) with
  the CloudRift-aliasing environment note inline. `extraction_max_retries`
  stays `null` (genuinely unspecified) — **the config still fails loudly**
  (`UnspecifiedPaperDetailError`) if that field is read without being set,
  exactly as before; only the two fields with an actual documented
  resolution path changed.
- `configs/reconstructed_default.yaml`: `llm.provider: openai`,
  `extraction_max_retries: 2` added (reproduction choice, documented).
- **New** `configs/providers/{openai,gemini,cohere,fireworks,cloudrift}_current.yaml`
  — one per provider, identical partitioning/correction/execution/dataset
  sections to `reconstructed_default.yaml`, each with an inline `STATUS on
  this workstation` comment stating plainly whether it is currently
  runnable here (openai: no: gemini: no; cohere: yes, live-verified;
  fireworks: untested; cloudrift: yes, live-verified) — see §15.1.

### 15.10 Updated blocker list

| Item | Classification | Status |
|---|---|---|
| Hugging Face NLP4LP access | — | **RESOLVED** (§13.2, unchanged) |
| Gurobi | — | **RESOLVED** (§13.3, unchanged) |
| NLP4LP loader suffixed/missing-file ids | — | **RESOLVED** this milestone (§15 loader fix; 6 genuinely-missing ids now raise a clean `MissingStructuredDataError` instead of a generic error) |
| Provider abstraction + `G_extr` | — | **RESOLVED** this milestone |
| Real OpenAI credentials on this workstation | **BLOCKER** (for `pamop_paper_faithful` execution specifically) | Not resolvable by this investigation — requires a genuine OpenAI key to be configured somewhere this pipeline runs; the paper-faithful *design decision* (§7.2/§15.9) is unaffected and stands regardless |
| Real Gemini credentials on this workstation | NON-BLOCKING (Gemini is one of five current-model variants, not required for any milestone) | `GOOGLE_API_KEY`/`GEMINI_API_KEY` present as names, empty as values |
| AMPL / `amplpy` | **BLOCKER** (for paper-faithful generation/execution, unchanged from §13.10) | Still not installed; not needed until the modeling/execution stage |
| Exact identity of PaMOP's 67 problems | NON-BLOCKING UNCERTAINTY (unchanged from §13) | — |
| Auth-error-vs-generic-error retry gap (§15.6) | NON-BLOCKING, minor | Documented, not fixed this pass |
| Missing prompt templates for `G_mod`/`G_exe`/`G_rev`/`G_comp`/`G_remod` | IMPLEMENTATION CHOICE | Next milestone |

### 15.11 Files added/changed (this pass)

Added: `baselines/pamop/llm/{__init__.py, base.py, types.py, registry.py,
openai_provider.py, gemini_provider.py, cohere_provider.py,
fireworks_provider.py, cloudrift_provider.py, _openai_compatible.py}`,
`baselines/pamop/prompts/{__init__.py, PROVENANCE.md, extraction_v1.txt}`,
`baselines/pamop/extraction.py`,
`baselines/pamop/configs/providers/{openai,gemini,cohere,fireworks,cloudrift}_current.yaml`,
`baselines/pamop/tests/{test_llm.py, test_prompts.py, test_extraction.py}`.
Changed: `baselines/pamop/config.py` (§15.9), `baselines/pamop/data.py`
(loader fix, §15's own section above), `baselines/pamop/configs/{paper_faithful,reconstructed_default}.yaml`
(§15.9), `baselines/pamop/representations.py` (`from_llm_extraction`),
`baselines/pamop/README.md`, `baselines/pamop/tests/test_data.py`
(loader-fix regression tests), `docs/PAMOP_REPRODUCTION_PLAN.md` (this
section). No other file in the repository was touched.

### 15.12 Exact next milestone

Self-augmented leaf-node modeling (`G_mod`, paper eq. 3) and its bottom-up
merge into a complete model (eq. 4): another LLM-touching stage, consuming
Milestone 1's partition tree + Milestone 2's `StructuredProblem`/vagueness
scores as input, producing per-leaf constraint formulas. This is the last
stage before AMPL generation becomes meaningful (the paper generates AMPL
text directly as the leaf-modeling output, §2.5/§13.10), so the AMPL/Gurobi
question (still blocked on installing `amplpy`, §13.10/§15.10) becomes live
at that point, not before. Should ship with its own reconstructed prompt
(`prompts/leaf_modeling_v1.txt`, same PROVENANCE.md discipline as this
milestone) and reuse the existing `llm/` provider abstraction unchanged.

---

## 16. Implementation Status: Milestone 3, Self-Augmented Modeling and Merge

**Follow-up implementation pass, 2026-08-11 (fourth same-day follow-up).**
Implements `G_mod` (eq. 3), the bottom-up merge (eq. 4), and a first-class
Azure OpenAI provider. No manuscript file touched, no benchmark result
changed, no AMPL generation/execution implemented, no full 269-block run.

### 16.1 Azure OpenAI environment status

**Authentication: PASS.** This workstation has a real, working Microsoft
Azure OpenAI resource, distinct from the CloudRift-aliased "OpenAI" key
found in Milestone 2 (§15.1) — confirmed by a live call, not assumed.

| Item | Value |
|---|---|
| Endpoint found | YES — `AZURE_OPENAI_ENDPOINT` (also mirrored as the generic `AZURE_API_BASE`), an OpenAI-compatible `.../openai/v1` path |
| Deployment(s) found | YES, two, both live-verified: `gpt-4.1-mini` (env `AZURE_OPENAI_DEPLOYMENT`) and `gpt-5.4` (env `AZURE_OPENAI_STRONG_DEPLOYMENT`) |
| GPT-4-family deployment exists | YES — `gpt-4.1-mini` |
| Underlying model/version | Discoverable and confirmed live: the API itself echoes back the exact served snapshot in its response — `gpt-4.1-mini` resolves to **`gpt-4.1-mini-2025-04-14`**; `gpt-5.4` resolves to `gpt-5.4-2026-03-05` (not GPT-4-family, recorded for completeness only) |
| API version | Two candidate env vars found, `AZURE_OPENAI_API_VERSION=2024-10-21` and `AZURE_API_VERSION=2024-12-01-preview`; the OpenAI-compatible `.../openai/v1` endpoint style used here did not require passing either explicitly and worked without it |
| Full deployment enumeration | **Not possible** with the available credential — the standard `/openai/deployments?api-version=...` listing endpoint returned HTTP 404 with this inference-plane key (deployment listing typically needs management-plane/ARM access, which was not sought). Only the two deployments named in environment variables were confirmed to exist; others may exist on this Azure resource without being independently discoverable here. |
| Credential aliasing | `AZURE_API_KEY` and `AZURE_OPENAI_API_KEY` are the same value (confirmed via hash comparison, never printed) — two naming conventions for one credential, both supported by the new provider |

**No secret value was printed, logged, or committed at any point in this
investigation.**

### 16.2 Azure OpenAI as a first-class provider

`baselines/pamop/llm/azure_openai_provider.py` (`AzureOpenAIProvider`,
registered as `"azure_openai"`): same `generate(prompt, ModelConfig) ->
LLMResponse` contract as every other provider. Two things needed to be
solved specifically for this workstation's resource, both covered by
regression tests:

1. **Endpoint style**: this resource's endpoint already includes
   `/openai/v1`, matching the OpenAI-compatible client style already used
   for Fireworks/CloudRift (`OpenAI(api_key=..., base_url=...)`), not the
   older `AzureOpenAI(azure_endpoint=...)` client shape (which would double
   up the path). Verified live before committing to this design.
2. **`max_tokens` vs. `max_completion_tokens`**: `gpt-5.4` (not GPT-4-family,
   found incidentally while checking the "strong" deployment) rejects the
   standard `max_tokens` parameter and requires `max_completion_tokens`.
   The provider tries `max_tokens` first and transparently retries once
   with the renamed parameter on that specific error, so callers never need
   to know which convention a given deployment uses. `gpt-4.1-mini` (the
   paper-faithful deployment) accepts plain `max_tokens` — the fallback
   exists for robustness, not because the primary path needs it.

`LLMResponse` gained a new field, `underlying_model` (`None` for providers
that don't echo one back), populated here with the exact served snapshot
(`gpt-4.1-mini-2025-04-14`) — this is the mechanism that makes "underlying
GPT model/version if discoverable" an actual recorded fact per call, not
just a one-time investigation note.

### 16.3 Paper-faithful model decision (revised from Milestone 2)

**Superseded**: §15.9's `gpt-4-0613` via direct OpenAI is **no longer the
active `paper_faithful.yaml` choice** — that workstation had no working
direct-OpenAI credential at all (§15.1), so it could never execute. It
remains documented, unchanged, as the choice for an environment with
genuine direct OpenAI access instead (`paper_faithful.yaml` inline comment
points there).

**New primary reproduction path**: `provider: azure_openai`, `model:
gpt-4.1-mini`, per task policy (verified GPT-4-family access takes
priority, no silent fallback to a non-GPT-4-family provider). Classified
explicitly as a **closest available GPT-4-family reproduction
configuration** — not exact reproduction:

- The paper says only "GPT-4," no version. "GPT-4.1" is a materially later
  model generation than plain "GPT-4," and "mini" is explicitly a smaller
  tier within that generation — this is the closest thing to a GPT-4-family
  model this workstation can actually call, not a match to what the authors
  used.
- A larger/older GPT-4-family Azure deployment may exist on this resource
  without being discoverable given the available credential (§16.1) — this
  choice reflects what could be *confirmed*, not necessarily everything
  that *exists*.
- `temperature: 0.2` (PAPER-SPECIFIED) is preserved unchanged.

### 16.4 `G_mod` implementation (eq. 3)

`baselines/pamop/modeling.py::model_leaf` / `model_all_leaves`:

- Inputs exactly as eq. 3 specifies: global summary `g`, the **full**
  global variable/parameter list `t_v` (not filtered to the node), and only
  this leaf's own constraint descriptions.
- Output is requested as plain AMPL text (not JSON, unlike `G_extr` —
  deliberately, since the paper describes eq. 3's output as code, "we
  directly generate code in the modeling language instead of formulas,"
  not structured multi-field data).
- **Vague-constraint augmentation** (paper: "when modeling nodes containing
  vague constraints, we can incorporate information from their parent and
  sibling nodes"): triggered per leaf when any of its constraints'
  `vagueness_score` (from `G_extr`, Milestone 2) is at or above
  `config.llm.vague_threshold` — REPRODUCTION CHOICE (paper gives no
  number). Augmentation content is the parent's and siblings' constraint
  *descriptions* (also a reproduction choice — paper doesn't specify the
  exact form; passing already-modeled sibling output isn't reliably
  available under a bottom-up traversal order, so descriptions were chosen
  over attempting to sequence sibling calls to guarantee available output).
- **Validation** (`validate_leaf_output`): minimal, explicitly heuristic —
  non-empty, contains at least one `;`. No AMPL parser exists this
  milestone (§16.7). Failure retries (asking again) up to
  `config.llm.modeling_max_retries` — REPRODUCTION CHOICE, paper doesn't
  specify a modeling-stage retry count either (distinct from
  `extraction_max_retries`, Milestone 2, and from `max_correction_iterations`,
  the paper's own later solver-debug loop, not yet implemented).
- **Unresolved-reference diagnostic** (`_find_unresolved_references`):
  heuristic, non-fatal scan for identifier-shaped tokens in a leaf's output
  that match neither a declared variable/parameter name nor a common AMPL
  keyword — excludes AMPL constraint labels (`subject to c1: ...`) by
  construction, a false-positive source caught and fixed during this
  milestone's own smoke testing (§16.9). This is diagnostic only, never a
  hard validation failure or an attempt at automatic repair.
- **Serializable symbol provenance**: each `LeafModelResult` records
  `referenced_global_symbols`, the declared `t_v` names that appear in the
  generated fragment. This is a lightweight downstream aid, not an AMPL
  parse and not a validation gate.

### 16.5 Bottom-up merge (paper mechanism, no LLM call)

`baselines/pamop/modeling.py::merge_bottom_up`: the paper is explicit that
this step is **not** an LLM call at internal tree layers — "we can directly
merge the modeled formulas" — implemented here as a genuine recursive
tree-walk (not a flattened leaf-order concatenation, even though the two
would produce identical text under a fixed traversal order) so the
procedure mirrors the paper's own "layer by layer from the bottom up"
description and stays auditable per internal node. Confirms directly, per
task's specific questions:

- **What's passed upward**: each leaf's modeled AMPL text, verbatim.
- **Concatenated, summarized, regenerated, or reconciled?** Concatenated —
  the paper explicitly relies on a "minimal conflict" assumption rather
  than reconciling anything.
- **Do parent (non-root, non-leaf) nodes invoke the LLM?** No — only leaves
  (eq. 3) and the root's own final step (eq. 4) do.
- **Are variables/parameters deduplicated?** Not applicable at this stage —
  leaves never declare variables/parameters in this design (only reference
  them); declarations happen once, at the root, in eq. 4.
- **How are conflicts handled?** Not reconciled automatically (matching the
  paper's own assumption) — only surfaced as diagnostics
  (`MergedModel.symbol_conflicts`) for unresolved references, duplicate
  leaf constraint labels, and leaf fragments that contain `param`/`var`
  declarations even though leaf `G_mod` should emit constraints only. The
  root Eq. 4 parser also rejects duplicate `param`/`var` declarations where
  detectable in its reconstructed four-section output. None of these checks
  rewrites or repairs model text. This is a deliberately conservative
  REPRODUCTION CHOICE: "improve on PaMOP" was explicitly out of scope for
  the paper-faithful path per task instructions.
- **Are objective pieces merged incrementally?** No — the objective is
  handled exactly once, at the root (eq. 4); eq. 3's leaves never touch it.

### 16.6 Root completion (eq. 4)

`baselines/pamop/modeling.py::model_root_objective`: one additional call
with `(g, t_v, t_o, m_c)` — global summary, full variable list, objective
text, and the already-merged constraint text — producing the complete
`M = (m_p, m_v, m_o, m_c)`. Output is requested as four labeled sections
(`### PARAMETERS` / `VARIABLES` / `OBJECTIVE` / `CONSTRAINTS`) — an
explicit **REPRODUCTION CHOICE** structuring decision (the paper describes
`M` only as an abstract tuple, never mandates an output format), made
specifically to give the next milestone's AMPL renderer reliable section
boundaries to consume (§16.7). Validation requires all four headers present
in order and non-empty `OBJECTIVE`/`CONSTRAINTS` sections; retries up to
`modeling_max_retries` on failure, same policy as leaf modeling.

### 16.7 AMPL interface boundary (prepared, not implemented)

`baselines/pamop/ampl_interface.py`: an `AmplRenderer` `Protocol` (`render`
/ `solve` method signatures only, no implementation) documenting exactly
what the next milestone must consume from `MergedModel` — its four
AMPL-flavored text fields — and stating plainly that none of them are
validated as syntactically correct AMPL yet (no parser exists this
milestone). A `naive_concatenation_preview` helper exists only for
human/debug inspection, explicitly **not** a renderer.

**AMPL/`amplpy` were not installed** — every check this milestone needed
(prompt construction, section parsing, heuristic reference-checking)
operates on plain text and required no AMPL runtime; installation becomes
necessary only once something needs to actually *execute* a model, which
is the next milestone's job.

### 16.8 Prompt provenance

Two new reconstructed templates, same discipline as Milestone 2's
`extraction_v1.txt`: `prompts/modeling_leaf_v1.txt` (eq. 3) and
`prompts/modeling_root_v1.txt` (eq. 4), both version/content-hashed via the
existing `prompts.load_prompt` mechanism, both documented in
`prompts/PROVENANCE.md` with an explicit per-template
paper-specified-vs-reconstructed breakdown (§16.4/§16.6 above summarize the
same table). No new prompt text was recoverable from any source this
milestone — re-reading the official IJCAI paper and the already-audited
local source trail exposed no author-supplied prompt wording, and no new
lead specific to `G_mod`/eq. 4 emerged.

### 16.9 A real bug found and fixed during this milestone's own testing

The first smoke test of the symbol-conflict diagnostic
(`_find_unresolved_references`) flagged AMPL constraint labels (e.g. `c1`
in `subject to c1: ...`) as "unresolved variable references" — a genuine
false-positive class, not a hypothetical one (surfaced immediately on the
very first hand-written test problem). Fixed by excluding tokens matching
the `identifier:` label pattern before scanning for unresolved references
(`_CONSTRAINT_LABEL_RE`); a dedicated regression test
(`test_model_leaf_does_not_flag_constraint_labels_as_unresolved`) now
covers this. Documented here per this project's practice of surfacing real
issues found during work rather than only reporting a clean final state.

### 16.10 Mocked test results

`baselines/pamop/tests/test_modeling.py` (25 tests), `test_ampl_interface.py`
(3 tests), and Azure-specific tests in `test_llm.py` — all
network-free, synthetic problems and hand-built partition trees only
(never gated NLP4LP text). Full suite: **134/134 passed** (up from 98 at
commit `f472542`), 0 skipped in this environment (network reachable).
Coverage against the task's explicit checklist: `G_mod` prompt construction
✅, leaf-node serialization ✅, malformed LLM output ✅, retry handling ✅,
one-leaf case ✅, two-leaf merge ✅, multi-level (3-leaf, 2-internal-node)
tree merge ✅ (hand-built tree, so the merge order is exactly known and
independent of `partition.py`'s own clustering behavior), unresolved-
reference handling ✅ (including the constraint-label false-positive fix,
§16.9), deterministic merge order ✅, provider metadata preservation ✅,
Azure provider configuration parsing with mocked credentials ✅, no secret
leakage ✅ (`MergedModel.to_dict()` and `LLMResponse` both checked),
duplicate leaf label handling ✅, leaf-declaration conflict diagnostics ✅,
and duplicate root declaration rejection ✅.

### 16.11 Tiny live Azure smoke test — full pipeline, no AMPL

Ran the complete `G_extr -> partition tree -> G_mod -> merge` pipeline live
against one real NLP4LP problem (id 1, from `pamop_possible_269`), using
`reconstructed_default.yaml`'s selected paper-faithful provider/deployment
(`azure_openai`, `gpt-4.1-mini`, temperature 0.2). **AMPL generation was
never invoked.** No raw problem text appears below or in any committed
file.

| Problem id | Status | Underlying model | Constraints | Leaves | Symbol conflicts | Total latency (s) |
|---|---|---|---:|---:|---:|---:|
| 1 | success | `gpt-4.1-mini-2025-04-14` | 4 | 4 | 0 | 11.756 |

The extraction and root-completion calls both succeeded on the first
validation attempt.

**Token/cost summary** (one-problem total, both directions):
extraction 1,009 tokens; leaf-modeling 1,564 tokens; root-completion 794
tokens — **3,367 tokens total** across the full pipeline. The Azure
inference API did not report actual cost, and no pricing API was queried.
No `sbatch` job was needed (the full pipeline completed in under 12
seconds of aggregate LLM latency).

### 16.12 Suspicious prior-work overlap check — repeated for this milestone, **nothing concerning found**

Extended the same audit (§14.6, §15.8) to `G_mod`/merge-specific surface
area: self-augmented leaf modeling, parent/sibling vague-context
augmentation, bottom-up text merge, and the symbol-conflict/unresolved-
reference heuristic.

- **Self-augmented modeling vs. schema-conditioned scalar instantiation**:
  mechanically opposite. `G_mod` *generates* fresh AMPL text per constraint
  group via an LLM call, conditioned on a growing textual context (global
  summary + full variable list + optionally parent/sibling descriptions);
  our manuscript's scalar instantiation *assigns* already-extracted numeric
  mentions to slots of an already-*retrieved* fixed schema — no generative
  step, no LLM, no textual-context augmentation concept anywhere in that
  pipeline. No mechanism overlap.
- **Structural verification**: `G_mod`'s unresolved-reference heuristic and
  our `formulation/verify.py` are both instances of the generic,
  independently-obvious "sanity-check before solving" pattern (same
  conclusion as §14.6's "basic inspection" comparison) — not a distinctive
  shared mechanism.
- **"Deterministic repair"** (explicitly asked about this milestone): no
  counterpart exists in PaMOP's `G_mod`/merge stage to compare against our
  pipeline's actual deterministic-repair layers (no-reuse compatibility
  slot assignment, lexicon-based role/admissibility refinement) — this
  milestone's merge is literal, unmodified concatenation (§16.5) and the
  symbol-conflict check never repairs anything, only flags it. There is no
  shared mechanism here, only a shared checklist term.
- **Grounding-bottleneck analysis / unusual metric design**: no counterpart
  in `G_mod`/merge at all — this stage produces model text, not evaluation
  metrics (that remains a future, unimplemented stage). No overlap.
- **Chronology**: unchanged from §14.6/§15.8.

**Conclusion: no concerning overlap found**, consistent with both prior
milestones. No accusation, implicit or explicit, is made in either
direction.

### 16.13 Files added/changed (this pass)

Added: `baselines/pamop/llm/azure_openai_provider.py`,
`baselines/pamop/modeling.py`, `baselines/pamop/ampl_interface.py`,
`baselines/pamop/prompts/modeling_leaf_v1.txt`,
`baselines/pamop/prompts/modeling_root_v1.txt`,
`baselines/pamop/configs/providers/azure_openai_current.yaml`,
`baselines/pamop/tests/{test_modeling.py, test_ampl_interface.py}`.
Changed: `baselines/pamop/llm/{types.py, base.py, registry.py}`
(`underlying_model` field, Azure registration), `baselines/pamop/config.py`
(`modeling_max_retries`, `vague_threshold` fields),
`baselines/pamop/configs/{paper_faithful,reconstructed_default}.yaml`
(§16.3), all 5 pre-existing `configs/providers/*.yaml` (new fields added
for consistency), `baselines/pamop/prompts/PROVENANCE.md` (two new
template entries), `baselines/pamop/tests/{test_llm.py, test_extraction.py}`
(Azure tests; one hardcoded-model-string test fixed to track config instead),
`baselines/pamop/README.md`, `docs/PAMOP_REPRODUCTION_PLAN.md` (this
section). No other file in the repository was touched.

### 16.14 Exact next milestone

The error-correction loop (paper §3.3, "Error correction"): basic
inspection (regex syntax check + parameter-vs-data verification, not yet
implemented — no AMPL parser exists, §16.7), the solver-debug loop (`G_exe`,
eq. 5 — requires actually executing a model, which requires AMPL/`amplpy`
acquisition, still deferred per §13.10), and reverse translation
(`G_rev`/`G_comp`/`G_remod`, eq. 6). This is also the point where
`ampl_interface.py`'s `AmplRenderer` Protocol (§16.7) needs its first real
implementation, and where the `max_correction_iterations: 5`
(PAPER-SPECIFIED) budget from `paper_faithful.yaml` finally gets used by
running code rather than just being validated as loadable. AMPL/`amplpy`
acquisition becomes a genuine blocker starting at this milestone, not
before.

---

## 17. Implementation Status: Milestone 4, AMPL Execution and Correction Loop

**Follow-up implementation pass, 2026-08-11 (fifth same-day follow-up).**
Implements the final core PaMOP execution/correction milestone: AMPL
rendering from `MergedModel`, AMPL/Gurobi execution (`G_exe` in this
scaffold), reconstructed `G_rev`/`G_comp`/`G_remod`, and a bounded
correction trace. No manuscript file touched, no full 269-block run, and no
claim of exact PaMOP 67-problem reproduction.

### 17.1 Paper-specified vs reconstructed details

- **PAPER-SPECIFIED:** AMPL is the generated modeling language; AMPL calls
  Gurobi; generated model + original data are solved; modeling is correct
  only when the solver returns an optimal solution that meets the problem
  requirements; maximum failed correction iterations is 5; solver errors
  are fed back during correction; reverse translation/comparison/remodeling
  use `G_rev`, `G_comp`, and `G_remod`.
- **HIGH-CONFIDENCE INFERENCE:** execution failures should be separated from
  environment/data failures so unavailable AMPL, solver/license problems,
  HF failures, and Azure authentication failures are not counted as model
  failures.
- **REPRODUCTION CHOICE:** exact regex/static checks, all correction prompt
  wording, JSON response schemas, subprocess-based AMPL invocation,
  `model_error`/`data_error`/`environment_error` categories, and the
  correction trace schema.

### 17.2 AMPL environment status

Installed user-locally into the existing Gurobi venv:

```bash
/home/soroush/.venvs/gurobi/bin/python -m pip install --upgrade amplpy
/home/soroush/.venvs/gurobi/bin/python -m amplpy.modules install highs gurobi
```

No repository file stores licenses or credentials. AMPL module install
status:

| Item | Status |
|---|---|
| AMPL available | YES — AMPL module `base`, version `20260809` |
| `amplpy` available | YES — in `/home/soroush/.venvs/gurobi` |
| AMPL parse/load check | PASS |
| AMPL→Gurobi trivial solve | PASS — objective `12.0`, `solve_result=solved` |
| AMPL→HiGHS | Installed but one timed solve check hit the timeout; Gurobi is the paper-faithful backend and works, so HiGHS was not used further |

Live execution commands set:

```bash
PAMOP_AMPLPY_PYTHON=/home/soroush/.venvs/gurobi/bin/python
```

### 17.3 Renderer and static validation

Added `baselines/pamop/ampl/`:

- `renderer.py`: renders the four fields of `MergedModel`
  (`parameters_text`, `variables_text`, `objective_text`, `constraints_text`)
  into a single AMPL text artifact. It removes markdown fences/section
  headers and preserves generated AMPL content; it does not semantically
  rewrite the model.
- `validator.py`: reconstructed "basic inspection" checks for non-empty
  model text, duplicate `param`/`var` declarations, missing variables,
  missing/multiple objectives, missing/duplicate constraints, unresolved
  symbols, malformed expressions, and unparsed semicolon statements.
- `types.py`: serializable diagnostics and render/execution result types.

The validator is intentionally not a full AMPL parser. AMPL itself remains
the execution authority.

### 17.4 `G_exe` execution

`baselines/pamop/ampl/executor.py::AmplExecutor` runs AMPL through:

```bash
python -m amplpy.modules run ampl model.run
```

It records:

- parse/model-load/solver-invocation success;
- solver status (`solve_result`);
- objective value when displayed;
- runtime;
- structured diagnostics;
- stdout/stderr tails only, not full logs;
- `model_error`, `data_error`, or `environment_error`.

Only `model_error` enters the correction loop. Environment/data failures
stop the trace and are not sent to `G_remod`.

### 17.5 `G_rev`, `G_comp`, `G_remod`, and correction loop

Added `baselines/pamop/correction.py` plus reconstructed prompts:

- `prompts/correction_review_v1.txt` (`G_rev`-style review): reviews the
  AMPL model and execution diagnostics, returning a JSON diagnosis and
  actionable feedback.
- `prompts/correction_compare_v1.txt` (`G_comp`): returns a binary
  `needs_remodel` decision and targeted issues.
- `prompts/correction_remodel_v1.txt` (`G_remod`): returns a full corrected
  AMPL model and a changes list.

The loop is:

```text
MergedModel -> render -> G_exe
if model_error and corrections_used < 5:
  G_rev -> G_comp -> G_remod -> G_exe
stop on success, non-model failure, comparison decline, or max=5
```

Each `CorrectionIteration` records iteration number, AMPL hash, execution
status/category, review/comparison/remodel metadata, prompt hashes,
provider/model/underlying-model metadata, tokens, and latency. `CorrectionTrace`
serializes the complete run without credentials or gated problem text.

### 17.6 Metrics prepared for future evaluation

The trace/result schema now supports the eventual PaMOP comparison metrics:

- AMPL generation/render success;
- execution rate;
- correction-needed rate;
- mean correction iterations;
- solver success;
- feasibility/solver status;
- objective produced;
- final model success;
- token usage;
- API cost if externally priced;
- latency.

PaMOP's published values remain literature facts only: **62.3% accuracy**
and **86.8% execution rate**. They are not reproduced results from this
codebase.

### 17.7 Mocked tests and static checks

Added network-free tests for AMPL rendering, LP objective rendering, MILP
variable declarations, bounds, duplicate symbols, unresolved symbols,
malformed expressions, execution-result parsing, infeasible-result parsing,
environment/model classification, correction-loop success on the initial
attempt, correction-loop success after retry, max-5 termination, no
correction on environment failure, `G_rev` prompt construction, `G_comp`,
`G_remod` parsing, and correction-trace serialization.

Full PaMOP suite: **152/152 passed**.

Additional static checks:

```bash
python -m ruff check baselines/pamop --ignore E402
python -m compileall -q baselines/pamop
git diff --check
```

`ruff` still needs `--ignore E402` because the existing test files use a
local `sys.path` bootstrap before imports.

### 17.8 Live infrastructure smoke

Trivial AMPL model:

```ampl
var x >= 0;
var y >= 0;
maximize profit: 3*x + 2*y;
subject to cap: x + y <= 4;
```

Result through AMPL→Gurobi:

| Field | Value |
|---|---|
| Parse success | YES |
| Model load success | YES |
| Solver invocation success | YES |
| Solver status | `solved` |
| Objective | `12.0` |
| Runtime | `0.047s` |

### 17.9 Tiny live NLP4LP smoke test

Ran one accessible NLP4LP item, id `1`, using raw `description.txt` in
memory only (not committed), Azure OpenAI `gpt-4.1-mini`, temperature
`0.2`, and AMPL/Gurobi execution.

Pipeline:

```text
problem -> G_extr -> partition tree -> G_mod -> bottom-up merge
-> AMPL render -> G_exe -> G_rev/G_comp/G_remod -> G_exe
```

Result:

| Field | Value |
|---|---|
| Initial render static validation | PASS |
| Initial execution | model failure requiring correction |
| Correction iterations | 1 |
| Final parse/model load/solver invocation | PASS / PASS / PASS |
| Final solver status | `solved` |
| Objective produced | `5016000000.0` |
| Final success | YES, execution-level only |
| Total LLM tokens | `4070` |
| Correction-stage tokens | `2235` |
| LLM latency | `15.932s` |
| Final AMPL runtime | `0.061s` |
| Cost | not reported by Azure API |

Important limitation: this is a smoke test of infrastructure and correction
plumbing, not a semantic correctness claim. The objective value is recorded
as solver output, not validated against the original problem requirement.

### 17.10 Our-method comparison preserved

The comparison dimension is now sharper:

- Our main method remains external-LLM-free at inference time: retrieval,
  deterministic scalar grounding, and verification.
- PaMOP reproduction now demonstrably uses multiple LLM calls, AMPL/Gurobi
  execution, model-version-dependent Azure access, correction-loop tokens,
  and per-query API latency/cost exposure.

This is documentation for a future manuscript revision only. The manuscript
was not modified.

### 17.11 Overlap check — nothing concerning found

Repeated the narrow overlap audit for execution/correction:

- **Structural verification:** generic overlap only. PaMOP's basic
  inspection checks AMPL text before solver execution; our method's
  structural verification checks schema/grounding readiness. Same broad
  engineering idea, different artifacts and purpose.
- **Correction/repair:** generic overlap only. PaMOP correction is LLM-based
  remodeling from AMPL/solver diagnostics. Our deterministic grounding
  repair/refinement assigns numeric evidence to schema slots without a
  generative model.
- **Retrieval-assisted instantiation / deterministic grounding:** no
  distinctive PaMOP counterpart in this stage.
- **Evaluation metrics:** generic overlap only. Execution rate and solver
  success are standard for generated optimization models; our
  InstantiationReady/Coverage/TypeMatch metrics target scalar grounding.

Conclusion: **no suspicious overlap found**.

### 17.12 Readiness and next milestone

The reproduction is ready for a deliberately small benchmark evaluation
over a handful of `pamop_possible_269` ids, with strict exclusion/accounting
for:

- known `MissingStructuredDataError` ids (`28`, `51`, `57`, `123`, `126`,
  `135`);
- HF access/data failures;
- Azure auth/provider failures;
- AMPL/amplpy/Gurobi/license failures.

Next milestone: run a small, explicitly non-67, non-full-269 evaluation
slice and report execution/correction metrics from the new trace schema.
Do not claim PaMOP's published 67-problem result until exact subset,
historical GPT-4 identity, and prompt uncertainty are resolved.

---

## 18. Pilot Benchmark Status: Controlled Small Slice

**Update, 2026-08-11 (same day, sixth follow-up):** the first controlled
pilot benchmark slice has been defined, but the benchmark did **not** execute
because this shell has no Slurm `sbatch` command on `PATH`. No local fallback
was used.

### 18.1 Exact Pilot Slice

Selected from `pamop_possible_269`, excluding the six known missing structured
data ids (`28`, `51`, `57`, `123`, `126`, `135`):

```text
14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262
```

The selection is deterministic (`pamop-pilot-v1`) and uses only non-gated
metadata: LP/MILP, objective sense, variable/parameter/constraint counts,
numeric-mention count, partition node count/depth, gold-code availability,
and stable SHA-256 tie-breaking. It deliberately covers LP and MILP,
maximize/minimize objectives, simple and multi-constraint cases, low/high
numeric-mention buckets, and small/large partition trees.

Committed artifacts:

- `results/pamop/pilot/selected_ids.json`
- `results/pamop/pilot/per_problem.csv`
- `results/pamop/pilot/summary.json`
- `results/pamop/pilot/failure_analysis.csv`
- `results/pamop/pilot/correction_analysis.csv`
- `results/pamop/pilot/comparison_with_ours.csv`
- `results/pamop/pilot/run_metadata.json`
- `docs/PAMOP_PILOT_BENCHMARK.md`

No gated raw NLP4LP text, API keys, HF tokens, or AMPL/Gurobi license
material are included.

### 18.2 Execution and Correction Behavior

Attempted submission:

```bash
sbatch batch/pamop/run_pamop_pilot.sbatch
```

Observed result:

```text
sbatch: command not found
```

Therefore no selected problem was evaluated. Execution/correction metrics are
all zero or not evaluable in this blocked attempt:

- selected problems: 18
- evaluated problems: 0
- initial AMPL execution success rate: 0.0
- final execution success rate: 0.0
- correction rescue count: 0
- total LLM tokens: 0
- semantic correctness: not evaluable

This is an environment failure, not a PaMOP model failure.

### 18.3 Remaining Reproduction Uncertainties

Unresolved uncertainties remain unchanged:

- PaMOP's exact 67 NLP4LP problem ids are still unknown.
- The historical GPT-4 snapshot/deployment used by PaMOP is still unknown.
- The original PaMOP prompt wording is not public.
- Feasible AMPL execution must not be treated as semantic accuracy.
- Gold comparison can provide solver/objective evidence where `optimus-code.py`
  is available, but full model-structure equivalence is not yet established.

### 18.4 Larger Evaluation Recommendation

Decision gate: **B. FIX SYSTEMATIC ISSUE FIRST**.

Do not start a larger run until the controlled pilot can execute through
Slurm and produce per-problem PaMOP/correction/gold-comparison metrics. The
next exact action is to run:

```bash
sbatch batch/pamop/run_pamop_pilot.sbatch
```

from a Slurm login node with Azure OpenAI, HF, AMPL, and Gurobi configured.
