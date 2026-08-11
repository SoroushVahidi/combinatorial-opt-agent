# PaMOP reproduction (independent, unofficial)

This is our own independent reproduction scaffold for PaMOP, built entirely
from the published paper text. **No official public PaMOP implementation was
found** despite an extensive search (GitHub, GitLab, Hugging Face Spaces,
arXiv, OpenReview, author pages, curated LLM4Opt paper lists) — see
[`docs/PAMOP_REPRODUCTION_PLAN.md`](../../docs/PAMOP_REPRODUCTION_PLAN.md)
section 1 for the full search record. Nothing in this directory is, or
claims to be, the authors' own code.

**Citation:**
Xiaotian Pan, Junhao Fang, Feng Wu, Sijia Zhang, Yi-Xiang Hu, Shaoang Li,
Xiang-Yang Li. "Guiding Large Language Models in Modeling Optimization
Problems via Question Partitioning." *IJCAI 2025*, pp. 2657–2665.
DOI: [10.24963/ijcai.2025/296](https://doi.org/10.24963/ijcai.2025/296).

**Published PaMOP numbers (NLP4LP, GPT-4), for literature reference only:**

| Metric | Value |
|---|---|
| Accuracy | **62.3%** |
| Code executability ("Execution Rate") | **86.8%** |

> These two numbers are **PUBLISHED PAPER RESULTS**, taken from the paper's
> own Table 1, **not results this code has produced or reproduced**. Nothing
> in this repository should ever caption a number from this scaffold as
> matching, reproducing, or being comparable to these two figures until the
> full pipeline (LLM stage, AMPL generation, solver execution — none of
> which exist yet, see "Not implemented yet" below) is built, run on a
> confirmed-equivalent problem subset, and explicitly labeled as such.

## What is implemented in this milestone

Only the **non-LLM question-partitioning stage** (paper section 3.2):

- `representations.py` — the `StructuredProblem` data structure (the
  paper's root-node content: objective text, constraint texts, variable/
  parameter texts, a global summary).
- `partition.py` — the actual algorithm:
  - **Independent-set separation** at the root: a bipartite graph between
    constraints and variables (edges = keyword-match confidence), split into
    connected components.
  - **Constraint clustering** at every deeper layer: the paper's exact
    distance formula (eq. 2, `d_ij = 1/(s_ij+eps) - 1/(1+eps)`) over a
    weighted combination of three similarity signals (adjacency/context,
    TF-IDF keyword overlap, embedding cosine similarity), recursed until
    each leaf has "a small number of constraints or highly similar ones."
- `config.py` + `configs/*.yaml` — every constant the paper leaves
  unspecified is a named config field, not a hidden literal (see below).
- `data.py` — an NLP4LP loader restricted to the **269-entry
  `pamop_possible_269` subset** (see "Dataset scope" below).
- `run_partitioning.py` — a diagnostics-only CLI for running the pipeline
  over a live subset (no gated text is ever written to a committed file).

## Not implemented yet

Deliberately out of scope for this milestone (interfaces are not even
stubbed beyond what `config.py`'s `execution`/`correction`/`llm` sections
declare):

- The LLM-based "structured extraction" call (`G_extr`) that the paper uses
  to *produce* a `StructuredProblem` from raw free text. This milestone
  instead builds `StructuredProblem` directly from NLP4LP's own pre-existing
  structured fields (see "Reconstructed choices" below) so partitioning can
  be built and tested without an LLM call.
- Self-augmented leaf-node modeling (`G_mod`) and AMPL generation.
- The three-layer correction loop (basic inspection, solver-debug `G_exe`,
  reverse translation `G_rev`/`G_comp`/`G_remod`).
- AMPL generation and Gurobi execution/validation.
- Any evaluation against PaMOP's published Accuracy / Execution-Rate / CE /
  RE metrics.

## Dataset scope: `pamop_possible_269`, never "PaMOP's 67"

PaMOP cites an NLP4LP release dated **2024-05-13** (54 LP + 13 MILP, 67
problems total — see report section 13.4). That exact release is **not**
identifiable inside the `udell-lab/NLP4LP` snapshot this repository can
access (it has grown to 361 problems and does not preserve the original
numbering). What *is* established (report section 13.6–13.8):

- HF problem ids **1–269** existed continuously since the dataset's first
  Hugging Face upload (2024-11-02) — i.e. before PaMOP was ever published,
  so PaMOP's 67 problems, whatever they are, can only be drawn from here.
- HF problem ids **270–361** were added 2026-02-12 through 2026-02-27 — six
  months *after* PaMOP's IJCAI 2025 publication — and **cannot** be part of
  its evaluation set under any interpretation.

`data.py` therefore exposes exactly one subset, `SUBSET_POSSIBLE_269`
(string value `"pamop_possible_269"`), covering ids 1–269. **There is no
`"pamop_67"` subset anywhere in this codebase, and there never should be**
— it would misrepresent an unverified 269-problem superset as PaMOP's
confirmed exact evaluation set. `data.assert_not_post_pamop()` raises
`PostPamopIdError` for any id ≥ 270, and `test_data.py` asserts that
`"pamop_67"` is rejected as an unknown subset name. Exact membership of
PaMOP's 67 within this 269-problem block remains **unresolved** — see report
section 13.5 for the two archival sources (`nlp4lp.vercel.app`, an
OpenReview supplementary link) that could resolve it but are both gated
behind interactive-only verification challenges.

**No claim of exact PaMOP-subset reproduction is permitted anywhere in this
codebase or its outputs until that exact-subset question, and the missing
prompt/model details listed below, are resolved.**

## Configuration: two configs, by design

- **`configs/paper_faithful.yaml`** — only values the paper actually states
  (`temperature: 0.2`, `max_correction_iterations: 5`,
  `generation_target: AMPL`, `solver_backend: gurobi_via_ampl`). Every other
  partitioning/correction/LLM constant is `null`. **Running the partitioning
  stage with this config raises `UnspecifiedPaperDetailError`** — it does
  not silently substitute a guess. This is intentional: it makes "the paper
  doesn't say" a loud, structural fact of the codebase rather than a comment
  someone can miss.
- **`configs/reconstructed_default.yaml`** — every field filled with a
  documented choice so development/testing can run today. Every
  non-paper-specified value is marked `# REPRODUCTION CHOICE` inline with a
  one-line justification.

## Reconstructed choices (this milestone)

Full sourcing and A/B/C classification lives in
[`docs/PAMOP_REPRODUCTION_PLAN.md`](../../docs/PAMOP_REPRODUCTION_PLAN.md)
sections 2 and 9. Summary for what's actually implemented here:

| Detail | Paper says | This milestone's choice | Why |
|---|---|---|---|
| Structured extraction (`G_extr`) | LLM call, prompt not given | Read directly from NLP4LP's own `objective`/`constraints`/`variables`/`parameters` fields | Bridges the paper's LLM extraction step with data the dataset already ships, so partitioning can be tested without implementing the LLM stage yet. **Not** a reproduction of `G_extr` itself. |
| Independent-set graph-search algorithm | "we apply graph search algorithms" (unnamed) | Connected components over the bipartite constraint–variable graph | Simplest algorithm consistent with "separate independent subgraphs" |
| Vector-similarity source | GloVe (Wikipedia 2014), variant unspecified | TF-IDF cosine similarity (`embedding_source: tfidf_fallback`) | No GloVe vectors are provisioned on this workstation yet; pluggable `VectorSimilarityProvider` interface so a real GloVe provider can be swapped in later without touching call sites |
| Keyword top-k (`k`) | "top k", value not given | `tfidf_top_k: 10` | Common default for top-k keyword schemes |
| Distance epsilon (eq. 2) | "a small fixed value", not given | `epsilon: 0.01` | Keeps eq. (2) well-scaled for similarities in [0, 1] |
| Per-layer similarity weights | "different [weights] to different layers", none given | `root` vs `default` weight sets in config | Two-tier scheme distinguishing the first split from deeper ones |
| Clustering algorithm | Distance metric given (eq. 2), clustering rule not named | Agglomerative, average linkage, cut at the leaf-similarity threshold | Assigns every point to *some* cluster — matches the paper's explicit "noise points... treated as potentially relevant... rather than removed" |
| Leaf stop conditions | "a small number of constraints" / "highly similar ones", no numbers | `leaf_stop_min_constraints: 3`, `leaf_stop_similarity_threshold: 0.6` | Documented placeholders pending any better-justified values |

## Running it

```bash
# Unit tests (synthetic data only, no network/gated access needed):
python -m pytest baselines/pamop/tests -v

# Diagnostics-only smoke run against a small live sample of the 269-block
# (requires HF_TOKEN with udell-lab/NLP4LP access; writes only aggregate
# numbers, never raw problem text, to --out):
python -m baselines.pamop.run_partitioning \
    --config baselines/pamop/configs/reconstructed_default.yaml \
    --subset pamop_possible_269 \
    --limit 20 \
    --out /tmp/pamop_smoke_summary.json
```

## Known limitation surfaced by the smoke run

A handful of ids in the 1–269 range (e.g. `3`, `28`, `51`, ...) exist in the
Hugging Face dataset only under a suffixed form (`3-infeasible`,
`28-unsolved`, ...), not the bare numeric id `data.py` currently fetches.
The smoke run correctly counts these as per-problem failures rather than
crashing (see `run_partitioning.py`'s failure tracking) — this is a known,
minor loader gap, not a partitioning-algorithm bug. A future milestone
should resolve suffixed ids to their real path before counting them as
`PostPamopIdError`-eligible or not.

## Next milestone

See [`docs/PAMOP_REPRODUCTION_PLAN.md`](../../docs/PAMOP_REPRODUCTION_PLAN.md)
section 14 for implementation status and the exact next step (the LLM-based
`G_extr` structured-extraction stage, gated behind acquiring an LLM API
configuration decision per report section 7.3).
