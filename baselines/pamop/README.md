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
> the full pipeline (LLM stage, AMPL generation, and solver execution — now
> present as an independent reconstruction, see "Evaluation boundary" below)
> is run on a confirmed-equivalent problem subset and explicitly labeled as
> such.

## What is implemented so far

**Milestone 1 — non-LLM question-partitioning stage** (paper section 3.2):

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

**Milestone 2 — LLM-based structured extraction (`G_extr`), paper section
3.2** and the provider abstraction it (and every future LLM-touching stage)
needs:

- `llm/` — a provider-agnostic interface (`generate(prompt, ModelConfig) ->
  LLMResponse`) with adapters for **OpenAI, Google Gemini, Cohere,
  Fireworks AI, and CloudRift AI**. See "LLM providers" below for which of
  these actually have a working credential on any given workstation — that
  is an environment fact, checked at runtime, not something to assume.
- `prompts/` — versioned, content-hashed prompt templates. See "Prompt
  status" below.
- `extraction.py` — calls a provider with the extraction prompt, strictly
  validates the JSON response against the paper's four required fields
  (`t_o`/`t_c`/`t_v`/`g`) plus a per-constraint vagueness score, and retries
  (asking again, never silently repairing content) on validation failure.
  Produces a `StructuredProblem` that feeds directly into Milestone 1's
  `partition.build_partition_tree` — verified wired end-to-end, live and in
  tests.

**Milestone 3 — self-augmented leaf modeling (`G_mod`, eq. 3) and bottom-up
merge (eq. 4), paper section 3.3**, plus a first-class **Azure OpenAI**
provider:

- `llm/azure_openai_provider.py` — Microsoft Azure OpenAI, same
  `generate()` contract as every other provider; handles this workstation's
  `.../openai/v1`-style endpoint and a `max_tokens`-vs-`max_completion_tokens`
  parameter quirk transparently. **This is now the primary paper-faithful
  LLM path** (a working GPT-4-family credential, unlike direct OpenAI —
  see "LLM providers" below).
- `modeling.py` — `model_leaf`/`model_all_leaves` (eq. 3: global summary +
  full variable list + this leaf's own constraints -> AMPL constraint
  text, with parent/sibling context augmentation for vague constraints),
  `merge_bottom_up` (paper: "directly merge... layer by layer from the
  bottom up" — literal concatenation, no LLM call at internal tree nodes),
  `model_root_objective` (eq. 4: the one additional call, at the root, that
  produces the objective and completes the model), and
  `build_merged_model` orchestrating all three into one `MergedModel`.
- `prompts/modeling_leaf_v1.txt`, `prompts/modeling_root_v1.txt` — two more
  reconstructed, versioned, hashed templates (same `PROVENANCE.md`
  discipline as `G_extr`'s).
- `ampl_interface.py` — the consumption contract the next milestone's AMPL
  renderer must implement (`AmplRenderer` Protocol, no implementation) —
  see "AMPL readiness" below.

Verified wired end-to-end, live, on one real NLP4LP problem: `G_extr` ->
partition tree -> `G_mod` (all leaves) -> merge -> eq. 4 root completion.

**Milestone 4 — AMPL execution and correction loop**, paper section 3.3:

- `ampl/renderer.py` — renders `MergedModel` into AMPL text.
- `ampl/validator.py` — reconstructed static checks for duplicate symbols,
  unresolved references, missing objective/variables/constraints, and
  malformed expressions.
- `ampl/executor.py` — `G_exe` execution wrapper around AMPL/Gurobi, with
  structured parse/load/solve status and model/data/environment failure
  classification.
- `correction.py` — correction trace and reconstructed `G_rev`, `G_comp`,
  and `G_remod` JSON-prompt stages, capped by the paper-specified
  `max_correction_iterations: 5`.
- `prompts/correction_{review,compare,remodel}_v1.txt` — reconstructed,
  versioned, hashed correction prompts.

Live infrastructure and tiny NLP4LP smoke tests pass with Azure OpenAI
`gpt-4.1-mini` and AMPL/Gurobi. The existing six-instance `gpt-5.4`
deployment diagnostic is also recorded in
`results/pamop/fidelity_diagnostic_gpt5/`; it improved objective-value proxy
success from 1/6 to 4/5 evaluable. Neither is an exact reproduction of
PaMOP's 67-problem result.

## Evaluation boundary

- The complete paper-defined Accuracy judgment is not reproducible from the
  available artifacts: the authors' exact 67-instance membership, prompts,
  model snapshot, and converted AMPL `data.dat` files are unavailable.
- The runner records execution, feasibility, objective production, runtime,
  correction, and token metrics. Objective equality is labeled only as an
  `OBJECTIVE_VALUE_PROXY_ONLY`, never as full semantic correctness.
- `tools/pamop_pilot_benchmark.py` preserves generated leaf/root/rendered AMPL
  artifacts and correction remodel outputs in each incremental JSON trace,
  while avoiding gated source text and prompts.

## LLM providers: what actually works, checked, not assumed

| Provider | Env var(s) | Status (verified) |
|---|---|---|
| **Azure OpenAI** | `AZURE_OPENAI_API_KEY`/`AZURE_API_KEY` + `AZURE_OPENAI_ENDPOINT`/`AZURE_API_BASE` | **Working** — live-verified, GPT-4-family (`gpt-4.1-mini`, resolves to underlying snapshot `gpt-4.1-mini-2025-04-14`). **Primary paper-faithful path as of Milestone 3.** |
| OpenAI (direct) | `OPENAI_API_KEY` | **Not usable for OpenAI** — the key present here is byte-identical to `CLOUDRIFT_API_KEY` (a CloudRift key aliased into OpenAI's env-var names); `openai_provider.py` always forces the real `api.openai.com` URL, which correctly then rejects this key with HTTP 401 |
| Gemini | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | **Not usable** — both variables exist but `GOOGLE_API_KEY`'s value is an empty string; `get_env_token()` correctly treats that as unset |
| Cohere | `COHERE_API_KEY` / `CO_API_KEY` | **Working** — live-verified |
| Fireworks AI | `FIREWORKS_API_KEY` | Present, well-formed, **not live-tested** |
| CloudRift AI | `CLOUDRIFT_API_KEY` | **Working** — live-verified (same key as the "direct OpenAI" one above, correctly used against CloudRift's own endpoint this time) |

None of this is a code defect — every provider's `ProviderAuthError` (key
absent) fires correctly, and the empty-Gemini-key case has a dedicated
regression test (`test_llm.py::test_get_env_token_treats_empty_string_as_unset`)
precisely because it's the kind of thing that's easy to assume away.
**Check your own environment's actual state before trusting this table** —
it describes one workstation at one point in time, not a permanent fact.
Full deployment enumeration on the Azure resource was not possible with the
available (inference-plane only) credential — only the two deployments
named in environment variables (`gpt-4.1-mini`, and `gpt-5.4` which is
**not** GPT-4-family) were confirmed to exist.

## Prompt status: reconstructed, never claimed as the authors' wording

No PaMOP prompt text exists in any public source (`docs/PAMOP_REPRODUCTION_PLAN.md`
§1, re-checked this milestone in §15.3 — still nothing). Every file under
`prompts/` is a full reconstruction; `prompts/PROVENANCE.md` separates, per
template, exactly which output *fields* are paper-specified (for
`extraction_v1.txt`: the four fields `t_o`/`t_c`/`t_v`/`g` and the
vagueness score, section 3.2) from the wording used to ask for them (all
reconstructed). Every template is content-hashed at load time
(`prompts.load_prompt`), and every `LLMResponse` produced from a call using
it carries that hash — so any future prompt-wording change is traceable in
run outputs, and no two differently-worded runs can be silently confused
with each other.

## Reproducibility metadata policy

Every LLM call, through every provider, returns the same `LLMResponse`
shape: provider, exact model id, timestamp, temperature/top_p/max_tokens,
prompt/completion/total token counts, latency, retry count, prompt hash,
finish reason. **No field ever holds an API key, token, or other secret**
(`test_llm.py::test_llm_response_never_has_a_field_that_looks_like_a_secret`
enforces this structurally). This is the metadata every future run's
results table should be built from — never hand-recorded or approximated.

## Our method's non-LLM inference distinction

PaMOP requires an LLM at inference time for every stage of its pipeline —
extraction, modeling, and correction alike. This repository's own
benchmarked pipeline (schema retrieval + deterministic scalar grounding,
see `manuscript/main.tex`) does not call an external generative LLM/API at
inference time at all. See `docs/PAMOP_REPRODUCTION_PLAN.md` §15.7 for the
fuller, carefully-hedged comparison (reproducibility, cost, model-version
stability, offline feasibility, privacy) — written for a future manuscript
revision to draw on, not a claim that either design is simply "better."

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
| Structured extraction (`G_extr`) prompt wording | LLM call, prompt not given | `prompts/extraction_v1.txt` — full reconstruction | See `prompts/PROVENANCE.md`; the four output fields and the vagueness score are paper-specified, the wording is ours |
| Vagueness score scale | "assign a vagueness score to each constraint" — no scale given | `[0, 1]` float | Simplest bounded scale consistent with "score" |
| `representations.from_nlp4lp_record` (Milestone 1's non-LLM bridge) | — | Still present, used only where no LLM call is wanted (e.g. `run_partitioning.py`'s smoke run) | Kept alongside `from_llm_extraction`, not replaced by it — the two are separate, clearly-named construction paths for `StructuredProblem` |
| Independent-set graph-search algorithm | "we apply graph search algorithms" (unnamed) | Connected components over the bipartite constraint–variable graph | Simplest algorithm consistent with "separate independent subgraphs" |
| Vector-similarity source | GloVe (Wikipedia 2014), variant unspecified | TF-IDF cosine similarity (`embedding_source: tfidf_fallback`) | No GloVe vectors are provisioned on this workstation yet; pluggable `VectorSimilarityProvider` interface so a real GloVe provider can be swapped in later without touching call sites |
| Keyword top-k (`k`) | "top k", value not given | `tfidf_top_k: 10` | Common default for top-k keyword schemes |
| Distance epsilon (eq. 2) | "a small fixed value", not given | `epsilon: 0.01` | Keeps eq. (2) well-scaled for similarities in [0, 1] |
| Per-layer similarity weights | "different [weights] to different layers", none given | `root` vs `default` weight sets in config | Two-tier scheme distinguishing the first split from deeper ones |
| Clustering algorithm | Distance metric given (eq. 2), clustering rule not named | Agglomerative, average linkage, cut at the leaf-similarity threshold | Assigns every point to *some* cluster — matches the paper's explicit "noise points... treated as potentially relevant... rather than removed" |
| Leaf stop conditions | "a small number of constraints" / "highly similar ones", no numbers | `leaf_stop_min_constraints: 3`, `leaf_stop_similarity_threshold: 0.6` | Documented placeholders pending any better-justified values |
| `G_mod` (eq. 3) prompt wording | LLM call, prompt not given | `prompts/modeling_leaf_v1.txt` — full reconstruction; requests plain AMPL text output, not JSON (paper: "we directly generate code... instead of formulas") | See `prompts/PROVENANCE.md`; the three input categories (g, full t_v, node-local t_c) and AMPL-as-output are paper-specified, the wording is ours |
| Vague-constraint augmentation threshold | "when modeling nodes containing vague constraints" — no numeric cutoff on `vagueness_score` | `vague_threshold: 0.5` | Midpoint of the `[0,1]` scale (itself already a reproduction choice from `G_extr`) |
| Vague-constraint augmentation content | "incorporate information from... parent and sibling nodes" — form not specified | Parent's + siblings' constraint *descriptions* | Already-modeled sibling *output* isn't reliably available under a bottom-up call order; descriptions are always available from `G_extr` |
| Eq. 4 root output structure | `M = (m_p, m_v, m_o, m_c)`, an abstract tuple — no output *format* mandated | Four labeled `### SECTION` blocks (`PARAMETERS`/`VARIABLES`/`OBJECTIVE`/`CONSTRAINTS`) | Gives the future AMPL renderer reliable section boundaries to parse |
| Merge conflict handling | Paper assumes "minimal conflict," does not describe handling any | Diagnostic-only (`MergedModel.symbol_conflicts`) for unresolved references, duplicate leaf labels, and leaf-side declarations; duplicate root declarations are rejected; never auto-repaired | "Improve on PaMOP" is out of scope for the paper-faithful path; only the paper's own literal "directly merge" is implemented |
| `G_mod`/eq.-4-specific retry count | Not discussed (distinct from `max_correction_iterations`, the later solver-debug loop) | `modeling_max_retries: 2` | Same rationale as `extraction_max_retries` |

## Running it

```bash
# Unit tests (mocked/synthetic data only, no network/gated/API access needed):
python -m pytest baselines/pamop/tests -v

# Diagnostics-only smoke run against a small live sample of the 269-block
# (requires HF_TOKEN with udell-lab/NLP4LP access; writes only aggregate
# numbers, never raw problem text, to --out):
python -m baselines.pamop.run_partitioning \
    --config baselines/pamop/configs/reconstructed_default.yaml \
    --subset pamop_possible_269 \
    --limit 20 \
    --out /tmp/pamop_smoke_summary.json

# A handful of tests hit the live, gated NLP4LP dataset (not an LLM API) and
# are marked @pytest.mark.requires_network -- included by default when
# network is reachable, auto-skipped otherwise (see tests/conftest.py):
python -m pytest baselines/pamop/tests -v -m requires_network
```

The `llm`/`extraction` test suites are entirely mocked (fake in-process
providers) and never make a network or API call, by design — a tiny live
LLM smoke test was run manually for this milestone instead (not part of the
default test suite, since it costs real API tokens); see
`docs/PAMOP_REPRODUCTION_PLAN.md` §16.11 for the latest full-pipeline
Azure smoke result.

## Known limitation (fixed) and its actual root cause

An earlier version of this milestone reported a loader failure and
attributed it to bare-vs-suffixed id paths (e.g. `3` vs `3-infeasible`) in
an older, cached dataset snapshot. Re-checking the **current live**
`udell-lab/NLP4LP` snapshot found that upstream's 2026-04-20 "Final revision
cleanup" commit already renamed every suffixed directory back to a bare
numeric id (0 suffixed directories remain). `data.py` still resolves
suffixed variants defensively (`_resolve_problem_info_path`, in case a
future snapshot reintroduces them), but that was not the actual failure.

The **real** cause: 6 of the 269 pre-PaMOP ids — **28, 51, 57, 123, 126,
135** — have a problem directory (`description.txt`, `metadata.json`,
`optimus-code.py`) but genuinely **no `problem_info.json` anywhere**, bare
or suffixed. These are almost certainly the dataset's own historical
"-infeasible"/"-unsolved" entries, renamed by the cleanup commit without
ever having had a structured formulation generated. There is no data to
recover for these 6 ids — `data.py` now raises a distinct
`MissingStructuredDataError` (a `FileNotFoundError` subclass) for them
instead of leaking a generic `RemoteEntryNotFoundError`, so callers can
count and report this cleanly rather than mistake it for an auth/network
failure. See `baselines/pamop/tests/test_data.py` for both a mocked
regression test of the resolution logic and a live regression test
(`@pytest.mark.requires_network`) pinned to these exact 6 ids.

## AMPL execution and correction

AMPL is installed user-locally in `~/.venvs/gurobi` via `amplpy` modules
(`base`, `highs`, `gurobi`). The default unit tests do not require AMPL or
network access; live execution uses:

```bash
PAMOP_AMPLPY_PYTHON=/home/soroush/.venvs/gurobi/bin/python
```

`ampl/executor.py` classifies failures as:

- `model_error` — syntax/formulation/solver-status failures that may enter
  the PaMOP correction loop.
- `data_error` — missing external data files or dataset records; not
  remodeled.
- `environment_error` — missing AMPL/amplpy, solver/license failures,
  provider/auth failures, or timeouts; not remodeled.

Correction trace entries record the AMPL hash, execution status, prompt
hashes, provider/model metadata, token counts, latency, review/comparison
results, remodeling status, and final success/failure. No raw credentials
or gated NLP4LP text are serialized by these trace types.

## Next milestone

See [`docs/PAMOP_REPRODUCTION_PLAN.md`](../../docs/PAMOP_REPRODUCTION_PLAN.md)
sections 14–17 for implementation status. Next: a deliberately small
benchmark evaluation on the `pamop_possible_269` superset, excluding the
six known `MissingStructuredDataError` ids from model-failure counts, and
reporting execution/correction metrics without claiming PaMOP's exact
67-problem result.
