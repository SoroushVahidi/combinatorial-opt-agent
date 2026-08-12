# ORLM baseline (scaffold only — not runnable in this environment)

**Status: SCAFFOLDED, 2026-08-12 (Phase 4).** No model weights downloaded,
no inference run. This directory holds the adapter/runner interface and
environment documentation so a future agent with GPU access can start
immediately, per `docs/BASELINE_IMPLEMENTATION_ROADMAP.md`'s ranking
(ORLM is next after PaMOP).

## Citation

Chenyu Huang, Zhengyang Tang, Shixi Hu, Ruoqing Jiang, Xin Zheng, Dongdong
Ge, Benyou Wang, Zizhuo Wang. "ORLM: A Customizable Framework in Training
Large Language Models for Automated Optimization Modeling."
arXiv:[2405.17743](https://arxiv.org/abs/2405.17743) (May 2024, v5 April
2025); accepted at *Operations Research* (2025).

## Official code and weights (verified 2026-08-12, primary sources)

- **Code:** [github.com/Cardinal-Operations/ORLM](https://github.com/Cardinal-Operations/ORLM)
  — public, active (272 stars, last push Sept 2025), **Apache-2.0**
  license, default branch `master` (not `main`).
- **Weights:** only **`CardinalOperations/ORLM-LLaMA-3-8B`** is confirmed
  publicly retrievable on HuggingFace (8.03B params, bf16, `llama3`
  license — this is a Meta license term, separate from the code's
  Apache-2.0). The paper and repo README also name `ORLM-Mistral-7B` and
  `ORLM-Deepseek-Math-7B-Base` checkpoints with published benchmark
  scores, but **neither resolves as a public HF repo** under any
  plausible path — verify independently before relying on either; treat
  as unconfirmed, not as "also available."
- **Correction to earlier planning note:** `docs/BASELINE_IMPLEMENTATION_ROADMAP.md`
  originally said "code+weights public" without deep verification. This
  is accurate for the code and for exactly one of the three named
  checkpoints (LLaMA-3-8B), not for all three.

## Dependencies / environment

- `requirements.txt` (upstream) pins `vllm==0.3.2` (Feb 2024 — old;
  likely needs an isolated env for CUDA/torch compatibility) and
  `openai==0.28.1` (legacy SDK, appears used only by the repo's optional
  GPT-baseline comparison scripts, not ORLM inference itself).
  torch/transformers/deepspeed/accelerate/peft are unpinned upstream.
- **`coptpy`** — Cardinal Operations' own commercial solver (COPT). ORLM's
  official generation target is COPT solver code, **not** Pyomo,
  GurobiPy, or a plain LP file. This is an additional solver dependency
  parallel to this repo's existing Gurobi/AMPL setup
  (`~/.venvs/gurobi`) — a COPT license (community/academic tiers likely
  exist but were not independently verified) is required to actually
  *execute* any code ORLM generates, separate from running the LLM itself.
- **GPU:** 8B params in bf16 ≈ 16GB for weights alone; with an 8192-token
  max sequence length, a single 24GB-class consumer/workstation GPU
  (RTX 3090/4090, A5000) is plausible for **inference**. Multi-GPU/
  DeepSpeed is used for *training* in the original paper, not required
  for using the published checkpoint as-is.
- **Not currently available on this workstation** (verified 2026-08-12):
  no GPU provisioned for this environment, no COPT license configured.
  This scaffold is therefore interface/documentation only.

## Input / output format (from the official prompt template)

ORLM expects a **fixed prompt template**, approximately:

```
Below is an operations research question. Build a mathematical model and
corresponding python code using `coptpy` that appropriately addresses the
question.
# Question:
{Question}
# Response:
```

Output is free-text: a mathematical model description followed by
`coptpy` (COPT solver) Python code — not a solver-agnostic LP file, not
GurobiPy, not Pyomo.

## Does ORLM cover NLP4LP?

**No.** The paper evaluates on NL4OPT, MAMO (EasyLP/ComplexLP), and
IndustryOR only (self-reported: 85.7% / 82.3% / 37.4% / 38.0%, micro-avg
71.4%). There is no published NLP4LP overlap. Adapting ORLM to NLP4LP
requires:
1. Wrapping each NLP4LP query in ORLM's exact prompt template (see
   `data_adapter.py` below — interface defined, not yet populated with
   the verified exact template string from the upstream repo).
2. A new execution/scoring harness: run the generated `coptpy` code,
   capture whether it parses/executes/solves, and — if a fair comparison
   to this repo's own metrics is wanted — a bridge from "solved
   correctly" to something comparable to (but not identical to)
   `InstantiationReady`. See "Fair comparison" below for why this is not
   a drop-in metric swap.

## Fair comparison caveats (do not conflate metrics)

ORLM solves a **broader, different task** than this repository's core
pipeline: full NL → mathematical-model-and-code generation, vs. this
repo's schema-conditioned scalar parameter grounding given an already-known
template. They are not directly comparable on `InstantiationReady`/
`Coverage`/`TypeMatch`. Comparable and incomparable dimensions:

| Dimension | This repo's pipeline | ORLM |
|---|---|---|
| End-to-end model generation | No (schema-conditioned slot-filling only) | Yes |
| Executable formulation | Restricted subset only (20/331, SciPy HiGHS) | Yes (COPT), for problems where generation succeeds |
| Solver success as a metric | Secondary (structural check first) | Primary reported metric |
| External generative LLM required | No (core pipeline) | Yes (8B LLM, GPU) |
| GPU requirement | None | Yes (~16-24GB) |
| Deterministic | Yes | No (LLM sampling, unless temperature=0 and still not guaranteed deterministic) |
| Runtime cost per query | Milliseconds, CPU | Seconds-minutes, GPU + LLM generation |
| Schema/template requirement | Yes (fixed catalog) | No (open-ended generation) |

A fair write-up, if this is ever run, should report ORLM's own metrics
(execution rate, accuracy vs. gold objective) as a **separate table**,
explicitly not folded into `InstantiationReady` comparisons, with the
above table's caveats stated alongside.

## Directory contents

- `config.py` — `OrlmConfig` dataclass (deployment path, prompt template,
  GPU/COPT requirements as documented fields, not hidden constants).
- `data_adapter.py` — interface (`build_orlm_prompt(nlp4lp_query: str) -> str`)
  for wrapping an NLP4LP query in ORLM's prompt format. **Not yet
  implemented against the verified exact upstream template string** —
  the template above is reconstructed from public documentation, not
  copy-pasted from the upstream repo's own prompt file (which should be
  fetched and diffed against this reconstruction before first real use).
- `runner.py` — interface (`OrlmRunner.generate(prompt: str) -> str`) for
  the actual model call. **Not implemented** — requires `vllm`/
  `transformers` + the downloaded checkpoint + a GPU, none available here.
- `output_normalizer.py` — interface
  (`parse_orlm_output(raw: str) -> OrlmParsedOutput`) for splitting ORLM's
  free-text response into the model description and the `coptpy` code
  block, and a hook for execution-outcome normalization.

## Exact first practical smoke-test milestone

1. Provision a single 24GB-class GPU and a COPT license (community tier
   if available).
2. `git clone https://github.com/Cardinal-Operations/ORLM` (or wrap it as
   a dependency, not vendored into this repo — do not copy large upstream
   code unnecessarily) in an isolated environment matching its pinned
   `vllm==0.3.2`.
3. Download `CardinalOperations/ORLM-LLaMA-3-8B` from HuggingFace.
4. Verify `data_adapter.py`'s reconstructed prompt template against the
   upstream repo's actual prompt file; fix any discrepancy.
5. Run **one** NLP4LP query through the reference generation script using
   the exact upstream prompt template.
6. Verify the generated `coptpy` code at least *parses* — this alone is a
   meaningful first checkpoint before attempting full solver execution.

Do not attempt steps 5-6 without 1-2 in place; do not attempt a
benchmark-scale run before the single-query smoke test succeeds.
