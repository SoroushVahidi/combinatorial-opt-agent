## Benchmark: Ported retrieval/UI/augmentation improvements (commit c6e5336)

**Scope:** Measure the impact of the Copilot-port changes that landed in `c6e5336` without adding new methods:

- Embedding cache fix (index built once and reused).
- Short-query expansion in retrieval baselines and `retrieval.search.search`.
- Written-number recognition / data augmentation support for NLP4LP downstream.
- Graceful missing-formulation handling (UX only).

All numbers below are from the current `main` after the port; ablations are done by toggling existing flags or inspecting coverage, not by rewriting methods.

---

## 1. Retrieval metrics: short-query expansion ON vs OFF

**Entry point:** `training/evaluate_retrieval.py` (SBERT-based retrieval over the catalog).

**Changes made for ablation:**

- Added CLI flag:

  ```bash
  --no-short-query-expansion
  ```

- Hooked into `retrieval.search.search` via:

  ```python
  search_results = search(
      query,
      catalog=catalog,
      model=model,
      embeddings=embeddings,
      top_k=k,
      expand_short_queries=not args.no_short_query_expansion,
  )
  ```

**Commands run (in venv):**

```bash
# Short-query expansion ON (default)
python -m training.evaluate_retrieval \
  --num 300 --seed 999 --regenerate \
  --results-dir comparison_reports/retrieval_short_on

# Short-query expansion OFF (ablated)
python -m training.evaluate_retrieval \
  --num 300 --seed 999 \
  --eval-file data/processed/eval_500.jsonl \
  --results-dir comparison_reports/retrieval_short_off \
  --no-short-query-expansion
```

**Dataset:** 300 eval instances generated from the catalog using existing `generate_queries_for_problem` (mix of short templates and longer NL queries).

**Results (from `comparison_reports/retrieval_short_on/eval.json` and `..._off/eval.json`):**

| Setting                     | P@1   | P@5   | MRR@10 | nDCG@10 | Coverage@10 |
|-----------------------------|-------|-------|--------|---------|-------------|
| **Expansion ON (default)**  | 0.973 | 0.997 | 0.985  | 0.989   | 1.000       |
| **Expansion OFF (ablated)** | 0.990 | 0.997 | 0.994  | 0.995   | 1.000       |

**Interpretation:**

- On this SBERT-based catalog evaluation, **disabling** short-query expansion slightly **increases** P@1 and MRR@10 on this 300-instance sample.
- Coverage@10 is already saturated (1.0) in both cases; P@5 is identical.
- Conclusion: for the current SBERT setup and this evaluation protocol, short-query expansion **does not provide a clear quantitative gain** and may slightly hurt pure SBERT retrieval on the generated eval set. It remains a heuristic that may help very short, hand-typed queries in the UI, but there is no evidence yet that it improves formal retrieval metrics.

---

## 2. Written-number recognition: coverage impact

**Entry point:** `tools/nlp4lp_downstream_utility.py` (no method changes for the benchmark).

We measured how often the new written-number logic (`_word_to_number`, updated `_extract_num_tokens`, `_extract_num_mentions`) adds *additional* numeric tokens/mentions on the existing NLP4LP eval queries.

**Script (run inline):**

```python
from pathlib import Path
import json
import tools.nlp4lp_downstream_utility as dutil
from tools.nlp4lp_downstream_utility import NUM_TOKEN_RE

p = Path("data/processed/nlp4lp_eval_orig.jsonl")
total_queries = 0
extra_token_mentions = 0
extra_mention_mentions = 0
with p.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        q = (obj.get("query") or "").strip()
        if not q:
            continue
        total_queries += 1

        toks = dutil._extract_num_tokens(q, "orig")
        word_only = [t for t in toks if not NUM_TOKEN_RE.fullmatch(str(t.raw).strip())]
        if word_only:
            extra_token_mentions += 1

        ments = dutil._extract_num_mentions(q, "orig")
        word_m = [m for m in ments if not NUM_TOKEN_RE.fullmatch(str(m.tok.raw).strip())]
        if word_m:
            extra_mention_mentions += 1

print("Total eval queries:", total_queries)
print("Queries with additional written-number NumToks:", extra_token_mentions)
print("Queries with additional written-number mentions:", extra_mention_mentions)
```

**Output:**

- `Total eval queries: 331`
- `Queries with additional written-number NumToks: 145`
- `Queries with additional written-number mentions: 145`

**Interpretation:**

- Approximately **44%** of NLP4LP eval queries (145 / 331) contain at least one **new numeric token/mention** that is *only* visible via the written-number logic.
- This closes a large coverage gap in the numeric-mention extraction stage (Bottleneck 3) and ensures that downstream constrained assignment and optimization-role methods can “see” many more numeric cues.
- This is a **coverage and input-quality improvement**; we did **not** re-run the full downstream benchmark in this step, so we do **not** claim specific changes to Exact20 or InstantiationReady here.

---

## 3. Embedding cache / index build: runtime impact

The embedding cache fix ensures that SBERT-based retrieval builds the catalog index once and reuses it via the `embeddings` argument to `retrieval.search.search`. In the *old* pattern, every query built a fresh index when `embeddings=None`.

We attempted to benchmark this directly by timing 100 calls with `embeddings=None` vs 100 calls with a pre-built `embeddings` array. However:

- On the Wulver login node, loading the SBERT model in a small ad-hoc script **without** a full SLURM GPU allocation repeatedly hit resource constraints (`can't start new thread`) due to HuggingFace/Transformers’ internal thread pools.
- Within the existing `training.evaluate_retrieval` entry point (which runs under a proper SLURM job in practice), the index is already built once and reused; the cost is dominated by model loading and embedding, not by per-query overhead.

**Practical outcome:**

- We **cannot provide a clean wall-clock comparison** for old-vs-new index usage from the login node without more invasive thread-pool tuning or a dedicated SLURM job just for timing.
- Qualitatively, the pattern is correct: `search()` with a pre-built `embeddings` array avoids the O(N_catalog) per-query re-encoding cost that would otherwise occur when `embeddings=None`.
- For actual deployed usage (web app / SLURM jobs), this change is an unambiguous **latency and load improvement**, even though we have not quantified the factor precisely here.

---

## 4. Graceful missing-formulation handling

This is a UX-only change in `retrieval.search.format_problem_and_ip`:

- When a catalog problem lacks a structured `formulation` (no variables, constraints, or objective), the function now returns:

  > **Formulation not yet available.** This problem is in the catalog but its structured ILP has not been added yet. The description above may still help you understand the problem structure.

  rather than rendering empty collapsible sections.

We did **not** run a quantitative benchmark here since this does not affect numeric metrics; it improves:

- Honesty: we do not imply a full IP where none exists.
- User experience: the UI clearly explains the situation instead of showing blank sections.

---

## 5. Summary and honest conclusion

**What was benchmarked:**

- Retrieval metrics via `training.evaluate_retrieval` on 300 SBERT-based eval instances:
  - Short-query expansion ON vs OFF (using `--no-short-query-expansion`).
- Written-number recognition coverage:
  - Fraction of NLP4LP eval queries where the new logic yields **additional numeric tokens/mentions**.
- Embedding cache:
  - Confirmed structurally (via code path) but not numerically benchmarked due to resource limits on Wulver login nodes.

**Main metric differences:**

- Short-query expansion:
  - On this SBERT eval sample, turning expansion **OFF** slightly **improves** P@1 and MRR@10; Coverage@10 and P@5 are unchanged.
  - There is **no evidence** from this test that short-query expansion improves SBERT retrieval metrics; its value is likely in UI ergonomics for very short, human-typed queries rather than in formal benchmark gains.
- Written-number recognition:
  - Roughly **44%** of NLP4LP eval queries gain at least one additional numeric token/mention from written-number handling.
  - This significantly improves coverage of numeric cues and should help downstream methods, but we **have not re-run** the full downstream benchmark here, so we **do not claim** specific gains in Exact20 or InstantiationReady.

**Runtime differences:**

- Embedding cache:
  - We were unable to obtain a stable, quantifiable old-vs-new timing comparison on the login node due to `can't start new thread` limitations in a standalone script.
  - However, the change clearly eliminates repeated index builds in hot loops and is structurally a strict latency improvement in any environment where `search()` was previously being called without pre-built `embeddings`.

**Honest conclusion:**

- The ported changes in `c6e5336`:
  - **Improve robustness and input quality** (written-number recognition; graceful missing-formulation handling).
  - **Improve infrastructure and UX** (embedding index reuse; PDF upload; short-query expansion for UI ergonomics).
  - Do **not** currently show a clear, quantitative retrieval-metric gain from short-query expansion on the SBERT eval we ran; if anything, the heuristic slightly lowers P@1/MRR in that specific setting.
  - Likely reduce query latency and model load in practice via the embedding cache fix, though we have not attached a precise speedup factor due to environment constraints.
- No new methods were added for this benchmark; we only toggled existing knobs and inspected coverage. Further work would be needed to:
  - Quantify written-number impact on full NLP4LP downstream metrics (TypeMatch, InstantiationReady, Exact20).
  - Revisit short-query expansion tuning or restrict it to specific baselines/entry points where it is demonstrably helpful.

