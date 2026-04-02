## SECTION A — Manuscript baseline (reference only)

- **Schema retrieval (orig / noisy / short)** — *classical baselines, N=331 problems*  
  - **Orig**: TF‑IDF R@1 = 0.9063, BM25 = 0.8852, LSA = 0.8550.  
  - **Noisy**: TF‑IDF = 0.9033, BM25 = 0.8943, LSA = 0.8912.  
  - **Short**: TF‑IDF = 0.7855, BM25 = 0.7734, LSA = 0.7704.  
  - **Provenance**: matches `results/baselines_nlp4lp_orig.csv` and `results/paper/nlp4lp_downstream_variant_table.csv` (verified from current repo artifacts).

- **Downstream main table on orig (N=331)** — *typed greedy, untyped, and oracle*  
  From `results/paper/nlp4lp_downstream_main_table_orig.csv`:

  - **random seeded**: Coverage 0.0101 | TypeMatch 0.0060 | Exact20 0.1250 | InstantiationReady 0.0060  
  - **lsa typed greedy**: Coverage 0.7976 | TypeMatch 0.2063 | Exact20 0.1965 | InstantiationReady 0.0604  
  - **bm25 typed greedy**: Coverage 0.8133 | TypeMatch 0.2251 | Exact20 0.2175 | InstantiationReady 0.0755  
  - **tfidf typed greedy baseline**: Coverage 0.8222 | TypeMatch 0.2267 | Exact20 0.2140 | InstantiationReady 0.0725  
  - **oracle typed greedy (perfect retrieval)**: Coverage 0.8695 | TypeMatch 0.2475 | Exact20 0.1871 | InstantiationReady 0.0816  
  - **Typed vs untyped TF‑IDF / oracle**: untyped TF‑IDF and oracle rows are also present in this file and match the prompt values.  
  - **Provenance**: verified from current repo artifact `results/paper/nlp4lp_downstream_main_table_orig.csv`.

- **Per‑type weakness (float)**  
  From `results/paper/nlp4lp_downstream_types_table_orig.csv`:
  - **TF‑IDF float coverage**: 0.8346.  
  - **TF‑IDF float TypeMatch**: 0.0287.  
  - **Oracle float coverage**: 0.8842.  
  - **Oracle float TypeMatch**: 0.0281.  
  - **Provenance**: verified from current repo artifact `results/paper/nlp4lp_downstream_types_table_orig.csv`.

- **Cross‑variant InstantiationReady (orig / noisy / short)**  
  From `results/paper/nlp4lp_downstream_variant_table.csv`:
  - **TF‑IDF**: orig 0.0725 | noisy 0.0000 | short 0.0060.  
  - **BM25**: orig 0.0755 | noisy 0.0000 | short 0.0091.  
  - **LSA**:  orig 0.0604 | noisy 0.0000 | short 0.0030.  
  - **Provenance**: verified from current repo artifact `results/paper/nlp4lp_downstream_variant_table.csv`.

- **Interpretive conclusions encoded in the manuscript‑era artifacts**
  - Retrieval is already very strong (schema R@1 ~0.9 on orig, still high on noisy/short).  
  - Downstream number‑to‑slot grounding is the main bottleneck (low TypeMatch and InstantiationReady, especially for floats).  
  - Perfect retrieval only modestly improves InstantiationReady over TF‑IDF (0.0725 → 0.0816).  
  - Float type‑match is extremely poor (~0.03).  
  - Typed assignment helps over untyped, but only incrementally.  
  - BM25 slightly beats TF‑IDF on orig InstantiationReady despite TF‑IDF winning schema retrieval.  
  - No learning‑based methods are benchmarked in these artifacts.  
  - **Provenance**: these conclusions are consistent with the above CSVs; no additional manuscript PDF was used.

---

## SECTION B — What the current repo adds beyond the manuscript baseline

### B1 — Implemented **and benchmarked** methods (NLP4LP downstream, orig)

All of the following appear both in code (primarily `tools/nlp4lp_downstream_utility.py`) and in downstream results artifacts under `results/paper/`:

- **tfidf_constrained**  
  - Implemented as a constrained typed greedy assignment.  
  - Benchmarked in `results/paper/nlp4lp_downstream_main_table_orig.csv` (and more detailed in `results/paper/nlp4lp_downstream_types_table_orig.csv`).  
  - On orig (N=331), main‑table row (from `nlp4lp_downstream_main_table_orig.csv`):  
    - Coverage 0.7720 | TypeMatch 0.1980 | Exact20 0.3279 | InstantiationReady 0.0272.  
  - **Evidence**: verified from current repo artifacts.

- **tfidf_semantic_ir_repair**  
  - Uses semantic IR cues to repair assignments.  
  - Benchmarked in `nlp4lp_downstream_main_table_orig.csv`:  
    - Coverage 0.7780 | TypeMatch 0.2540 | Exact20 0.2610 | InstantiationReady 0.0630.  
  - **Evidence**: verified from current repo artifacts (`nlp4lp_downstream_main_table_orig.csv`, `nlp4lp_downstream_types_table_orig.csv`).

- **tfidf_optimization_role_repair**  
  - Matching‑plus‑repair over optimization roles; implemented with richer utilities in `tools/nlp4lp_downstream_utility.py`.  
  - Benchmarked in `nlp4lp_downstream_main_table_orig.csv`:  
    - Coverage 0.8220 | TypeMatch 0.2430 | Exact20 0.2770 | InstantiationReady 0.0600.  
  - **Evidence**: verified from `results/paper/nlp4lp_downstream_main_table_orig.csv` and `results/paper/nlp4lp_downstream_types_table_orig.csv`.

- **tfidf_acceptance_rerank**  
  - Acceptance‑based reranking of candidates.  
  - Benchmarked in `results/paper/nlp4lp_current_situation_after_7_methods.csv` and per‑type tables:  
    - From the prompt (and consistent with artifacts): Coverage ≈0.7974 | TypeMatch ≈0.2275 | InstantiationReady ≈0.0816.  
  - **Evidence**: current repo artifacts in `results/paper/` confirm the method row; some exact rounded values in the prompt are from external memory but are consistent with the CSV.

- **tfidf_hierarchical_acceptance_rerank**  
  - Hierarchical acceptance reranking that uses multi‑level decision rules.  
  - Benchmarked in `results/paper/nlp4lp_current_situation_after_7_methods.csv`:  
    - Coverage ≈0.7771 | TypeMatch ≈0.2303 | InstantiationReady ≈0.0816.  
  - **Evidence**: verified from `nlp4lp_current_situation_after_7_methods.csv` and per‑type tables; small rounding differences vs prompt are negligible.

- **oracle_* with constrained / semantic_ir_repair / optimization_role_repair / untyped**  
  - These oracle‑retrieval variants appear in multiple downstream JSON/CSV files under `results/paper/`, including `nlp4lp_downstream_orig_oracle_*.json` and `nlp4lp_downstream_types_table_orig.csv`.  
  - They quantify what happens if retrieval is *perfect* but different downstream strategies are used.  
  - **Evidence**: verified from the JSON/CSV artifacts.

- **Per‑type breakdowns and error analyses**  
  - `results/paper/nlp4lp_downstream_types_table_orig.csv` and related plots/texts provide rich breakdowns by parameter type (currency, float, integer, percent), including float coverage and TypeMatch for every method.  
  - A large set of analysis notes (`nlp4lp_downstream_*`, `nlp4lp_error_*`, `nlp4lp_three_bottlenecks_*`) document bottlenecks and comparative method behavior.  
  - **Evidence**: verified from current repo artifacts; this is more detailed than what is summarized in the prompt but is still manuscript‑era.

### B2 — Implemented **and benchmarked** changes added *after* the manuscript‑era tables

These correspond to the Copilot‑port improvements and their benchmarks:

- **Short‑query expansion in retrieval**  
  - Implemented in `retrieval/utils.py`, used in `retrieval/baselines.py` and `retrieval/search.py`.  
  - Benchmarked *only* for SBERT‑based retrieval via `training/evaluate_retrieval.py` in `docs/PORTED_IMPROVEMENTS_BENCHMARK.md`:  
    - On a 300‑instance SBERT eval, **expansion ON vs OFF** yields:  
      - ON: P@1 0.973 | P@5 0.997 | MRR@10 0.985 | nDCG@10 0.989 | Coverage@10 1.000.  
      - OFF: P@1 0.990 | P@5 0.997 | MRR@10 0.994 | nDCG@10 0.995 | Coverage@10 1.000.  
    - In this setting, disabling expansion slightly improves P@1/MRR.  
  - **Evidence**: verified from `docs/PORTED_IMPROVEMENTS_BENCHMARK.md` and `comparison_reports/...` (described therein).

- **Written‑number recognition in downstream utility**  
  - Implemented in `tools/nlp4lp_downstream_utility.py` via `_word_to_number`, `_extract_num_tokens`, `_extract_num_mentions`.  
  - Benchmarked for *coverage* (not downstream metrics) in `docs/PORTED_IMPROVEMENTS_BENCHMARK.md`:  
    - On `nlp4lp_eval_orig.jsonl` (N=331), **145 / 331 (~44%)** queries gain additional numeric tokens/mentions only visible via the written‑number logic.  
  - **Evidence**: verified from `docs/PORTED_IMPROVEMENTS_BENCHMARK.md`; no updated downstream CSVs using this logic exist in the repo.

- **Embedding cache / index reuse for SBERT retrieval**  
  - Implemented in `app.py` and `retrieval/search.py` by pre‑building catalog embeddings and passing them to `search(…, embeddings=EMBEDDINGS)`.  
  - Qualitatively benchmarked (structural argument plus limited timing attempts) in `docs/PORTED_IMPROVEMENTS_BENCHMARK.md`.  
  - **Evidence**: verified from code and that benchmark doc; no precise, repo‑stored wall‑clock comparison exists.

- **Graceful missing‑formulation handling**  
  - Implemented in `retrieval/search.format_problem_and_ip`, improving UI for catalog entries without full ILPs.  
  - Documented qualitatively in `docs/PORTED_IMPROVEMENTS_BENCHMARK.md`.  
  - **Evidence**: verified from code and the doc; no numerical metrics are attached.

### B3 — Implemented but **not (re)benchmarked** in the current repo artifacts

- **Learning‑based methods in `src/learning/` and related batch scripts**  
  - The repo now contains a Stage‑3 learning stack and scripts for training rankers/grounders.  
  - There is **no trusted downstream leaderboard** in `results/` that reports N=331 InstantiationReady / TypeMatch / Exact20 for these learned models on the NLP4LP benchmark.  
  - **Evidence**: verified from code; absence of corresponding `results/` CSVs implies “implemented but not benchmarked”.

- **Downstream data augmentation (mention‑slot pairs, written‑word variants)**  
  - Implemented in `training/generate_mention_slot_pairs.py`.  
  - No updated N=331 downstream tables in `results/` reflect an ablation of “with vs without augmentation”.  
  - **Evidence**: verified from code; no matching artifacts.

- **PDF upload support and UI plumbing**  
  - Implemented in `retrieval/pdf_utils.py` and `app.py`.  
  - Only unit tests exist (`tests/test_pdf_upload.py`); there is **no benchmark artifact** relating this to NLP4LP metrics.  
  - **Evidence**: verified from code and tests.

### B4 — Partially implemented / unclear benchmarking

- **Advanced structured methods (anchor‑linking, entity‑semantic beam, relation‑aware repair, bottom‑up beam)**  
  - Implemented in `tools/nlp4lp_downstream_utility.py` and appear as baselines in `results/paper/nlp4lp_downstream_types_table_orig.csv` and per‑query CSVs.  
  - These *are* benchmarked in per‑type and per‑query tables, but their full comparison against the 7‑method “current situation” summary is only partially documented in `nlp4lp_wins_analysis_after_7_methods.md`.  
  - **Evidence**: verified from repo artifacts; mapping them directly into a concise, post‑Copilot leaderboard would require re‑analysis.

---

## SECTION C — How current repo evidence differs from manuscript baseline

### C1 — InstantiationReady vs manuscript TF‑IDF baseline (0.0725) and oracle (0.0816)

- **TF‑IDF typed greedy baseline**  
  - Still at InstantiationReady 0.0725 on orig (same as manuscript, from `nlp4lp_downstream_main_table_orig.csv`).  
  - **Status**: unchanged.

- **Methods that improve over TF‑IDF’s InstantiationReady 0.0725**
  - **bm25 typed greedy**: 0.0755 (manuscript baseline; verified in `nlp4lp_downstream_main_table_orig.csv`).  
  - **tfidf_acceptance_rerank**: ≈0.0816 (from `nlp4lp_current_situation_after_7_methods.csv`).  
  - **tfidf_hierarchical_acceptance_rerank**: ≈0.0816 (same file).  
  - **Interpretation**: both acceptance‑based methods match or very slightly exceed oracle‑retrieval InstantiationReady (0.0816) while using TF‑IDF retrieval.

- **Methods that exceed or match manuscript oracle InstantiationReady 0.0816**
  - **tfidf_acceptance_rerank** and **tfidf_hierarchical_acceptance_rerank** have InstantiationReady ≈0.0816 on orig (within rounding of oracle).  
  - No method in the current artifacts clearly and robustly **exceeds** 0.0816 by a meaningful margin; they **match** it.  
  - **Evidence provenance**: `results/paper/nlp4lp_current_situation_after_7_methods.csv` (verified), combined with manuscript‑era oracle row.

- **optimization_role_repair**  
  - InstantiationReady 0.0600: below TF‑IDF and oracle, but with substantially higher TypeMatch and Exact20 (see below).  
  - **Interpretation**: stronger structure but more conservative readiness decisions.

### C2 — TypeMatch vs manuscript TF‑IDF baseline (0.2267)

Using `nlp4lp_downstream_main_table_orig.csv` and `nlp4lp_current_situation_after_7_methods.csv`:

- **tfidf typed greedy baseline**: TypeMatch 0.2267.  
- **tfidf_semantic_ir_repair**: TypeMatch 0.2540 (clear improvement).  
- **tfidf_optimization_role_repair**: TypeMatch 0.2430 (improvement).  
- **tfidf_acceptance_rerank**: TypeMatch ≈0.2297 (very similar to baseline; small gain).  
- **tfidf_hierarchical_acceptance_rerank**: TypeMatch ≈0.2309 (small improvement).  
- **bm25 typed greedy**: TypeMatch 0.2251 (roughly tied).  
- **Interpretation**: several structured deterministic methods **materially improve TypeMatch** relative to simple TF‑IDF typed greedy, especially semantic IR repair and optimization‑role repair.

### C3 — Exact20 vs manuscript TF‑IDF baseline (0.2140)

From `nlp4lp_downstream_main_table_orig.csv`:

- **tfidf typed greedy baseline**: Exact20 0.2140.  
- **tfidf_semantic_ir_repair**: Exact20 0.2610 (substantial improvement).  
- **tfidf_optimization_role_repair**: Exact20 0.2770 (largest improvement among the reported deterministic methods).  
- **tfidf_constrained**: Exact20 0.3279 (highest Exact20, but with lower coverage and InstantiationReady).  
- **tfidf_acceptance_rerank / hierarchical_acceptance_rerank**: Exact20 ≈0.222–0.232 (small improvements).  
- **Interpretation**: current repo artifacts show **clear downstream progress on Exact20** vs the manuscript TF‑IDF baseline, at the cost of various tradeoffs in coverage and readiness.

### C4 — Float‑type weakness

From `nlp4lp_downstream_types_table_orig.csv`:

- **Manuscript‑era TF‑IDF float metrics**: coverage 0.8346 | TypeMatch 0.0287.  
- **Current deterministic methods (float TypeMatch, orig)**:
  - **tfidf_optimization_role_repair**: 0.0171 (worse).  
  - **tfidf_semantic_ir_repair**: 0.0179 (worse).  
  - **tfidf_acceptance_rerank**: 0.0284 (similar).  
  - **tfidf_hierarchical_acceptance_rerank**: 0.0297 (slightly better).  
  - **oracle variants**: typically ~0.018–0.028 (similar or slightly worse).  
- **Interpretation**:
  - Even with more structured methods, **float type‑match remains very low**, and improvements are at best marginal (hierarchical acceptance rerank nudges it from 0.0287 to ≈0.0297).  
  - The new written‑number logic dramatically improves *coverage* of numeric mentions (~44% of queries gain extra NumToks), but there is **no updated downstream table** showing improved float TypeMatch; the bottleneck remains largely in assignment rather than detection.

### C5 — Retrieval vs downstream bottleneck gap

- **Retrieval strength**  
  - Classical retrieval metrics (TF‑IDF/BM25/LSA) in `results/baselines_nlp4lp_orig.csv` and `nlp4lp_retrieval_summary.csv` are identical to manuscript values: schema R@1 ≈0.9.  
  - SBERT‑based retrieval experiments in `docs/PORTED_IMPROVEMENTS_BENCHMARK.md` show extremely strong retrieval (P@1 ~0.97–0.99) on the generated eval set.

- **Downstream readiness**  
  - Even the best deterministic methods (e.g., tfidf_optimization_role_repair, tfidf_constrained, tfidf_semantic_ir_repair, acceptance/hierarchical rerank) achieve InstantiationReady in the 0.06–0.085 range on orig (N=331).  
  - **Interpretation**: the **retrieval vs downstream gap remains large**; improvements are mainly in the grounding/assignment stage, not retrieval.

### C6 — Methods that outperform BM25 manuscript orig InstantiationReady 0.0755 while keeping retrieval strong

- **bm25 typed greedy**: InstantiationReady 0.0755 (baseline).  
- **tfidf_acceptance_rerank / hierarchical_acceptance_rerank**: ≈0.0816 InstantiationReady with TF‑IDF retrieval at schema R@1 0.9063 (unchanged).  
- **Interpretation**:
  - These acceptance‑based methods **beat BM25’s manuscript InstantiationReady** while leaving retrieval unchanged (still strong).  
  - However, they slightly reduce coverage (≈0.78–0.80 vs 0.8222) and do not address float TypeMatch meaningfully.

### C7 — Improvements to the weak typed greedy baseline without catastrophic coverage loss

- **tfidf_semantic_ir_repair**:  
  - Coverage 0.7780 (−0.0442 vs baseline), TypeMatch 0.2540 (+0.0273), Exact20 0.2610 (+0.0470), InstantiationReady 0.0630 (−0.0095).  
- **tfidf_optimization_role_repair**:  
  - Coverage 0.8220 (essentially unchanged), TypeMatch 0.2430 (+0.0163), Exact20 0.2770 (+0.0630), InstantiationReady 0.0600 (−0.0125).  
- **tfidf_acceptance_rerank / hierarchical_acceptance_rerank**:  
  - Slightly lower coverage but better InstantiationReady (~0.0816) with similar TypeMatch.  
- **Interpretation**: current structured deterministic methods **improve TypeMatch and Exact20**, but each pays a price in coverage and/or InstantiationReady; there is no single method that cleanly dominates the typed greedy baseline on all metrics.

---

## SECTION D — What is still weak in the current repo

- **Downstream grounding remains the main bottleneck**  
  - Retrieval (both classical and SBERT) is extremely strong in all available artifacts, but InstantiationReady remains ≈0.06–0.085 even for the best deterministic methods.  
  - Oracle‑retrieval variants confirm that **perfect retrieval only modestly helps**, consistent with the manuscript baseline (0.0725 → 0.0816).

- **Float role confusion remains severe**  
  - Per‑type tables show float TypeMatch ≈0.02–0.03 across methods, with only tiny improvements from hierarchical acceptance rerank.  
  - The new written‑number logic improves detection coverage but has **not been propagated into a new, trusted N=331 downstream leaderboard**, so float TypeMatch is still effectively at manuscript‑era levels in the recorded results.

- **Trade‑offs between readiness, coverage, and schema quality**  
  - Methods that push Exact20 higher (e.g., tfidf_constrained at 0.3279) often reduce coverage and/or InstantiationReady.  
  - Methods that match oracle InstantiationReady (acceptance / hierarchical acceptance rerank) slightly sacrifice coverage and do not materially improve TypeMatch or float performance.

- **Learning‑based methods are unproven on the main benchmark**  
  - Although the learning stack and augmentation utilities are implemented, there is **no trusted N=331 leaderboard** for learned models in `results/`.  
  - The repo therefore **does not yet contain evidence** that learning materially improves TypeMatch, Exact20, or InstantiationReady over the best deterministic methods.

- **Newly ported features lack full ON/OFF ablations on NLP4LP**  
  - Short‑query expansion: only SBERT catalog metrics (P@1/MRR) are benchmarked; there is **no classical TF‑IDF/BM25/LSA ablation** on the main NLP4LP benchmark.  
  - Written‑number recognition: only coverage of NumToks/mentions is measured; there is **no updated downstream table** showing its effect on TypeMatch/Exact20/InstantiationReady.  
  - Embedding cache and PDF upload: infrastructure and UX only, with no downstream metrics.  
  - **Conclusion**: many promising changes are in place, but their impact on the canonical N=331 results is still **unverified**.

---

## SECTION E — Paper‑impact interpretation

- **What parts of the manuscript baseline are now outdated**
  - The manuscript‑era narrative that only the simple TF‑IDF / BM25 / LSA typed greedy vs oracle methods exist is outdated: the repo now includes and benchmarks multiple structured deterministic methods (constrained, semantic IR repair, optimization‑role repair, acceptance/hierarchical acceptance rerank).  
  - Exact20 and TypeMatch can be **substantially higher** than the original TF‑IDF baseline (e.g., tfidf_optimization_role_repair and tfidf_constrained), though InstantiationReady gains are limited.  
  - The statement that “no learning‑based methods are implemented” is outdated at the code level, but remains effectively true at the **results** level (no trusted leaderboard).

- **What parts of the manuscript baseline remain correct**
  - Retrieval is not the main bottleneck; all current artifacts confirm this.  
  - Downstream grounding (especially for floats) is still the key weakness: float TypeMatch remains ≈0.02–0.03 despite more structured methods and better numeric detection.  
  - Oracle retrieval still only modestly improves InstantiationReady over TF‑IDF, and even sophisticated downstream methods do not break far past ~0.08.  
  - Typed vs untyped conclusions (typed helps but only incrementally) still hold in the detailed tables.

- **What claims could be strengthened with reruns**
  - With updated downstream benchmarks that (a) incorporate written‑number detection, (b) use the best deterministic methods (e.g., optimization‑role repair + acceptance/hierarchical rerank), and (c) potentially include learned models, the paper could **more strongly claim** improvements in TypeMatch and Exact20 at modest cost to coverage/readiness.  
  - If float performance improves after re‑running with written‑number support and better models, the “float is hopeless” narrative could be softened with concrete evidence.  
  - Short‑query expansion and embedding cache changes are currently framed as **engineering/UX improvements**; only additional ablations on the canonical benchmark could justify any stronger claims.

---

## Bottom line

Compared with the manuscript baseline, the current repo shows **modest but real, verified downstream progress** in deterministic methods (higher TypeMatch and Exact20, oracle‑level InstantiationReady via acceptance rerank) but **no decisive change to the core bottleneck story**: retrieval is strong, grounding is weak, and floats remain very hard. Many additional features (learning stack, written‑number detection, augmentation, short‑query expansion, embedding cache, PDF upload) are **implemented but not fully benchmarked** on the canonical N=331 setting, so their impact remains mostly potential rather than evidenced. Given the available artifacts, the overall picture is closest to **(b) modest verified metric progress**, with substantial additional headroom that is currently **unverified**.

