# Paper Scope vs Demo Scope

This document clarifies what is part of the **benchmark-evaluated research pipeline**
(EAAI manuscript) versus what is **demo-only** or **exploratory** and outside the
paper evaluation.

---

## Benchmark-evaluated research pipeline (paper scope)

The following components are evaluated in the EAAI manuscript and have documented
benchmark results:

| Component | Location | Evidence |
|-----------|----------|---------|
| TF-IDF schema retrieval | `retrieval/baselines.py` | Table 1: Schema R@1 = 0.9094 |
| BM25 / LSA retrieval baselines | `retrieval/baselines.py` | Table 1 |
| Short-query expansion | `retrieval/utils.py` | Table 1 (short variant) |
| Typed greedy grounding | `tools/nlp4lp_downstream_utility.py` | Table 1: Coverage 0.822, TypeMatch 0.226 |
| Constrained assignment | `tools/nlp4lp_downstream_utility.py` | Table 1: Exact20 0.328 |
| Optimization-role repair | `tools/nlp4lp_downstream_utility.py` | Table 1: TypeMatch 0.243 |
| LP structural consistency checks | `formulation/verify.py` | Table 2 (60-instance subset) |
| SciPy HiGHS solver execution | `tools/run_eaai_final_solver_attempt.py` | Table 4 (20-instance subset) |
| Failure taxonomy | `analysis/eaai_*_report.md` | Table 5 |

**Dataset:** NLP4LP, `orig` variant, 331 test queries.  
**Reproducibility:** All scripts documented in `HOW_TO_REPRODUCE.md`.  
**Artifacts:** `results/paper/eaai_camera_ready_tables/` and `results/paper/eaai_camera_ready_figures/`.

---

## Demo-only features (outside paper scope)

The following features exist in the repository but are **not** evaluated in the
benchmark and should **not** be cited as having benchmark-validated performance:

| Feature | Location | Status |
|---------|----------|--------|
| Gradio web UI | `app.py` | Demo only |
| LLM-based formulation generation (unknown problems) | `app.py` | Demo only; not benchmarked |
| GAMSPy / Pyomo / PuLP solver code generation | `app.py` | Demo only |
| Feedback collection server | `feedback_server.py` | Demo only |
| Hugging Face Spaces deployment | `deploy_to_hf.py`, `demo/README_Spaces.md` | Demo only |
| Telemetry / query logging | `telemetry.py` | Demo only |
| Dense retrieval (E5, BGE, SBERT fine-tuned) | `retrieval/baselines.py` | Supplementary only |

---

## Exploratory / experimental features

The following components exist in the codebase but are not yet fully evaluated:

| Feature | Location | Status |
|---------|----------|--------|
| Global Consistency Grounding (GCG) | `tools/nlp4lp_downstream_utility.py` | Unit-tested; full benchmark needs HF gold data |
| Learned retrieval fine-tuning | `src/learning/`, `training/` | Future work; GPU + HF data required |
| Number-role repair subsystem | `src/features/`, `src/analysis/` | Experimental; partially integrated |
| Hybrid BM25+TF-IDF retriever | `retrieval/baselines.py` | Evaluated in audit; not in main tables |

---

## Summary

> **For paper citations:** use only components in the "Benchmark-evaluated" table above.  
> **For demo use:** any feature in `app.py` and the `demo/` directory.  
> **For exploratory/future work:** see EAAI manuscript Limitations and Future Work sections.
