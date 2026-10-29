# Natural-language optimization research and manuscripts

## Current JDSA manuscript

**Retrieval-Assisted Scalar Grounding for Natural-Language Optimization: A Diagnostic Study**

Public pre-submission author-review version for the **International Journal of Data Science and Analytics**; no journal acceptance or publication is implied.
Scientific source: corrected commit `d3583fa28fcdb87f7a446f764ca3225ab5f855d3`.
Canonical current package: [`manuscript/jdsa-stage3/`](https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/main/manuscript/jdsa-stage3).

- [Current manuscript PDF](https://github.com/SoroushVahidi/combinatorial-opt-agent/raw/refs/heads/main/manuscript/jdsa-stage3/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf)
- [Current supplement — Online Resource 1](https://github.com/SoroushVahidi/combinatorial-opt-agent/raw/refs/heads/main/manuscript/jdsa-stage3/ESM_1.pdf)
- [Matching editable submission-source ZIP](https://github.com/SoroushVahidi/combinatorial-opt-agent/raw/refs/heads/main/manuscript/jdsa-stage3/JDSA_Stage3_Submission_Source.zip)
- [Build instructions](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/main/manuscript/jdsa-stage3/README.md)
- [Historical manuscript v1.0.0 release](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0): immutable earlier version, superseded scientifically by the current corrected manuscript.
- [Immutable scientific evidence release](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0)
- [Exact Hugging Face evidence](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0)

Saved-record reanalysis is distinct from complete experimental reproduction and independent semantic validation. Restricted originals and historical evidence limitations remain as stated in the manuscript. Download the current PDF pair together for cross-document links. Separate SNCS, DKE and KAIS materials remain historical journal packages.

Historical companion repository for the paper **"Retrieval-Assisted Instantiation of Natural-Language Optimization Problems"**, submitted to **SN Computer Science** (Springer Nature) on 2026-08-26, submission ID SNCS-D-26-07548. The repository implements and evaluates a two-stage pipeline — fixed-catalog schema retrieval (TF--IDF / BM25 / LSA / BGE-M3) followed by deterministic, inference-time-LLM-free scalar-parameter grounding — on the **NLP4LP** benchmark (331 queries, 335-candidate schema catalog), together with restricted structural, solver-backed, and external-baseline validation subsets. This is a research artifact repository, not a production product.

## Historical SN Computer Science manuscript status

- **Submission status:** Submitted to SN Computer Science (SNCS-D-26-07548), August 2026.
- **SN Computer Science manuscript source:** [`manuscript/sncs/main.tex`](manuscript/sncs/main.tex) (SN Computer Science / Springer Nature `sn-jnl` template), migrated 2026-08-27 from the corrected [`manuscript/dke/main.tex`](manuscript/dke/main.tex) (Data & Knowledge Engineering / Elsevier `elsarticle` template, an earlier submission attempt for the same paper).
- **Older, superseded manuscript versions:** `manuscript/main.tex` / `manuscript/submission_package/main.tex` (a still-earlier Knowledge and Information Systems / KAIS attempt) and the original EAAI/Elsevier draft. These contain a **retracted headline number** (InstantiationReady $=0.5287$) that does not reproduce from the current codebase — see [`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`](docs/BASELINE_STALENESS_AUDIT_2026-08-12.md). **Do not cite numbers from these files.**
- **Full audit trail:** [`docs/SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md`](docs/SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md), [`docs/SNCS_STAGE2_LOCAL_WULVER_GITHUB_AUDIT_2026-08-27.md`](docs/SNCS_STAGE2_LOCAL_WULVER_GITHUB_AUDIT_2026-08-27.md), and [`docs/SNCS_STAGE3_MANUSCRIPT_SPRINGER_REPOSITORY_FINALIZATION_2026-08-27.md`](docs/SNCS_STAGE3_MANUSCRIPT_SPRINGER_REPOSITORY_FINALIZATION_2026-08-27.md) document every manuscript-vs-repository consistency check performed before submission.
- **Start here for navigation:** [`docs/DKE_SOURCE_OF_TRUTH.md`](docs/DKE_SOURCE_OF_TRUTH.md).

## Historical SN Computer Science headline results

| Metric | TF--IDF (ratio-aware) | BGE-M3 (ratio-aware) | Oracle |
|---|---|---|---|
| Schema R@1 | 0.9094 | 0.9456 | 1.0000 |
| Coverage | 0.8886 | 0.9154 | 0.9416 |
| TypeMatch | 0.8665 | 0.8946 | 0.9230 |
| Exact20 (on hits) | 0.2614 | 0.2436 | 0.2505 |
| InstantiationReady | 0.8006 | 0.8248 | 0.8489 |
| StrictInstantiationReady | 0.7704 | 0.8006 | 0.8489 |

All four rows share the 331-query benchmark denominator and a uniform Exact20 aggregation convention (see [`docs/SNCS_RESULT_MANIFEST_2026-08-27.json`](docs/SNCS_RESULT_MANIFEST_2026-08-27.json)). Every value above is independently reproducible from a small deterministic script — see **Reproducing the results** below.

## Where things live

| What | Where |
|---|---|
| **Authoritative downstream results** | `results/final_resubmission_method/`, `results/oracle_recomputation_2026-08-15/`, `results/dense_retrieval_bge_m3/` |
| **Statistical significance verification** | `tools/recompute_dke_significance.py` → `results/final_resubmission_method/significance_recomputation_2026-08-27.json` (reproduces every diff/CI/$p$ in the manuscript's significance table exactly) |
| **Exact20 denominator root-cause + fix** | `tools/audit_exact20_denominator.py`, `tools/compute_uniform_exact20.py` |
| **Residual-error decomposition** (replaces the old heuristic error taxonomy) | `tools/recompute_residual_error_analysis.py` → `results/final_resubmission_method/residual_error_analysis_2026-08-27.json` |
| **Structural / solver-backed subsets (60 / 269 / 20 instances)** | `results/paper/eaai_camera_ready_tables/table2*.csv`, `table3*.csv`, `table4*.csv` |
| **External baseline provenance** (PaMOP, ORLM, OptMATH, generic LLM, DeepOR, OR-R1) | `docs/*_PROVENANCE.md`, `results/external_baseline_comparison/`, `results/optmath/`, `results/orlm/`, `results/pamop/`, `results/generic_llm/` |
| **Full manuscript-claim-to-artifact map** | [`docs/SNCS_REPRODUCIBILITY.md`](docs/SNCS_REPRODUCIBILITY.md) |
| **Historical / superseded results** (EAAI/KAIS-era, pre-2026-08-13 ratio-aware patch) | `results/eswa_revision/`, `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`, `results/CANONICAL_RESULTS.md` — all banner-marked as historical; see [`docs/DKE_SOURCE_OF_TRUTH.md`](docs/DKE_SOURCE_OF_TRUTH.md) |
| **Negative results** (max-weight matching, selective reranking, learned grounding) | [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md), `results/max_weight_matching_validation/`, `results/selective_grounding_rerank/`, `results/learned_grounding_p0/` |

## Reproducing the results

Four deterministic, CPU-only, no-API-key Python scripts reproduce the manuscript's corrected/verified numbers directly from already-committed per-query artifacts:

```bash
python3 tools/audit_exact20_denominator.py          # Exact20 denominator root-cause
python3 tools/compute_uniform_exact20.py             # uniform Exact20 across all 4 arms
python3 tools/recompute_residual_error_analysis.py   # exact residual-error decomposition
python3 tools/recompute_dke_significance.py          # all 5 significance-table rows
```

**Reproducing from raw data** (rather than verifying from committed artifacts) requires:

| Goal | Requirement |
|---|---|
| Full NLP4LP rerun (retrieval + grounding) | Approved Hugging Face account + `HF_TOKEN` for the gated `udell-lab/NLP4LP` dataset |
| BGE-M3 dense retrieval | GPU + `sentence-transformers` |
| External LLM baselines (PaMOP / OptMATH / generic LLM) | Azure OpenAI API access; Gurobi license for OptMATH/generic-LLM solver execution |
| Structural/solver-backed subsets | No gated access needed for structural checks; the 20-instance solver-backed subset uses SciPy's open-source HiGHS backend (no Gurobi required) |

See [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md) (EAAI-era reproduction guide; still valid for environment setup, historically scoped to the older result tables) and [`docs/SNCS_REPRODUCIBILITY.md`](docs/SNCS_REPRODUCIBILITY.md) (current claim-to-artifact map).

## Setup

```bash
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```bash
cp .env.example .env   # add HF_TOKEN=hf_... for gated NLP4LP access (never commit .env)
# Request dataset access: https://huggingface.co/datasets/udell-lab/NLP4LP
```

Tech stack: Python 3.10+, scikit-learn, rank-bm25, optional sentence-transformers, SciPy, pytest, GitHub Actions, optional SLURM on NJIT's Wulver HPC cluster ([`docs/wulver.md`](docs/wulver.md)).

## Repository structure (high level)

```
manuscript/jdsa/      canonical current JDSA manuscript, supplement and editable sources
manuscript/sncs/      preserved SN Computer Science manuscript package
manuscript/dke/       corrected DKE/Elsevier manuscript (migration source, historical target venue)
manuscript/main.tex   older KAIS-era manuscript (superseded, retracted headline number)
tools/                deterministic pipeline + verification/reproduction scripts
results/              frozen result artifacts (see table above for current vs. historical)
docs/                 provenance, audit, and status documentation
data/                 dataset adapters/catalogs (gated NLP4LP data itself is not redistributed)
```

## What this repository does not claim

- Full open-domain optimization-model generation from natural language.
- Benchmark-wide solver readiness — the solver-backed evidence is restricted to a 20-instance compatibility-filtered subset (SciPy HiGHS; Gurobi not required or used for the paper's own numbers).
- A head-to-head ranking against end-to-end LLM optimization-modeling systems — the external-baseline comparison (Section on external baselines in the manuscript) is explicitly scoped as contextual, not a leaderboard.
- Open-domain generalization of schema retrieval beyond the fixed NLP4LP catalog.

## License · acknowledgments · contact

**License:** [MIT](LICENSE) · © Soroush Vahidi

**Acknowledgments:** NL4Opt, Gurobi (examples), GAMS, MIPLIB, OR-Library, Pyomo; NJIT Computer Science.

**Contact:** [sv96@njit.edu](mailto:sv96@njit.edu) · [@SoroushVahidi](https://github.com/SoroushVahidi)

**Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
