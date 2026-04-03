# Dataset expansion plan — Text2Zinc & CP-Bench

> **NLP4LP remains the canonical, paper-evaluated benchmark** in this repository. Text2Zinc and CP-Bench (DCP-Bench-Open) are **external-validation targets** for future work—not a rewrite of the EAAI headline pipeline.

## Why add these datasets?

- **External validity:** NLP4LP is a fixed LP-structured catalog benchmark. Text2Zinc and CP-Bench probe **different target formalisms** (MiniZinc / constraint programs) and problem sources.
- **Honest scope:** Integration here means **adapters, staging scripts, and documentation** so evaluation can be designed deliberately—not automatic parity with Tables 1–5.

## How they differ from NLP4LP

| Aspect | NLP4LP (canonical) | Text2Zinc | CP-Bench (DCP-Bench-Open) |
|--------|--------------------|-----------|---------------------------|
| Primary signal | NL + fixed schema catalog + scalar slots | NL → MiniZinc + data | Reference **CPMPy** Python in public sample |
| Schema retrieval over shared catalog | Core paper metric | Not the same catalog | No shared NLP4LP schema |
| Scalar grounding (our utility) | Defined & measured | Partial overlap possible | **Not** aligned (no NLP4LP-style slots in `sample_test.jsonl`) |
| Access | Gated HF `udell-lab/NLP4LP` | Gated HF `skadio/text2zinc` | Public GitHub (Apache-2.0) |

## What can be evaluated immediately

- **NLP4LP:** Full pipeline as documented; camera-ready tables in `results/paper/eaai_camera_ready_tables/`.
- **Text2Zinc:** After HF access, **staging** via `scripts/get_text2zinc.py` → JSONL under `data/external/text2zinc/`. Adapter can load rows and expose `InternalExample` fields; **no new headline metrics** are claimed until an eval spec + runs exist.
- **CP-Bench:** `python scripts/datasets/get_cp_bench_open.py` stages `sample_test.jsonl`. Adapter reports **reference code coverage** via `tools/run_dataset_benchmarks.py --dataset cp_bench` once files exist—not NLP4LP retrieval/grounding accuracy.

## What requires adaptation

- **Text2Zinc:** Map MiniZinc/data artifacts to any future metric; optional LLM or symbolic stages; licensing/access on Hugging Face.
- **CP-Bench:** Problems are **constraint-modeling** oriented; the public sample JSONL has **no natural-language statement** comparable to NLP4LP queries. Bridging NL (if added upstream or derived) to CPMPy or to our LP-focused checks is **research/engineering work**.

## Claims **not** supported yet

- **No** Text2Zinc or CP-Bench numbers in camera-ready EAAI tables unless explicitly added after real runs.
- **No** statement that the NLP4LP pipeline “works out of the box” on these datasets.
- **No** implication that adapter integration equals completed benchmarking.

## Pointers

- Machine-readable status: [`data/dataset_registry.json`](../data/dataset_registry.json)
- Implementation status: [`DATASET_EXPANSION_STATUS.md`](DATASET_EXPANSION_STATUS.md)
- Staging scripts: [`scripts/datasets/README.md`](../scripts/datasets/README.md)
