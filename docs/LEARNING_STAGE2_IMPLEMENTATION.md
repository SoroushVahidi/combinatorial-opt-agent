# Learning Stage 2 Implementation

Stage 2 adds **NL4Opt auxiliary supervision** for the downstream number-to-slot grounding bottleneck. The **target** task remains NLP4LP pairwise slot-ranking and final evaluation; NL4Opt is used only to teach entity association, bound direction, and number-role classification in a way that is honest about supervision differences (see **docs/NL4OPT_AUX_SUPERVISION.md**).

## 1. NL4Opt auxiliary tasks and label sets

Three auxiliary tasks are derived from NL4Opt raw data (`data_external/raw/nl4opt_competition/generation_data/{train,dev,test}.jsonl`):

| Task | Goal | Label set | Source |
|------|-----|-----------|--------|
| **Entity / variable association** | Which variable/entity a numeric mention belongs to | `gold_variable_id` (one of `candidate_variable_ids`) | PARAM/LIMIT spans + obj_declaration.terms / const_declarations.var + var_mention_to_first_var |
| **Bound direction** | Lower vs upper vs equality vs other | `lower`, `upper`, `equality`, `other` | LIMIT spans + const_declarations.direction and .operator |
| **Number-role** | Objective coefficient / limit / rhs / ratio / other | `objective_coeff`, `limit`, `rhs_total`, `ratio`, `other` | Span label (PARAM/LIMIT) + obj_declaration / const_declarations.type |

Exact derivation rules and direct vs approximate mappings are in **docs/NL4OPT_AUX_SUPERVISION.md**. We do **not** invent NLP4LP-style slot-fill labels for NL4Opt.

## 2. Implemented components

### A) NL4Opt auxiliary data builder

- **src/learning/build_nl4opt_aux_data.py** — Reads NL4Opt generation_data JSONL, emits:
  - **Entity:** `entity_{train,dev,test}.jsonl` — one row per PARAM/LIMIT with resolved `gold_variable_id`; only rows with a resolved variable are written.
  - **Bound:** `bound_{train,dev,test}.jsonl` — one row per LIMIT with `gold_bound_label` in `{lower, upper, equality, other}`.
  - **Role:** `role_{train,dev,test}.jsonl` — one row per PARAM/LIMIT with `gold_role_label` in `{objective_coeff, limit, rhs_total, ratio, other}`.
- **Output dir:** `artifacts/learning_aux_data/nl4opt/` — JSONL files plus `summary.json` and `summary.md` (counts per split/task).

### B) Multitask model and trainer

- **src/learning/models/multitask_grounder.py** — `MultitaskGrounderModule`: shared encoder + pairwise head (NLP4LP) + optional auxiliary heads (entity: linear to fixed max candidates; bound: 4 classes; role: 5 classes). Same encoder used for pairwise and aux.
- **src/learning/train_multitask_grounder.py** — Two modes:
  - **pretrain_then_finetune:** (1) Train auxiliary heads on NL4Opt only; (2) Finetune pairwise head on NLP4LP (encoder + pairwise head only; aux heads not used in finetune).
  - **joint:** Alternate NLP4LP pairwise batches and NL4Opt auxiliary batches in one training loop.
- **Flags:** `--use_nl4opt_entity`, `--use_nl4opt_bound`, `--use_nl4opt_role`; `--aux_loss_weight_entity/bound/role`; `--pretrain_steps`, `--finetune_steps`, `--joint_steps`.
- **Export for eval:** Saves `pairwise_only.pt` (encoder + pairwise head state) so **eval_nlp4lp_pairwise_ranker** and **eval_bottleneck_slices** can load the pairwise part of a multitask run. Eval script prefers `pairwise_only.pt` over `checkpoint.pt` when present.

### C) Bottleneck-slice evaluation

- **src/learning/eval_bottleneck_slices.py** — Evaluates on NLP4LP ranker data with heuristic slices:
  - **multiple_float_like:** instances with ≥3 distinct numeric mentions.
  - **lower_upper_cues:** instances with operator_cue_match or slot_role lower_bound/upper_bound.
  - **multi_entity:** instances with ≥2 slots.
  - **overall:** all instances.
- Compares **rule baseline** and any number of **run dirs** (e.g. NLP4LP-only run, NL4Opt-augmented run). For each run it loads `pairwise_only.pt` or `checkpoint.pt`, scores all pairs once, then aggregates pairwise/slot/exact metrics per slice.
- **Outputs:** `artifacts/learning_runs/bottleneck_slices/slice_metrics.json`, `bottleneck_slice_report.md`; per-run copies under `artifacts/learning_runs/<run_name>/bottleneck_slices/slice_metrics.json`.

### D) Batch and scripts

- **batch/learning/build_nl4opt_aux_data.sbatch** — Build NL4Opt aux data; env: `OUTPUT_DIR`, `SEED`, `MAX_EXAMPLES`.
- **batch/learning/train_multitask_grounder.sbatch** — Train multitask; env: `RUN_NAME`, `MODE`, `USE_ENTITY`, `USE_BOUND`, `USE_ROLE`, `PRETRAIN_STEPS`, `FINETUNE_STEPS`, `JOINT_STEPS`, etc.
- **batch/learning/eval_bottleneck_slices.sbatch** — Eval slices; env: `DATA_DIR`, `OUT_DIR`, `SPLIT`, `RUN_DIRS`.
- **scripts/learning/run_build_nl4opt_aux_data.sh**, **run_train_multitask_grounder.sh**, **run_eval_bottleneck_slices.sh** — Wrappers (optional `--sbatch`; run_dirs passed as args for eval).

Logs: **logs/learning/**.

## 3. How to run

### Build NL4Opt auxiliary data

```bash
python -m src.learning.build_nl4opt_aux_data --output_dir artifacts/learning_aux_data/nl4opt --seed 42
# Or: ./scripts/learning/run_build_nl4opt_aux_data.sh
# Batch: sbatch batch/learning/build_nl4opt_aux_data.sbatch
# With limit: MAX_EXAMPLES=100 sbatch batch/learning/build_nl4opt_aux_data.sbatch
```

### Pretrain-then-finetune (NL4Opt aux → NLP4LP pairwise)

```bash
python -m src.learning.train_multitask_grounder \
  --mode pretrain_then_finetune \
  --run_name multitask_run0 \
  --use_nl4opt_entity --use_nl4opt_bound --use_nl4opt_role \
  --nlp4lp_data_dir artifacts/learning_ranker_data/nlp4lp \
  --nl4opt_aux_dir artifacts/learning_aux_data/nl4opt \
  --save_dir artifacts/learning_runs \
  --pretrain_steps 50 --finetune_steps 100
# Or: USE_ENTITY=1 USE_BOUND=1 USE_ROLE=1 sbatch batch/learning/train_multitask_grounder.sbatch
```

### Joint multitask training

```bash
python -m src.learning.train_multitask_grounder \
  --mode joint \
  --run_name joint_run0 \
  --use_nl4opt_entity --use_nl4opt_bound --use_nl4opt_role \
  --joint_steps 150
# Or: MODE=joint USE_ENTITY=1 USE_BOUND=1 USE_ROLE=1 sbatch batch/learning/train_multitask_grounder.sbatch
```

### Evaluate bottleneck slices

```bash
python -m src.learning.eval_bottleneck_slices \
  --data_dir artifacts/learning_ranker_data/nlp4lp \
  --out_dir artifacts/learning_runs \
  --split test \
  --run_dirs rule_baseline run0 multitask_run0
# Or: ./scripts/learning/run_eval_bottleneck_slices.sh rule_baseline run0 multitask_run0
# Batch: RUN_DIRS="rule_baseline run0 multitask_run0" sbatch batch/learning/eval_bottleneck_slices.sbatch
```

Rule baseline: use `--run_dirs rule_baseline` (no checkpoint; script uses handcrafted rule scores).

### Evaluate NLP4LP pairwise (standard) on a multitask run

```bash
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir artifacts/learning_ranker_data/nlp4lp \
  --run_dir artifacts/learning_runs/multitask_run0 \
  --split test
```

The eval script loads `pairwise_only.pt` from the run dir when present, so the pairwise head is evaluated without aux heads.

## 4. Limitations

- **NL4Opt ≠ NLP4LP:** We do not claim NL4Opt has exact slot-filling labels. Auxiliary tasks are entity association, bound direction, and number-role only.
- **TAT-QA / FinQA:** Not used in Stage 2 for multitask training; may be added later with clear separation.
- **Slices:** Bottleneck slices are heuristic (mention count, cue presence, slot count); they are not gold-annotated “hard” subsets.
- **Stability:** First implementation is kept simple; hyperparameters (steps, aux weights, batch size) may need tuning per setup.

## 5. Relation to Stage 1

- Stage 1 pipeline is unchanged: **build_common_corpus** → **validate** → **build_nlp4lp_ranker_data** → **train_nlp4lp_pairwise_ranker** → **eval_nlp4lp_pairwise_ranker**.
- Stage 2 is additive: NL4Opt aux data build, multitask trainer, bottleneck-slice eval, and corresponding batch/scripts. The same pairwise ranker interface and decoding are used for both NLP4LP-only and NL4Opt-augmented runs.
