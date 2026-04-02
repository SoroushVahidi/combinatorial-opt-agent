# Valid first learning run — record

## Purpose

First **scientifically valid** learning run for NLP4LP pairwise mention-slot ranker: distinct train/dev/test, no test-as-train fallback, fixed seed, integrity checks, and held-out test evaluation.

## Date / time

Submitted: 2025-03-09 (job 854613).

## Task / path chosen

- **Model:** NLP4LP pairwise mention-slot ranker (DistilRoBERTa + linear head; text-only).
- **Entrypoint:** `src.learning.train_nlp4lp_pairwise_ranker`; evaluation: `src.learning.eval_nlp4lp_pairwise_ranker`.

## Datasets and splits

- **Source:** Local NLP4LP test corpus built from `data/processed/nlp4lp_eval_orig.jsonl` and gold (NLP4LP_GOLD_CACHE or HF test gold).
- **Split strategy:** Single corpus file `artifacts/learning_corpus/nlp4lp_test.jsonl` is split by **instance_id** with **seed 42** into train 70% / dev 15% / test 15% via `src.learning.split_nlp4lp_corpus_for_benchmark`. No overlap between splits.
- **Ranker data:** `artifacts/learning_ranker_data/nlp4lp/{train,dev,test}.jsonl` produced by `build_nlp4lp_pairwise_ranker_data` from the above corpus splits.

## Split integrity

- **Checked:** Before training, `src.learning.verify_split_integrity` confirms train/dev/test files exist and pairwise content hashes differ (no train==test or dev==test). Run fails if integrity check fails (benchmark mode; no fallback).
- **Confirmation:** Train, dev, and test are distinct (see job log for line counts and "Split integrity OK"). Example counts (after one local run): corpus 330 instances → train 230 / dev 50 / test 50 instances; ranker pairs: train 9729, dev 2230, test 2339.

## Data prep commands

1. Build test corpus (if missing):  
   `python -m src.learning.build_common_grounding_corpus --dataset nlp4lp --split test --output_dir artifacts/learning_corpus`
2. Split for benchmark:  
   `python -m src.learning.split_nlp4lp_corpus_for_benchmark --corpus_dir artifacts/learning_corpus --seed 42`
3. Build ranker data:  
   `python -m src.learning.build_nlp4lp_pairwise_ranker_data --corpus_dir artifacts/learning_corpus --output_dir artifacts/learning_ranker_data/nlp4lp`
4. Integrity check:  
   `python -m src.learning.verify_split_integrity --data_dir artifacts/learning_ranker_data/nlp4lp`

## Training command

```bash
python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name valid_first_run \
  --data_dir artifacts/learning_ranker_data/nlp4lp \
  --save_dir artifacts/learning_runs \
  --encoder distilroberta-base \
  --seed 42 \
  --epochs 1 \
  --max_steps 200 \
  --lr 2e-5 \
  --batch_size 8
```

## Evaluation command

```bash
python -m src.learning.eval_nlp4lp_pairwise_ranker \
  --data_dir artifacts/learning_ranker_data/nlp4lp \
  --run_dir artifacts/learning_runs/valid_first_run \
  --split test \
  --out_dir artifacts/learning_runs/valid_first_run
```

## Cluster resources

- **Partition:** gpu  
- **QoS:** standard  
- **Account:** ikoutis  
- **GPU:** 1  
- **Time:** 2h  
- **Mem:** 24G  
- **CPUs:** 4  

## Job and artifacts

- **Script:** `batch/learning/train_nlp4lp_valid_first_run.sbatch`
- **Job ID:** 854613
- **Log (stdout):** `logs/learning/train_nlp4lp_valid_first_run_854613.out`
- **Log (stderr):** `logs/learning/train_nlp4lp_valid_first_run_854613.err`
- **Output dir:** `artifacts/learning_runs/valid_first_run/`
- **Checkpoint:** `artifacts/learning_runs/valid_first_run/checkpoint.pt`
- **Config:** `artifacts/learning_runs/valid_first_run/config.json`
- **Git rev:** `artifacts/learning_runs/valid_first_run/git_rev.txt`
- **Test metrics:** `artifacts/learning_runs/valid_first_run/metrics.json`, `metrics.md`
- **Test predictions:** `artifacts/learning_runs/valid_first_run/predictions.jsonl`

## Run status

pending (submitted as job 854613)

## Split integrity proof (example run)

- **Corpus (after split):** `artifacts/learning_corpus/nlp4lp_{train,dev,test}.jsonl` — train 230 instances, dev 50, test 50 (330 total).
- **Ranker data:** `artifacts/learning_ranker_data/nlp4lp/{train,dev,test}.jsonl` — train 9729 lines, dev 2230, test 2339.
- **Check:** `verify_split_integrity` hashes each file; train≠test, train≠dev, dev≠test. Run exits 1 if any pair is identical.

## Scientific validity

- **Benchmark-valid for held-out evaluation:** Yes, provided the job completed after passing the split-integrity check and evaluation was run on the held-out test split only. Train and test are disjoint by construction (instance_id split with fixed seed).
- **Comparable to deterministic baselines:** Test metrics (pairwise accuracy, slot selection accuracy, exact slot-fill, type-match) are on the same test split; comparison to rule baseline on the same split is valid once baseline is evaluated on this same test set.

## Caveats

- Split is derived from the **same** source as the original “test” set (local NLP4LP eval); we do not introduce a separate external test set. The important guarantee is that train/dev/test are disjoint.
- No NL4Opt or other datasets are used in this run.
- Single seed (42) and one run; no variance reported.
