"""
End-to-end synthetic-data generation and retrieval-model training pipeline.

Usage (from the project root):

  # Full pipeline: build splits → generate synthetic pairs → fine-tune model
  python -m training.run_pipeline

  # Skip model training (only regenerate data)
  python -m training.run_pipeline --skip-train

  # Dry run: report what would be generated/trained without writing anything
  python -m training.run_pipeline --dry-run

  # Customise hyperparameters
  python -m training.run_pipeline --instances-per-problem 50 --epochs 3 --batch-size 16

Steps
-----
1. Build (or refresh) train/dev/test splits from the full problem catalog.
   Output: data/processed/splits.json  (committed; regenerates deterministically).

2. Generate synthetic (query, passage) training pairs for the train split only,
   so there is no leakage from eval problems into training data.
   Output: data/processed/training_pairs.jsonl  (gitignored; regenerated on demand).

3. Fine-tune ``sentence-transformers/all-MiniLM-L6-v2`` (or a custom base model)
   on the generated pairs using MultipleNegativesRankingLoss with regularisation
   and validation-based early stopping.
   Output: data/models/retrieval_finetuned/final  (gitignored; regenerated on demand).

The fine-tuned model is automatically used by ``retrieval/search.py`` when the
directory ``data/models/retrieval_finetuned/final`` exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _project_root() -> Path:
    return ROOT


def step_build_splits(
    splits_path: Path,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    dry_run: bool,
) -> dict[str, list[str]]:
    """Step 1 – build or refresh train/dev/test problem-ID splits."""
    from training.splits import build_splits, write_splits, load_catalog

    catalog = load_catalog()
    splits = build_splits(
        catalog,
        seed=seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
    )
    n_train = len(splits["train"])
    n_dev = len(splits["dev"])
    n_test = len(splits["test"])
    if dry_run:
        print(
            f"[dry-run] Would write splits: train={n_train}, dev={n_dev}, test={n_test}"
            f" → {splits_path}"
        )
    else:
        write_splits(splits, splits_path)
        print(
            f"[splits] train={n_train}, dev={n_dev}, test={n_test} → {splits_path}"
        )
    return splits


def step_generate_pairs(
    splits: dict[str, list[str]],
    pairs_path: Path,
    instances_per_problem: int,
    seed: int,
    dry_run: bool,
) -> int:
    """Step 2 – generate synthetic (query, passage) pairs for the train split."""
    from training.generate_samples import generate_all_samples

    train_ids = splits.get("train", [])
    if dry_run:
        # Estimate count without actually generating
        estimated = len(train_ids) * instances_per_problem
        print(
            f"[dry-run] Would generate ~{estimated} pairs"
            f" ({instances_per_problem} × {len(train_ids)} train problems)"
            f" → {pairs_path}"
        )
        return estimated

    pairs = generate_all_samples(
        seed=seed,
        instances_per_problem=instances_per_problem,
        include_real_world=False,  # train split only → no real-world bleed
        split_problem_ids=train_ids,
    )
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, "w", encoding="utf-8") as f:
        for q, passage in pairs:
            f.write(json.dumps({"query": q, "passage": passage}, ensure_ascii=False) + "\n")
    print(
        f"[generate] {len(pairs)} pairs written"
        f" ({instances_per_problem} per problem target, {len(train_ids)} train problems)"
        f" → {pairs_path}"
    )
    return len(pairs)


def step_train(
    pairs_path: Path,
    output_dir: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    val_ratio: float,
    max_steps: int,
    dry_run: bool,
) -> None:
    """Step 3 – fine-tune the retrieval model on the generated pairs."""
    if dry_run:
        print(
            f"[dry-run] Would fine-tune '{base_model}' on {pairs_path}"
            f" for up to {epochs} epochs"
            f" (batch={batch_size}, lr={lr}) → {output_dir}"
        )
        return

    # Import training deps lazily so the rest of the pipeline works without them
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer  # noqa: F401
    except ImportError as exc:
        print(
            f"[train] Missing training dependency: {exc}\n"
            "Install with: pip install sentence-transformers datasets 'accelerate>=1.1.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    # train_retrieval applies the PyTorch 2.2 get_default_device patch at import time
    from training.train_retrieval import load_pairs, pairs_to_dataset, train_val_split

    import torch
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    pairs = load_pairs(pairs_path)
    print(f"[train] Loaded {len(pairs)} pairs; splitting {int(val_ratio*100)}% by problem for val…")
    train_pairs, val_pairs = train_val_split(pairs, val_ratio=val_ratio, seed=42)
    print(f"[train] train={len(train_pairs)}, val={len(val_pairs)}")

    train_dataset = pairs_to_dataset(train_pairs)
    eval_dataset = pairs_to_dataset(val_pairs) if val_pairs else None

    model = SentenceTransformer(base_model)
    loss = MultipleNegativesRankingLoss(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    training_kw: dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "learning_rate": lr,
        "max_steps": max_steps if max_steps > 0 else -1,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "logging_steps": 50,
        "save_steps": 300,
        "save_total_limit": 2,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
    }
    if eval_dataset is not None:
        training_kw["eval_strategy"] = "steps"
        training_kw["eval_steps"] = 300
        training_kw["load_best_model_at_end"] = True
        training_kw["metric_for_best_model"] = "eval_loss"
        training_kw["greater_is_better"] = False

    training_args = SentenceTransformerTrainingArguments(**training_kw)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    try:
        trainer.train()
        final_dir = output_dir / "final"
        model.save(str(final_dir))
        print(f"[train] Model saved → {final_dir}")
    except Exception:
        import traceback
        err = traceback.format_exc()
        crash_log = output_dir / "crash.log"
        crash_log.write_text(err, encoding="utf-8")
        print(f"[train] CRASHED — traceback written to {crash_log}", file=sys.stderr)
        raise


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run all pipeline steps."""
    p = argparse.ArgumentParser(
        description="Generate synthetic training data and fine-tune the retrieval model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Output paths
    p.add_argument(
        "--splits-out",
        type=Path,
        default=None,
        help="Path for splits JSON (default: data/processed/splits.json)",
    )
    p.add_argument(
        "--pairs-out",
        type=Path,
        default=None,
        help="Path for training pairs JSONL (default: data/processed/training_pairs.jsonl)",
    )
    p.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Directory for fine-tuned model (default: data/models/retrieval_finetuned)",
    )
    # Generation params
    p.add_argument(
        "--instances-per-problem",
        type=int,
        default=100,
        help="Synthetic (query, passage) pairs to generate per training problem",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for splits and generation")
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of catalog problems assigned to the train split",
    )
    p.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Fraction of catalog problems assigned to the dev split",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of catalog problems assigned to the test split",
    )
    # Training hyperparams
    p.add_argument(
        "--base-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model ID to fine-tune",
    )
    p.add_argument("--epochs", type=int, default=4, help="Maximum training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Per-device training batch size")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps (-1 = determined by epochs)",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of training pairs (by problem) held out for validation",
    )
    # Pipeline control
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Only build splits and generate pairs; skip model fine-tuning",
    )
    p.add_argument(
        "--skip-splits",
        action="store_true",
        help="Skip rebuilding splits (use existing data/processed/splits.json)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files or training",
    )
    args = p.parse_args(argv)

    root = _project_root()
    splits_path = args.splits_out or (root / "data" / "processed" / "splits.json")
    pairs_path = args.pairs_out or (root / "data" / "processed" / "training_pairs.jsonl")
    model_out = args.model_out or (root / "data" / "models" / "retrieval_finetuned")

    print("=== Synthetic data generation + retrieval training pipeline ===")

    # ── Step 1: splits ────────────────────────────────────────────────────────
    if args.skip_splits and splits_path.exists():
        print(f"[splits] Skipped (using existing {splits_path})")
        with open(splits_path, encoding="utf-8") as f:
            splits = json.load(f)
    else:
        splits = step_build_splits(
            splits_path=splits_path,
            seed=args.seed,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            dry_run=args.dry_run,
        )

    # ── Step 2: generate pairs ────────────────────────────────────────────────
    n_pairs = step_generate_pairs(
        splits=splits,
        pairs_path=pairs_path,
        instances_per_problem=args.instances_per_problem,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    # ── Step 3: train ─────────────────────────────────────────────────────────
    if args.skip_train:
        print("[train] Skipped (--skip-train)")
    elif n_pairs == 0:
        print("[train] Skipped (no training pairs were generated)")
    else:
        step_train(
            pairs_path=pairs_path,
            output_dir=model_out,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            val_ratio=args.val_ratio,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("=== Dry run complete (no files were written) ===")
    else:
        final = model_out / "final"
        if not args.skip_train:
            if final.exists():
                print(f"=== Pipeline complete. Fine-tuned model at: {final} ===")
            else:
                print(f"=== Splits and pairs generated. Run without --skip-train to fine-tune. ===")
        else:
            print(f"=== Splits and pairs generated. Run without --skip-train to fine-tune. ===")


if __name__ == "__main__":
    main()
