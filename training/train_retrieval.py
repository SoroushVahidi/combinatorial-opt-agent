"""
Fine-tune a sentence-transformers model on (query, passage) pairs for better retrieval.
Uses MultipleNegativesRankingLoss. Run on GPU for speed.
"""
from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Load (query, passage) pairs from JSONL."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append((obj["query"], obj["passage"]))
    return pairs


def pairs_to_dataset(pairs: list[tuple[str, str]]):
    """Convert (query, passage) pairs to HuggingFace Dataset with sentence1, sentence2."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Training requires: pip install datasets") from None
    return Dataset.from_list([
        {"sentence1": q, "sentence2": p} for q, p in pairs
    ])


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Fine-tune retrieval model on synthetic pairs")
    p.add_argument("--data", type=Path, default=None, help="Training pairs JSONL (default: data/processed/training_pairs.jsonl)")
    p.add_argument("--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Base model to fine-tune")
    p.add_argument("--output-dir", type=Path, default=None, help="Where to save the fine-tuned model")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 = use epochs)")
    args = p.parse_args()

    root = _project_root()
    data_path = args.data or root / "data" / "processed" / "training_pairs.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. Run: python -m training.generate_samples"
        )
    output_dir = args.output_dir or root / "data" / "models" / "retrieval_finetuned"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(data_path)
    print(f"Loaded {len(pairs)} training pairs")
    train_dataset = pairs_to_dataset(pairs)

    model = SentenceTransformer(args.base_model)
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    try:
        trainer.train()
        model.save(str(output_dir / "final"))
        print(f"Model saved to {output_dir / 'final'}")
    except Exception:
        import sys
        import traceback
        err = traceback.format_exc()
        print("TRAINING CRASHED", file=sys.stderr)
        traceback.print_exc()
        crash_log = output_dir / "crash.log"
        crash_log.write_text(err, encoding="utf-8")
        print(f"Traceback written to {crash_log}", file=sys.stderr)
        print("\n--- CRASH ---\n" + err)  # stdout so it appears in .out
        raise


if __name__ == "__main__":
    main()
