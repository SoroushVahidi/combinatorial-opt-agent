"""
Train a small binary classifier for mention-slot compatibility (cross-encoder style).
Input: (mention_context, slot_name). Output: 0/1. Uses HuggingFace Trainer.
Run from repo root. Data: data/processed/mention_slot_pairs.jsonl (and _dev.jsonl).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _project_root() -> Path:
    return ROOT


def load_pairs(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Train mention-slot scorer (binary classification)")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "mention_slot_pairs.jsonl")
    p.add_argument("--data-dev", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mention_slot_scorer")
    p.add_argument("--model", type=str, default="nreimers/MiniLM-L6-H384-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--smoke", action="store_true", help="Smoke test: 10 steps, no real data")
    args = p.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"Import error: {e}. Need: pip install transformers datasets", file=sys.stderr)
        sys.exit(1)

    if args.smoke:
        print("Smoke test: creating synthetic data and training 10 steps")
        train_data = [{"sentence1": "budget of 5000 dollars", "sentence2": "TotalBudget", "label": 1}] * 50
        train_data += [{"sentence1": "budget of 5000 dollars", "sentence2": "UnitCost", "label": 0}] * 50
        eval_data = train_data[:20]
        model_name = args.model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        except Exception as e:
            print(f"Model load failed ({e}), using random init for smoke test")
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                model = AutoModelForSequenceClassification.from_config(config)
            except ImportError as ie:
                print(f"PyTorch not available ({ie}). Writing smoke placeholder and exiting.")
                (output_dir / "metrics.json").write_text(json.dumps({"smoke": True, "pytorch_available": False}, indent=2))
                return
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        max_steps = 10
        eval_steps = 5
    else:
        data_path = args.data
        if not data_path.exists():
            print(f"Data not found: {data_path}. Run: python -m training.generate_mention_slot_pairs", file=sys.stderr)
            sys.exit(1)
        train_list = load_pairs(data_path)
        dev_path = args.data_dev or data_path.parent / (data_path.stem + "_dev.jsonl")
        if dev_path.exists():
            eval_list = load_pairs(dev_path)
        else:
            n = len(train_list)
            eval_list = train_list[-max(1, n // 10) :]
            train_list = train_list[: -max(1, n // 10)]
        print(f"Train {len(train_list)}, Eval {len(eval_list)}")
        train_dataset = Dataset.from_list(train_list)
        eval_dataset = Dataset.from_list(eval_list)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        except Exception as e:
            print(f"Model load failed: {e}. Try --smoke for pipeline check.", file=sys.stderr)
            sys.exit(1)
        max_steps = -1
        eval_steps = 200

    def tokenize(examples):
        out = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        out["labels"] = examples["label"]
        return out

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    train_dataset.set_format("torch")
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)
    eval_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=max_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps,
        save_steps=eval_steps if eval_dataset else 500,
        save_total_limit=2,
        load_best_model_at_end=bool(eval_dataset),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    metrics = trainer.evaluate() if eval_dataset else {}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {output_dir / 'final'}, metrics: {metrics}")


if __name__ == "__main__":
    main()
