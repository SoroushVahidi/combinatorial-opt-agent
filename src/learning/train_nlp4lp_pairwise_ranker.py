#!/usr/bin/env python3
"""Minimal training script for NLP4LP pairwise mention-slot ranker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.learning.models.features import row_to_feature_vector
from src.learning.models.pairwise_ranker import PairwiseRanker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp")
    ap.add_argument("--run_name", type=str, default="run0")
    ap.add_argument("--encoder", type=str, default="distilroberta-base")
    ap.add_argument("--use_features", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--save_dir", type=Path, default=ROOT / "artifacts" / "learning_runs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--init_checkpoint", type=Path, default=None, help="Load this checkpoint before training (for two-stage: aux then NLP4LP)")
    args = ap.parse_args()
    run_dir = args.save_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.jsonl"
    dev_path = args.data_dir / "dev.jsonl"
    if not train_path.exists():
        print(f"Train data not found: {train_path}", file=sys.stderr)
        print("Run build_nlp4lp_pairwise_ranker_data.py first.", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("No training rows.", file=sys.stderr)
        sys.exit(1)
    ranker = PairwiseRanker(
        encoder_name=args.encoder,
        use_structured_features=args.use_features,
    )
    if args.init_checkpoint and args.init_checkpoint.exists():
        ranker.load(str(args.init_checkpoint))
        print(f"Loaded init checkpoint: {args.init_checkpoint}", file=sys.stderr)
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("Training requires torch. Saving config only.", file=sys.stderr)
        config = {
            "encoder": args.encoder,
            "use_features": args.use_features,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "data_dir": str(args.data_dir),
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config written to {run_dir / 'config.json'}")
        return
    if ranker.model is None:
        print("Model not built (transformers/torch missing?). Saving config only.", file=sys.stderr)
        config = {"encoder": args.encoder, "use_features": args.use_features}
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker.model = ranker.model.to(device)
    ranker.model.train()
    optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=args.lr)
    step = 0
    train_size = len(rows)
    # Reproducibility: seed only affects data order if we shuffled; we iterate in order
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    losses: list[float] = []
    for epoch in range(args.epochs):
        for i in range(0, min(len(rows), args.max_steps * args.batch_size), args.batch_size):
            batch = rows[i : i + args.batch_size]
            if not batch:
                break
            texts = []
            feats = []
            labels = []
            for row in batch:
                text = ranker._format_pair(
                    row.get("slot_name", ""),
                    row.get("slot_role"),
                    row.get("mention_surface", ""),
                    row.get("sentence_or_context"),
                )
                texts.append(text)
                feats.append(row_to_feature_vector(row) if args.use_features else None)
                labels.append(float(row.get("label", 0)))
            inputs = ranker.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if args.use_features and feats and feats[0] is not None:
                feat_t = torch.tensor(feats, dtype=torch.float32, device=device)
            else:
                feat_t = None
            logits = ranker.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                feature_vector=feat_t,
            )
            label_t = torch.tensor(labels, dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            loss_val = float(loss.item())
            losses.append(loss_val)
            if step % 50 == 0 or step <= 3:
                print(f"  step {step} loss={loss_val:.4f}")
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break
    ranker.save(str(run_dir / "checkpoint.pt"))
    dev_count = 0
    if dev_path.exists():
        with open(dev_path, encoding="utf-8") as f:
            dev_count = sum(1 for line in f if line.strip())
    config = {
        "encoder": args.encoder,
        "use_features": args.use_features,
        "steps": step,
        "seed": args.seed,
        "train_size": train_size,
        "dev_size": dev_count,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "final_loss": losses[-1] if losses else None,
    }
    if args.init_checkpoint and args.init_checkpoint.exists():
        config["init_checkpoint"] = str(args.init_checkpoint)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved to {run_dir} (steps={step}, train_size={train_size}, final_loss={config['final_loss']})")


if __name__ == "__main__":
    main()
