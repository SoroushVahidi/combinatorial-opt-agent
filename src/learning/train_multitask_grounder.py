#!/usr/bin/env python3
"""Train with NL4Opt auxiliary tasks: pretrain-then-finetune or joint multitask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.learning.models.features import row_to_feature_vector
from src.learning.models.multitask_grounder import (
    BOUND_LABELS,
    BOUND_N,
    MAX_ENTITY_CANDIDATES,
    ROLE_LABELS,
    ROLE_N,
    MultitaskGrounderModule,
)

try:
    import torch
    from transformers import AutoTokenizer
except ImportError:
    torch = None
    AutoTokenizer = None


def _default_encoder() -> str:
    return "distilroberta-base"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain_then_finetune", "joint"], default="pretrain_then_finetune")
    ap.add_argument("--run_name", type=str, default="multitask_run0")
    ap.add_argument("--encoder", type=str, default=_default_encoder())
    ap.add_argument("--nlp4lp_data_dir", type=Path, default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp")
    ap.add_argument("--nl4opt_aux_dir", type=Path, default=ROOT / "artifacts" / "learning_aux_data" / "nl4opt")
    ap.add_argument("--save_dir", type=Path, default=ROOT / "artifacts" / "learning_runs")
    ap.add_argument("--use_nl4opt_entity", action="store_true")
    ap.add_argument("--use_nl4opt_bound", action="store_true")
    ap.add_argument("--use_nl4opt_role", action="store_true")
    ap.add_argument("--aux_loss_weight_entity", type=float, default=1.0)
    ap.add_argument("--aux_loss_weight_bound", type=float, default=1.0)
    ap.add_argument("--aux_loss_weight_role", type=float, default=1.0)
    ap.add_argument("--pretrain_steps", type=int, default=50)
    ap.add_argument("--finetune_steps", type=int, default=100)
    ap.add_argument("--joint_steps", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    if not torch:
        print("torch/transformers required", file=sys.stderr)
        sys.exit(1)
    if MultitaskGrounderModule is None:
        print("MultitaskGrounderModule not available (torch/transformers missing in models.multitask_grounder)", file=sys.stderr)
        sys.exit(1)
    run_dir = args.save_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultitaskGrounderModule(
        encoder_name=args.encoder,
        use_structured_features=False,
        use_entity_head=args.use_nl4opt_entity,
        use_bound_head=args.use_nl4opt_bound,
        use_role_head=args.use_nl4opt_role,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    config = {
        "mode": args.mode,
        "encoder": args.encoder,
        "use_entity": args.use_nl4opt_entity,
        "use_bound": args.use_nl4opt_bound,
        "use_role": args.use_nl4opt_role,
    }
    if args.mode == "pretrain_then_finetune":
        # Phase 1: aux tasks only on NL4Opt
        for task_name, use_it, steps in [
            ("entity", args.use_nl4opt_entity, args.pretrain_steps),
            ("bound", args.use_nl4opt_bound, args.pretrain_steps),
            ("role", args.use_nl4opt_role, args.pretrain_steps),
        ]:
            if not use_it:
                continue
            path = args.nl4opt_aux_dir / f"{task_name}_train.jsonl"
            if not path.exists():
                print(f"Skip pretrain {task_name}: {path} not found", file=sys.stderr)
                continue
            examples = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
            if not examples:
                continue
            step = 0
            for _ in range(steps):
                i = step % len(examples)
                ex = examples[i]
                text = (ex.get("problem_text") or "")[:400]
                if task_name == "entity":
                    mention = ex.get("mention_surface", "")
                    ctx = ex.get("local_context", "")
                    text = f"{text} [SEP] {mention} {ctx}"[:512]
                else:
                    surface = ex.get("surface", ex.get("mention_surface", ""))
                    ctx = ex.get("local_context", "")
                    text = f"{text} [SEP] {surface} {ctx}"[:512]
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                if task_name == "entity":
                    cand_ids = ex.get("candidate_variable_ids") or []
                    gold_id = ex.get("gold_variable_id")
                    if not cand_ids or gold_id not in cand_ids:
                        step += 1
                        continue
                    gold_idx = cand_ids.index(gold_id)
                    logits = model.forward_aux_entity(inp["input_ids"], inp.get("attention_mask"))
                    n_cand = min(len(cand_ids), MAX_ENTITY_CANDIDATES)
                    if gold_idx >= n_cand:
                        step += 1
                        continue
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :n_cand],
                        torch.tensor([gold_idx], device=device, dtype=torch.long),
                    )
                    loss = loss * args.aux_loss_weight_entity
                elif task_name == "bound":
                    gold = ex.get("gold_bound_label", "other")
                    if gold not in BOUND_LABELS:
                        gold = "other"
                    gold_idx = BOUND_LABELS.index(gold)
                    logits = model.forward_aux_bound(inp["input_ids"], inp.get("attention_mask"))
                    loss = torch.nn.functional.cross_entropy(
                        logits,
                        torch.tensor([gold_idx], device=device, dtype=torch.long),
                    )
                    loss = loss * args.aux_loss_weight_bound
                else:
                    gold = ex.get("gold_role_label", "other")
                    if gold not in ROLE_LABELS:
                        gold = "other"
                    gold_idx = ROLE_LABELS.index(gold)
                    logits = model.forward_aux_role(inp["input_ids"], inp.get("attention_mask"))
                    loss = torch.nn.functional.cross_entropy(
                        logits,
                        torch.tensor([gold_idx], device=device, dtype=torch.long),
                    )
                    loss = loss * args.aux_loss_weight_role
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
        # Phase 2: finetune pairwise on NLP4LP
        train_path = args.nlp4lp_data_dir / "train.jsonl"
        if not train_path.exists():
            print(f"NLP4LP train not found: {train_path}", file=sys.stderr)
        else:
            rows = []
            with open(train_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            for step in range(args.finetune_steps):
                i = step % len(rows)
                row = rows[i]
                slot_name = row.get("slot_name", "")
                slot_role = row.get("slot_role")
                surface = row.get("mention_surface", "")
                ctx = row.get("sentence_or_context", "")
                text = f"[SLOT] {slot_name} [SEP] {surface} {ctx}"[:512]
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                label = float(row.get("label", 0))
                logits = model.forward_pairwise(inp["input_ids"], inp.get("attention_mask"))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(),
                    torch.tensor([label], device=device, dtype=torch.float32),
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
    else:
        # Joint: alternate aux and pairwise
        aux_examples = {"entity": [], "bound": [], "role": []}
        for task in ("entity", "bound", "role"):
            path = args.nl4opt_aux_dir / f"{task}_train.jsonl"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            aux_examples[task].append(json.loads(line))
        nlp4lp_rows = []
        if (args.nlp4lp_data_dir / "train.jsonl").exists():
            with open(args.nlp4lp_data_dir / "train.jsonl", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        nlp4lp_rows.append(json.loads(line))
        step = 0
        for _ in range(args.joint_steps):
            do_pairwise = nlp4lp_rows and (step % 3 == 0)
            if do_pairwise:
                row = nlp4lp_rows[step % len(nlp4lp_rows)]
                slot_name = row.get("slot_name", "")
                slot_role = row.get("slot_role")
                surface = row.get("mention_surface", "")
                ctx = row.get("sentence_or_context", "")
                text = f"[SLOT] {slot_name} [SEP] {surface} {ctx}"[:512]
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                label = float(row.get("label", 0))
                logits = model.forward_pairwise(inp["input_ids"], inp.get("attention_mask"))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(),
                    torch.tensor([label], device=device, dtype=torch.float32),
                )
            if not do_pairwise:
                task = "bound" if args.use_nl4opt_bound and aux_examples["bound"] else (
                    "role" if args.use_nl4opt_role and aux_examples["role"] else (
                        "entity" if args.use_nl4opt_entity and aux_examples["entity"] else None
                    )
                )
                if task is None or not aux_examples[task]:
                    step += 1
                    continue
                ex = aux_examples[task][step % len(aux_examples[task])]
                text = (ex.get("problem_text") or "")[:400]
                surface = ex.get("surface", ex.get("mention_surface", ""))
                ctx = ex.get("local_context", "")
                text = f"{text} [SEP] {surface} {ctx}"[:512]
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                if task == "bound":
                    gold_idx = BOUND_LABELS.index(ex.get("gold_bound_label", "other") if ex.get("gold_bound_label") in BOUND_LABELS else "other")
                    logits = model.forward_aux_bound(inp["input_ids"], inp.get("attention_mask"))
                    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([gold_idx], device=device, dtype=torch.long)) * args.aux_loss_weight_bound
                elif task == "role":
                    gold_idx = ROLE_LABELS.index(ex.get("gold_role_label", "other") if ex.get("gold_role_label") in ROLE_LABELS else "other")
                    logits = model.forward_aux_role(inp["input_ids"], inp.get("attention_mask"))
                    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([gold_idx], device=device, dtype=torch.long)) * args.aux_loss_weight_role
                else:
                    cand_ids = ex.get("candidate_variable_ids") or []
                    gold_id = ex.get("gold_variable_id")
                    if not cand_ids or gold_id not in cand_ids:
                        step += 1
                        continue
                    n_cand = min(len(cand_ids), MAX_ENTITY_CANDIDATES)
                    gold_idx = min(cand_ids.index(gold_id), n_cand - 1)
                    logits = model.forward_aux_entity(inp["input_ids"], inp.get("attention_mask"))
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :n_cand],
                        torch.tensor([gold_idx], device=device, dtype=torch.long),
                    ) * args.aux_loss_weight_entity
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    # Save full state
    torch.save({
        "model_state": model.state_dict(),
        "encoder_name": args.encoder,
        "use_entity_head": args.use_nl4opt_entity,
        "use_bound_head": args.use_nl4opt_bound,
        "use_role_head": args.use_nl4opt_role,
    }, run_dir / "checkpoint.pt")
    # Export pairwise-only for eval_nlp4lp_pairwise_ranker (encoder + head)
    sd = model.state_dict()
    pairwise_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            pairwise_sd[k] = v
        if k.startswith("pairwise_head."):
            pairwise_sd[k.replace("pairwise_head.", "head.")] = v
    torch.save({
        "model_state": pairwise_sd,
        "encoder_name": args.encoder,
    }, run_dir / "pairwise_only.pt")
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved to {run_dir} (checkpoint.pt, pairwise_only.pt, config.json)")


if __name__ == "__main__":
    main()
