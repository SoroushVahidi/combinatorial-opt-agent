#!/usr/bin/env python3
import json
import random
import ast
import re
from pathlib import Path
from datasets import load_dataset
import sys
import os

# Add tools to path to import nlp4lp_downstream_utility
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from tools.nlp4lp_downstream_utility import _extract_num_mentions

OUT_DIR = ROOT / "results" / "external_validation" / "optmath"
OUT_DIR.mkdir(parents=True, exist_ok=True)

class NumberVisitor(ast.NodeVisitor):
    def __init__(self):
        self.numbers = set()
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            self.numbers.add(float(node.value))
        self.generic_visit(node)
    # Exclude negative sign in unary op for simplicity, though can be added.
    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, (int, float)):
                self.numbers.add(-float(node.operand.value))
        self.generic_visit(node)

def extract_gold_numbers_from_code(code_str: str) -> set:
    v = NumberVisitor()
    try:
        v.visit(ast.parse(code_str))
    except SyntaxError:
        pass
    return v.numbers

def extract_code_block(text: str) -> str:
    # finds python code blocks
    matches = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    return ""

def main():
    print("Loading shushulei/OptMATH-Train...")
    ds = load_dataset("shushulei/OptMATH-Train", split="train")
    
    # Deterministic sampling
    random.seed(0)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    sample_indices = indices[:1000]
    
    results = []
    
    log_path = OUT_DIR / "per_query_metrics.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for idx in sample_indices:
            row = ds[idx]
            nl_input = row.get("input", "")
            output_str = row.get("output", "")
            
            code_block = extract_code_block(output_str)
            gold_numbers = extract_gold_numbers_from_code(code_block)
            
            # Remove zeros or ones which are very common in models (e.g. range(0, n), x > 0)
            gold_numbers = {n for n in gold_numbers if n not in (0.0, 0, 1.0, 1)}
            
            # extract using frozen pipeline
            mentions = _extract_num_mentions(nl_input, "orig")
            extracted_numbers = set()
            for m in mentions:
                if m.tok.value is not None:
                    extracted_numbers.add(float(m.tok.value))
            
            # Intersection
            matched = gold_numbers.intersection(extracted_numbers)
            
            recall = len(matched) / len(gold_numbers) if len(gold_numbers) > 0 else 1.0
            
            res = {
                "idx": idx,
                "gold_numbers": list(gold_numbers),
                "extracted_numbers": list(extracted_numbers),
                "matched_numbers": list(matched),
                "recall": recall,
                "num_gold": len(gold_numbers),
                "num_matched": len(matched)
            }
            results.append(res)
            f.write(json.dumps(res) + "\n")
            
    # Aggregate
    total_gold = sum(r["num_gold"] for r in results)
    total_matched = sum(r["num_matched"] for r in results)
    macro_recall = total_matched / total_gold if total_gold > 0 else 0
    average_recall = sum(r["recall"] for r in results) / len(results)
    
    agg = {
        "dataset": "shushulei/OptMATH-Train",
        "sample_size": 1000,
        "macro_recall": macro_recall,
        "average_recall": average_recall,
        "total_gold_numbers": total_gold,
        "total_matched_numbers": total_matched
    }
    
    with open(OUT_DIR / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
        
    print(f"Validation complete. Macro Recall: {macro_recall:.4f}")

if __name__ == "__main__":
    main()
