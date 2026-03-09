#!/usr/bin/env python3
"""Print training environment diagnostics: Python, torch, transformers, CUDA, data dirs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    print("=== Training environment check ===\n")
    print("sys.executable:", sys.executable)
    print("python version:", sys.version.split()[0])
    cwd = Path.cwd()
    print("cwd:", cwd)

    # Torch
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device_count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"  device_{i}:", torch.cuda.get_device_name(i))
        else:
            print("device_count: 0")
    except ImportError as e:
        print("torch: MISSING:", e)
        return 1

    # Transformers
    try:
        import transformers
        print("transformers:", transformers.__version__)
    except ImportError as e:
        print("transformers: MISSING:", e)
        return 1

    # Expected input dirs (relative to repo or cwd)
    data_dir = ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp"
    aux_dir = ROOT / "artifacts" / "learning_aux_data" / "nl4opt"
    for name, p in [("learning_ranker_data/nlp4lp", data_dir), ("learning_aux_data/nl4opt", aux_dir)]:
        train_f = p / "train.jsonl"
        test_f = p / "test.jsonl"
        exists = p.exists()
        train_ok = train_f.exists()
        test_ok = test_f.exists()
        print(f"data_dir {name}: exists={exists} train.jsonl={train_ok} test.jsonl={test_ok}")

    print("\n=== Check complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
