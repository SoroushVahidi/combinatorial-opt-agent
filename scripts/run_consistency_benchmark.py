"""Runner script for consistency benchmark."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.consistency_benchmark import run_benchmark, write_benchmark_outputs, SYNTHETIC_CASES


def main():
    print("Running consistency benchmark...")
    results, per_case = run_benchmark(SYNTHETIC_CASES)

    output_dir = str(ROOT / "outputs" / "consistency_benchmark")
    write_benchmark_outputs(results, per_case, output_dir=output_dir)

    print(f"\n=== Summary ===")
    print(f"Total cases: {results['n_cases']}")
    print(f"Correct cases: {results['n_correct']}")
    print(f"Wrong cases: {results['n_wrong']}")
    print(f"\nOLD checker:")
    print(f"  FPR (correct flagged): {results['old']['fpr_correct']:.2%}")
    print(f"  Recall (wrong caught): {results['old']['recall_wrong']:.2%}")
    print(f"\nREPAIRED checker:")
    print(f"  FPR (correct flagged): {results['repaired']['fpr_correct']:.2%}")
    print(f"  Recall (wrong caught): {results['repaired']['recall_wrong']:.2%}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
