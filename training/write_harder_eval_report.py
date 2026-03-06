"""Write a one-page markdown report summarizing harder evaluation results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Write harder_eval_report.md")
    p.add_argument("--results-dir", type=Path, default=None)
    args = p.parse_args()

    results_dir = Path(args.results_dir or ROOT / "results")

    # Dataset sizes
    eval_test = ROOT / "data" / "processed" / "eval_test.jsonl"
    eval_masked = ROOT / "data" / "processed" / "eval_test_masked.jsonl"
    eval_nlp4lp = ROOT / "data" / "processed" / "nlp4lp_eval.jsonl"
    eval_optibench = ROOT / "data" / "processed" / "optibench_eval.jsonl"

    sizes = {
        "test_normal": _count_jsonl(eval_test),
        "test_masked": _count_jsonl(eval_masked),
        "nlp4lp": _count_jsonl(eval_nlp4lp),
        "optibench": _count_jsonl(eval_optibench),
    }

    # Coverage
    cov_nlp4lp = {}
    cov_optibench = {}
    nlp4lp_cov_path = results_dir / "nlp4lp_coverage.json"
    opti_cov_path = results_dir / "optibench_coverage.json"
    if nlp4lp_cov_path.exists():
        cov_nlp4lp = json.loads(nlp4lp_cov_path.read_text(encoding="utf-8"))
    if opti_cov_path.exists():
        cov_optibench = json.loads(opti_cov_path.read_text(encoding="utf-8"))

    # Metrics
    m_normal = _load_csv(results_dir / "baselines_test_normal.csv")
    m_masked = _load_csv(results_dir / "baselines_test_masked.csv")
    m_nlp4lp = _load_csv(results_dir / "baselines_nlp4lp.csv")
    m_optibench = _load_csv(results_dir / "baselines_optibench.csv")

    report_path = results_dir / "harder_eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Harder evaluation report\n\n")
        f.write("## Dataset sizes\n\n")
        f.write("- **test_normal**: {} queries\\n".format(sizes["test_normal"]))
        f.write("- **test_masked**: {} queries\\n".format(sizes["test_masked"]))
        f.write("- **NLP4LP**: {} mapped queries\\n".format(sizes["nlp4lp"]))
        f.write("- **OptiBench**: {} mapped queries\\n\n".format(sizes["optibench"]))

        f.write("## Mapping coverage (external)\\n\\n")
        if cov_nlp4lp:
            f.write("- **NLP4LP**: {mapped}/{total} mapped ({coverage:.2%})\\n".format(**cov_nlp4lp))
        else:
            f.write("- **NLP4LP**: coverage data not available\\n")
        if cov_optibench:
            f.write("- **OptiBench**: {mapped}/{total} mapped ({coverage:.2%})\\n\n".format(**cov_optibench))
        else:
            f.write("- **OptiBench**: coverage data not available\\n\n")

        def _write_table(title: str, rows: list[dict]):
            if not rows:
                f.write(f"### {title} (no data)\\n\\n")
                return
            f.write(f"### {title}\\n\\n")
            cols = ["baseline", "P@1", "MRR@10"]
            f.write("| baseline | P@1 | MRR@10 |\\n")
            f.write("|----------|-----|--------|\\n")
            for r in rows:
                f.write("| {baseline} | {P@1} | {MRR@10} |\\n".format(**r))
            f.write("\\n")

        f.write("## Metrics (P@1, MRR@10)\\n\\n")
        _write_table("Test (normal)", m_normal)
        _write_table("Test (name-masked)", m_masked)
        _write_table("NLP4LP (mapped subset)", m_nlp4lp)
        _write_table("OptiBench (mapped subset)", m_optibench)

        f.write("## Interpretation (brief)\\n\\n")
        f.write("- **Lexical baselines (BM25, TF-IDF)** typically degrade more on the name-masked and external eval sets,\n")
        f.write("  especially when direct lexical cues (problem name, aliases, numbers) are removed.\\n")
        f.write("- **SBERT/SBERT-finetuned** (when available) are expected to be more robust to these shifts,\n")
        f.write("  as they rely less on exact string matches and more on semantic similarity.\\n")
        f.write("- The gap between normal vs. masked / external performance quantifies how brittle the retrieval\n")
        f.write("  stack is to distribution shift — a key point to highlight to Q1 reviewers.\\n")

    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

