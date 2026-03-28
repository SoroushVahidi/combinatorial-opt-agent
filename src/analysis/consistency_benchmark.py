"""Consistency benchmark: compare old vs repaired number-role checkers on synthetic cases."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path

SYNTHETIC_CASES = [
    # 7 correct cases where old checker incorrectly flags (answer is computed, not literally in question)
    {
        "case_id": "correct_01",
        "question": "A factory produces 12 widgets per hour. How many widgets are produced in 5 hours?",
        "reasoning": "12 widgets/hour × 5 hours = 60 widgets",
        "answer": "60",
        "is_correct": True,
        "failure_type": None,
    },
    {
        "case_id": "correct_02",
        "question": "You have 100 dollars and spend 37 dollars. How much is left?",
        "reasoning": "100 - 37 = 63 dollars remaining",
        "answer": "63",
        "is_correct": True,
        "failure_type": None,
    },
    {
        "case_id": "correct_03",
        "question": "A store needs at least 20 boxes but no more than 50 boxes. The optimal order is the minimum required. How many boxes should be ordered?",
        "reasoning": "The minimum is 20 boxes per the constraint.",
        "answer": "20",
        "is_correct": True,
        "failure_type": None,
    },
    {
        "case_id": "correct_04",
        "question": "There are 3 workers each earning 40 dollars per day. Total payroll after 5 days?",
        "reasoning": "3 workers × 40 dollars/day × 5 days = 600 dollars",
        "answer": "600",
        "is_correct": True,
        "failure_type": None,
    },
    {
        "case_id": "correct_05",
        "question": "A farmer has 8 acres. Each acre yields 15 bushels. Maximize the total harvest.",
        "reasoning": "8 × 15 = 120 bushels total harvest",
        "answer": "120",
        "is_correct": True,
        "failure_type": None,
    },
    {
        "case_id": "correct_06",
        "question": "Revenue is 250 dollars per unit. Variable cost is 180 dollars per unit. Fixed cost is 1000 dollars. Maximize profit at 30 units.",
        "reasoning": "Profit = (250 - 180) × 30 - 1000 = 70 × 30 - 1000 = 2100 - 1000 = 1100",
        "answer": "1100",
        "is_correct": True,
        "failure_type": None,
    },
    {
        "case_id": "correct_07",
        "question": "Minimize cost: produce at least 10 units. Each unit costs 5 dollars to produce.",
        "reasoning": "Minimum cost = 10 × 5 = 50 dollars",
        "answer": "50",
        "is_correct": True,
        "failure_type": None,
    },
    # 4 wrong: intermediate_as_final
    {
        "case_id": "wrong_intermediate_01",
        "question": "A truck carries 8 boxes per trip. Maximize boxes delivered in 6 trips.",
        "reasoning": "Intermediate step: 8 boxes per trip. Final answer: 8",
        "answer": "8",
        "is_correct": False,
        "failure_type": "intermediate_as_final",
    },
    {
        "case_id": "wrong_intermediate_02",
        "question": "Profit per unit is 15 dollars. Maximize profit selling 20 units.",
        "reasoning": "Revenue per unit is 15. Total = 15",
        "answer": "15",
        "is_correct": False,
        "failure_type": "intermediate_as_final",
    },
    {
        "case_id": "wrong_intermediate_03",
        "question": "A worker earns 12 dollars per hour. Must work at least 40 hours. What is minimum weekly pay?",
        "reasoning": "Rate is 12 per hour. Answer: 40",
        "answer": "40",
        "is_correct": False,
        "failure_type": "intermediate_as_final",
    },
    {
        "case_id": "wrong_intermediate_04",
        "question": "Each machine costs 200 dollars to run. Minimize cost for 3 machines over 5 days.",
        "reasoning": "Cost per machine is 200. Answer: 200",
        "answer": "200",
        "is_correct": False,
        "failure_type": "intermediate_as_final",
    },
    # 4 wrong: wrong_target_quantity
    {
        "case_id": "wrong_target_01",
        "question": "Maximize revenue: sell at most 100 units at 25 dollars each.",
        "reasoning": "Revenue = 100 × 25. Answer: 100",
        "answer": "100",
        "is_correct": False,
        "failure_type": "wrong_target_quantity",
    },
    {
        "case_id": "wrong_target_02",
        "question": "Minimize waste: produce no more than 50 items. Each item generates 2 kg waste.",
        "reasoning": "Maximum items is 50. Answer: 50",
        "answer": "50",
        "is_correct": False,
        "failure_type": "wrong_target_quantity",
    },
    {
        "case_id": "wrong_target_03",
        "question": "A project requires at least 5 workers. Each worker costs 300 dollars. What is the minimum cost?",
        "reasoning": "Need at least 5 workers. Answer: 5",
        "answer": "5",
        "is_correct": False,
        "failure_type": "wrong_target_quantity",
    },
    {
        "case_id": "wrong_target_04",
        "question": "Maximize profit: each product yields 40 dollars profit. Sell no fewer than 10 units.",
        "reasoning": "Minimum units is 10. Answer: 10",
        "answer": "10",
        "is_correct": False,
        "failure_type": "wrong_target_quantity",
    },
]


def _old_checker_flag(case: dict) -> bool:
    """Flag if answer value is not found as a literal number in the question text."""
    return case["answer"] not in case["question"]


def _repaired_checker_flag(case: dict) -> bool:
    """Use full pipeline; flag only if suspicious with confidence != 'low'."""
    from src.features.number_role_features import extract_number_mentions, annotate_relevance
    from src.features.number_role_repair import (
        repair_number_roles, calibrate_required_flags, detect_suspicious_missing_roles
    )

    mentions = extract_number_mentions(case["question"])
    annotated = annotate_relevance(mentions, case["question"])
    repaired = repair_number_roles(case["question"], annotated)
    calibrated = calibrate_required_flags(case["question"], repaired)
    result = detect_suspicious_missing_roles(case["question"], case["reasoning"], calibrated)
    return result["suspicious_missing"] and result["confidence"] != "low"


def run_benchmark(cases: list[dict]) -> tuple[dict, list[dict]]:
    """
    Run both checkers on all cases.
    Returns (results_dict, per_case_list).

    results_dict keys: n_cases, n_correct, n_wrong, old, repaired
    old/repaired keys: fpr_correct, recall_wrong, n_flagged_correct, n_flagged_wrong
    """
    per_case = []

    correct_cases = [c for c in cases if c["is_correct"]]
    wrong_cases = [c for c in cases if not c["is_correct"]]

    for case in cases:
        old_flagged = _old_checker_flag(case)
        repaired_flagged = _repaired_checker_flag(case)

        is_correct = case["is_correct"]

        # old_correct: did old checker make the right call?
        if is_correct:
            old_correct = not old_flagged
            repaired_correct = not repaired_flagged
        else:
            old_correct = old_flagged
            repaired_correct = repaired_flagged

        per_case.append({
            "case_id": case["case_id"],
            "is_correct": is_correct,
            "failure_type": case["failure_type"],
            "old_flagged": old_flagged,
            "repaired_flagged": repaired_flagged,
            "old_correct": old_correct,
            "repaired_correct": repaired_correct,
        })

    n_correct = len(correct_cases)
    n_wrong = len(wrong_cases)

    # FPR: fraction of correct cases that were flagged (false positives)
    old_flagged_correct = sum(1 for r in per_case if r["is_correct"] and r["old_flagged"])
    repaired_flagged_correct = sum(1 for r in per_case if r["is_correct"] and r["repaired_flagged"])

    # Recall: fraction of wrong cases that were flagged (true positives)
    old_flagged_wrong = sum(1 for r in per_case if not r["is_correct"] and r["old_flagged"])
    repaired_flagged_wrong = sum(1 for r in per_case if not r["is_correct"] and r["repaired_flagged"])

    fpr_old = old_flagged_correct / n_correct if n_correct > 0 else 0.0
    fpr_repaired = repaired_flagged_correct / n_correct if n_correct > 0 else 0.0
    recall_old = old_flagged_wrong / n_wrong if n_wrong > 0 else 0.0
    recall_repaired = repaired_flagged_wrong / n_wrong if n_wrong > 0 else 0.0

    results = {
        "n_cases": len(cases),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "old": {
            "fpr_correct": fpr_old,
            "recall_wrong": recall_old,
            "n_flagged_correct": old_flagged_correct,
            "n_flagged_wrong": old_flagged_wrong,
        },
        "repaired": {
            "fpr_correct": fpr_repaired,
            "recall_wrong": recall_repaired,
            "n_flagged_correct": repaired_flagged_correct,
            "n_flagged_wrong": repaired_flagged_wrong,
        },
    }

    return results, per_case


def write_benchmark_outputs(results: dict, per_case: list[dict], output_dir: str) -> None:
    """Write benchmark outputs to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # summary.json
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # per_candidate_results.csv
    with open(out / "per_candidate_results.csv", "w", newline="") as f:
        fieldnames = ["case_id", "is_correct", "failure_type", "old_flagged", "repaired_flagged", "old_correct", "repaired_correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_case)

    # failure_type_summary.csv
    from collections import defaultdict
    by_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "old_correct": 0, "repaired_correct": 0})
    for r in per_case:
        ft = r["failure_type"] if r["failure_type"] else "correct"
        by_type[ft]["total"] += 1
        by_type[ft]["old_correct"] += int(r["old_correct"])
        by_type[ft]["repaired_correct"] += int(r["repaired_correct"])

    with open(out / "failure_type_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["failure_type", "total", "old_correct", "repaired_correct"])
        writer.writeheader()
        for ft, stats in sorted(by_type.items()):
            writer.writerow({"failure_type": ft, **stats})

    # role_signal_summary.csv
    with open(out / "role_signal_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "old", "repaired"])
        writer.writeheader()
        writer.writerow({
            "metric": "fpr_correct",
            "old": f"{results['old']['fpr_correct']:.4f}",
            "repaired": f"{results['repaired']['fpr_correct']:.4f}",
        })
        writer.writerow({
            "metric": "recall_wrong",
            "old": f"{results['old']['recall_wrong']:.4f}",
            "repaired": f"{results['repaired']['recall_wrong']:.4f}",
        })
        writer.writerow({
            "metric": "n_flagged_correct",
            "old": results["old"]["n_flagged_correct"],
            "repaired": results["repaired"]["n_flagged_correct"],
        })
        writer.writerow({
            "metric": "n_flagged_wrong",
            "old": results["old"]["n_flagged_wrong"],
            "repaired": results["repaired"]["n_flagged_wrong"],
        })
