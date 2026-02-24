"""
Analyze collected chats and flags to guide improvements to the retrieval model.

Usage:
    python analyze_feedback.py

This script reads:
  - data/feedback/chat_logs.jsonl  (one JSON record per interaction)
  - any CSV files under data/feedback/ produced by Gradio flagging

and prints simple summaries:
  - total number of chats
  - most common queries
  - most frequently returned problems
  - flagged counts by reason and by problem
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FEEDBACK_DIR = ROOT / "data" / "feedback"
CHAT_LOG_PATH = FEEDBACK_DIR / "chat_logs.jsonl"


def load_chat_logs() -> list[dict]:
    records: list[dict] = []
    if not CHAT_LOG_PATH.exists():
        return records
    with CHAT_LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing.
                continue
    return records


def load_flag_csv_files() -> list[dict]:
    """Load all rows from CSV files created by Gradio flagging."""
    rows: list[dict] = []
    if not FEEDBACK_DIR.exists():
        return rows
    for csv_path in FEEDBACK_DIR.glob("*.csv"):
        try:
            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows.extend(reader)
        except Exception:
            continue
    return rows


def summarize_chats(records: list[dict]) -> None:
    print("=== Chat Logs Summary ===")
    print(f"Total interactions: {len(records)}")
    if not records:
        print("No chat logs found.")
        print()
        return

    query_counter = Counter()
    problem_counter = Counter()

    for rec in records:
        query = (rec.get("query") or "").strip()
        if query:
            query_counter[query] += 1
        for res in rec.get("results") or []:
            pid = res.get("problem_id") or "<unknown>"
            problem_counter[pid] += 1

    print("\nTop queries:")
    for query, count in query_counter.most_common(10):
        print(f"  {count:4d} × {query}")

    print("\nMost frequently returned problems (by id):")
    for pid, count in problem_counter.most_common(10):
        print(f"  {count:4d} × {pid}")

    print()


def summarize_flags(rows: list[dict]) -> None:
    print("=== Flagged Interactions Summary ===")
    print(f"Total flagged rows: {len(rows)}")
    if not rows:
        print("No flagged data found.")
        print()
        return

    reason_counter = Counter()
    problem_counter = Counter()
    # Map reason -> problem_id -> count, where possible.
    reason_by_problem: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        # Gradio by default stores the selected flagging option in a column
        # usually named like "flag" or the last column; try a few keys.
        reason = (
            row.get("flag")
            or row.get("flagging")
            or row.get("label")
            or row.get("Unnamed: 0")
            or ""
        )
        reason = reason.strip()
        if not reason and row:
            # Fallback: last column value.
            *_, last_val = row.values()
            reason = (last_val or "").strip()

        if reason:
            reason_counter[reason] += 1

        # Try to infer a problem id from the output markdown if present.
        output_text = (row.get("Answer") or row.get("output") or "").strip()
        problem_id = None
        if "id:" in output_text:
            # Very loose heuristic; you can tighten this later if you embed ids.
            # Right now most outputs don't expose ids, so this will often be empty.
            problem_id = output_text.split("id:", 1)[1].split()[0].strip()

        if problem_id:
            problem_counter[problem_id] += 1
            if reason:
                reason_by_problem[problem_id][reason] += 1

    print("\nFlags by reason:")
    for reason, count in reason_counter.most_common():
        print(f"  {count:4d} × {reason}")

    if problem_counter:
        print("\nMost flagged problems (heuristic, if ids are present in outputs):")
        for pid, count in problem_counter.most_common(10):
            print(f"  {count:4d} × {pid}")
            for reason, rcount in reason_by_problem[pid].most_common():
                print(f"         {rcount:4d} × {reason}")

    print()


def main() -> None:
    chats = load_chat_logs()
    flags = load_flag_csv_files()
    summarize_chats(chats)
    summarize_flags(flags)


if __name__ == "__main__":
    main()

