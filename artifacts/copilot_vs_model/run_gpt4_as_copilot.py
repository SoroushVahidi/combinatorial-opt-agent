#!/usr/bin/env python3
"""Step 4: Automate Copilot collection via the OpenAI GPT-4 API.

This script sends all 26 PENDING benchmark cases to the OpenAI Chat Completions
API (gpt-4o or gpt-4-turbo-preview) using the same prompt template as the human
Copilot collection workflow.  Responses are ingested into copilot_outputs.jsonl
automatically — no manual copy-paste needed.

Usage
-----
    # Set your API key:
    export OPENAI_API_KEY="sk-..."

    # Run all pending cases (dry-run to preview without writing):
    python artifacts/copilot_vs_model/run_gpt4_as_copilot.py --dry-run

    # Run for real (writes to copilot_outputs.jsonl):
    python artifacts/copilot_vs_model/run_gpt4_as_copilot.py

    # Run for a single case:
    python artifacts/copilot_vs_model/run_gpt4_as_copilot.py --case-id nlp4lp_test_0

    # After all cases are ingested, re-score:
    python artifacts/copilot_vs_model/score_comparison.py

Requirements
------------
    pip install openai

Notes
-----
* The script uses the same JSON prompt already present in copilot_prompts/<case_id>.txt.
* GPT-4o is used by default (same capability class as GitHub Copilot Chat).  Pass
  --model gpt-4-turbo-preview to use the older GPT-4 Turbo checkpoint.
* A temperature of 0 is used for determinism.
* The script retries with exponential back-off on rate-limit errors (up to 5 times).
* Already-filled cases (status != PENDING) are skipped unless --force is given.
* Estimated cost: ~$0.05-0.10 per case with gpt-4o, ~$1.50 total for 26 cases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PROMPTS_DIR   = ROOT / "artifacts" / "copilot_vs_model" / "copilot_prompts"
OUTPUTS_FILE  = ROOT / "artifacts" / "copilot_vs_model" / "copilot_outputs.jsonl"
BENCH_FILE    = ROOT / "artifacts" / "copilot_vs_model" / "benchmark_cases.jsonl"
DEFAULT_MODEL = "gpt-4o"
MAX_RETRIES   = 5


# ── helpers ───────────────────────────────────────────────────────────────────

def load_outputs() -> dict[str, dict]:
    """Load copilot_outputs.jsonl keyed by case_id."""
    outputs: dict[str, dict] = {}
    if OUTPUTS_FILE.exists():
        for line in OUTPUTS_FILE.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                outputs[obj["case_id"]] = obj
    return outputs


def save_outputs(outputs: dict[str, dict]) -> None:
    """Write copilot_outputs.jsonl sorted by the original benchmark order."""
    bench_order = [json.loads(l)["case_id"] for l in BENCH_FILE.read_text().splitlines() if l.strip()]
    with OUTPUTS_FILE.open("w") as f:
        for cid in bench_order:
            if cid in outputs:
                f.write(json.dumps(outputs[cid]) + "\n")


def is_pending(entry: dict) -> bool:
    return entry.get("parse_error") == "PENDING" or entry.get("status") == "PENDING"


def call_openai(client, model: str, prompt_text: str) -> tuple[str, dict | None, str | None]:
    """Call OpenAI Chat Completions; return (raw_text, parsed_dict | None, error | None)."""
    messages = [
        {"role": "system", "content": "You are an expert in mathematical optimization."},
        {"role": "user",   "content": prompt_text},
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        raw = ""  # ensure raw is defined before the try block in all exception paths
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            parsed = json.loads(raw)
            return raw, parsed, None
        except json.JSONDecodeError as exc:
            return raw, None, f"JSON parse error: {exc}"
        except Exception as exc:
            err_str = str(exc)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = 2 ** attempt
                print(f"    Rate limit — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                return raw, None, f"API error: {exc}"
    return "", None, f"Exhausted {MAX_RETRIES} retries"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Automate Copilot collection via GPT-4 API")
    ap.add_argument("--case-id",  default=None, help="Run a single case (default: all pending)")
    ap.add_argument("--model",    default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    ap.add_argument("--dry-run",  action="store_true", help="Print prompts but do not call the API")
    ap.add_argument("--force",    action="store_true", help="Re-run cases that are already filled")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # Lazily import openai (not in standard deps)
    if not args.dry_run:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
        except ImportError:
            print("ERROR: openai package not installed.  Run: pip install openai")
            sys.exit(1)
    else:
        client = None  # type: ignore

    outputs = load_outputs()

    # Determine which cases to process
    if args.case_id:
        case_ids = [args.case_id]
    else:
        bench_cases = [json.loads(l)["case_id"] for l in BENCH_FILE.read_text().splitlines() if l.strip()]
        case_ids = [
            cid for cid in bench_cases
            if cid in outputs and is_pending(outputs[cid])
        ]
        if not case_ids:
            print("All cases are already filled.  Use --force to re-run them.")
            return

    print(f"Processing {len(case_ids)} case(s) with model {args.model} …")

    success = errors = skipped = 0

    for cid in case_ids:
        prompt_file = PROMPTS_DIR / f"{cid}.txt"
        if not prompt_file.exists():
            print(f"  [SKIP] {cid}: no prompt file at {prompt_file}")
            skipped += 1
            continue

        if not args.force and cid in outputs and not is_pending(outputs[cid]):
            print(f"  [SKIP] {cid}: already filled (use --force to overwrite)")
            skipped += 1
            continue

        prompt_text = prompt_file.read_text()

        if args.dry_run:
            print(f"  [DRY-RUN] {cid}")
            print(f"    Prompt ({len(prompt_text)} chars) — would call {args.model}")
            continue

        print(f"  [{cid}] calling {args.model} …", end=" ", flush=True)
        raw, parsed, err = call_openai(client, args.model, prompt_text)

        if err:
            print(f"FAIL  ({err})")
            outputs[cid] = {
                "case_id":     cid,
                "model":       args.model,
                "raw_response": raw,
                "parsed":      None,
                "parse_error": err,
            }
            errors += 1
        else:
            n_slots = len(parsed.get("slot_value_assignments", {}))
            print(f"OK  ({n_slots} slot assignments)")
            outputs[cid] = {
                "case_id":     cid,
                "model":       args.model,
                "raw_response": raw,
                "parsed":      parsed,
                "parse_error": None,
            }
            success += 1

        # Save after every case so partial results are not lost
        save_outputs(outputs)
        # Brief pause between successful calls as a courtesy to the API.
        # Rate-limit back-off is handled separately in call_openai().
        if not err:
            time.sleep(0.3)

    print(f"\nDone.  Success: {success}  Errors: {errors}  Skipped: {skipped}")
    if success > 0 and not args.dry_run:
        print(f"Re-run scorer:  python {ROOT / 'artifacts/copilot_vs_model/score_comparison.py'}")


if __name__ == "__main__":
    main()
