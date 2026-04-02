#!/usr/bin/env python3
"""Mandatory Gemini preflight before long NLP4LP batch jobs.

Validates API key, lists models with generateContent, checks configured model,
runs one minimal generateContent call, and optionally estimates API call budget.

Exit codes:
  0  success
  1  configuration / model-not-listed / generic failure
  2  hard zero quota (limit:0) — do not start a long benchmark
  3  pick-model style total failure when --pick-instead is used
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.llm_baselines import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    GeminiHardQuotaError,
    classify_gemini_quota_failure,
    gemini_fallback_candidates,
    gemini_preflight,
    is_gemini_hard_zero_quota_error,
    list_gemini_models_with_generate_content,
    pick_first_gemini_model_with_quota,
    resolve_gemini_model_from_config,
    write_gemini_selection_artifact,
)


def _estimate_nlp4lp_calls(*, n_queries: int, n_variants: int) -> dict[str, int]:
    """Two generateContent calls per query (retrieval + instantiation)."""
    per_variant = max(0, int(n_queries)) * 2
    return {
        "queries_per_variant": int(n_queries),
        "variants": int(n_variants),
        "estimated_generate_content_calls": per_variant * int(n_variants),
        "per_variant_calls": per_variant,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Gemini API preflight for NLP4LP batch jobs")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--model", type=str, default=None, help="Override model (else GEMINI_MODEL / yaml)")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write structured preflight result to this path (parent dirs created).",
    )
    p.add_argument(
        "--pick-instead",
        action="store_true",
        help="Run pick-model probes over yaml/env fallback chain; sets chosen model in result.",
    )
    p.add_argument(
        "--write-selected",
        type=Path,
        default=None,
        help="With --pick-instead, write selection artifact (same schema as tools/llm_baselines pick-model).",
    )
    p.add_argument(
        "--estimate-calls",
        action="store_true",
        help="Include NLP4LP call budget (331 queries × 2 stages × variants) in JSON output.",
    )
    p.add_argument("--estimate-queries", type=int, default=331, help="Queries per variant for --estimate-calls")
    p.add_argument("--estimate-variants", type=int, default=3, help="Variants (orig/noisy/short) for estimate")
    p.add_argument(
        "--allow-unlisted",
        action="store_true",
        help="Allow model not returned by list_models (still runs probe unless --no-probe).",
    )
    p.add_argument("--no-probe", action="store_true", help="Only list_models check; no generateContent probe")
    args = p.parse_args()

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print("ERROR: GEMINI_API_KEY is not set.", file=sys.stderr)
        raise SystemExit(1)

    result: dict = {
        "ok": False,
        "picked_model": None,
        "listed_models_sample": [],
        "preflight": {},
        "estimate": None,
        "failure_class": None,
    }

    try:
        listed = list_gemini_models_with_generate_content()
        result["listed_models_sample"] = [r["name"] for r in listed[:40]]
        result["listed_models_count"] = len(listed)

        if args.pick_instead:
            cand_list = gemini_fallback_candidates(args.config)
            chosen, err, failures = pick_first_gemini_model_with_quota(candidates=cand_list)
            result["pick_candidates"] = cand_list
            result["pick_failures"] = [{"model": n, "error": e[:800]} for n, e in failures]
            if not chosen:
                result["failure_class"] = "pick_all_failed"
                result["last_error"] = err
                if args.write_selected is not None:
                    write_gemini_selection_artifact(
                        args.write_selected,
                        ok=False,
                        model=None,
                        candidates_ordered=cand_list,
                        failures=failures,
                        source="scripts/gemini_preflight",
                    )
                print(json.dumps(result, indent=2))
                if args.output_json:
                    args.output_json.parent.mkdir(parents=True, exist_ok=True)
                    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                raise SystemExit(3)
            result["picked_model"] = chosen
            os.environ["GEMINI_MODEL"] = chosen
            if args.write_selected is not None:
                write_gemini_selection_artifact(
                    args.write_selected,
                    ok=True,
                    model=chosen,
                    candidates_ordered=cand_list,
                    failures=failures,
                    source="scripts/gemini_preflight",
                )

        model_try = args.model or result.get("picked_model") or resolve_gemini_model_from_config(args.config)
        pf = gemini_preflight(
            model_try,
            config_path=args.config,
            probe_generate=not args.no_probe,
            require_listed=not args.allow_unlisted,
        )
        result["preflight"] = pf
        result["ok"] = True
        result["resolved_model"] = pf.get("model") or model_try

        if args.estimate_calls:
            result["estimate"] = _estimate_nlp4lp_calls(
                n_queries=args.estimate_queries,
                n_variants=args.estimate_variants,
            )
            est = result["estimate"]["estimated_generate_content_calls"]
            print(
                f"INFO: Full orig+noisy+short NLP4LP Gemini baseline needs ~{est} generateContent calls "
                f"({args.estimate_queries} queries × 2 stages × {args.estimate_variants} variants).",
                file=sys.stderr,
            )

        print(json.dumps(result, indent=2))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("OK: gemini_preflight passed.", file=sys.stderr)

    except GeminiHardQuotaError as e:
        result["ok"] = False
        result["failure_class"] = "hard_zero_quota"
        result["error"] = str(e)
        print(json.dumps(result, indent=2))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"HARD_QUOTA: {e}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as e:
        result["ok"] = False
        result["failure_class"] = classify_gemini_quota_failure(e)
        if is_gemini_hard_zero_quota_error(e):
            result["failure_class"] = "hard_zero_quota"
        result["error"] = str(e)
        print(json.dumps(result, indent=2))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
