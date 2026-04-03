#!/usr/bin/env python3
"""Mandatory Mistral API preflight before long NLP4LP batch jobs.

Checks MISTRAL_API_KEY, runs one minimal chat completion (JSON mode), and optionally
estimates API call budget for the two-stage baseline.

Exit codes:
  0  success
  1  missing key / configuration / API failure (incl. 401/403/429)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.llm_baselines import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    resolve_mistral_model_from_config,
)


def _estimate_nlp4lp_calls(*, n_queries: int, n_variants: int) -> dict[str, int]:
    """Two chat.completions per query (retrieval + instantiation)."""
    per_variant = max(0, int(n_queries)) * 2
    return {
        "queries_per_variant": int(n_queries),
        "variants": int(n_variants),
        "estimated_chat_completions": per_variant * int(n_variants),
        "per_variant_calls": per_variant,
    }


def _run_probe(model: str) -> tuple[bool, str | None, dict]:
    try:
        from mistralai.client import Mistral
        from mistralai.client.errors import MistralError
    except Exception as e:  # pragma: no cover
        return False, f"mistralai import failed: {e}", {}

    key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
    if not key:
        return False, "MISTRAL_API_KEY is not set", {}

    client = Mistral(api_key=key)
    try:
        resp = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": '{"ping": true}'},
            ],
            temperature=0.0,
            max_tokens=32,
            response_format={"type": "json_object"},
        )
    except MistralError as e:
        return False, str(e), {"status_code": getattr(e, "status_code", None)}
    except Exception as e:
        return False, str(e), {}

    choice0 = resp.choices[0] if getattr(resp, "choices", None) else None
    msg = getattr(choice0, "message", None) if choice0 else None
    content = (getattr(msg, "content", None) or "").strip()
    return True, None, {"response_preview": content[:200]}


def main() -> None:
    p = argparse.ArgumentParser(description="Mistral API preflight for NLP4LP batch jobs")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--model", type=str, default=None, help="Override model (else MISTRAL_MODEL / yaml)")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write structured preflight result to this path (parent dirs created).",
    )
    p.add_argument(
        "--estimate-calls",
        action="store_true",
        help="Include NLP4LP call budget (331 queries × 2 stages × variants) in JSON output.",
    )
    p.add_argument("--estimate-queries", type=int, default=331, help="Queries per variant for --estimate-calls")
    p.add_argument("--estimate-variants", type=int, default=3, help="Variants for estimate (orig/noisy/short)")
    args = p.parse_args()

    if not (os.environ.get("MISTRAL_API_KEY") or "").strip():
        print("ERROR: MISTRAL_API_KEY is not set.", file=sys.stderr)
        raise SystemExit(1)

    model = (args.model or "").strip() or resolve_mistral_model_from_config(args.config)
    ts = datetime.now(timezone.utc).isoformat()
    result: dict = {
        "ok": False,
        "provider": "mistral",
        "model": model,
        "timestamp_utc": ts,
        "probe": {},
        "estimate": None,
        "error": None,
    }

    ok, err, probe_meta = _run_probe(model)
    result["probe"] = probe_meta
    if not ok:
        result["error"] = err
        print(json.dumps(result, indent=2))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"ERROR: {err}", file=sys.stderr)
        raise SystemExit(1)

    result["ok"] = True
    if args.estimate_calls:
        result["estimate"] = _estimate_nlp4lp_calls(
            n_queries=args.estimate_queries,
            n_variants=args.estimate_variants,
        )
        est = result["estimate"]["estimated_chat_completions"]
        print(
            f"INFO: Full {args.estimate_variants}-variant NLP4LP Mistral baseline needs ~{est} chat completions "
            f"({args.estimate_queries} queries × 2 stages × {args.estimate_variants} variants).",
            file=sys.stderr,
        )

    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("OK: mistral_preflight passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
