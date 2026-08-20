"""Run the pinned OptMATH checkpoint on the fixed NLP4LP pilot or common-18 set.

Resumable CLI launcher around the existing OptMATH baseline code. Mirrors
`scripts/run_orlm_inference.py`. Never fabricates rows: only the real
Transformers backend (or an explicitly requested, clearly separated mock
backend) writes rows, and `--mock` routes to a separate output path so real
outputs can never contain mock rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from baselines.optmath.config import OPTMATH_PRIMARY_MODEL, OPTMATH_PROMPT_VERSION, OptmathConfig
from baselines.optmath.data_adapter import OptmathInputRecord
from baselines.optmath.pipeline import JsonlResultStore
from baselines.optmath.runner import OptmathRunner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/nlp4lp_eval_orig.jsonl"
DEFAULT_MANIFEST = ROOT / "baselines/optmath/manifests/nlp4lp_common_manifest.json"
DEFAULT_OUTPUT = ROOT / "results/optmath/pilot_official_checkpoint"
DEFAULT_GOLD_CACHE = ROOT / "results/eswa_revision/00_env/nlp4lp_gold_cache.json"
OPTMATH_CHECKPOINT_REVISION = "617fe77"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_gold_objectives(gold_cache: Path | None) -> dict[str, object]:
    """Maps nlp4lp_test_N -> gold objective value (or None) when available.

    The gold cache layout is ``{"split": ..., "gold_by_id": {query_id: {...}}}``
    where ``gold_by_id`` is itself keyed directly by query_id (not by split).
    """
    if gold_cache is None or not gold_cache.exists():
        return {}
    try:
        cache = json.loads(gold_cache.read_text(encoding="utf-8"))
        gold_by_id = cache["gold_by_id"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return {}
    out: dict[str, object] = {}
    for query_id, entry in gold_by_id.items():
        if isinstance(entry, dict):
            solution = entry.get("solution") or {}
            out[query_id] = solution.get("objective")
        else:
            out[query_id] = None
    return out


def _load_records(input_path: Path, manifest_path: Path, subset: str, gold_objectives: dict[str, object]) -> list[OptmathInputRecord]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = manifest["pilot_ids"] if subset == "pilot" else manifest["future_evaluation_ids"]
    by_query_id: dict[str, dict] = {}
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_query_id[str(row["query_id"])] = row

    records: list[OptmathInputRecord] = []
    for problem_id in ids:
        query_id = f"nlp4lp_test_{int(problem_id) - 1}"
        row = by_query_id.get(query_id)
        if row is None:
            raise KeyError(f"missing query for manifest problem_id={problem_id} ({query_id})")
        gold = {"relevant_doc_id": row.get("relevant_doc_id"), "query_id": query_id}
        if query_id in gold_objectives:
            gold["gold_objective"] = gold_objectives[query_id]
        records.append(
            OptmathInputRecord(
                problem_id=str(problem_id),
                dataset="nlp4lp",
                raw_text=str(row["query"]).strip(),
                gold_metadata=gold,
                source_metadata={"input_query_id": query_id, "manifest_problem_id": int(problem_id)},
            )
        )
    return records


def _write_metadata(path: Path, *, args: argparse.Namespace, records: list[OptmathInputRecord], git_sha: str, mock: bool) -> None:
    config = OptmathConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device_map=args.device_map,
        dtype=args.dtype,
        seed=args.seed,
    )
    payload = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "mode": "mock" if mock else "real",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "upstream_revision": config.upstream_revision,
        "prompt_version": config.prompt_version,
        "prompt_template_sha256": hashlib.sha256(_prompt_template_text().encode("utf-8")).hexdigest(),
        "generation": config.generation_dict(),
        "solver": config.solver,
        "timeout_seconds": config.timeout_seconds,
        "numerical_tolerance": config.numerical_tolerance,
        "subset": args.subset,
        "problem_ids": [int(r.problem_id) for r in records],
        "input_path": str(args.input),
        "manifest_path": str(args.manifest),
        "gold_cache_path": str(args.gold_cache) if args.gold_cache else None,
        "output_path": str(args.output),
        "device": args.device,
        "runner": "TransformersBackend (repository adaptation of official checkpoint)",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prompt_template_text() -> str:
    from baselines.optmath.config import OPTMATH_SYSTEM_PROMPT, OPTMATH_USER_TEMPLATE
    return OPTMATH_SYSTEM_PROMPT + "\n" + OPTMATH_USER_TEMPLATE


class _MockBackend:
    """Clearly-labeled fake backend; used ONLY when --mock is requested."""

    def __init__(self) -> None:
        from baselines.optmath.config import OPTMATH_SYSTEM_PROMPT
        self._code = (
            "import gurobipy as gp\nfrom gurobipy import GRB\n"
            "model = gp.Model()\nx = model.addVar(lb=0)\n"
            "model.setObjective(x, GRB.MAXIMIZE)\nmodel.addConstr(x <= 1)\n"
            "model.optimize()\nprint(model.objVal)\n"
        )
        self._system = OPTMATH_SYSTEM_PROMPT

    def generate(self, prompt, config):
        assert prompt.system == self._system
        return "MOCK OUTPUT\n```python\n" + self._code + "```", {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pinned OptMATH checkpoint on the fixed NLP4LP pilot/common-18 set.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", choices=("pilot", "common18"), default="pilot")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "results.jsonl")
    parser.add_argument("--gold-cache", type=Path, default=DEFAULT_GOLD_CACHE, help="Optional NLP4LP gold cache for gold_objective metadata.")
    parser.add_argument("--model-id", default=OPTMATH_PRIMARY_MODEL)
    parser.add_argument("--model-revision", default=OPTMATH_CHECKPOINT_REVISION)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mock", action="store_true", help="Run the separate mock/test backend; results go to a clearly separate .mock output path.")
    parser.add_argument(
        "--backfill-gold",
        action="store_true",
        help="Do not run inference; only attach gold_objective metadata from the gold cache to existing result rows "
        "that are missing it. Gold objectives are gold-standard metadata (never a model output), so backfilling them "
        "is idempotent and does not alter generation/parse/static evidence.",
    )
    return parser


def _backfill_gold(output: Path, gold_objectives: dict[str, object]) -> tuple[int, int]:
    """Attach gold_objective to existing rows missing it. Returns (rows_seen, rows_updated)."""
    if not output.exists():
        return 0, 0
    lines = output.read_text(encoding="utf-8").splitlines()
    updated = 0
    out_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("gold_objective") is None:
            query_id = f"nlp4lp_test_{int(row['problem_id']) - 1}"
            if query_id in gold_objectives:
                row["gold_objective"] = gold_objectives[query_id]
                updated += 1
        out_lines.append(json.dumps(row))
    output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(out_lines), updated


def main() -> int:
    args = build_parser().parse_args()
    args.input = args.input.resolve()
    args.manifest = args.manifest.resolve()
    if args.gold_cache is not None:
        args.gold_cache = args.gold_cache.resolve()
    if args.mock:
        args.output = args.output.with_suffix(".mock.jsonl")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    gold_objectives = _load_gold_objectives(args.gold_cache)

    if args.backfill_gold:
        rows_seen, rows_updated = _backfill_gold(args.output, gold_objectives)
        print(json.dumps({"event": "gold_backfilled", "output": str(args.output), "rows_seen": rows_seen, "rows_updated": rows_updated}), flush=True)
        return 0

    records = _load_records(args.input, args.manifest, args.subset, gold_objectives)
    git_sha = _git_sha()
    print(json.dumps({"event": "records_loaded", "subset": args.subset, "mode": "mock" if args.mock else "real", "problem_ids": [int(r.problem_id) for r in records]}), flush=True)

    metadata_path = args.output.with_name("run_metadata.json")
    if not metadata_path.exists():
        _write_metadata(metadata_path, args=args, records=records, git_sha=git_sha, mock=args.mock)

    config = OptmathConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device_map=args.device_map,
        dtype=args.dtype,
        seed=args.seed,
    )
    backend = _MockBackend() if args.mock else None
    runner = OptmathRunner(config=config, backend=backend)
    store = JsonlResultStore(args.output)
    print(json.dumps({"event": "starting_checkpoint_runner", "mode": "mock" if args.mock else "real", "model_id": config.model_id, "model_revision": config.model_revision, "device_map": config.device_map, "max_new_tokens": config.max_new_tokens, "temperature": config.temperature}), flush=True)
    results = store.append_unfinished(records, runner, git_sha=git_sha)
    print(json.dumps({"output": str(args.output), "attempted": len(results), "completed_ids": sorted(store.completed_ids())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())