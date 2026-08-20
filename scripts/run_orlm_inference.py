"""Run the pinned ORLM checkpoint on the fixed NLP4LP pilot or common-18 set."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from baselines.orlm.config import ORLM_CHECKPOINT_REVISION, OrlmConfig
from baselines.orlm.data_adapter import OrlmInputRecord
from baselines.orlm.pipeline import JsonlResultStore
from baselines.orlm.runner import OrlmRunner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/nlp4lp_eval_orig.jsonl"
DEFAULT_MANIFEST = ROOT / "baselines/orlm/manifests/nlp4lp_common_manifest.json"
DEFAULT_OUTPUT = ROOT / "results/orlm/pilot_official_checkpoint"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_records(input_path: Path, manifest_path: Path, subset: str) -> list[OrlmInputRecord]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = manifest["pilot_ids"] if subset == "pilot" else manifest["future_evaluation_ids"]
    by_query_id: dict[str, dict] = {}
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_query_id[str(row["query_id"])] = row

    records: list[OrlmInputRecord] = []
    for problem_id in ids:
        query_id = f"nlp4lp_test_{int(problem_id) - 1}"
        row = by_query_id.get(query_id)
        if row is None:
            raise KeyError(f"missing query for manifest problem_id={problem_id} ({query_id})")
        records.append(
            OrlmInputRecord(
                problem_id=str(problem_id),
                source=str(input_path.relative_to(ROOT)),
                raw_text=str(row["query"]).strip(),
                gold_metadata={"relevant_doc_id": row.get("relevant_doc_id"), "query_id": query_id},
                source_metadata={"input_query_id": query_id, "manifest_problem_id": int(problem_id)},
            )
        )
    return records


def _write_metadata(path: Path, *, args: argparse.Namespace, records: list[OrlmInputRecord], git_sha: str) -> None:
    config = OrlmConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        dtype=args.dtype,
        seed=args.seed,
    )
    payload = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "prompt_version": config.prompt_version,
        "prompt_template_sha256": hashlib.sha256(config.prompt_template.encode("utf-8")).hexdigest(),
        "generation": config.generation_dict(),
        "solver": config.solver,
        "subset": args.subset,
        "problem_ids": [int(r.problem_id) for r in records],
        "input_path": str(args.input),
        "manifest_path": str(args.manifest),
        "output_path": str(args.output),
        "device": args.device,
        "runner": "TransformersBackend (repository adaptation of official checkpoint)",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", choices=("pilot", "common18"), default="pilot")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "results.jsonl")
    parser.add_argument("--model-id", default="CardinalOperations/ORLM-LLaMA-3-8B")
    parser.add_argument("--model-revision", default=ORLM_CHECKPOINT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.input = args.input.resolve()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = _load_records(args.input, args.manifest, args.subset)
    git_sha = _git_sha()
    print(json.dumps({"event": "records_loaded", "subset": args.subset, "problem_ids": [int(r.problem_id) for r in records]}), flush=True)
    metadata_path = args.output.with_name("run_metadata.json")
    if not metadata_path.exists():
        _write_metadata(metadata_path, args=args, records=records, git_sha=git_sha)
    config = OrlmConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        dtype=args.dtype,
        seed=args.seed,
    )
    runner = OrlmRunner(config=config)
    store = JsonlResultStore(args.output)
    print(json.dumps({"event": "starting_checkpoint_runner", "model_id": config.model_id, "model_revision": config.model_revision, "device_map": config.device_map, "max_new_tokens": config.max_new_tokens}), flush=True)
    results = store.append_unfinished(records, runner, git_sha=git_sha)
    print(json.dumps({"output": str(args.output), "attempted": len(results), "completed_ids": sorted(store.completed_ids())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
