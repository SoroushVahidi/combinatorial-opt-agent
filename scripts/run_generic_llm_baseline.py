"""Run the GENERAL_PURPOSE_LLM_BASELINE on the fixed NLP4LP pilot/common-18 set.

Resumable CLI launcher. Real mode calls the Azure OpenAI `gpt-5.4`
deployment; `--mock` uses a clearly-labeled fake backend that writes to a
separate `.mock.jsonl` path so real outputs can never contain mock rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from baselines.generic_llm.config import GENERIC_LLM_DEPLOYMENT, GENERIC_LLM_SYSTEM_PROMPT, GENERIC_LLM_USER_TEMPLATE, GenericLLMConfig
from baselines.generic_llm.pipeline import JsonlResultStore
from baselines.generic_llm.prompt import build_prompt
from baselines.optmath.data_adapter import OptmathInputRecord

from scripts.run_optmath_inference import _load_gold_objectives, _load_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/nlp4lp_eval_orig.jsonl"
DEFAULT_MANIFEST = ROOT / "baselines/optmath/manifests/nlp4lp_common_manifest.json"
DEFAULT_OUTPUT = ROOT / "results/generic_llm/common18_official"
DEFAULT_GOLD_CACHE = ROOT / "results/eswa_revision/00_env/nlp4lp_gold_cache.json"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_metadata(path: Path, *, args: argparse.Namespace, records: list[OptmathInputRecord], git_sha: str, mock: bool) -> None:
    config = GenericLLMConfig(
        provider=args.provider,
        deployment=args.deployment,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    payload = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "mode": "mock" if mock else "real",
        "provider": config.provider,
        "deployment": config.deployment,
        "prompt_version": config.prompt_version,
        "prompt_template_sha256": hashlib.sha256((GENERIC_LLM_SYSTEM_PROMPT + "\n" + GENERIC_LLM_USER_TEMPLATE).encode("utf-8")).hexdigest(),
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
        "label": "GENERAL_PURPOSE_LLM_BASELINE",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _MockBackend:
    """Clearly-labeled fake backend; used ONLY when --mock is requested."""

    def __init__(self) -> None:
        self._code = (
            "import gurobipy as gp\nfrom gurobipy import GRB\n"
            "model = gp.Model()\nx = model.addVar(lb=0)\n"
            "model.setObjective(x, GRB.MAXIMIZE)\nmodel.addConstr(x <= 1)\n"
            "model.optimize()\nprint(model.objVal)\n"
        )

    def generate(self, prompt, config):
        return "MOCK OUTPUT\n```python\n" + self._code + "```", {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}


def _run_mock(record: OptmathInputRecord, config: GenericLLMConfig, *, git_sha: str | None = None):
    from baselines.generic_llm.pipeline import run_one
    from baselines.generic_llm import runner as generic_runner
    from unittest.mock import patch
    with patch.object(generic_runner, "generate", new=lambda prompt, cfg: _mock_generation(prompt)):
        return run_one(record, config, git_sha=git_sha)


def _mock_generation(prompt):
    from baselines.generic_llm.runner import GenerationResult
    return GenerationResult(
        raw_output="MOCK OUTPUT\n```python\n" + _MockBackend()._code + "```",
        status="COMPLETED", provider="azure_openai", deployment="mock-deployment",
        underlying_model="mock-model", prompt_sha256=prompt.user_sha256,
        runtime_seconds=0.001, prompt_tokens=5, completion_tokens=12,
        total_tokens=17, finish_reason="stop", retry_count=0,
        environment={"python": "mock", "platform": "mock"},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GENERAL_PURPOSE_LLM_BASELINE on the fixed NLP4LP pilot/common-18 set.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--subset", choices=("pilot", "common18"), default="common18")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "results.jsonl")
    parser.add_argument("--gold-cache", type=Path, default=DEFAULT_GOLD_CACHE)
    parser.add_argument("--provider", default="azure_openai")
    parser.add_argument("--deployment", default=GENERIC_LLM_DEPLOYMENT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--mock", action="store_true", help="Use the clearly-labeled mock backend; writes to a separate .mock output path.")
    return parser


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
    records = _load_records(args.input, args.manifest, args.subset, gold_objectives)
    git_sha = _git_sha()
    print(json.dumps({"event": "records_loaded", "subset": args.subset, "mode": "mock" if args.mock else "real", "problem_ids": [int(r.problem_id) for r in records]}), flush=True)

    metadata_path = args.output.with_name("run_metadata.json")
    if not metadata_path.exists():
        _write_metadata(metadata_path, args=args, records=records, git_sha=git_sha, mock=args.mock)

    config = GenericLLMConfig(
        provider=args.provider,
        deployment=args.deployment,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    store = JsonlResultStore(args.output)
    print(json.dumps({"event": "starting_api_runner", "mode": "mock" if args.mock else "real", "provider": config.provider, "deployment": config.deployment, "temperature": config.temperature, "max_tokens": config.max_tokens}), flush=True)

    results = []
    if args.mock:
        for record in records:
            if record.problem_id in store.completed_ids():
                continue
            result = _run_mock(record, config, git_sha=git_sha)
            store.append(result)
            results.append(result)
    else:
        results = store.append_unfinished(records, config, git_sha=git_sha)

    print(json.dumps({"output": str(args.output), "attempted": len(results), "completed_ids": sorted(store.completed_ids())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())