"""Lightweight ORLM inference-to-validation pipeline and resumable JSONL store."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from baselines.orlm.config import OrlmConfig
from baselines.orlm.data_adapter import OrlmInputRecord
from baselines.orlm.output_normalizer import parse_orlm_output
from baselines.orlm.result_schema import OrlmResult
from baselines.orlm.runner import OrlmRunner
from baselines.orlm.static_validation import validate_coptpy_code


def run_one(record: OrlmInputRecord, runner: OrlmRunner, *, git_sha: str | None = None) -> OrlmResult:
    config = runner.config
    prompt = record.to_upstream_example(config)["prompt"]
    generation = runner.generate(prompt)
    result = OrlmResult.from_generation(record.problem_id, record.source, record.raw_text_sha256, config.prompt_version, generation)
    result.git_sha = git_sha
    result.timestamp_utc = datetime.now(timezone.utc).isoformat()
    if generation.status != "COMPLETED":
        return result
    result.parsed = parse_orlm_output(generation.raw_output)
    result.static_validation = validate_coptpy_code(result.parsed.coptpy_code)
    if result.static_validation.status != "STATIC_VALID":
        result.error_category = "python_syntax_failure" if not result.static_validation.python_syntax_valid else "static_validation_failure"
    return result


class JsonlResultStore:
    """Append-only store with problem-id resume semantics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def completed_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids: set[str] = set()
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ids.add(str(json.loads(line)["problem_id"]))
        return ids

    def append(self, result: OrlmResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(result.to_json() + "\n")

    def append_unfinished(self, records: Iterable[OrlmInputRecord], runner: OrlmRunner, *, git_sha: str | None = None) -> list[OrlmResult]:
        completed = self.completed_ids()
        results = []
        for record in records:
            if record.problem_id in completed:
                continue
            result = run_one(record, runner, git_sha=git_sha)
            self.append(result)
            results.append(result)
        return results
