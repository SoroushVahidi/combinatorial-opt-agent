"""Mockable OptMATH pipeline and append-only resume store."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from baselines.optmath.data_adapter import OptmathInputRecord
from baselines.optmath.result_schema import OptmathResult
from baselines.optmath.output_normalizer import parse_output
from baselines.optmath.runner import OptmathRunner
from baselines.optmath.static_validation import validate_code


def run_one(record: OptmathInputRecord, runner: OptmathRunner, *, git_sha: str | None = None) -> OptmathResult:
    prompt = record.prompt(runner.config)
    generation = runner.generate(prompt)
    result = OptmathResult(record.problem_id, record.dataset, record.raw_text_sha256, runner.config.model_id, runner.config.model_revision, runner.config.upstream_revision, prompt.version, prompt.user_sha256, generation, git_sha=git_sha, timestamp_utc=datetime.now(timezone.utc).isoformat(), gold_objective=record.gold_metadata.get("gold_objective"), error_category=generation.error_category)
    if generation.status == "COMPLETED":
        result.parsed = parse_output(generation.raw_output)
        result.static_validation = validate_code(result.parsed.generated_code)
        if result.static_validation.status != "STATIC_VALID": result.error_category = "syntax_invalid" if not result.static_validation.syntax_valid else "structural_invalid"
    return result


class JsonlResultStore:
    def __init__(self, path: str | Path): self.path = Path(path)
    def completed_ids(self) -> set[str]:
        if not self.path.exists(): return set()
        with self.path.open(encoding="utf-8") as handle: return {str(json.loads(line)["problem_id"]) for line in handle if line.strip()}
    def append(self, result: OptmathResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle: handle.write(result.to_json() + "\n")
    def append_unfinished(self, records: Iterable[OptmathInputRecord], runner: OptmathRunner, *, git_sha: str | None = None) -> list[OptmathResult]:
        completed = self.completed_ids()
        results = []
        for record in records:
            if record.problem_id in completed:
                continue
            result = run_one(record, runner, git_sha=git_sha)
            self.append(result)
            results.append(result)
        return results
