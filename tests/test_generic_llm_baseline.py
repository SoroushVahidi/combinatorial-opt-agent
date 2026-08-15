"""Focused tests for the GENERAL_PURPOSE_LLM_BASELINE package and CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.generic_llm.config import GENERIC_LLM_DEPLOYMENT, GENERIC_LLM_PROMPT_VERSION, GenericLLMConfig
from baselines.generic_llm.prompt import build_prompt
from baselines.generic_llm.runner import GenerationResult
from baselines.optmath.data_adapter import OptmathInputRecord

from scripts.run_generic_llm_baseline import _run_mock, _write_metadata


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_is_fixed_zero_shot_without_gold():
    bundle = build_prompt("  Maximize x. \n")
    assert bundle.system == "You are an expert in operations research and optimization."
    assert "# Problem:" in bundle.user
    assert "Maximize x." in bundle.user
    assert bundle.version == GENERIC_LLM_PROMPT_VERSION
    assert "gold" not in bundle.user.lower()
    assert "answer" not in bundle.user.lower()
    assert bundle.user.count("```python") == 2


def test_mock_run_produces_parseable_static_valid_row():
    record = OptmathInputRecord(problem_id="14", dataset="nlp4lp", raw_text="Maximize x subject to x <= 1.", gold_metadata={"gold_objective": 1.0})
    config = GenericLLMConfig()
    result = _run_mock(record, config, git_sha="test")
    assert result.generation.status == "COMPLETED"
    assert result.generation.raw_output.startswith("MOCK OUTPUT")
    assert result.parsed is not None and result.parsed.code_block_found
    assert result.static_validation is not None and result.static_validation.status == "STATIC_VALID"
    assert result.checkpoint == GENERIC_LLM_DEPLOYMENT


def test_failed_generation_is_not_fabricated_as_success():
    from baselines.pamop.llm.azure_openai_provider import AzureOpenAIProvider
    from unittest.mock import patch

    def boom(self, prompt, config):
        raise RuntimeError("simulated api failure")

    record = OptmathInputRecord(problem_id="23", dataset="nlp4lp", raw_text="x", gold_metadata={})
    with patch.object(AzureOpenAIProvider, "generate", new=boom):
        from baselines.generic_llm.pipeline import run_one
        result = run_one(record, GenericLLMConfig(), git_sha="test")
    assert result.generation.status == "FAILED"
    assert result.generation.error_category == "api_call_failed"
    assert result.parsed is None
    assert result.objective_proxy_status == "NOT_EVALUABLE"


def test_generation_result_roundtrip():
    g = GenerationResult("out", "COMPLETED", "azure_openai", "gpt-5.4", "gpt-5.4-snapshot", "h", 1.0, 10, 20, 30, "stop", 0)
    d = g.to_dict()
    assert d["underlying_model"] == "gpt-5.4-snapshot"
    assert d["total_tokens"] == 30


def test_metadata_records_deployment_and_prompt(tmp_path):
    out = tmp_path / "results.jsonl"
    import argparse
    args = argparse.Namespace(input=Path("/in.jsonl"), manifest=Path("/m.json"), gold_cache=Path("/g.json"),
                              subset="common18", output=out, provider="azure_openai", deployment="gpt-5.4",
                              temperature=0.0, max_tokens=8192, top_p=None)
    records = [OptmathInputRecord("14", "nlp4lp", "text", gold_metadata={})]
    meta_path = out.with_name("run_metadata.json")
    _write_metadata(meta_path, args=args, records=records, git_sha="abc", mock=False)
    meta = json.loads(meta_path.read_text())
    assert meta["label"] == "GENERAL_PURPOSE_LLM_BASELINE"
    assert meta["deployment"] == "gpt-5.4"
    assert meta["prompt_version"] == GENERIC_LLM_PROMPT_VERSION
    assert meta["mode"] == "real"
    assert meta["problem_ids"] == [14]


def test_cli_mock_routes_to_separate_output(tmp_path):
    nlp4lp = tmp_path / "nlp4lp.jsonl"
    lines = [json.dumps({"query_id": f"nlp4lp_test_{i}", "query": f"Problem text {i}", "relevant_doc_id": f"nlp4lp_test_{i}"}) for i in range(0, 270)]
    nlp4lp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pilot_ids": [14, 23, 34, 59, 69, 72], "future_evaluation_ids": [14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262]}), encoding="utf-8")
    out = tmp_path / "out" / "results.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_generic_llm_baseline",
         "--input", str(nlp4lp), "--manifest", str(manifest),
         "--subset", "pilot", "--output", str(out), "--mock"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    mock_path = out.with_suffix(".mock.jsonl")
    assert mock_path.exists()
    assert not out.exists()
    rows = [json.loads(l) for l in mock_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 6
    assert all(r["generation"]["raw_output"].startswith("MOCK OUTPUT") for r in rows)
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert meta["mode"] == "mock"


def test_cli_resume_skips_completed(tmp_path):
    nlp4lp = tmp_path / "nlp4lp.jsonl"
    lines = [json.dumps({"query_id": f"nlp4lp_test_{i}", "query": f"Problem text {i}", "relevant_doc_id": f"nlp4lp_test_{i}"}) for i in range(0, 270)]
    nlp4lp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pilot_ids": [14, 23, 34, 59, 69, 72], "future_evaluation_ids": [14, 23, 34, 59, 69, 72]}), encoding="utf-8")
    out = tmp_path / "out" / "results.jsonl"
    cmd = [sys.executable, "-m", "scripts.run_generic_llm_baseline",
           "--input", str(nlp4lp), "--manifest", str(manifest),
           "--subset", "pilot", "--output", str(out), "--mock"]
    first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    assert '"attempted": 6' in first.stdout
    second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stderr
    assert '"attempted": 0' in second.stdout