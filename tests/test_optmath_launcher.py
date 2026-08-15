"""Focused tests for the OptMATH resumable CLI launcher and its resume store."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.optmath.config import OPTMATH_PRIMARY_MODEL, OPTMATH_PROMPT_VERSION
from baselines.optmath.pipeline import JsonlResultStore
from baselines.optmath.runner import OptmathRunner

from scripts.run_optmath_inference import _MockBackend, _load_gold_objectives, _load_records


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def nlp4lp_input(tmp_path: Path) -> Path:
    path = tmp_path / "nlp4lp_eval_orig.jsonl"
    lines = []
    for i in range(0, 270):
        lines.append(json.dumps({"query_id": f"nlp4lp_test_{i}", "query": f"Problem text {i}", "relevant_doc_id": f"nlp4lp_test_{i}"}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "pilot_ids": [14, 23, 34, 59, 69, 72],
        "future_evaluation_ids": [14, 23, 34, 59, 69, 72, 84, 88, 96, 117, 190, 202, 208, 219, 232, 237, 254, 262],
    }), encoding="utf-8")
    return path


@pytest.fixture
def gold_cache(tmp_path: Path) -> Path:
    """Matches the real gold-cache layout: `gold_by_id` keyed directly by query_id."""
    path = tmp_path / "gold_cache.json"
    path.write_text(json.dumps({
        "split": "test",
        "gold_by_id": {"nlp4lp_test_13": {"solution": {"objective": 12.0}}},
    }), encoding="utf-8")
    return path


def test_load_records_pilot_and_common18(nlp4lp_input, manifest, gold_cache):
    gold = _load_gold_objectives(gold_cache)
    pilot = _load_records(nlp4lp_input, manifest, "pilot", gold)
    assert [r.problem_id for r in pilot] == ["14", "23", "34", "59", "69", "72"]
    assert all(r.raw_text.startswith("Problem text ") for r in pilot)
    common18 = _load_records(nlp4lp_input, manifest, "common18", gold)
    assert len(common18) == 18
    assert [r.problem_id for r in common18] == ["14", "23", "34", "59", "69", "72", "84", "88", "96", "117", "190", "202", "208", "219", "232", "237", "254", "262"]
    gold_by_id = {r.problem_id: r.gold_metadata.get("gold_objective") for r in common18}
    assert gold_by_id["14"] == 12.0
    assert gold_by_id["23"] is None


def test_missing_manifest_query_raises(nlp4lp_input, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pilot_ids": [999], "future_evaluation_ids": [999]}), encoding="utf-8")
    with pytest.raises(KeyError):
        _load_records(nlp4lp_input, manifest, "pilot", {})


def test_append_unfinished_skips_completed(tmp_path):
    store = JsonlResultStore(tmp_path / "results.jsonl")
    record = _load_records(
        ROOT / "data/processed/nlp4lp_eval_orig.jsonl",
        ROOT / "baselines/optmath/manifests/nlp4lp_common_manifest.json",
        "pilot", {},
    )[0]
    runner = OptmathRunner(backend=_MockBackend())
    first = store.append_unfinished([record], runner, git_sha="test")
    assert len(first) == 1
    second = store.append_unfinished([record], runner, git_sha="test")
    assert second == []
    assert store.completed_ids() == {"14"}


def test_mock_backend_is_labeled():
    from baselines.optmath.config import OPTMATH_SYSTEM_PROMPT
    from baselines.optmath.prompt import PromptBundle
    backend = _MockBackend()
    bundle = PromptBundle(OPTMATH_SYSTEM_PROMPT, "Question?", "v1", "hash")
    raw, tokens = backend.generate(bundle, None)
    assert raw.startswith("MOCK OUTPUT")
    assert tokens["total_tokens"] == 17


def test_cli_mock_mode_routes_to_separate_output(tmp_path, nlp4lp_input, manifest):
    out = tmp_path / "out" / "results.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_optmath_inference",
         "--input", str(nlp4lp_input), "--manifest", str(manifest),
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
    assert meta["model_id"] == OPTMATH_PRIMARY_MODEL
    assert meta["model_revision"] == "617fe77"
    assert meta["prompt_version"] == OPTMATH_PROMPT_VERSION
    assert meta["generation"]["temperature"] == 0.8
    assert meta["generation"]["max_new_tokens"] == 8192


def test_cli_resume_skips_completed(tmp_path, nlp4lp_input, manifest):
    out = tmp_path / "out" / "results.jsonl"
    cmd = [sys.executable, "-m", "scripts.run_optmath_inference",
           "--input", str(nlp4lp_input), "--manifest", str(manifest),
           "--subset", "pilot", "--output", str(out), "--mock"]
    first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    assert '"attempted": 6' in first.stdout
    second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stderr
    assert '"attempted": 0' in second.stdout


def test_cli_backfill_gold_attaches_objective_only(tmp_path, nlp4lp_input, manifest, gold_cache):
    # Run with an unavailable gold cache first, so rows are generated without gold.
    out = tmp_path / "out" / "results.jsonl"
    cmd = [sys.executable, "-m", "scripts.run_optmath_inference",
           "--input", str(nlp4lp_input), "--manifest", str(manifest),
           "--subset", "pilot", "--output", str(out), "--mock",
           "--gold-cache", str(tmp_path / "missing_gold.json")]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    mock_out = out.with_suffix(".mock.jsonl")
    rows = [json.loads(l) for l in mock_out.read_text().splitlines() if l.strip()]
    assert all(r["gold_objective"] is None for r in rows)
    # Backfill attaches gold from the cache without touching generation evidence.
    backfill = subprocess.run(
        [sys.executable, "-m", "scripts.run_optmath_inference", "--backfill-gold",
         "--output", str(mock_out), "--subset", "pilot", "--gold-cache", str(gold_cache)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert backfill.returncode == 0, backfill.stderr
    assert '"rows_updated": 1' in backfill.stdout  # only nlp4lp_test_13 has gold in the fixture
    rows = [json.loads(l) for l in mock_out.read_text().splitlines() if l.strip()]
    by_id = {r["problem_id"]: r["gold_objective"] for r in rows}
    assert by_id["14"] == 12.0
    assert by_id["23"] is None
    # generation evidence untouched by backfill
    assert all(r["generation"]["raw_output"].startswith("MOCK OUTPUT") for r in rows)
    # Idempotent: a second backfill updates nothing.
    again = subprocess.run(
        [sys.executable, "-m", "scripts.run_optmath_inference", "--backfill-gold",
         "--output", str(mock_out), "--subset", "pilot", "--gold-cache", str(gold_cache)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert '"rows_updated": 0' in again.stdout