"""Smoke tests for the parts of the ORLM scaffold that don't need a GPU/model."""
from __future__ import annotations

import pytest

from baselines.orlm.config import OrlmConfig
from baselines.orlm.data_adapter import build_orlm_prompt
from baselines.orlm.output_normalizer import parse_orlm_output
from baselines.orlm.runner import OrlmRunner


def test_build_orlm_prompt_wraps_question():
    prompt = build_orlm_prompt("Maximize profit given two resources.")
    assert "Maximize profit given two resources." in prompt
    assert "# Question:" in prompt
    assert "# Response:" in prompt


def test_build_orlm_prompt_strips_whitespace():
    prompt = build_orlm_prompt("  padded query  \n")
    assert "padded query" in prompt
    assert "  padded query  " not in prompt


def test_parse_orlm_output_extracts_fenced_code_block():
    raw = "Some model description.\n```python\nimport coptpy\nmodel = coptpy.Envr()\n```\n"
    parsed = parse_orlm_output(raw)
    assert parsed.code_block_found is True
    assert "import coptpy" in parsed.coptpy_code
    assert "Some model description." in parsed.model_description


def test_parse_orlm_output_no_code_block():
    parsed = parse_orlm_output("Just a description, no code.")
    assert parsed.code_block_found is False
    assert parsed.coptpy_code is None


def test_runner_generate_not_implemented():
    runner = OrlmRunner(config=OrlmConfig())
    with pytest.raises(NotImplementedError):
        runner.generate("some prompt")
