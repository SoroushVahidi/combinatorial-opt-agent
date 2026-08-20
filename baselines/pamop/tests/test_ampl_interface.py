"""Tests for the (deliberately unimplemented) AMPL interface boundary.

No AMPL/Gurobi call is possible here by design -- this module defines a
consumption contract only, see baselines/pamop/ampl_interface.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.pamop.ampl_interface import AmplRenderer, naive_concatenation_preview
from baselines.pamop.llm.types import LLMResponse
from baselines.pamop.modeling import MergedModel
from baselines.pamop.prompts import load_prompt


def _fake_response() -> LLMResponse:
    return LLMResponse(
        text="", provider="fake", model="fake-model", timestamp="2026-01-01T00:00:00+00:00",
        temperature=0.2, top_p=None, max_tokens=None, prompt_tokens=1, completion_tokens=1,
        total_tokens=2, latency_seconds=0.0, retry_count=0, prompt_hash="deadbeef",
    )


def _merged_model() -> MergedModel:
    template = load_prompt("modeling_root_v1.txt")
    return MergedModel(
        problem_id="p1",
        parameters_text="param cap;",
        variables_text="var x >= 0;",
        objective_text="minimize cost: x;",
        constraints_text="subject to c1: x <= cap;",
        leaf_results=(),
        root_llm_response=_fake_response(),
        root_prompt_template=template,
        symbol_conflicts=(),
        config_hash="testhash",
        provenance={},
    )


def test_ampl_renderer_is_a_protocol_not_an_implementation():
    """No concrete AmplRenderer exists yet -- confirming the Protocol has
    no instantiable default implementation."""
    import inspect

    assert inspect.isclass(AmplRenderer)
    # Protocols are not meant to be instantiated directly.


def test_naive_concatenation_preview_includes_all_four_sections():
    preview = naive_concatenation_preview(_merged_model())
    assert "param cap;" in preview
    assert "var x >= 0;" in preview
    assert "minimize cost: x;" in preview
    assert "subject to c1: x <= cap;" in preview


def test_naive_concatenation_preview_skips_empty_sections():
    model = _merged_model()
    import dataclasses

    model = dataclasses.replace(model, parameters_text="")
    preview = naive_concatenation_preview(model)
    assert "param cap;" not in preview
    assert "var x >= 0;" in preview
