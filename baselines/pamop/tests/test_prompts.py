"""Tests for versioned prompt-template loading (baselines/pamop/prompts/)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from baselines.pamop.prompts import load_prompt
from baselines.pamop.llm.base import prompt_hash


def test_load_prompt_parses_name_and_version():
    template = load_prompt("extraction_v1.txt")
    assert template.name == "extraction"
    assert template.version == "v1"


def test_load_prompt_hash_matches_content_hash():
    template = load_prompt("extraction_v1.txt")
    assert template.content_hash == prompt_hash(template.text)


def test_load_prompt_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist_v1.txt")


def test_extraction_prompt_renders_with_problem_text():
    template = load_prompt("extraction_v1.txt")
    rendered = template.render(problem_text="A factory makes widgets.")
    assert "A factory makes widgets." in rendered
    assert "{problem_text}" not in rendered


def test_extraction_prompt_is_explicitly_marked_as_reconstruction():
    template = load_prompt("extraction_v1.txt")
    assert "REPRODUCTION CHOICE" in template.text


def test_extraction_prompt_requests_the_four_paper_specified_fields():
    """The four fields (t_o, t_c, t_v, g) and the vagueness score are
    PAPER-SPECIFIED requirements (section 3.2); the wording asking for them
    is a reconstruction, but the fields themselves must be present."""
    template = load_prompt("extraction_v1.txt")
    for required in ("global_summary", "objective_text", "constraints", "variables", "vagueness_score"):
        assert required in template.text
