"""Tests for G_mod leaf modeling (eq. 3) and bottom-up merge (eq. 4).

Fake in-process providers and hand-built partition trees only -- never a
real network/API call, and never gated NLP4LP text.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from baselines.pamop.config import load_config, reconstructed_default_path
from baselines.pamop.llm.base import LLMProvider
from baselines.pamop.modeling import (
    ModelingValidationError,
    build_merged_model,
    merge_bottom_up,
    model_all_leaves,
    model_leaf,
    model_root_objective,
    validate_leaf_output,
)
from baselines.pamop.partition import PartitionNode, PartitionTree
from baselines.pamop.representations import synthetic_structured_problem

VALID_ROOT_TEXT = """### PARAMETERS
param cap;

### VARIABLES
var x >= 0;

### OBJECTIVE
minimize cost: x;

### CONSTRAINTS
{constraints}
"""


class _ScriptedProvider(LLMProvider):
    """Returns each entry of ``responses`` in order, one per call (cyclic
    fake keyed by call index -- mirrors tests/test_extraction.py's helper)."""

    name = "scripted_modeling"

    def __init__(self, responses: list[str], **kwargs):
        super().__init__(**kwargs)
        self._responses = list(responses)
        self.prompts_seen: list[str] = []
        self.calls = 0

    def _call(self, prompt, config):
        self.prompts_seen.append(prompt)
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return {"text": text, "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 10}


class _RoutingProvider(LLMProvider):
    """Routes to a leaf-fragment response for leaf prompts and a full
    root-section response for the root prompt, based on prompt content --
    lets a single provider serve a whole build_merged_model() call."""

    name = "routing"

    def __init__(self, leaf_fragment_by_hint: dict[str, str], root_text: str, **kwargs):
        super().__init__(**kwargs)
        self._leaf_by_hint = leaf_fragment_by_hint
        self._root_text = root_text
        self.calls = 0

    def _call(self, prompt, config):
        self.calls += 1
        if "Already-modeled constraints" in prompt:
            return {"text": self._root_text, "finish_reason": "stop"}
        for hint, fragment in self._leaf_by_hint.items():
            if hint in prompt:
                return {"text": fragment, "finish_reason": "stop"}
        raise AssertionError(f"no matching leaf fragment for prompt: {prompt[:200]}")


@pytest.fixture(scope="module")
def config():
    return load_config(reconstructed_default_path())


def _problem(vague_first: bool = False):
    p = synthetic_structured_problem(
        "synth_merge",
        global_summary="A toy two-part logistics problem.",
        objective_text="Minimize total cost.",
        constraint_texts=[
            "Train unloading volume must not exceed dock capacity.",
            "Crane loading volume must not exceed ship capacity.",
        ],
        variables=[
            ("dock_capacity", "Maximum containers the dock can receive"),
            ("train_unload_volume", "Containers unloaded from trains"),
            ("ship_capacity", "Maximum containers a ship can receive"),
            ("crane_load_volume", "Containers loaded onto ships"),
        ],
    )
    if vague_first:
        from dataclasses import replace

        constraints = (replace(p.constraints[0], vagueness_score=0.9),) + p.constraints[1:]
        from baselines.pamop.representations import StructuredProblem

        p = StructuredProblem(
            problem_id=p.problem_id, global_summary=p.global_summary, objective_text=p.objective_text,
            constraints=constraints, variables=p.variables, source=p.source,
        )
    return p


def _one_leaf_tree(problem) -> PartitionTree:
    root = PartitionNode(node_id="node_0", parent_id=None, depth=0, node_type="leaf",
                          constraint_group=tuple(range(len(problem.constraints))))
    return PartitionTree(problem_id=problem.problem_id, root_id="node_0",
                          nodes={"node_0": root}, config_hash="testhash", provenance={})


def _three_leaf_tree(problem) -> PartitionTree:
    """root -> [A, B]; A -> [leaf1, leaf2]; B -> [leaf3]. Deliberately
    hand-built (not produced by build_partition_tree) so the merge order is
    exactly known and independent of clustering behavior."""
    nodes = {
        "root": PartitionNode("root", None, 0, "root", (0, 1, 2), children=("A", "B")),
        "A": PartitionNode("A", "root", 1, "internal", (0, 1), children=("leaf1", "leaf2")),
        "B": PartitionNode("B", "root", 1, "leaf", (2,)),
        "leaf1": PartitionNode("leaf1", "A", 2, "leaf", (0,)),
        "leaf2": PartitionNode("leaf2", "A", 2, "leaf", (1,)),
    }
    return PartitionTree(problem_id=problem.problem_id, root_id="root", nodes=nodes,
                          config_hash="testhash", provenance={})


# ---------------------------------------------------------------------
# validate_leaf_output
# ---------------------------------------------------------------------


def test_validate_leaf_output_accepts_ampl_like_text():
    assert validate_leaf_output("subject to c1: x <= 5;") == "subject to c1: x <= 5;"


def test_validate_leaf_output_strips_code_fence():
    assert validate_leaf_output("```ampl\nsubject to c1: x <= 5;\n```") == "subject to c1: x <= 5;"


def test_validate_leaf_output_rejects_empty():
    with pytest.raises(ModelingValidationError):
        validate_leaf_output("   ")


def test_validate_leaf_output_rejects_text_without_semicolon():
    with pytest.raises(ModelingValidationError):
        validate_leaf_output("this is just prose with no AMPL statement at all")


# ---------------------------------------------------------------------
# model_leaf -- prompt construction, augmentation, retries
# ---------------------------------------------------------------------


def test_model_leaf_prompt_includes_global_summary_and_variables(config):
    problem = _problem()
    tree = _one_leaf_tree(problem)
    provider = _ScriptedProvider(["subject to c1: x <= 5;"])
    model_leaf(tree.nodes[tree.root_id], tree, problem, provider, config)
    prompt = provider.prompts_seen[0]
    assert problem.global_summary in prompt
    assert "dock_capacity" in prompt
    assert "train_unload_volume" in prompt


def test_model_leaf_does_not_augment_when_not_vague(config):
    problem = _problem(vague_first=False)
    tree = _one_leaf_tree(problem)
    provider = _ScriptedProvider(["subject to c1: x <= 5;"])
    result = model_leaf(tree.nodes[tree.root_id], tree, problem, provider, config)
    assert result.used_augmentation is False
    assert "Additional context" not in provider.prompts_seen[0]


def test_model_leaf_augments_when_vague(config):
    problem = _problem(vague_first=True)
    tree = _three_leaf_tree(problem)
    # leaf1 covers constraint index 0 (the vague one); its sibling under A is leaf2 (index 1)
    provider = _ScriptedProvider(["subject to c1: x <= 5;"])
    result = model_leaf(tree.nodes["leaf1"], tree, problem, provider, config)
    assert result.used_augmentation is True
    assert "Additional context" in provider.prompts_seen[0]
    # the sibling's constraint description should appear as context
    assert problem.constraints[1].description in provider.prompts_seen[0]


def test_model_leaf_retries_on_malformed_output_then_succeeds(config):
    problem = _problem()
    tree = _one_leaf_tree(problem)
    provider = _ScriptedProvider(["no semicolon here", "subject to c1: x <= 5;"])
    result = model_leaf(tree.nodes[tree.root_id], tree, problem, provider, config)
    assert result.validation_attempts == 2


def test_model_leaf_gives_up_after_modeling_max_retries(config):
    problem = _problem()
    tree = _one_leaf_tree(problem)
    provider = _ScriptedProvider(["still no semicolon"] * 10)
    with pytest.raises(ModelingValidationError):
        model_leaf(tree.nodes[tree.root_id], tree, problem, provider, config)
    assert provider.calls == config.llm.modeling_max_retries + 1


def test_model_leaf_flags_unresolved_reference():
    from baselines.pamop.config import load_config, reconstructed_default_path

    config = load_config(reconstructed_default_path())
    problem = _problem()
    tree = _one_leaf_tree(problem)
    provider = _ScriptedProvider(["subject to c1: totally_undeclared_var <= 5;"])
    result = model_leaf(tree.nodes[tree.root_id], tree, problem, provider, config)
    assert "totally_undeclared_var" in result.unresolved_references


def test_model_leaf_does_not_flag_constraint_labels_as_unresolved(config):
    problem = _problem()
    tree = _one_leaf_tree(problem)
    provider = _ScriptedProvider(["subject to my_label: dock_capacity >= train_unload_volume;"])
    result = model_leaf(tree.nodes[tree.root_id], tree, problem, provider, config)
    assert "my_label" not in result.unresolved_references


def test_leaf_serialization_records_referenced_global_symbols(config):
    problem = _problem()
    tree = _one_leaf_tree(problem)
    provider = _ScriptedProvider(["subject to c1: dock_capacity >= train_unload_volume;"])
    result = model_leaf(tree.nodes[tree.root_id], tree, problem, provider, config)
    assert result.referenced_global_symbols == ("dock_capacity", "train_unload_volume")
    serialized = result.to_dict()
    assert serialized["referenced_global_symbols"] == ["dock_capacity", "train_unload_volume"]


# ---------------------------------------------------------------------
# merge_bottom_up
# ---------------------------------------------------------------------


def test_merge_one_leaf_returns_its_own_fragment(config):
    problem = _problem()
    tree = _one_leaf_tree(problem)
    provider = _ScriptedProvider(["subject to c1: x <= 5;"])
    leaf_results = model_all_leaves(tree, problem, provider, config)
    assert merge_bottom_up(tree, leaf_results) == "subject to c1: x <= 5;"


def test_merge_multi_level_tree_concatenates_in_tree_order(config):
    problem = synthetic_structured_problem(
        "synth_3leaf", global_summary="s", objective_text="o",
        constraint_texts=["c0 text", "c1 text", "c2 text"],
        variables=[("x", "var x")],
    )
    tree = _three_leaf_tree(problem)
    fragments = {
        "leaf1": "subject to c0: x <= 1;",
        "leaf2": "subject to c1: x <= 2;",
        "B": "subject to c2: x <= 3;",
    }
    provider = _ScriptedProvider(list(fragments.values()))
    # deterministic call order is sorted(tree.leaves(), key=node_id): B, leaf1, leaf2
    leaf_results = model_all_leaves(tree, problem, provider, config)
    merged = merge_bottom_up(tree, leaf_results)
    # A = leaf1 then leaf2 (children order); root = A then B (children order)
    expected = "\n".join([
        leaf_results["leaf1"].ampl_fragment,
        leaf_results["leaf2"].ampl_fragment,
        leaf_results["B"].ampl_fragment,
    ])
    assert merged == expected


def test_model_all_leaves_visits_in_deterministic_node_id_order(config):
    problem = synthetic_structured_problem(
        "synth_order", global_summary="s", objective_text="o",
        constraint_texts=["c0", "c1", "c2"], variables=[("x", "v")],
    )
    tree = _three_leaf_tree(problem)
    provider = _ScriptedProvider(["subject to c: x <= 1;"] * 3)
    model_all_leaves(tree, problem, provider, config)
    # sorted node ids among {"B", "leaf1", "leaf2"} -> B, leaf1, leaf2
    assert provider.calls == 3


# ---------------------------------------------------------------------
# model_root_objective (eq. 4) -- section parsing / validation / retries
# ---------------------------------------------------------------------


def test_model_root_objective_parses_all_four_sections(config):
    problem = _problem()
    text = VALID_ROOT_TEXT.format(constraints="subject to c1: x <= 5;")
    provider = _ScriptedProvider([text])
    sections, response, template, attempts = model_root_objective(problem, "subject to c1: x <= 5;", provider, config)
    assert sections["OBJECTIVE"] == "minimize cost: x;"
    assert "subject to c1" in sections["CONSTRAINTS"]
    assert attempts == 1


def test_model_root_objective_rejects_missing_section():
    from baselines.pamop.config import load_config, reconstructed_default_path

    config = load_config(reconstructed_default_path())
    problem = _problem()
    incomplete = "### PARAMETERS\nparam cap;\n### OBJECTIVE\nminimize x;\n### CONSTRAINTS\nsubject to c1: x<=1;\n"
    provider = _ScriptedProvider([incomplete])
    with pytest.raises(ModelingValidationError):
        model_root_objective(problem, "subject to c1: x<=1;", provider, config)


def test_model_root_objective_rejects_out_of_order_sections():
    from baselines.pamop.config import load_config, reconstructed_default_path

    config = load_config(reconstructed_default_path())
    problem = _problem()
    swapped = (
        "### CONSTRAINTS\nsubject to c1: x<=1;\n"
        "### PARAMETERS\nparam cap;\n"
        "### VARIABLES\nvar x;\n"
        "### OBJECTIVE\nminimize x;\n"
    )
    provider = _ScriptedProvider([swapped])
    with pytest.raises(ModelingValidationError):
        model_root_objective(problem, "subject to c1: x<=1;", provider, config)


def test_model_root_objective_rejects_duplicate_declarations(config):
    problem = _problem()
    duplicate = (
        "### PARAMETERS\nparam cap;\n"
        "### VARIABLES\nvar cap >= 0;\n"
        "### OBJECTIVE\nminimize cost: cap;\n"
        "### CONSTRAINTS\nsubject to c1: cap <= 1;\n"
    )
    provider = _ScriptedProvider([duplicate])
    with pytest.raises(ModelingValidationError, match="duplicate"):
        model_root_objective(problem, "subject to c1: cap<=1;", provider, config)


def test_model_root_objective_retries_then_succeeds(config):
    problem = _problem()
    bad = "not the right format at all"
    good = VALID_ROOT_TEXT.format(constraints="subject to c1: x <= 5;")
    provider = _ScriptedProvider([bad, good])
    sections, response, template, attempts = model_root_objective(problem, "subject to c1: x <= 5;", provider, config)
    assert attempts == 2


# ---------------------------------------------------------------------
# build_merged_model -- full eq. 3 + merge + eq. 4 pipeline
# ---------------------------------------------------------------------


def test_build_merged_model_end_to_end_two_leaf(config):
    problem = _problem()
    tree = _three_leaf_tree_two_constraints(problem)

    # Use a routing provider: leaf calls return a fragment keyed by which
    # constraint text is in the prompt; the root call is detected by its
    # own distinctive prompt text ("Already-modeled constraints"). The
    # root's expected text is precomputed since it depends on the merged
    # leaf output, which is itself deterministic given fixed leaf fragments.
    merged_constraints_expected = (
        "subject to c0: train_unload_volume <= dock_capacity;\n"
        "subject to c1: crane_load_volume <= ship_capacity;"
    )
    provider = _RoutingProvider(
        leaf_fragment_by_hint={
            "Train unloading": "subject to c0: train_unload_volume <= dock_capacity;",
            "Crane loading": "subject to c1: crane_load_volume <= ship_capacity;",
        },
        root_text=VALID_ROOT_TEXT.format(constraints=merged_constraints_expected),
    )

    merged = build_merged_model(tree, problem, provider, config)
    assert merged.constraints_text == merged_constraints_expected
    assert merged.objective_text == "minimize cost: x;"
    assert len(merged.leaf_results) == 2
    assert merged.symbol_conflicts == ()


def test_build_merged_model_reports_duplicate_leaf_constraint_labels(config):
    problem = _problem()
    tree = _three_leaf_tree_two_constraints(problem)
    provider = _RoutingProvider(
        leaf_fragment_by_hint={
            "Train unloading": "subject to repeated_label: train_unload_volume <= dock_capacity;",
            "Crane loading": "subject to repeated_label: crane_load_volume <= ship_capacity;",
        },
        root_text=VALID_ROOT_TEXT.format(
            constraints=(
                "subject to repeated_label: train_unload_volume <= dock_capacity;\n"
                "subject to repeated_label: crane_load_volume <= ship_capacity;"
            )
        ),
    )
    merged = build_merged_model(tree, problem, provider, config)
    assert any("repeated_label" in conflict for conflict in merged.symbol_conflicts)


def test_build_merged_model_reports_leaf_declarations(config):
    problem = _problem()
    tree = _three_leaf_tree_two_constraints(problem)
    provider = _RoutingProvider(
        leaf_fragment_by_hint={
            "Train unloading": "param stray;\nsubject to c0: train_unload_volume <= dock_capacity;",
            "Crane loading": "subject to c1: crane_load_volume <= ship_capacity;",
        },
        root_text=VALID_ROOT_TEXT.format(
            constraints=(
                "param stray;\nsubject to c0: train_unload_volume <= dock_capacity;\n"
                "subject to c1: crane_load_volume <= ship_capacity;"
            )
        ),
    )
    merged = build_merged_model(tree, problem, provider, config)
    assert any("stray" in conflict for conflict in merged.symbol_conflicts)


def _three_leaf_tree_two_constraints(problem) -> PartitionTree:
    """root -> [leafA, leafB], two leaves, one constraint each."""
    nodes = {
        "root": PartitionNode("root", None, 0, "root", (0, 1), children=("leafA", "leafB")),
        "leafA": PartitionNode("leafA", "root", 1, "leaf", (0,)),
        "leafB": PartitionNode("leafB", "root", 1, "leaf", (1,)),
    }
    return PartitionTree(problem_id=problem.problem_id, root_id="root", nodes=nodes,
                          config_hash="testhash", provenance={})


def test_build_merged_model_is_deterministic(config):
    problem = _problem()
    tree = _three_leaf_tree_two_constraints(problem)
    merged_constraints_expected = (
        "subject to c0: train_unload_volume <= dock_capacity;\n"
        "subject to c1: crane_load_volume <= ship_capacity;"
    )

    def make_merged():
        provider = _RoutingProvider(
            leaf_fragment_by_hint={
                "Train unloading": "subject to c0: train_unload_volume <= dock_capacity;",
                "Crane loading": "subject to c1: crane_load_volume <= ship_capacity;",
            },
            root_text=VALID_ROOT_TEXT.format(constraints=merged_constraints_expected),
        )
        return build_merged_model(tree, problem, provider, config)

    m1, m2 = make_merged(), make_merged()
    assert m1.to_dict() == m2.to_dict()


def test_merged_model_to_dict_never_contains_a_secret_looking_field(config):
    problem = _problem()
    tree = _three_leaf_tree_two_constraints(problem)
    merged_constraints_expected = (
        "subject to c0: train_unload_volume <= dock_capacity;\n"
        "subject to c1: crane_load_volume <= ship_capacity;"
    )
    provider = _RoutingProvider(
        leaf_fragment_by_hint={
            "Train unloading": "subject to c0: train_unload_volume <= dock_capacity;",
            "Crane loading": "subject to c1: crane_load_volume <= ship_capacity;",
        },
        root_text=VALID_ROOT_TEXT.format(constraints=merged_constraints_expected),
    )
    merged = build_merged_model(tree, problem, provider, config)
    serialized = str(merged.to_dict())
    assert "api_key" not in serialized.lower()
    assert "secret" not in serialized.lower()
