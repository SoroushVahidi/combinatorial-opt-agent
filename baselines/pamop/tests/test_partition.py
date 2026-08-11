"""Tests for the non-LLM partitioning stage (baselines/pamop/partition.py).

Uses only hand-written synthetic problems -- never gated NLP4LP text -- per
the task's instruction to avoid committing gated content in test fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from baselines.pamop.config import load_config, reconstructed_default_path
from baselines.pamop.partition import PartitionTree, build_partition_tree
from baselines.pamop.representations import synthetic_structured_problem


@pytest.fixture(scope="module")
def config():
    return load_config(reconstructed_default_path())


def _two_block_problem():
    """A synthetic problem with two clearly disjoint constraint/variable
    groups, loosely inspired (in *shape* only, not text) by the paper's own
    published Figure 1 seaport example: an "unloading" block and a
    "storage/crane" block that share no variables."""
    return synthetic_structured_problem(
        "synthetic_two_block",
        global_summary="A toy problem with two independent subsystems.",
        objective_text="Minimize total operating cost.",
        constraint_texts=[
            "Train unloading volume must not exceed dock capacity each period.",
            "Dock capacity limits how many containers trains can unload per period.",
            "Crane loading volume must not exceed ship capacity each period.",
            "Ship capacity limits how many containers cranes can load per period.",
            "Total crane rental spending must stay within the crane budget.",
        ],
        variables=[
            ("dock_capacity", "Maximum containers the dock can receive from trains"),
            ("train_unload_volume", "Containers unloaded from trains at the dock"),
            ("ship_capacity", "Maximum containers a ship can receive from cranes"),
            ("crane_load_volume", "Containers loaded onto ships by cranes"),
            ("crane_budget", "Total budget available for crane rentals"),
        ],
    )


def _single_tiny_problem():
    return synthetic_structured_problem(
        "synthetic_tiny",
        global_summary="A trivially small problem.",
        objective_text="Maximize profit.",
        constraint_texts=["Total production must not exceed factory capacity."],
        variables=[("capacity", "Factory production capacity")],
    )


def test_deterministic_partitioning(config):
    problem = _two_block_problem()
    tree_a = build_partition_tree(problem, config)
    tree_b = build_partition_tree(problem, config)
    assert tree_a.to_dict() == tree_b.to_dict()


def test_valid_parent_child_relationships(config):
    tree = build_partition_tree(_two_block_problem(), config)
    assert tree.nodes[tree.root_id].parent_id is None
    for node in tree.nodes.values():
        if node.node_id == tree.root_id:
            continue
        assert node.parent_id in tree.nodes, f"{node.node_id} has dangling parent {node.parent_id}"
        assert node.node_id in tree.nodes[node.parent_id].children

    for node in tree.nodes.values():
        for child_id in node.children:
            assert tree.nodes[child_id].parent_id == node.node_id


def test_no_cycles_and_full_reachability(config):
    tree = build_partition_tree(_two_block_problem(), config)
    visited: set[str] = set()
    stack = [tree.root_id]
    while stack:
        node_id = stack.pop()
        assert node_id not in visited, f"cycle detected at {node_id}"
        visited.add(node_id)
        stack.extend(tree.nodes[node_id].children)
    assert visited == set(tree.nodes), "not every node is reachable from the root exactly once"


def test_every_constraint_assigned_exactly_once_across_leaves(config):
    problem = _two_block_problem()
    tree = build_partition_tree(problem, config)
    seen: list[int] = []
    for leaf in tree.leaves():
        seen.extend(leaf.constraint_group)
    assert sorted(seen) == list(range(len(problem.constraints)))


def test_independent_set_separates_disjoint_blocks(config):
    """The two-block synthetic problem shares no variables between its two
    halves, so independent-set separation at the root should find >= 2
    components -- this is the mechanism the paper describes for the root
    node specifically, distinct from clustering at deeper layers."""
    problem = _two_block_problem()
    tree = build_partition_tree(problem, config)
    root = tree.nodes[tree.root_id]
    assert len(root.children) >= 2, (
        "expected independent-set separation to split the two disjoint "
        "constraint blocks into at least two root children"
    )


def test_tiny_problem_becomes_a_single_leaf(config):
    problem = _single_tiny_problem()
    tree = build_partition_tree(problem, config)
    assert len(tree.nodes) == 1
    only_node = tree.nodes[tree.root_id]
    assert only_node.node_type == "leaf"
    assert only_node.constraint_group == (0,)


def test_leaf_depths_are_nondecreasing_from_root(config):
    tree = build_partition_tree(_two_block_problem(), config)
    for node in tree.nodes.values():
        if node.parent_id is not None:
            assert node.depth == tree.nodes[node.parent_id].depth + 1


def test_serialization_round_trip(config):
    tree = build_partition_tree(_two_block_problem(), config)
    restored = PartitionTree.from_dict(tree.to_dict())
    assert restored.to_dict() == tree.to_dict()


def test_serialized_tree_never_contains_raw_constraint_text(config):
    """The serialized tree must reference constraints by index only -- no
    gated/raw problem text should ever end up in a committed artifact."""
    problem = _two_block_problem()
    tree = build_partition_tree(problem, config)
    serialized = tree.to_json()
    for constraint in problem.constraints:
        assert constraint.description not in serialized


def test_config_hash_changes_when_partitioning_config_changes(config):
    tree_a = build_partition_tree(_two_block_problem(), config)
    from dataclasses import replace

    other_partitioning = replace(config.partitioning, tfidf_top_k=3)
    other_config = replace(config, partitioning=other_partitioning)
    tree_b = build_partition_tree(_two_block_problem(), other_config)
    assert tree_a.config_hash != tree_b.config_hash
