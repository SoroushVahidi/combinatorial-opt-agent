"""Non-LLM question-partitioning stage (PaMOP section 3.2).

Implements, deterministically and without any LLM call:

  * "Separating independent sets" at the root node -- a bipartite graph
    between constraints and variables (edges = keyword-match confidence),
    partitioned into connected components via graph search (eq. unnamed,
    paper section 3.2 "Separating independent sets").
  * "Clustering constraint sets" at every deeper layer -- eq. (2)'s distance
    formula over a weighted combination of three similarity signals
    (adjacency/context, TF-IDF keyword overlap, embedding cosine), consumed
    by a clustering algorithm the paper does not name.

Every numeric constant the paper does not specify is read from
``PamopConfig.partitioning`` (see config.py) -- nothing here is hard-coded.
Where this module makes a specific algorithmic choice the paper leaves open
(e.g. *which* graph-search / clustering algorithm), the choice is marked
``# REPRODUCTION CHOICE`` in-line and explained in baselines/pamop/README.md.

This module does not call an LLM, does not generate AMPL, and does not touch
a solver -- see baselines/pamop/README.md "Not implemented yet".
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PamopConfig
from .representations import StructuredProblem

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")


# --------------------------------------------------------------------------
# Similarity signals (paper section 3.2, "Clustering constraint sets")
# --------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _fit_tfidf(texts: list[str]) -> TfidfVectorizer:
    # scikit-learn's default English stop-word-free tokenizer; deterministic
    # given fixed input, no randomness.
    vec = TfidfVectorizer()
    vec.fit(texts if any(t.strip() for t in texts) else ["placeholder"])
    return vec


def top_k_keywords(text: str, vectorizer: TfidfVectorizer, top_k: int) -> set[str]:
    """PAPER-SPECIFIED mechanism: "keywords ... extracted using TF-IDF", "top k"."""
    if not text.strip():
        return set()
    row = vectorizer.transform([text])
    if row.nnz == 0:
        return set()
    terms = np.array(vectorizer.get_feature_names_out())
    row_arr = row.toarray()[0]
    order = np.argsort(-row_arr)
    order = order[row_arr[order] > 0][:top_k]
    return set(terms[order])


def keyword_similarity(kw_i: set[str], kw_j: set[str], top_k: int) -> float:
    """PAPER-SPECIFIED signal: "number of common keywords in the top k",
    normalized to [0, 1] by ``top_k`` -- REPRODUCTION CHOICE for the exact
    normalization (paper gives a raw count, not a bounded similarity)."""
    if top_k <= 0:
        return 0.0
    shared = len(kw_i & kw_j)
    return min(1.0, shared / top_k)


def adjacency_similarity(i: int, j: int) -> float:
    """PAPER-SPECIFIED idea, REPRODUCTION CHOICE functional form: "constraints
    are typically extracted in order, so adjacent ones are likely to be
    similar ... we prioritize higher similarity scores for adjacent
    constraints." No equation is given for this signal in the paper; this
    implements a smooth decay with rank distance, bounded in (0, 1]."""
    return 1.0 / (1.0 + abs(i - j))


class VectorSimilarityProvider:
    """Cosine similarity between two texts under some embedding.

    PAPER-SPECIFIED signal source: GloVe (Wikipedia 2014) cosine similarity.
    No GloVe vectors are provisioned on this workstation (see
    docs/PAMOP_REPRODUCTION_PLAN.md section 9/13); ``embedding_source`` in
    the reconstructed config is "tfidf_fallback", implemented here, which
    substitutes TF-IDF cosine similarity for the GloVe signal specifically.
    A real GloVe-backed provider can be added later behind this same
    interface without changing ``partition.py``'s call sites.
    """

    def __init__(self, texts: list[str]):
        self._vectorizer = _fit_tfidf(texts)
        self._matrix = self._vectorizer.transform(texts)
        self._index = {t: i for i, t in enumerate(texts)}

    def similarity(self, text_i: str, text_j: str) -> float:
        vi = self._matrix[self._index[text_i]]
        vj = self._matrix[self._index[text_j]]
        num = float((vi.multiply(vj)).sum())
        denom = float(np.sqrt(vi.multiply(vi).sum()) * np.sqrt(vj.multiply(vj).sum()))
        if denom == 0.0:
            return 0.0
        return max(0.0, min(1.0, num / denom))


def combined_similarity(
    i: int,
    j: int,
    constraints: list[str],
    keyword_sets: list[set[str]],
    vector_provider: VectorSimilarityProvider,
    weights: dict[str, float],
    tfidf_top_k: int,
) -> float:
    """Weighted average of the three signals (paper: "weighted averages of
    these three similarity measures", exact weights REPRODUCTION CHOICE, see
    config ``similarity_weights_by_layer``)."""
    adj = adjacency_similarity(i, j)
    kw = keyword_similarity(keyword_sets[i], keyword_sets[j], tfidf_top_k)
    vec = vector_provider.similarity(constraints[i], constraints[j])
    total_weight = weights["adjacency"] + weights["keyword"] + weights["vector"]
    if total_weight <= 0:
        raise ValueError("similarity weights must sum to a positive number")
    return (
        weights["adjacency"] * adj + weights["keyword"] * kw + weights["vector"] * vec
    ) / total_weight


def distance(similarity: float, epsilon: float) -> float:
    """Eq. (2), PAPER-SPECIFIED exactly: d_ij = 1/(s_ij+eps) - 1/(1+eps)."""
    return 1.0 / (similarity + epsilon) - 1.0 / (1.0 + epsilon)


# --------------------------------------------------------------------------
# Union-find (no new dependency for graph connectivity)
# --------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, items: list[Any]):
        self._parent = {x: x for x in items}

    def find(self, x: Any) -> Any:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


# --------------------------------------------------------------------------
# Independent-set separation (paper section 3.2, root node only)
# --------------------------------------------------------------------------


def independent_set_separation(
    problem: StructuredProblem,
    constraint_indices: list[int],
    config: PamopConfig,
) -> list[list[int]]:
    """PAPER-SPECIFIED mechanism: bipartite graph over {constraints} u
    {variables}, edges = keyword-match confidence >= threshold, split into
    connected components ("apply graph search algorithms to separate
    independent subgraphs"). REPRODUCTION CHOICE: connected components is the
    specific graph-search algorithm used (paper does not name one).
    """
    top_k = config.require("partitioning", "tfidf_top_k")
    threshold = config.require("partitioning", "bipartite_edge_confidence_threshold")

    constraint_texts = [problem.constraints[i].description for i in constraint_indices]
    variable_texts = [v.description for v in problem.variables]
    corpus = constraint_texts + variable_texts
    vectorizer = _fit_tfidf(corpus)

    c_keywords = [top_k_keywords(t, vectorizer, top_k) for t in constraint_texts]
    v_keywords = [top_k_keywords(t, vectorizer, top_k) for t in variable_texts]

    c_nodes = [("c", idx) for idx in constraint_indices]
    v_nodes = [("v", vi) for vi in range(len(problem.variables))]
    uf = _UnionFind(c_nodes + v_nodes)

    for ci, c_idx in enumerate(constraint_indices):
        for vi in range(len(problem.variables)):
            shared = len(c_keywords[ci] & v_keywords[vi])
            if shared >= threshold:
                uf.union(("c", c_idx), ("v", vi))

    groups: dict[Any, list[int]] = {}
    for c_idx in constraint_indices:
        root = uf.find(("c", c_idx))
        groups.setdefault(root, []).append(c_idx)

    return [sorted(g) for g in groups.values()]


# --------------------------------------------------------------------------
# Constraint clustering (paper section 3.2, deeper layers)
# --------------------------------------------------------------------------


def cluster_constraints(
    problem: StructuredProblem,
    constraint_indices: list[int],
    config: PamopConfig,
    layer: str,
) -> list[list[int]]:
    """PAPER-SPECIFIED distance metric (eq. 2) over a combined similarity;
    REPRODUCTION CHOICE clustering algorithm (agglomerative, average linkage
    by default) since the paper specifies the metric but not the clustering
    rule that consumes it. "Noise points ... treated as potentially relevant
    ... rather than removed" is honored by construction: agglomerative
    clustering with a distance threshold assigns every point to some
    cluster, never discards one."""
    if len(constraint_indices) <= 1:
        return [list(constraint_indices)]

    epsilon = config.require("partitioning", "epsilon")
    tfidf_top_k = config.require("partitioning", "tfidf_top_k")
    weights_by_layer = config.require("partitioning", "similarity_weights_by_layer")
    weights = weights_by_layer.get(layer, weights_by_layer.get("default"))
    if weights is None:
        raise ValueError(f"no similarity weights configured for layer {layer!r} or 'default'")
    algorithm = config.require("partitioning", "clustering_algorithm")
    stop_similarity = config.require("partitioning", "leaf_stop_similarity_threshold")

    texts = [problem.constraints[i].description for i in constraint_indices]
    vectorizer = _fit_tfidf(texts)
    keyword_sets = [top_k_keywords(t, vectorizer, tfidf_top_k) for t in texts]
    vector_provider = VectorSimilarityProvider(texts)

    n = len(constraint_indices)
    dist = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            sim = combined_similarity(
                a, b, texts, keyword_sets, vector_provider, weights, tfidf_top_k
            )
            d = distance(sim, epsilon)
            dist[a, b] = dist[b, a] = d

    if algorithm != "agglomerative_average_linkage":
        raise ValueError(f"unsupported clustering_algorithm: {algorithm!r}")

    labels = _agglomerative_cluster(dist, distance(stop_similarity, epsilon))

    groups: dict[int, list[int]] = {}
    for local_i, label in enumerate(labels):
        groups.setdefault(label, []).append(constraint_indices[local_i])
    return [sorted(g) for g in groups.values()]


def _agglomerative_cluster(dist: np.ndarray, distance_threshold: float) -> list[int]:
    """Average-linkage agglomerative clustering, cut at ``distance_threshold``.

    Uses scipy if available (already a repo dependency); falls back to a
    tiny pure-Python implementation otherwise so this module has no hard new
    dependency.
    """
    n = dist.shape[0]
    if n == 1:
        return [0]
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(dist, checks=False)
        z = linkage(condensed, method="average")
        labels = fcluster(z, t=distance_threshold, criterion="distance")
        return [int(x) for x in labels]
    except ImportError:
        return _agglomerative_cluster_pure_python(dist, distance_threshold)


def _agglomerative_cluster_pure_python(dist: np.ndarray, distance_threshold: float) -> list[int]:
    n = dist.shape[0]
    clusters = {i: {i} for i in range(n)}

    def avg_linkage(a: set[int], b: set[int]) -> float:
        return float(np.mean([dist[i, j] for i in a for j in b]))

    while len(clusters) > 1:
        ids = list(clusters.keys())
        best = None
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                d = avg_linkage(clusters[ids[x]], clusters[ids[y]])
                if best is None or d < best[0]:
                    best = (d, ids[x], ids[y])
        if best is None or best[0] > distance_threshold:
            break
        _, a, b = best
        clusters[a] |= clusters[b]
        del clusters[b]

    labels = [0] * n
    for label, members in enumerate(clusters.values()):
        for i in members:
            labels[i] = label
    return labels


# --------------------------------------------------------------------------
# Partition tree
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionNode:
    node_id: str
    parent_id: str | None
    depth: int
    node_type: str  # "root" | "internal" | "leaf"
    constraint_group: tuple[int, ...]
    children: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "node_type": self.node_type,
            "text_span_ids": list(self.constraint_group),
            "constraint_group": list(self.constraint_group),
            "children": list(self.children),
        }


@dataclass(frozen=True)
class PartitionTree:
    problem_id: str
    root_id: str
    nodes: dict[str, PartitionNode]
    config_hash: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "root": self.root_id,
            "nodes": [self.nodes[nid].to_dict() for nid in sorted(self.nodes)],
            "config_hash": self.config_hash,
            "provenance": self.provenance,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartitionTree":
        nodes = {}
        for raw in data["nodes"]:
            nodes[raw["node_id"]] = PartitionNode(
                node_id=raw["node_id"],
                parent_id=raw["parent_id"],
                depth=raw["depth"],
                node_type=raw["node_type"],
                constraint_group=tuple(raw["constraint_group"]),
                children=tuple(raw["children"]),
            )
        return cls(
            problem_id=data["problem_id"],
            root_id=data["root"],
            nodes=nodes,
            config_hash=data["config_hash"],
            provenance=data["provenance"],
        )

    def leaves(self) -> list[PartitionNode]:
        return [n for n in self.nodes.values() if n.node_type == "leaf"]

    def max_depth(self) -> int:
        return max(n.depth for n in self.nodes.values())


def _config_hash(config: PamopConfig) -> str:
    p = config.partitioning
    payload = json.dumps(
        {
            "config_kind": config.config_kind,
            "tfidf_top_k": p.tfidf_top_k,
            "epsilon": p.epsilon,
            "similarity_weights_by_layer": p.similarity_weights_by_layer,
            "clustering_algorithm": p.clustering_algorithm,
            "independent_set_algorithm": p.independent_set_algorithm,
            "bipartite_edge_confidence_threshold": p.bipartite_edge_confidence_threshold,
            "leaf_stop_min_constraints": p.leaf_stop_min_constraints,
            "leaf_stop_similarity_threshold": p.leaf_stop_similarity_threshold,
            "embedding_source": p.embedding_source,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_partition_tree(problem: StructuredProblem, config: PamopConfig) -> PartitionTree:
    """PAPER-SPECIFIED structure: root -> independent-set separation
    (section 3.2) -> recursive constraint clustering at every deeper layer
    until each leaf has "a small number of constraints or highly similar
    ones" (thresholds: config.partitioning.leaf_stop_*).
    """
    min_constraints = config.require("partitioning", "leaf_stop_min_constraints")
    stop_similarity = config.require("partitioning", "leaf_stop_similarity_threshold")

    all_indices = list(range(len(problem.constraints)))
    nodes: dict[str, PartitionNode] = {}
    node_counter = [0]

    def new_node_id() -> str:
        node_counter[0] += 1
        return f"node_{node_counter[0] - 1}"

    root_id = new_node_id()
    nodes[root_id] = PartitionNode(
        node_id=root_id,
        parent_id=None,
        depth=0,
        node_type="root",
        constraint_group=tuple(all_indices),
    )

    def is_leaf_ready(indices: list[int], layer: str) -> bool:
        if len(indices) <= min_constraints:
            return True
        # "highly similar" check: average pairwise combined similarity.
        if len(indices) < 2:
            return True
        texts = [problem.constraints[i].description for i in indices]
        vectorizer = _fit_tfidf(texts)
        tfidf_top_k = config.require("partitioning", "tfidf_top_k")
        keyword_sets = [top_k_keywords(t, vectorizer, tfidf_top_k) for t in texts]
        vector_provider = VectorSimilarityProvider(texts)
        weights_by_layer = config.require("partitioning", "similarity_weights_by_layer")
        weights = weights_by_layer.get(layer, weights_by_layer.get("default"))
        sims = []
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                sims.append(
                    combined_similarity(a, b, texts, keyword_sets, vector_provider, weights, tfidf_top_k)
                )
        return bool(sims) and (sum(sims) / len(sims)) >= stop_similarity

    def recurse(node_id: str, indices: list[int], depth: int, layer: str) -> None:
        if is_leaf_ready(indices, layer):
            nodes[node_id] = PartitionNode(
                node_id=node_id,
                parent_id=nodes[node_id].parent_id,
                depth=depth,
                node_type="leaf",
                constraint_group=tuple(indices),
            )
            return

        groups = cluster_constraints(problem, indices, config, layer)
        if len(groups) <= 1:
            # Clustering could not split this node further -- force a leaf
            # rather than looping forever on an unsplittable group.
            nodes[node_id] = PartitionNode(
                node_id=node_id,
                parent_id=nodes[node_id].parent_id,
                depth=depth,
                node_type="leaf",
                constraint_group=tuple(indices),
            )
            return

        children_ids = []
        for group in groups:
            child_id = new_node_id()
            children_ids.append(child_id)
            nodes[child_id] = PartitionNode(
                node_id=child_id,
                parent_id=node_id,
                depth=depth + 1,
                node_type="internal",
                constraint_group=tuple(group),
            )

        parent = nodes[node_id]
        nodes[node_id] = PartitionNode(
            node_id=node_id,
            parent_id=parent.parent_id,
            depth=parent.depth,
            node_type=parent.node_type,
            constraint_group=parent.constraint_group,
            children=tuple(children_ids),
        )

        for child_id, group in zip(children_ids, groups):
            recurse(child_id, group, depth + 1, layer="default")

    # Root uses independent-set separation, not clustering.
    root_groups = independent_set_separation(problem, all_indices, config)
    if len(root_groups) <= 1:
        # Independent-set separation found nothing to split (single
        # component) -- fall through to clustering directly on the root's
        # own constraint set, keeping root_id at depth 0. recurse() rewrites
        # root_id's node in place: either into a leaf (if already small/
        # similar enough) or into an internal node with clustered children.
        # layer="root" here (not "default") is the only place the config's
        # similarity_weights_by_layer.root entry is used -- every deeper
        # split after this one uses "default".
        recurse(root_id, all_indices, 0, layer="root")
    else:
        children_ids = []
        for group in root_groups:
            child_id = new_node_id()
            children_ids.append(child_id)
            nodes[child_id] = PartitionNode(
                node_id=child_id, parent_id=root_id, depth=1, node_type="internal",
                constraint_group=tuple(group),
            )
        nodes[root_id] = PartitionNode(
            node_id=root_id, parent_id=None, depth=0, node_type="root",
            constraint_group=tuple(all_indices), children=tuple(children_ids),
        )
        for child_id, group in zip(children_ids, root_groups):
            recurse(child_id, group, 1, layer="default")

    return PartitionTree(
        problem_id=problem.problem_id,
        root_id=root_id,
        nodes=nodes,
        config_hash=_config_hash(config),
        provenance={
            "config_kind": config.config_kind,
            "structured_problem_source": problem.source,
        },
    )
