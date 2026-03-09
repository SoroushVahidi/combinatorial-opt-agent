"""Decoding: per-slot argmax and optional one-to-one matching."""

from __future__ import annotations

from typing import Any


def argmax_per_slot(
    group_scores: list[tuple[str, str, float]],
) -> dict[str, str]:
    """group_scores: list of (group_id, mention_id, score). Return best mention_id per group (slot)."""
    best: dict[str, tuple[str, float]] = {}
    for gid, mid, sc in group_scores:
        if gid not in best or sc > best[gid][1]:
            best[gid] = (mid, sc)
    return {gid: best[gid][0] for gid in best}


def one_to_one_matching(
    group_scores: list[tuple[str, str, float]],
) -> dict[str, str]:
    """Greedy one-to-one: each mention assigned at most once (pick highest score first)."""
    # Sort by score descending
    sorted_list = sorted(group_scores, key=lambda x: -x[2])
    assigned_mentions: set[str] = set()
    result: dict[str, str] = {}
    for gid, mid, sc in sorted_list:
        if gid in result or mid in assigned_mentions:
            continue
        result[gid] = mid
        assigned_mentions.add(mid)
    return result
