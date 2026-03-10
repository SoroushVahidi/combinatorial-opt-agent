"""
Shared pytest fixtures and utilities for the combinatorial-opt-agent test suite.
"""
from __future__ import annotations

import socket

import pytest


def _is_network_available(host: str = "huggingface.co", port: int = 443, timeout: float = 2.0) -> bool:
    """Return True if the host is reachable (network is available)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def network_available() -> bool:
    """Session-scoped fixture: True if network is reachable."""
    return _is_network_available()


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests marked ``requires_network`` when offline."""
    if _is_network_available():
        return  # network is up — run everything
    skip_offline = pytest.mark.skip(reason="network unavailable (HuggingFace unreachable)")
    for item in items:
        if item.get_closest_marker("requires_network"):
            item.add_marker(skip_offline)


def tiny_catalog() -> list[dict]:
    """Minimal three-problem catalog for fast unit tests."""
    return [
        {
            "id": "p1",
            "name": "Knapsack",
            "aliases": ["0-1 knapsack"],
            "description": "Select items with weights and values to maximize value in capacity.",
        },
        {
            "id": "p2",
            "name": "Set Cover",
            "aliases": [],
            "description": "Choose minimum subsets to cover all elements.",
        },
        {
            "id": "p3",
            "name": "Vertex Cover",
            "aliases": [],
            "description": "Minimum vertices to cover every edge.",
        },
    ]

