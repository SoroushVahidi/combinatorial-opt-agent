"""Shared synthetic catalog helper; offline policy is in root conftest.py."""
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

