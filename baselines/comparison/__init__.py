"""Lightweight cross-baseline comparison harness.

Sits above each baseline's own native `result_schema.py`; never replaces
it. See README.md for the unified schema and metric taxonomy, and
docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md for the frozen protocol.
"""

from baselines.comparison.schema import CellState, UnifiedRow

__all__ = ["CellState", "UnifiedRow"]
