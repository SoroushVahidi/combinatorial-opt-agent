"""Lightweight, paper-faithful DeepOR inference interface."""

from .config import DeepORConfig
from .data_adapter import DeepORInputRecord, adapt_record
from .pipeline import run_mock_pipeline

__all__ = ["DeepORConfig", "DeepORInputRecord", "adapt_record", "run_mock_pipeline"]
