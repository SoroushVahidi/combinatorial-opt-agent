"""OR-R1 baseline lightweight implementation (2026-08-13).

Interface-only: no model weights are bundled or downloaded, no training or
inference is run by importing this package. See README.md for status and
docs/ORR1_PROVENANCE.md for the full fidelity matrix. Nothing here is wired
into the main pipeline or any evaluated result.
"""

from baselines.orr1.config import OrR1Config, pass1_config, pass8_config
from baselines.orr1.data_adapter import OrR1InputRecord, adapt_record
from baselines.orr1.pipeline import run_mock_pipeline

__all__ = ["OrR1Config", "pass1_config", "pass8_config", "OrR1InputRecord", "adapt_record", "run_mock_pipeline"]
