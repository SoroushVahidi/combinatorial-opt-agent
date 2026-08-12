"""ORLM inference runner interface.

NOT IMPLEMENTED. Calling generate() raises NotImplementedError -- this
module exists to define the interface a future agent should implement
once a GPU and the model weights are available (README.md "Exact first
practical smoke-test milestone"). Do not stub this out with a fake/mocked
response; a future agent should implement it against the real
vllm/transformers + CardinalOperations/ORLM-LLaMA-3-8B stack.
"""
from __future__ import annotations

from dataclasses import dataclass

from baselines.orlm.config import OrlmConfig


@dataclass
class OrlmRunner:
    config: OrlmConfig

    def generate(self, prompt: str) -> str:
        """Run one ORLM inference call and return the raw text response.

        Requires: a GPU with >= config.min_gpu_memory_gb, the model
        weights downloaded locally or accessible via vllm/transformers,
        and (to execute any generated code, not to generate it) a COPT
        license for coptpy. None of these are available in the
        environment this scaffold was written in.
        """
        raise NotImplementedError(
            "ORLM inference is not implemented in this scaffold -- no GPU or "
            "model weights are available in this environment. See "
            "baselines/orlm/README.md for the exact next milestone before "
            "implementing this method."
        )
