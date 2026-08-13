"""Non-executing normalization of OR-R1 free-text generations.

Mirrors the section checklist `02_grpo_train.py`'s `reward_with_reference`
uses for its format reward, and the ```python fence extraction both
`reward_with_reference`/`run_code` (training) and `eval/execute.py`
(evaluation) perform via `output.find("```python")` / `output.find("```", ...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from baselines.orr1.config import ORR1_FORMAT_FIELDS


@dataclass(frozen=True)
class OrR1ParsedOutput:
    raw_output: str
    coptpy_code: str | None
    code_block_found: bool
    format_fields_present: tuple[str, ...] = field(default_factory=tuple)
    format_reward: float = 0.0
    parser_status: str = "EMPTY"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_output": self.raw_output,
            "coptpy_code": self.coptpy_code,
            "code_block_found": self.code_block_found,
            "format_fields_present": list(self.format_fields_present),
            "format_reward": self.format_reward,
            "parser_status": self.parser_status,
            "warnings": list(self.warnings),
        }


def _extract_first_python_block(raw: str) -> str | None:
    """Official upstream logic: first-only, not longest/best-of-many.

    `run_code` in `02_grpo_train.py` and `eval/execute.py`'s code-field loader
    both use `output.find("```python")` then the *next* ` ``` `, i.e. the
    first fenced python block only -- unlike this repo's ORLM adapter, which
    selects the longest of several candidate blocks. That difference is
    intentional and preserved here for fidelity.
    """
    start = raw.find("```python")
    if start == -1:
        return None
    end = raw.find("```", start + 9)
    if end == -1:
        return None
    return raw[start:end].replace("```python", "").strip()


def parse_output(raw: str) -> OrR1ParsedOutput:
    if not isinstance(raw, str) or not raw.strip():
        return OrR1ParsedOutput(raw if isinstance(raw, str) else "", None, False, (), 0.0, "EMPTY")
    present = tuple(field_name for field_name in ORR1_FORMAT_FIELDS if raw.find(field_name) != -1)
    format_reward = len(present) / len(ORR1_FORMAT_FIELDS)
    code = _extract_first_python_block(raw)
    if code is None:
        return OrR1ParsedOutput(raw, None, False, present, format_reward, "NO_CODE", ("code_fence_not_found",))
    if not code.strip():
        return OrR1ParsedOutput(raw, None, False, present, format_reward, "NO_CODE", ("empty_code_fence",))
    return OrR1ParsedOutput(raw, code, True, present, format_reward, "CODE_EXTRACTED")
