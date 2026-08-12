"""Parse ORLM's free-text response into model description + coptpy code.

ORLM's response format (per its prompt template) is free text containing
a mathematical model description followed by a coptpy Python code block.
This module splits those apart; it does NOT execute the code or attempt
solver-outcome normalization -- that requires a COPT license and is a
separate, not-yet-built step (README.md "Fair comparison caveats").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


@dataclass
class OrlmParsedOutput:
    model_description: str
    coptpy_code: str | None
    code_block_found: bool


def parse_orlm_output(raw: str) -> OrlmParsedOutput:
    """Split ORLM's raw text response into description and code.

    This is a structural best-effort parse (fenced code block extraction)
    based on the documented response format; it has not been validated
    against real ORLM output, since no inference has been run in this
    environment. Verify and adjust against real output before relying on
    this for any reported metric.
    """
    match = _CODE_BLOCK_RE.search(raw)
    if match:
        code = match.group(1).strip()
        description = (raw[: match.start()] + raw[match.end():]).strip()
        return OrlmParsedOutput(model_description=description, coptpy_code=code, code_block_found=True)
    return OrlmParsedOutput(model_description=raw.strip(), coptpy_code=None, code_block_found=False)
