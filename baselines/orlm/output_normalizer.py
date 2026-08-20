"""Non-executing normalization of ORLM free-text generations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_CODE_BLOCK_RE = re.compile(r"```(?P<language>[A-Za-z0-9_+.-]*)?[ \t]*\n?(?P<code>.*?)```", re.DOTALL)
_INLINE_CODE_START = re.compile(r"(?m)^(?P<line>\s*(?:from|import)\s+coptpy\b.*)$")


@dataclass(frozen=True)
class OrlmParsedOutput:
    raw_output: str
    model_description: str
    coptpy_code: str | None
    code_block_found: bool
    code_blocks_seen: int = 0
    selected_block_index: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    parser_status: str = "EMPTY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_output": self.raw_output,
            "model_description": self.model_description,
            "coptpy_code": self.coptpy_code,
            "code_block_found": self.code_block_found,
            "code_blocks_seen": self.code_blocks_seen,
            "selected_block_index": self.selected_block_index,
            "warnings": list(self.warnings),
            "parser_status": self.parser_status,
        }


def _looks_like_code(language: str, code: str) -> bool:
    lowered = language.lower()
    return lowered in {"python", "py", "coptpy", ""} and (
        "coptpy" in code or "setobjective" in code.lower() or "model." in code
    )


def parse_orlm_output(raw: str) -> OrlmParsedOutput:
    """Extract the most plausible coptpy block without repairing its semantics."""
    if not isinstance(raw, str) or not raw.strip():
        return OrlmParsedOutput(raw_output=raw if isinstance(raw, str) else "", model_description="", coptpy_code=None, code_block_found=False, parser_status="EMPTY")
    matches = list(_CODE_BLOCK_RE.finditer(raw))
    warnings: list[str] = []
    if matches:
        candidates = [(i, m) for i, m in enumerate(matches) if _looks_like_code(m.group("language") or "", m.group("code"))]
        if not candidates:
            warnings.append("fenced_blocks_found_but_no_coptpy_like_block")
            return OrlmParsedOutput(raw, raw.strip(), None, True, len(matches), None, tuple(warnings), "NO_COPT_CODE")
        index, match = max(candidates, key=lambda item: len(item[1].group("code")))
        code = match.group("code").strip()
        description = (raw[: match.start()] + raw[match.end():]).strip()
        if len(candidates) > 1:
            warnings.append("multiple_coptpy_like_blocks_selected_longest")
        return OrlmParsedOutput(raw, description, code, True, len(matches), index, tuple(warnings), "CODE_EXTRACTED")

    inline = _INLINE_CODE_START.search(raw)
    if inline:
        code = raw[inline.start():].strip()
        description = raw[: inline.start()].strip()
        warnings.append("unfenced_coptpy_code_detected")
        return OrlmParsedOutput(raw, description, code, False, 0, None, tuple(warnings), "UNFENCED_CODE_EXTRACTED")
    return OrlmParsedOutput(raw, raw.strip(), None, False, 0, None, tuple(warnings), "NO_CODE")
