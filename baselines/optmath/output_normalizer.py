"""Normalization of OptMATH's official Python/Gurobi output."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_CODE_RE = re.compile(r"```(?P<language>[A-Za-z0-9_+.-]*)?[ \t]*\n?(?P<code>.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ParsedOutput:
    raw_output: str
    generated_code: str | None
    formulation_text: str
    code_block_found: bool
    blocks_seen: int
    selected_block_index: int | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    status: str = "EMPTY"

    def to_dict(self) -> dict[str, Any]:
        return {"raw_output": self.raw_output, "generated_code": self.generated_code, "formulation_text": self.formulation_text, "code_block_found": self.code_block_found, "blocks_seen": self.blocks_seen, "selected_block_index": self.selected_block_index, "warnings": list(self.warnings), "status": self.status}


def _is_gurobi(code: str) -> bool:
    lowered = code.lower()
    return "gurobipy" in lowered or "import gp" in lowered


def parse_output(raw: str) -> ParsedOutput:
    if not isinstance(raw, str) or not raw.strip():
        return ParsedOutput(raw if isinstance(raw, str) else "", None, "", False, 0, None, status="EMPTY")
    matches = list(_CODE_RE.finditer(raw))
    candidates = [(i, match) for i, match in enumerate(matches) if _is_gurobi(match.group("code"))]
    warnings: list[str] = []
    if candidates:
        index, match = max(candidates, key=lambda item: len(item[1].group("code")))
        if len(candidates) > 1:
            warnings.append("multiple_gurobi_blocks_selected_longest")
        return ParsedOutput(raw, match.group("code").strip(), (raw[:match.start()] + raw[match.end():]).strip(), True, len(matches), index, tuple(warnings), "CODE_EXTRACTED")
    if "import gurobipy" in raw or "import gp" in raw:
        start = raw.find("import gurobipy") if "import gurobipy" in raw else raw.find("import gp")
        warnings.append("unfenced_gurobi_code_detected")
        return ParsedOutput(raw, raw[start:].strip(), raw[:start].strip(), False, len(matches), None, tuple(warnings), "UNFENCED_CODE_EXTRACTED")
    if matches:
        warnings.append("fenced_blocks_found_but_no_gurobi_code")
    return ParsedOutput(raw, None, raw.strip(), bool(matches), len(matches), None, tuple(warnings), "NO_CODE")
