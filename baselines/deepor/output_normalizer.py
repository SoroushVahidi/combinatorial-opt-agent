"""Conservative extraction of Pyomo code from a DeepOR generation."""
from __future__ import annotations
import re
from dataclasses import dataclass
from .reasoning_parser import ReasoningTrace, parse_reasoning

@dataclass(frozen=True)
class ParsedOutput:
    raw_output: str
    reasoning: ReasoningTrace
    final_formulation: str
    generated_code: str | None
    status: str
    warnings: tuple[str, ...] = ()
    def to_dict(self): return {"raw_output": self.raw_output, "reasoning": self.reasoning.to_dict(), "final_formulation": self.final_formulation, "generated_code": self.generated_code, "status": self.status, "warnings": list(self.warnings)}

def parse_output(raw: str) -> ParsedOutput:
    trace = parse_reasoning(raw)
    blocks = list(re.finditer(r"```(?:python|py)?\s*\n?(.*?)```", trace.final_answer, re.I | re.S))
    candidates = [m for m in blocks if "pyomo" in m.group(1).lower() or "import" in m.group(1).lower()]
    if candidates:
        code = max(candidates, key=lambda m: len(m.group(1))).group(1).strip()
        return ParsedOutput(raw, trace, trace.final_answer, code, "CODE_EXTRACTED", trace.warnings)
    if "import pyomo" in trace.final_answer.lower() or "from pyomo" in trace.final_answer.lower():
        pos = min(x for x in (trace.final_answer.lower().find("import pyomo"), trace.final_answer.lower().find("from pyomo")) if x >= 0)
        return ParsedOutput(raw, trace, trace.final_answer, trace.final_answer[pos:].strip(), "UNFENCED_CODE_EXTRACTED", trace.warnings + ("unfenced_code",))
    return ParsedOutput(raw, trace, trace.final_answer, None, "FORMULATION_ONLY" if trace.final_answer else "EMPTY", trace.warnings + (("code_not_found",) if trace.final_answer else ()))
