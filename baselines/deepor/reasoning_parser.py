"""Parse observable reasoning/final-answer sections without semantic repair."""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class ReasoningTrace:
    raw_output: str
    reasoning: str
    final_answer: str
    status: str
    warnings: tuple[str, ...] = ()
    def to_dict(self): return {"raw_output": self.raw_output, "reasoning": self.reasoning, "final_answer": self.final_answer, "status": self.status, "warnings": list(self.warnings)}

def parse_reasoning(raw: str) -> ReasoningTrace:
    if not isinstance(raw, str) or not raw.strip(): return ReasoningTrace(raw if isinstance(raw, str) else "", "", "", "EMPTY")
    m = re.search(r"<think>(.*?)</think>", raw, re.I | re.S)
    if m:
        final = (raw[:m.start()] + raw[m.end():]).strip()
        final = re.sub(r"^\s*(final answer|answer)\s*:\s*", "", final, flags=re.I)
        return ReasoningTrace(raw, m.group(1).strip(), final, "REASONING_AND_FINAL" if final else "REASONING_ONLY", () if final else ("final_answer_missing",))
    markers = re.split(r"\n\s*(?:final answer|answer)\s*:\s*", raw, maxsplit=1, flags=re.I)
    if len(markers) == 2: return ReasoningTrace(raw, markers[0].strip(), markers[1].strip(), "REASONING_AND_FINAL")
    return ReasoningTrace(raw, "", raw.strip(), "FINAL_WITHOUT_REASONING", ("reasoning_delimiter_not_found",))
