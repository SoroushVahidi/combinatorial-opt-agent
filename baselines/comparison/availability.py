"""Per-system availability status: separate from performance, never conflated with it.

This is the answer to "does a genuine empirical NLP4LP result exist for
this system right now" -- distinct from any metric value. A report must
show these states explicitly rather than a blank or zero cell.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AvailabilityStatus:
    system: str
    status_label: str
    has_empirical_nlp4lp_rows: bool
    classification: str  # matches each baseline's own README/provenance classification string
    detail: str

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


AVAILABILITY: dict[str, AvailabilityStatus] = {
    "ours": AvailabilityStatus(
        "ours", "VALIDATED (whole-benchmark); common-18-subset status tracked separately",
        True, "PAPER_CORE_VALIDATED",
        "331-query whole-benchmark result exists (results/paper/eaai_camera_ready_tables/); "
        "a dedicated common-18-instance subset run is a separate, smaller extraction (see report §7).",
    ),
    "pamop": AvailabilityStatus(
        "pamop", "COMMON-18 COMPLETE (gpt-5.4, AMPL/HiGHS execution)",
        True, "PAMOP_COMMON18_COMPLETE",
        "results/pamop/fidelity_diagnostic_gpt5/: 18/18 common-18 rows, deployment gpt-5.4 "
        "temperature 0.2, 13/18 execution success, 5 AMPL parse failures, 8/11 evaluable "
        "objective-value-proxy success. Independent reconstruction, no official code.",
    ),
    "orlm": AvailabilityStatus(
        "orlm", "COMMON-18 COMPLETE (official checkpoint; execution blocked on coptpy)",
        True, "ORLM_COMMON18_COMPLETE_EXECUTION_BLOCKED",
        "results/orlm/common18_official_checkpoint/results.jsonl: 18/18 rows, official "
        "CardinalOperations/ORLM-LLaMA-3-8B revision 94fdc3c5738c6536d4880dc19a78f215529181c5, "
        "all generation/parse/static-validation complete. Solver execution and objective "
        "comparison are blocked because coptpy is not installed (COPTPY_MISSING); "
        "objective_proxy_status=NOT_EVALUABLE for all 18 rows.",
    ),
    "optmath": AvailabilityStatus(
        "optmath", "COMMON-18 COMPLETE (official checkpoint; execution complete)",
        True, "OPTMATH_COMMON18_COMPLETE_EXECUTION_COMPLETE",
        "results/optmath/common18_official_checkpoint/results.jsonl: 18/18 rows, official "
        "Aurora-Gem/OptMATH-Qwen2.5-7B revision 617fe77, all generation/parse/static-validation "
        "complete; gurobipy execution complete (15/18 COMPLETED, 3 genuine model-code failures), "
        "objective-proxy 6/18 agreement (tolerance 0.05).",
    ),
    "generic": AvailabilityStatus(
        "generic", "COMMON-18 COMPLETE (gpt-5.4, zero-shot gurobipy; execution complete)",
        True, "GENERIC_LLM_COMMON18_COMPLETE_EXECUTION_COMPLETE",
        "results/generic_llm/common18_official/results.jsonl: 18/18 rows, all generation/parse/"
        "static-validation complete, served model gpt-5.4-2026-03-05 (azure_openai deployment gpt-5.4); "
        "gurobipy execution complete (16/18 COMPLETED, 2 genuine model-code failures), objective-proxy "
        "10/18 agreement (tolerance 0.05). Generic-purpose API LLM with a fixed zero-shot gurobipy prompt "
        "(no optimization training).",
    ),
    "deepor": AvailabilityStatus(
        "deepor", "PAPER RECONSTRUCTION, OFFICIAL CHECKPOINT UNAVAILABLE",
        False, "DEEPOR_PAPER_RECONSTRUCTION_READY",
        "No official code, checkpoint, or requirements file located anywhere. Lightweight reconstruction "
        "is mock-tested only; zero empirical result rows exist and none can exist without a released artifact.",
    ),
    "orr1": AvailabilityStatus(
        "orr1", "OFFICIAL CODE INTEGRATED, CHECKPOINT UNAVAILABLE",
        False, "ORR1_CODE_INTEGRATED_CHECKPOINT_BLOCKED",
        "Official code verified (cited directly by the arXiv paper). No SFT/TGRPO/merged checkpoint "
        "released anywhere. Faithful reproduction additionally requires training TGRPO transductively "
        "over the evaluation set itself. Zero empirical result rows exist.",
    ),
}
