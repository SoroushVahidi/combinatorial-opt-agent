"""Static, per-system resource/fairness profiles.

These are recorded facts about each method's requirements, not a ranking.
Do not derive a "score" from this module; it exists so a reader can see at
a glance why systems are not directly comparable on compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResourceProfile:
    system: str
    compute: str  # e.g. "CPU-only" | "1x GPU >=24GB" | "multi-GPU"
    external_llm_api: bool
    deterministic: bool
    model_size_params: str  # human-readable, e.g. "N/A", "~8B", "gpt-5.4 (undisclosed)"
    solver: str
    test_time_learning: bool
    rollouts_per_problem: str  # human-readable, e.g. "1", "1 or 8", "N/A"
    training_required_for_faithful_result: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        d = self.__dict__.copy()
        d["notes"] = list(self.notes)
        return d


RESOURCE_PROFILES: dict[str, ResourceProfile] = {
    "ours": ResourceProfile(
        system="ours", compute="CPU-only", external_llm_api=False, deterministic=True,
        model_size_params="N/A (TF-IDF + deterministic rules)", solver="N/A (no solver call)",
        test_time_learning=False, rollouts_per_problem="1 (deterministic)",
        training_required_for_faithful_result=False,
        notes=("Fixed-catalog scalar grounding, not full NL-to-model generation.",
               "No GPU dependency in the evaluated method set."),
    ),
    "pamop": ResourceProfile(
        system="pamop", compute="CPU-only (API calls) + local AMPL/HiGHS", external_llm_api=True, deterministic=False,
        model_size_params="Undisclosed (Azure OpenAI gpt-5.4)", solver="AMPL + HiGHS (this repo's config)",
        test_time_learning=False, rollouts_per_problem="1 per attempt, plus a bounded G_exe/G_rev/G_comp/G_remod correction loop (max 5 iterations)",
        training_required_for_faithful_result=False,
        notes=("Independent reconstruction; no official PaMOP code was found.",
               "Correction loop means effective attempts per problem vary (1-6 in the empirical pilot)."),
    ),
    "orlm": ResourceProfile(
        system="orlm", compute="1x GPU, >=24GB VRAM", external_llm_api=False, deterministic=True,
        model_size_params="~8B (LLaMA-3)", solver="coptpy (COPT)",
        test_time_learning=False, rollouts_per_problem="1 (greedy, official topk=1)",
        training_required_for_faithful_result=False,
        notes=("Official code (Apache-2.0) and one public checkpoint confirmed; not evaluated on NLP4LP by its authors.",),
    ),
    "optmath": ResourceProfile(
        system="optmath", compute="1x GPU, >=24GB VRAM", external_llm_api=False, deterministic=False,
        model_size_params="~7B (Qwen2.5), optional 32B variant", solver="gurobipy (Gurobi)",
        test_time_learning=False, rollouts_per_problem="1 (official sampling, temperature=0.8)",
        training_required_for_faithful_result=False,
        notes=("Official code and checkpoint confirmed public.",),
    ),
    "deepor": ResourceProfile(
        system="deepor", compute="Unknown (official code unavailable; paper implies multi-GPU training)", external_llm_api=False,
        deterministic=True, model_size_params="~8B (Qwen3-8B, paper-specified)", solver="Pyomo (paper case study)",
        test_time_learning=True, rollouts_per_problem="1 (paper-specified greedy pass@1)",
        training_required_for_faithful_result=True,
        notes=("No official code, checkpoint, or requirements file located.",
               "SFT + GRPO with checklist-based reward, per the paper; not independently verifiable without code."),
    ),
    "orr1": ResourceProfile(
        system="orr1", compute="1x GPU >=24GB for inference; multi-GPU DeepSpeed ZeRO-3 for any training", external_llm_api=False,
        deterministic=True, model_size_params="8B (Qwen3-8B)", solver="coptpy (COPT)",
        test_time_learning=True, rollouts_per_problem="1 (pass@1) or 8 (pass@8/mj@8, official group size)",
        training_required_for_faithful_result=True,
        notes=("Official code confirmed public; no SFT/TGRPO/merged checkpoint released at any stage.",
               "TGRPO's official training data is transductive over all evaluation sets, including NLP4LP -- see docs/ORR1_PROVENANCE.md.",
               "Faithful reproduction is substantially more expensive than ORLM/OptMATH's inference-only path."),
    ),
}
