"""Mockable end-to-end path: adapter -> prompt -> rollouts -> TGRPO reward
-> normalizer -> static validation -> result schema -> evaluator.

No GPU, vLLM, coptpy, or network access is required; `run_mock_pipeline`
exercises every stage of the wiring described in the task's lifecycle trace
(training data -> SFT -> TGRPO -> merged checkpoint -> inference -> generated
code -> solver -> evaluation) using an injectable mock backend for the
inference/solver-dependent steps only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from baselines.orr1.config import OrR1Config, pass8_config
from baselines.orr1.data_adapter import adapt_record, build_orr1_prompt
from baselines.orr1.evaluator import compute_solving_accuracy
from baselines.orr1.output_normalizer import parse_output
from baselines.orr1.result_schema import OrR1Result
from baselines.orr1.rollout import score_group
from baselines.orr1.runner import GenerationResult, OrR1Runner
from baselines.orr1.static_validation import validate_code
from baselines.orr1.tgrpo_controller import reward_component_breakdown

_MOCK_CODE = (
    "```python\n"
    "import coptpy\n"
    "from coptpy import COPT\n"
    "env = coptpy.Envr()\n"
    "model = env.createModel('x')\n"
    "x = model.addVar(lb=0)\n"
    "model.setObjective(x, COPT.MAXIMIZE)\n"
    "model.addConstr(x <= {bound})\n"
    "model.solve()\n"
    "```"
)

_MOCK_OUTPUT = (
    "## Mathematical Model:\nMaximize x subject to a bound.\n\n"
    "## Decision Variables:\nx >= 0\n\n"
    "## Objective Function:\nmax x\n\n"
    "## Constraints:\nx <= {bound}\n\n"
    "## Python Code Solution Using `coptpy`:\n" + _MOCK_CODE
)


class _MockBackend:
    """Deterministic multi-rollout mock; last rollout dissents to exercise voting."""

    def generate(self, prompt: str, config: OrR1Config) -> tuple[list[str], dict[str, Any]]:
        outputs = []
        for i in range(config.rollouts):
            bound = 1 if i < config.rollouts - 1 or config.rollouts == 1 else 2  # one dissenting rollout when rollouts > 1
            outputs.append(_MOCK_OUTPUT.format(bound=bound))
        return outputs, {"prompt_tokens": 10, "generated_tokens": 30 * config.rollouts}


def run_mock_pipeline(record: dict[str, Any], *, config: OrR1Config | None = None, git_sha: str | None = None) -> dict[str, Any]:
    config = config or pass8_config(model_id="ORR1_PIPELINE_MOCK_OR_PROXY", checkpoint_stage="MERGED")
    adapted = adapt_record(record)
    if not adapted.supported:
        raise ValueError(adapted.reason)

    prompt = build_orr1_prompt(adapted.record.raw_text, config)
    generation = OrR1Runner(config, _MockBackend()).generate(prompt)
    timestamp = datetime.now(timezone.utc).isoformat()

    if generation.status != "COMPLETED":
        result = OrR1Result.from_generation(
            problem_id=adapted.record.problem_id, dataset=adapted.record.dataset,
            input_sha256=adapted.record.input_sha256, config=config, rollout_index=0,
            rollout_count=config.rollouts, raw_output="", generation=generation,
            git_sha=git_sha, timestamp_utc=timestamp,
        )
        return {"records": [result.to_dict()], "group_score": None}

    # Mock "execution": rollout i's code claims bound `i`'s objective value as its answer.
    parsed_list = [parse_output(raw) for raw in generation.raw_outputs]
    validations = [validate_code(p.coptpy_code) for p in parsed_list]
    mock_best_solutions = ["1" if i < config.rollouts - 1 or config.rollouts == 1 else "2" for i in range(config.rollouts)]

    records: list[OrR1Result] = []
    for i, (raw, parsed, validation, best) in enumerate(zip(generation.raw_outputs, parsed_list, validations, mock_best_solutions)):
        rewards = reward_component_breakdown(
            format_reward=parsed.format_reward, execution_best_solution=best,
            group_pred_answers=mock_best_solutions, own_pred_answer=best,
        )
        result = OrR1Result.from_generation(
            problem_id=adapted.record.problem_id, dataset=adapted.record.dataset,
            input_sha256=adapted.record.input_sha256, config=config, rollout_index=i,
            rollout_count=config.rollouts, raw_output=raw, generation=generation,
            git_sha=git_sha, timestamp_utc=timestamp,
        )
        result.parsed = parsed.to_dict()
        result.static_validation = validation.to_dict()
        result.rewards = rewards.to_dict()
        result.execution_attempted = True
        result.execution = {"status": "COMPLETED_WITH_SOLUTION", "best_solution": best}
        result.objective = float(best)
        result.gold_objective = adapted.record.gold_metadata.get("gold_objective")
        result.failure_category = None if validation.status == "STATIC_VALID" else "static_validation_failure"
        records.append(result)

    gold = adapted.record.gold_metadata.get("gold_objective")
    group_score = score_group(mock_best_solutions, gold).to_dict() if gold is not None else None
    return {"records": [r.to_dict() for r in records], "group_score": group_score}
