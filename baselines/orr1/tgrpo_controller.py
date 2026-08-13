"""TGRPO control-flow abstraction: the component that distinguishes OR-R1.

What the released code actually does (verified by reading `02_grpo_train.py`
end to end, not just the paper's prose):

  A. Rollout generation: for each training prompt, `GRPOTrainer` (TRL) samples
     `num_generations=8` completions from the *current* policy.
  B. Grouping: completions are consumed in fixed-size chunks of 8, one chunk
     per training question (`for i in range(0, len(completions), 8)`).
  C. Reward evaluation, three additive components per completion (see
     `reward_component_breakdown` below): format reward (fraction of the six
     `ORR1_FORMAT_FIELDS` headers present), valid-code reward (1.0 iff the
     extracted code executes and yields a best-solution string), and a
     *majority-voting* reward (1.0 iff the completion's own numeric answer
     equals its group's majority-voted answer). The ground-truth answer
     (`kwargs['answer']`) is read only for a logged CSV row -- it is never
     part of the reward. This is the paper's "Test-Time" framing: RL without
     labels, scored by self-consistency.
  D. Policy update: standard GRPO/PPO-style LoRA update (`use_peft`,
     `lora_r=16`, `lora_alpha=16`, `lora_target_modules=all-linear`,
     `learning_rate=1e-4`) via TRL's `GRPOTrainer.train()`, run to
     completion offline over `datasets/trainset/train_all.jsonl` -- there is
     no per-test-instance training loop and no online gradient step at
     evaluation time. `04_eval.sh` only calls `eval/generate.py` (pure vLLM
     inference) against the checkpoint produced by this offline stage.
  E. Candidate regeneration / checkpoint state: `save_steps=10`,
     `save_total_limit=100`; the final LoRA adapter is merged into the SFT
     checkpoint by `03_combine_lora.py` before evaluation.
  F. Test-set-as-training-data: `GRPO_TRAIN_DATA` is `train_all.jsonl`, which
     is *exactly* the union of every official `datasets/testset/*.jsonl`
     file (see `GRPO_TRANSDUCTIVE_LEAKAGE_NOTE` in `config.py`). So yes --
     per (F) in the task brief -- the test problems themselves are used as
     unlabeled TGRPO training data, for every official benchmark including
     NLP4LP, not merely "as applicable".

Net effect: "test-time RL" names the *label-free reward design*, not literal
online test-time weight updates. This module models the state machine and
reward computation without performing any gradient-heavy training; the
`mock_*` entry points exist purely to exercise state transitions in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from baselines.orr1.config import (
    GRPO_ADAM_BETA2, GRPO_LEARNING_RATE, GRPO_LORA_ALPHA, GRPO_LORA_R,
    GRPO_LORA_TARGET_MODULES, GRPO_MAX_COMPLETION_LENGTH, GRPO_MAX_PROMPT_LENGTH,
    GRPO_NUM_GENERATIONS, GRPO_NUM_TRAIN_EPOCHS, GRPO_TRAIN_DATA,
    GRPO_TRANSDUCTIVE_LEAKAGE_NOTE, GRPO_WARMUP_STEPS, GRPO_WEIGHT_DECAY,
)
from baselines.orr1.rollout import group_majority_vote

CHECKPOINT_STAGES = ("NONE", "BASE", "SFT", "GRPO_LORA", "MERGED")


@dataclass(frozen=True)
class TGRPOTrainingConfig:
    """Recorded, never executed by this repository (Phase 24: no training now)."""

    num_generations: int = GRPO_NUM_GENERATIONS
    lora_r: int = GRPO_LORA_R
    lora_alpha: int = GRPO_LORA_ALPHA
    lora_target_modules: str = GRPO_LORA_TARGET_MODULES
    learning_rate: float = GRPO_LEARNING_RATE
    num_train_epochs: int = GRPO_NUM_TRAIN_EPOCHS
    warmup_steps: int = GRPO_WARMUP_STEPS
    weight_decay: float = GRPO_WEIGHT_DECAY
    adam_beta2: float = GRPO_ADAM_BETA2
    max_prompt_length: int = GRPO_MAX_PROMPT_LENGTH
    max_completion_length: int = GRPO_MAX_COMPLETION_LENGTH
    train_data: str = GRPO_TRAIN_DATA
    is_transductive_over_eval_sets: bool = True
    transductive_note: str = GRPO_TRANSDUCTIVE_LEAKAGE_NOTE
    performs_online_per_instance_updates_at_eval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class RewardComponents:
    format_reward: float
    valid_code_reward: float
    voting_reward: float

    @property
    def total(self) -> float:
        return self.format_reward + self.valid_code_reward + self.voting_reward

    def to_dict(self) -> dict[str, Any]:
        return {"format_reward": self.format_reward, "valid_code_reward": self.valid_code_reward,
                "voting_reward": self.voting_reward, "total": self.total}


def reward_component_breakdown(
    *, format_reward: float, execution_best_solution: str | None, group_pred_answers: list[Any], own_pred_answer: Any,
) -> RewardComponents:
    """Port of `02_grpo_train.py`'s `reward_with_reference`, split into components.

    `execution_best_solution` / `group_pred_answers` come from running each
    completion's code (see `execution_harness.py`); ground truth is
    intentionally not a parameter -- upstream never uses it for reward.
    """
    valid_code_reward = 1.0 if execution_best_solution not in (None, "No Best Solution") else 0.0
    majority_answer = group_majority_vote(group_pred_answers)
    if own_pred_answer is None or own_pred_answer == "No Best Solution" or majority_answer is None:
        voting_reward = 0.0
    else:
        try:
            voting_reward = 1.0 if round(float(own_pred_answer)) == majority_answer else 0.0
        except (TypeError, ValueError):
            voting_reward = 0.0
    return RewardComponents(format_reward, valid_code_reward, voting_reward)


@dataclass(frozen=True)
class CheckpointState:
    """Tracks which stage produced a given model artifact; never mutated in place."""

    stage: str  # one of CHECKPOINT_STAGES
    base_model: str
    sft_path: str | None = None
    lora_adapter_path: str | None = None
    merged_path: str | None = None
    adaptation_scope: str = "NONE"  # NONE | GLOBAL_TRANSDUCTIVE_TRAINSET (official) | PER_PROBLEM_GROUP (isolated, non-official)
    owning_group_id: str | None = None  # Set only for a non-official per-group isolation experiment.

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def advance(self, stage: str, **updates: Any) -> "CheckpointState":
        if stage not in CHECKPOINT_STAGES:
            raise ValueError(f"unknown checkpoint stage: {stage}")
        return CheckpointState(**{**self.to_dict(), "stage": stage, **updates})


def mock_sft_step(state: CheckpointState, *, output_dir: str) -> CheckpointState:
    """State-only transition; performs no training. See config.SFT_* for the recorded hyperparameters."""
    return state.advance("SFT", sft_path=output_dir)


def mock_tgrpo_step(state: CheckpointState, *, output_dir: str, group_id: str | None = None) -> CheckpointState:
    """State-only transition for the (offline, transductive) TGRPO stage; performs no training."""
    if state.stage != "SFT":
        raise ValueError("TGRPO stage requires an SFT checkpoint as input, per 02_grpo_train.sh's $MODEL_NAME argument")
    scope = "PER_PROBLEM_GROUP" if group_id else "GLOBAL_TRANSDUCTIVE_TRAINSET"
    return state.advance("GRPO_LORA", lora_adapter_path=output_dir, adaptation_scope=scope, owning_group_id=group_id)


def mock_merge_step(state: CheckpointState, *, output_dir: str) -> CheckpointState:
    """State-only transition mirroring `03_combine_lora.py`; performs no merge."""
    if state.stage != "GRPO_LORA":
        raise ValueError("merge requires a GRPO LoRA adapter, per 03_combine_lora.py's positional arguments")
    return state.advance("MERGED", merged_path=output_dir)


def assert_no_cross_group_isolation_violation(states: list[CheckpointState]) -> None:
    """Fails if two PER_PROBLEM_GROUP-scoped states share a group id but different adapter paths.

    This is a guard for *our own* evaluation protocol, should we ever run an
    isolated (non-official) per-instance TGRPO variant: results attributed to
    one problem group must never silently carry another group's adapted
    weights. It does not apply to the official GLOBAL_TRANSDUCTIVE_TRAINSET
    scope, which is transductive by design (see module docstring, point F).
    """
    by_group: dict[str, set[str | None]] = {}
    for state in states:
        if state.adaptation_scope != "PER_PROBLEM_GROUP" or state.owning_group_id is None:
            continue
        by_group.setdefault(state.owning_group_id, set()).add(state.lora_adapter_path)
    for group_id, paths in by_group.items():
        if len(paths) > 1:
            raise AssertionError(f"group {group_id!r} is associated with multiple adapter paths: {paths}")
