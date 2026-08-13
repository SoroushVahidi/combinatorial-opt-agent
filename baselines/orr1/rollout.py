"""Group-of-N rollout aggregation: majority voting and Pass@k / mj@k.

Ports two official pieces exactly:
  * `Counter`-based `majority_voting` from `eval/execute.py` (first answer
    with the maximum count, by `Counter`'s first-seen-wins tie-break).
  * The `02_grpo_train.py` `reward_with_reference` in-training voting reward,
    which groups completions into fixed-size chunks of `GRPO_NUM_GENERATIONS`
    (8) and rewards a completion only if its own answer equals the group's
    majority-voted integer answer -- never the ground truth.
  * `eval/execute.py`'s pass@k ("is any of k roll-outs within tolerance of
    gold") and mj@k ("is the majority-voted roll-out within tolerance of
    gold") scoring, including its "No Best Solution" special case and its
    zero-gold absolute-tolerance branch.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

_NO_SOLUTION = "No Best Solution"


def majority_voting(pred_answers: list[Any]) -> Any:
    """Exact port of `eval/execute.py`'s `majority_voting`."""
    count = Counter(pred_answers)
    max_count = max(count.values())
    possible = [answer for answer, cnt in count.items() if cnt == max_count]
    return possible[0]


def _coerce_numeric(answer: Any) -> Any:
    if answer is None or answer == _NO_SOLUTION:
        return answer
    try:
        return round(float(answer))
    except (TypeError, ValueError):
        return answer


def group_majority_vote(pred_answers: list[Any]) -> Any | None:
    """Numeric-coerced majority vote over a rollout group; `None` if all missing."""
    coerced = [_coerce_numeric(a) for a in pred_answers if a is not None]
    if not coerced:
        return None
    return majority_voting(coerced)


def _within_tolerance(pred: Any, gold: Any, tolerance: float) -> bool:
    if gold == _NO_SOLUTION:
        return pred is not None and pred == gold
    gold_f = round(float(gold))
    if pred is None or pred == _NO_SOLUTION:
        return False
    pred_f = round(float(pred))
    if gold_f == 0:
        return abs(pred_f) <= tolerance
    return abs((pred_f - gold_f) / gold_f) <= tolerance


@dataclass(frozen=True)
class GroupScore:
    k: int
    pass_at_k: bool
    majority_answer: Any
    mj_at_k: bool

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "pass_at_k": self.pass_at_k, "majority_answer": self.majority_answer, "mj_at_k": self.mj_at_k}


def score_group(pred_answers: list[Any], gold_answer: Any, *, tolerance: float = 0.05) -> GroupScore:
    """Official `eval/execute.py` scoring for one question's rollout group."""
    k = len(pred_answers)
    pass_at_k = any(_within_tolerance(p, gold_answer, tolerance) for p in pred_answers)
    mj_answer = group_majority_vote(pred_answers)
    mj_at_k = _within_tolerance(mj_answer, gold_answer, tolerance)
    return GroupScore(k, pass_at_k, mj_answer, mj_at_k)
