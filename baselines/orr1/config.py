"""OR-R1 provenance and inference configuration.

Verified against the official repository (`SCUTE-ZZ/OR-R1`, commit
`9de48e3b22555e729ec032e7efd00ebaaa8e78d5`) on 2026-08-13. The arXiv HTML
source of the paper itself cites this exact GitHub URL as "Code", and the
GitHub repo was created 2025-11-11 / pushed 2025-11-12T02:47:45Z, one day
before the arXiv submission (2025-11-12T08:05:31Z) -- consistent with an
author-controlled camera-ready code drop. See docs/ORR1_PROVENANCE.md.

No OR-R1 checkpoint (base, SFT, LoRA, or merged) has been found on Hugging
Face, ModelScope, GitHub releases, or the repository itself. ``model_id`` is
therefore unset by default: a proxy must never be mistaken for OR-R1.

Critical methodology note: the official ``datasets/trainset/train_all.jsonl``
GRPO/TGRPO training file is *exactly* the union of all nine official test
sets (2634 rows = sum of all nine `datasets/testset/*.jsonl` sizes,
verified by direct line count and by set-based question-text matching; all
242 NLP4LP test questions are present verbatim). The paper states TGRPO
trains "on unlabeled test set data" using a majority-vote reward that never
reads the ground-truth answer (`utils`/`02_grpo_train.py`'s
`reward_with_reference` fetches ``kwargs['answer']`` only for logging, not
for the reward). OR-R1's primary reported metric is therefore transductive:
the TGRPO stage is trained directly on the same questions later scored for
Pass@1/Pass@8, without labels. This is not a bug in our reconstruction --
it is the officially released design -- but it is a first-order fairness
difference from every other baseline in this repository, which does not
train on the evaluation questions. See ``tgrpo_controller.py`` and
docs/ORR1_PROVENANCE.md for the full leakage analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

UPSTREAM_REPOSITORY = "https://github.com/SCUTE-ZZ/OR-R1"
UPSTREAM_DEFAULT_BRANCH = "master"
UPSTREAM_REVISION = "9de48e3b22555e729ec032e7efd00ebaaa8e78d5"
UPSTREAM_LICENSE = None  # No LICENSE file is present in the repository.

PAPER_TITLE = (
    "OR-R1: Automating Modeling and Solving of Operations Research "
    "Optimization Problem via Test-Time Reinforcement Learning"
)
PAPER_AUTHORS = ("Zezhen Ding", "Zhen Tan", "Jiheng Zhang", "Tianlong Chen")
PAPER_VENUE = "Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, No. 1"
PAPER_DOI = "10.1609/aaai.v40i1.36983"
PAPER_ARXIV_ID = "2511.09092"
PAPER_ARXIV_URL = "https://arxiv.org/abs/2511.09092"
PAPER_AAAI_URL = "https://ojs.aaai.org/index.php/AAAI/article/view/36983"
PAPER_PUBLICATION_DATE = "2026-03-14"

# Verbatim from upstream `02_grpo_train.py` / `eval/generate.py` (`TEMPLATE_q2mc_en`).
ORR1_PROMPT_TEMPLATE = (
    "Below is an operations research question. Build a mathematical model "
    "and corresponding python code using `coptpy` that appropriately "
    "addresses the question.\n\n"
    "# Question:\n"
    "{Question}\n\n"
    "# Response:\n"
)
ORR1_PROMPT_VERSION = "upstream-02_grpo_train-TEMPLATE_q2mc_en-v1"

# Verbatim from `02_grpo_train.py`'s `reward_with_reference`: the ordered
# section headers whose presence defines the format reward (count / 6).
ORR1_FORMAT_FIELDS = (
    "## Mathematical Model:",
    "## Decision Variables:",
    "## Objective Function:",
    "## Constraints:",
    "## Python Code Solution Using `coptpy`:",
    "```python",
)

# Appended by both `02_grpo_train.py`'s `run_code` and `eval/execute.py`
# before executing generated code. Requires the model's own code to expose a
# variable literally named `model`.
ORR1_ADD_SCRIPT = (
    '\nif model.status == COPT.OPTIMAL:\n'
    '    print(f"Just print the best solution: {model.objval}")\n'
    'else:\n'
    '    print("No Best Solution")'
)

BASE_MODEL = "Qwen/Qwen3-8B"

# `01_sft_train.sh` defaults.
SFT_LEARNING_RATE = 2e-5
SFT_MAX_SEQ_LENGTH = 8192
SFT_GRADIENT_ACCUMULATION_STEPS = 16
SFT_LR_SCHEDULER = "linear"
SFT_WARMUP_RATIO = 0.03
SFT_DEEPSPEED_CONFIG = "config/sft_config.json"  # ZeRO-3, CPU offload.
SFT_TRAIN_DATA = "datasets/OR-Instruct-Data-3K/OR-Instruct-Data-{sample}.json"
SFT_DATA_SOURCE = "CardinalOperations/OR-Instruct-Data-3K (from ORLM, Tang et al. 2024, arXiv:2405.17743, cc-by-nc-4.0)"

# `02_grpo_train.sh` / `02_grpo_train.py` defaults -- the TGRPO stage.
GRPO_NUM_GENERATIONS = 8  # Group size for GRPO / majority-vote reward.
GRPO_LORA_R = 16
GRPO_LORA_ALPHA = 16
GRPO_LORA_TARGET_MODULES = "all-linear"
GRPO_LEARNING_RATE = 1e-4
GRPO_LR_SCHEDULER = "cosine"
GRPO_NUM_TRAIN_EPOCHS = 1
GRPO_WARMUP_STEPS = 10
GRPO_WEIGHT_DECAY = 0.01
GRPO_ADAM_BETA2 = 0.95
GRPO_MAX_PROMPT_LENGTH = 2048
GRPO_MAX_COMPLETION_LENGTH = 6144
GRPO_DEEPSPEED_CONFIG = "config/grpo_config.json"  # ZeRO-3, CPU offload.
GRPO_TRAIN_DATA = "datasets/trainset/train_all.jsonl"
GRPO_REWARD_TIMEOUT_SECONDS = 10  # `reward_with_reference`'s `compile_script` default.
GRPO_TRANSDUCTIVE_LEAKAGE_NOTE = (
    "train_all.jsonl (2634 rows) is exactly the concatenation of all nine "
    "official datasets/testset/*.jsonl files (18+100+211+652+230+605+166+"
    "410+242=2634); every one of the 242 NLP4LP test questions is present "
    "verbatim. TGRPO's reward never reads the ground-truth answer, so this "
    "is unlabeled/self-consistency training directly on the eval questions, "
    "not a conventional held-out split."
)

# `04_eval.sh` -> `eval/eval.all.sh` -> per-dataset `eval.<name>.pass{1,8}.sh`.
EVAL_TIMEOUT_SECONDS = 600  # `eval/execute.py --timeout`.
EVAL_MAX_WORKERS = 16
EVAL_NUMERICAL_ERR_TOLERANCE = 0.05  # `eval/execute.py --numerical_err_tolerance` default.
EVAL_STOP_TOKENS = ("</s>", "<|endoftext|>", "<|im_end|>")
EVAL_MAX_TOKENS = 10000  # `eval/generate.py` hardcoded default.

PASS1_TOPK = 1
PASS1_DECODING_METHOD = "greedy"
PASS1_TEMPERATURE = 0.0
PASS1_TOP_P = 1.0

PASS8_TOPK = 8
PASS8_DECODING_METHOD = "sampling"
PASS8_TEMPERATURE = 0.7  # `eval/generate.py --temperature` default.
PASS8_TOP_P = 0.95  # `eval/generate.py --top_p` default.

NLP4LP_EVAL_DATASET = "datasets/testset/nlp4lp.jsonl"

CHECKPOINT_STATUS = "CHECKPOINT_NOT_RELEASED"


@dataclass(frozen=True)
class OrR1Config:
    """Provenance-carrying inference configuration. No default weights."""

    model_id: str | None = None
    model_path: str | None = None
    model_revision: str | None = None
    checkpoint_stage: str = "NONE"  # NONE | BASE | SFT | GRPO_LORA | MERGED
    lora_adapter_path: str | None = None
    prompt_template: str = ORR1_PROMPT_TEMPLATE
    prompt_version: str = ORR1_PROMPT_VERSION
    upstream_revision: str = UPSTREAM_REVISION
    solver: str = "coptpy"
    decoding_method: str = PASS1_DECODING_METHOD
    temperature: float = PASS1_TEMPERATURE
    top_p: float = PASS1_TOP_P
    topk: int = PASS1_TOPK
    stop_tokens: tuple[str, ...] = EVAL_STOP_TOKENS
    max_tokens: int = EVAL_MAX_TOKENS
    tensor_parallel_size: int = 1
    seed: int = 0
    timeout_seconds: int = EVAL_TIMEOUT_SECONDS
    numerical_err_tolerance: float = EVAL_NUMERICAL_ERR_TOLERANCE
    rollouts: int = PASS1_TOPK
    requires_chat_template: bool = True  # Official code wraps the prompt with `tokenizer.apply_chat_template`.
    min_gpu_memory_gb: int = 24
    finetuning_required: bool = True  # No merged/SFT/GRPO checkpoint is released; faithful reproduction requires training.
    requires_external_api: bool = False

    def generation_dict(self) -> dict[str, object]:
        return {
            "decoding_method": self.decoding_method,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "topk": self.topk,
            "stop_tokens": list(self.stop_tokens),
            "max_tokens": self.max_tokens,
            "tensor_parallel_size": self.tensor_parallel_size,
            "seed": self.seed,
            "rollouts": self.rollouts,
        }


def pass1_config(**overrides: object) -> OrR1Config:
    base = dict(decoding_method=PASS1_DECODING_METHOD, temperature=PASS1_TEMPERATURE,
                top_p=PASS1_TOP_P, topk=PASS1_TOPK, rollouts=PASS1_TOPK)
    base.update(overrides)
    return OrR1Config(**base)


def pass8_config(**overrides: object) -> OrR1Config:
    base = dict(decoding_method=PASS8_DECODING_METHOD, temperature=PASS8_TEMPERATURE,
                top_p=PASS8_TOP_P, topk=PASS8_TOPK, rollouts=PASS8_TOPK)
    base.update(overrides)
    return OrR1Config(**base)
