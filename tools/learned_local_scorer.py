"""P0 feature-augmented local mention-slot scorer.

This module is additive, following the same pattern as
``tools/search_structured_grounding.py`` and
``tools/hierarchical_structured_grounding.py``: it imports shared
extraction/scoring internals from ``tools/nlp4lp_downstream_utility.py``
(the canonical downstream pipeline) rather than duplicating them, and does
NOT modify that file.

Key difference from the prior learned ranker (NR10, see
``docs/NEGATIVE_RESULTS.md``): NR10 fine-tuned a text-only transformer
(``distilroberta-base``) on raw (slot name, mention surface, context) text,
with no access to the hand-engineered compatibility signals the rule-based
pipeline already computes. This module instead:

1. Reuses the canonical pipeline's own optimization-role scoring function
   (``_score_mention_slot_opt``) to obtain a rich, already-computed feature
   vector (type compatibility, operator/role/unit cues, total-vs-per-unit
   signals, bound-direction signals, etc.) for every (mention, slot) pair --
   the same signals the rule-based ``typed_greedy`` / ``optimization_role_repair``
   methods already use, but exposed as *inputs* to a learned model instead
   of hard-coded weights.
2. Adds ONE additional signal not previously available: a frozen
   sentence-embedding cosine similarity between the mention's local context
   and the slot's name/role description, using the same encoder the
   repository's own retrieval baselines already default to
   (``sentence-transformers/all-MiniLM-L6-v2``, see ``retrieval/search.py``).
   The encoder is not fine-tuned in this phase.
3. Feeds the resulting ~25-dimensional feature vector to a SMALL classifier
   (from a preregistered set: logistic regression, gradient-boosted trees,
   or a tiny MLP -- see ``scripts/learning/train_p0_classifier.py``), not a
   fine-tuned transformer, avoiding NR10's data-starvation failure mode
   (9,729 training pairs is small for full transformer fine-tuning but
   plausible for a low-parameter-count classifier).

Inference remains fully local: no external generative-LLM API is called at
any point in this module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nlp4lp_downstream_utility import (  # noqa: E402  (reuse canonical internals, do not modify that file)
    MentionOptIR,
    SlotOptIR,
    _build_slot_opt_irs,
    _extract_opt_role_mentions,
    _opt_role_validate_one,
    _score_mention_slot_opt,
)

# Fixed, documented feature schema. Boolean/flag features from
# `_score_mention_slot_opt`'s diagnostics dict are encoded as 1.0/0.0;
# count-valued features keep their integer magnitude; `hand_engineered_score`
# is the raw score `_score_mention_slot_opt` itself would have used for a
# greedy/typed-greedy decision -- included as one input feature among many,
# not as the sole signal.
ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "type_incompatible",
    "derived_count_non_count",
    "type_exact",
    "type_loose",
    "opt_role_overlap",
    "fragment_objective",
    "fragment_bound",
    "fragment_resource",
    "fragment_ratio",
    "operator_match",
    "ctx_overlap",
    "sent_overlap",
    "unit_match",
    "entity_resource_overlap",
    "total_match",
    "coefficient_match",
    "coeff_to_total_penalty",
    "total_to_coeff_penalty",
    "count_role_match",
    "count_to_non_count_penalty",
    "lower_bound_match",
    "upper_bound_match",
    "bound_direction_wrong",
    "weak_penalty",
    "hand_engineered_score",
)
EMBEDDING_FEATURE_NAME = "embedding_similarity"
ALL_FEATURE_NAMES: tuple[str, ...] = ENGINEERED_FEATURE_NAMES + (EMBEDDING_FEATURE_NAME,)

_COUNT_FEATURES = frozenset(
    {"opt_role_overlap", "ctx_overlap", "sent_overlap", "entity_resource_overlap"}
)

DEFAULT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def score_pair_engineered(m: MentionOptIR, s: SlotOptIR) -> dict[str, float]:
    """Reuse the canonical opt-role scorer; return a fixed-schema feature dict."""
    raw_score, diagnostics = _score_mention_slot_opt(m, s)
    out: dict[str, float] = {name: 0.0 for name in ENGINEERED_FEATURE_NAMES}
    out["hand_engineered_score"] = float(raw_score)
    for key, value in diagnostics.items():
        if key not in out:
            continue  # schema_prior/total_score intentionally excluded (constant / duplicate of raw_score)
        if key in _COUNT_FEATURES:
            out[key] = float(value)
        else:
            out[key] = 1.0 if value else 0.0
    return out


def slot_description_text(s: SlotOptIR) -> str:
    role_text = " ".join(sorted(s.slot_role_tags)) if s.slot_role_tags else ""
    op_text = " ".join(sorted(s.operator_preference)) if s.operator_preference else ""
    return f"{s.name} {role_text} {op_text} {s.expected_type}".strip()


def mention_context_text(m: MentionOptIR) -> str:
    ctx = " ".join(m.context_tokens) if m.context_tokens else ""
    return f"{m.raw_surface} {ctx}".strip()


def build_engineered_rows(query: str, expected_scalar: list[str]) -> tuple[list[MentionOptIR], list[SlotOptIR], list[dict[str, Any]]]:
    """Extract canonical mentions/slots for one query and score every pair.

    Returns (mentions, slots, rows) where each row has slot_name, mention_id,
    engineered feature dict, and the raw texts needed for the embedding
    feature (computed separately, in batch, by the caller for efficiency).
    """
    mentions = _extract_opt_role_mentions(query, variant="orig")
    slots = _build_slot_opt_irs(expected_scalar)
    rows: list[dict[str, Any]] = []
    for s in slots:
        for m in mentions:
            feats = score_pair_engineered(m, s)
            rows.append(
                {
                    "slot_name": s.name,
                    "mention_id": m.mention_id,
                    "engineered_features": feats,
                    "slot_text": slot_description_text(s),
                    "mention_text": mention_context_text(m),
                }
            )
    return mentions, slots, rows


class FrozenEncoder:
    """Thin wrapper around a frozen sentence-transformers encoder (not fine-tuned)."""

    def __init__(self, model_name: str = DEFAULT_ENCODER_NAME) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]):
        import numpy as np

        self._ensure_loaded()
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        emb = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=64)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return emb / norms

    def pair_similarities(self, texts_a: list[str], texts_b: list[str]):
        """Cosine similarity for aligned (a[i], b[i]) text pairs, batched and deduplicated."""
        import numpy as np

        unique_texts = sorted(set(texts_a) | set(texts_b))
        idx = {t: i for i, t in enumerate(unique_texts)}
        embeddings = self.encode(unique_texts)
        a_idx = [idx[t] for t in texts_a]
        b_idx = [idx[t] for t in texts_b]
        a_vecs = embeddings[a_idx]
        b_vecs = embeddings[b_idx]
        return np.sum(a_vecs * b_vecs, axis=1)


def feature_dict_to_vector(feats: dict[str, float], names: tuple[str, ...] = ALL_FEATURE_NAMES) -> list[float]:
    return [float(feats.get(name, 0.0)) for name in names]


def validate_assignment(slot_name: str, mention: MentionOptIR, slot: SlotOptIR, score: float) -> tuple[bool, float]:
    """Reuse the canonical single-assignment plausibility check (stage 6 primitive)."""
    return _opt_role_validate_one(slot_name, mention, slot, score)
