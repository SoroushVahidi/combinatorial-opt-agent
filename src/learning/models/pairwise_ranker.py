"""Pairwise (slot, mention) ranker: encoder + optional handcrafted features."""

from __future__ import annotations

from typing import Any

# Optional transformers; scaffold runs without it for validation
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _default_encoder_name() -> str:
    return "distilroberta-base"


class PairwiseRanker:
    """Score(slot, mention) with optional structured features."""

    def __init__(
        self,
        encoder_name: str | None = None,
        use_structured_features: bool = False,
        feature_dim: int = 5,
    ):
        self.encoder_name = encoder_name or _default_encoder_name()
        self.use_structured_features = use_structured_features
        self.feature_dim = feature_dim
        self.model = None
        self.tokenizer = None
        if _HAS_TORCH:
            self._build()

    def _build(self) -> None:
        if not _HAS_TORCH:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.model = _PairwiseRankerModule(
            encoder_name=self.encoder_name,
            use_structured_features=self.use_structured_features,
            feature_dim=self.feature_dim,
        )

    def _format_pair(self, slot_name: str, slot_role: str | None, mention_surface: str, context: str | None) -> str:
        parts = [f"[SLOT] {slot_name}"]
        if slot_role:
            parts.append(f"({slot_role})")
        parts.append("[SEP]")
        parts.append(mention_surface)
        if context:
            parts.append(context[:300])
        return " ".join(parts)

    def score(
        self,
        slot_name: str,
        slot_role: str | None,
        mention_surface: str,
        context: str | None = None,
        feature_vector: list[float] | None = None,
    ) -> float:
        """Return scalar score for (slot, mention). Without torch returns 0.0."""
        if not _HAS_TORCH or self.model is None:
            return 0.0
        text = self._format_pair(slot_name, slot_role, mention_surface, context)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        if self.use_structured_features and feature_vector is not None:
            feats = torch.tensor([feature_vector], dtype=torch.float32)
        else:
            feats = None
        with torch.no_grad():
            logit = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                feature_vector=feats,
            )
        return float(logit.squeeze().item())

    def save(self, path: str) -> None:
        if _HAS_TORCH and self.model is not None:
            torch.save({"model_state": self.model.state_dict(), "encoder_name": self.encoder_name}, path)

    def load(self, path: str) -> None:
        if not _HAS_TORCH:
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if self.model is None:
            self.encoder_name = ckpt.get("encoder_name", self.encoder_name)
            self._build()
        if self.model is not None and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"], strict=False)


if _HAS_TORCH:

    class _PairwiseRankerModule(nn.Module):
        def __init__(self, encoder_name: str, use_structured_features: bool = False, feature_dim: int = 5):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_name)
            hidden = self.encoder.config.hidden_size
            if use_structured_features:
                self.head = nn.Linear(hidden + feature_dim, 1)
            else:
                self.head = nn.Linear(hidden, 1)
            self.use_structured_features = use_structured_features

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any = None,
            feature_vector: Any = None,
        ) -> Any:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            if self.use_structured_features and feature_vector is not None:
                cls = torch.cat([cls, feature_vector.to(cls.device)], dim=-1)
            return self.head(cls).squeeze(-1)
