"""Multitask grounder: shared encoder + pairwise head + optional NL4Opt auxiliary heads."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Auxiliary label set sizes (from NL4Opt)
BOUND_LABELS = ("lower", "upper", "equality", "other")
ROLE_LABELS = ("objective_coeff", "limit", "rhs_total", "ratio", "other")
BOUND_N = 4
ROLE_N = 5
MAX_ENTITY_CANDIDATES = 32


def _default_encoder_name() -> str:
    return "distilroberta-base"


if _HAS_TORCH:
    class MultitaskGrounderModule(nn.Module):
        """Encoder + pairwise head + optional entity/bound/role heads for NL4Opt aux."""

        def __init__(
            self,
            encoder_name: str,
            use_structured_features: bool = False,
            feature_dim: int = 5,
            use_entity_head: bool = False,
            use_bound_head: bool = False,
            use_role_head: bool = False,
        ):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_name)
            hidden = self.encoder.config.hidden_size
            inp_dim = hidden + (feature_dim if use_structured_features else 0)
            self.pairwise_head = nn.Linear(inp_dim, 1)
            self.use_structured_features = use_structured_features
            self.use_entity_head = use_entity_head
            self.use_bound_head = use_bound_head
            self.use_role_head = use_role_head
            if use_entity_head:
                self.entity_head = nn.Linear(hidden, MAX_ENTITY_CANDIDATES)
            else:
                self.entity_head = None
            if use_bound_head:
                self.bound_head = nn.Linear(hidden, BOUND_N)
            else:
                self.bound_head = None
            if use_role_head:
                self.role_head = nn.Linear(hidden, ROLE_N)
            else:
                self.role_head = None

        def _encode(self, input_ids: Any, attention_mask: Any = None) -> torch.Tensor:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0, :]

        def forward_pairwise(
            self,
            input_ids: Any,
            attention_mask: Any = None,
            feature_vector: Any = None,
        ) -> torch.Tensor:
            cls = self._encode(input_ids, attention_mask)
            if self.use_structured_features and feature_vector is not None:
                cls = torch.cat([cls, feature_vector.to(cls.device)], dim=-1)
            return self.pairwise_head(cls).squeeze(-1)

        def forward_aux_entity(self, input_ids: Any, attention_mask: Any = None) -> torch.Tensor:
            if self.entity_head is None:
                raise RuntimeError("entity head not enabled")
            cls = self._encode(input_ids, attention_mask)
            return self.entity_head(cls)

        def forward_aux_bound(self, input_ids: Any, attention_mask: Any = None) -> torch.Tensor:
            if self.bound_head is None:
                raise RuntimeError("bound head not enabled")
            cls = self._encode(input_ids, attention_mask)
            return self.bound_head(cls)

        def forward_aux_role(self, input_ids: Any, attention_mask: Any = None) -> torch.Tensor:
            if self.role_head is None:
                raise RuntimeError("role head not enabled")
            cls = self._encode(input_ids, attention_mask)
            return self.role_head(cls)
else:
    MultitaskGrounderModule = None  # type: ignore[misc, assignment]
