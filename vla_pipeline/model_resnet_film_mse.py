"""ResNet18 + FiLM + direct action regression (single-step BC)."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from vla_pipeline.model_resnet_film_flow import ResNetFiLMEncoder, TextEncoder


class ResNetFiLMMSEPolicy(nn.Module):
    """Predict normalized 9-d action directly; simpler and more stable than flow matching."""

    def __init__(
        self,
        action_dim: int = 9,
        image_size: int = 128,
        lang_dim: int = 256,
        num_views: int = 3,
        hidden: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = 1
        self.image_size = image_size

        self.text_enc = TextEncoder(out_dim=lang_dim)
        self.vision_enc = ResNetFiLMEncoder(lang_dim=lang_dim, num_views=num_views)
        cond_dim = self.vision_enc.out_dim + lang_dim
        self.head = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, action_dim),
        )

    def encode_condition(self, views: torch.Tensor, texts: List[str]) -> torch.Tensor:
        lang = self.text_enc(texts)
        vis = self.vision_enc(views, lang)
        return torch.cat([vis, lang], dim=-1)

    def predict_action(self, views: torch.Tensor, texts: List[str]) -> torch.Tensor:
        return torch.tanh(self.head(self.encode_condition(views, texts)))

    def bc_loss(
        self,
        views: torch.Tensor,
        texts: List[str],
        action_chunk: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """action_chunk: B x H x D — uses first step only."""
        target = action_chunk[:, 0, :]
        pred = self.predict_action(views, texts)
        per = ((pred - target) ** 2).mean(dim=-1)
        if weights is not None:
            return (per * weights).mean()
        return per.mean()

    @torch.no_grad()
    def sample_action_chunk(self, views: torch.Tensor, texts: List[str]) -> torch.Tensor:
        action = self.predict_action(views, texts)
        return action.unsqueeze(1)

    def forward(self, views: torch.Tensor, texts: List[str]) -> torch.Tensor:
        return self.predict_action(views, texts)
