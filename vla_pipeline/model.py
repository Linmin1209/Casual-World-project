"""Lightweight multi-view + language BC model for CausalWorld."""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleTextEncoder(nn.Module):
    """Bag-of-characters encoder (no external HF deps)."""

    def __init__(self, vocab_size: int = 128, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, texts: list) -> torch.Tensor:
        batch = []
        for text in texts:
            codes = [min(127, ord(c)) for c in text[:512]]
            if not codes:
                codes = [0]
            t = torch.tensor(codes, device=self.embed.weight.device)
            emb = self.embed(t).mean(dim=0)
            batch.append(self.proj(emb))
        return torch.stack(batch, dim=0)


class CausalWorldBCVLA(nn.Module):
    def __init__(self, action_dim: int = 9, image_size: int = 128):
        super().__init__()
        c = 3 * 3  # 3 RGB views
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
        )
        self.text_enc = SimpleTextEncoder(out_dim=128)
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, views: torch.Tensor, texts: list) -> torch.Tensor:
        # views: B x 3 x 3 x H x W
        b, n, c, h, w = views.shape
        x = views.reshape(b, n * c, h, w)
        vis = self.cnn(x)
        lang = self.text_enc(texts)
        return self.head(torch.cat([vis, lang], dim=-1))
