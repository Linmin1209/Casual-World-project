"""ResNet18 + FiLM (language) + conditional Flow Matching for multi-step actions."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _resnet18_backbone():
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except (ImportError, AttributeError, TypeError):
        from torchvision.models import resnet18

        return resnet18(pretrained=True)


class TextEncoder(nn.Module):
    """Lightweight text encoder (no HuggingFace dependency)."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(128, 64)
        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        batch = []
        device = self.embed.weight.device
        for text in texts:
            codes = [min(127, ord(c)) for c in text[:512]] or [0]
            t = torch.tensor(codes, device=device, dtype=torch.long)
            batch.append(self.proj(self.embed(t).mean(dim=0)))
        return torch.stack(batch, dim=0)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta = nn.Linear(cond_dim, feat_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return g * x + b


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResNetFiLMEncoder(nn.Module):
    """Shared ResNet18 backbone with FiLM conditioning after layer4."""

    def __init__(self, lang_dim: int = 256, num_views: int = 3):
        super().__init__()
        self.num_views = num_views
        backbone = _resnet18_backbone()
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.film4 = FiLM(lang_dim, 512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512 * num_views

    def _forward_view(self, img: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        x = self.stem(img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.film4(x, lang)
        return self.pool(x).flatten(1)

    def forward(self, views: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        # views: B x V x 3 x H x W
        feats = []
        for v in range(views.shape[1]):
            feats.append(self._forward_view(views[:, v], lang))
        return torch.cat(feats, dim=1)


class FlowMatchingActionHead(nn.Module):
    """Predict vector field v(a_t, t | cond) for flattened action chunks."""

    def __init__(self, chunk_dim: int, cond_dim: int, hidden: int = 512, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(chunk_dim + cond_dim + time_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk_dim),
        )

    def forward(self, a_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(a_t.shape[0])
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        x = torch.cat([a_t, cond, t_emb], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        chunk_dim: int,
        steps: int = 10,
    ) -> torch.Tensor:
        b = cond.shape[0]
        a = torch.randn(b, chunk_dim, device=cond.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b,), 1.0 - i * dt, device=cond.device)
            v = self.forward(a, t, cond)
            a = a - dt * v
        return a


class ResNetFiLMFlowPolicy(nn.Module):
    """
    Multi-step action policy: predicts action chunk [H, action_dim] via flow matching.

    Training: conditional flow matching on flattened chunks.
    Inference: sample chunk, execute first action (queue rest in hybrid wrapper).
    """

    def __init__(
        self,
        action_dim: int = 9,
        action_horizon: int = 8,
        image_size: int = 128,
        lang_dim: int = 256,
        num_views: int = 3,
        fm_hidden: int = 512,
        fm_sample_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.fm_sample_steps = fm_sample_steps
        self.chunk_dim = action_horizon * action_dim

        self.text_enc = TextEncoder(out_dim=lang_dim)
        self.vision_enc = ResNetFiLMEncoder(lang_dim=lang_dim, num_views=num_views)
        cond_dim = self.vision_enc.out_dim + lang_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, fm_hidden),
            nn.ReLU(),
        )
        self.flow_head = FlowMatchingActionHead(
            chunk_dim=self.chunk_dim,
            cond_dim=fm_hidden,
            hidden=fm_hidden,
        )

    def encode_condition(self, views: torch.Tensor, texts: List[str]) -> torch.Tensor:
        lang = self.text_enc(texts)
        vis = self.vision_enc(views, lang)
        return self.cond_proj(torch.cat([vis, lang], dim=-1))

    def flow_matching_loss(
        self,
        views: torch.Tensor,
        texts: List[str],
        action_chunk: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        action_chunk: B x H x action_dim
        weights: B optional step-level weight (uses first step weight if provided)
        """
        b = action_chunk.shape[0]
        x0 = action_chunk.reshape(b, -1)
        x1 = torch.randn_like(x0)
        t = torch.rand(b, device=x0.device)
        x_t = (1.0 - t).unsqueeze(1) * x0 + t.unsqueeze(1) * x1
        target_v = x1 - x0
        cond = self.encode_condition(views, texts)
        pred_v = self.flow_head(x_t, t, cond)
        per = ((pred_v - target_v) ** 2).mean(dim=-1)
        if weights is not None:
            return (per * weights).mean()
        return per.mean()

    def aux_denoise_loss(
        self,
        views: torch.Tensor,
        texts: List[str],
        action_chunk: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-step denoise at t=1: stronger direct supervision on actions."""
        b = action_chunk.shape[0]
        x0 = action_chunk.reshape(b, -1)
        x1 = torch.randn_like(x0)
        t = torch.ones(b, device=x0.device)
        cond = self.encode_condition(views, texts)
        pred_v = self.flow_head(x1, t, cond)
        x0_hat = x1 - pred_v
        per = ((x0_hat - x0) ** 2).mean(dim=-1)
        if weights is not None:
            return (per * weights).mean()
        return per.mean()

    def training_loss(
        self,
        views: torch.Tensor,
        texts: List[str],
        action_chunk: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        aux_weight: float = 0.0,
    ) -> torch.Tensor:
        loss = self.flow_matching_loss(views, texts, action_chunk, weights=weights)
        if aux_weight > 0:
            loss = loss + aux_weight * self.aux_denoise_loss(
                views, texts, action_chunk, weights=weights
            )
        return loss

    @torch.no_grad()
    def sample_action_chunk(self, views: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Returns B x H x action_dim."""
        cond = self.encode_condition(views, texts)
        flat = self.flow_head.sample(cond, self.chunk_dim, steps=self.fm_sample_steps)
        flat = torch.tanh(flat)
        return flat.view(views.shape[0], self.action_horizon, self.action_dim)

    def forward(self, views: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """First action of sampled chunk (for API compatibility)."""
        chunk = self.sample_action_chunk(views, texts)
        return chunk[:, 0, :]


def build_policy_from_config(tcfg: dict, action_dim: int) -> nn.Module:
    model_type = tcfg.get("model_type", "resnet_film_flow")
    if model_type == "resnet_film_flow":
        return ResNetFiLMFlowPolicy(
            action_dim=action_dim,
            action_horizon=int(tcfg.get("action_horizon", 8)),
            image_size=int(tcfg.get("image_size", 128)),
            lang_dim=int(tcfg.get("lang_dim", 256)),
            num_views=int(tcfg.get("num_views", 3)),
            fm_hidden=int(tcfg.get("fm_hidden", 512)),
            fm_sample_steps=int(tcfg.get("fm_sample_steps", 10)),
        )
    if model_type == "resnet_film_mse":
        from vla_pipeline.model_resnet_film_mse import ResNetFiLMMSEPolicy

        return ResNetFiLMMSEPolicy(
            action_dim=action_dim,
            image_size=int(tcfg.get("image_size", 128)),
            lang_dim=int(tcfg.get("lang_dim", 256)),
            num_views=int(tcfg.get("num_views", 3)),
            hidden=int(tcfg.get("fm_hidden", 512)),
            dropout=float(tcfg.get("dropout", 0.0)),
        )
    if model_type == "cnn_mse":
        from vla_pipeline.model import CausalWorldBCVLA

        return CausalWorldBCVLA(action_dim=action_dim, image_size=int(tcfg.get("image_size", 128)))
    raise ValueError(f"Unknown model_type: {model_type}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze ResNet stem + layers (keep FiLM/text/flow head trainable)."""
    if not hasattr(model, "vision_enc"):
        return
    enc = model.vision_enc
    for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
        for p in getattr(enc, name).parameters():
            p.requires_grad = trainable


def load_policy_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = {
        "model_type": ckpt.get("model_type", "cnn_mse"),
        "action_dim": ckpt.get("action_dim", 9),
        "action_horizon": ckpt.get("action_horizon", 1),
        "image_size": ckpt.get("image_size", 128),
        "fm_sample_steps": ckpt.get("fm_sample_steps", 10),
    }
    tcfg = {
        "model_type": meta["model_type"],
        "action_horizon": meta["action_horizon"],
        "image_size": meta["image_size"],
        "fm_sample_steps": meta["fm_sample_steps"],
    }
    model = build_policy_from_config(tcfg, meta["action_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, meta
