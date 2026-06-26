"""Hybrid policy: VLA + privileged PPO teacher with protocol-aware residual blending."""
from __future__ import annotations

import os
from collections import deque
from typing import Callable, Deque, Dict, Optional

import numpy as np
import torch

from vla_pipeline.camera_utils import ensure_tool_cameras
from vla_pipeline.label_utils import build_instruction
from vla_pipeline.model_resnet_film_flow import load_policy_from_checkpoint
from vla_pipeline.torch_compat import resolve_vla_device

EASY_PROTOCOLS = frozenset({"P0", "P1", "P2", "P3", "P4", "P5"})


def effective_hybrid_beta(
    hybrid_cfg: dict,
    task_name: str,
    protocol_name: str,
) -> float:
    """Resolve residual blend weight for (task, protocol)."""
    if not hybrid_cfg:
        return 0.0
    task_map = (hybrid_cfg.get("beta_by_task_protocol") or {}).get(task_name) or {}
    if protocol_name in task_map:
        return float(task_map[protocol_name])
    beta_map = hybrid_cfg.get("beta_by_protocol") or {}
    if protocol_name in beta_map:
        return float(beta_map[protocol_name])
    alpha_map = hybrid_cfg.get("alpha_by_protocol") or {}
    if protocol_name in alpha_map:
        return 1.0 - float(alpha_map[protocol_name])
    if protocol_name in EASY_PROTOCOLS:
        return float(hybrid_cfg.get("beta_easy", 0.0))
    if hybrid_cfg.get("blend_mode", "residual") == "residual":
        return float(hybrid_cfg.get("beta_default", 0.25))
    alpha_map_t = hybrid_cfg.get("alpha_by_task") or {}
    alpha = float(
        alpha_map_t.get(
            task_name,
            hybrid_cfg.get("alpha_default", hybrid_cfg.get("alpha", 0.35)),
        )
    )
    return 1.0 - alpha


def build_hybrid_policy(hybrid_cfg, teacher_fn, task_name: str, image_size: int, device=None):
    alpha_map = hybrid_cfg.get("alpha_by_task") or {}
    alpha = float(
        alpha_map.get(
            task_name,
            hybrid_cfg.get("alpha_default", hybrid_cfg.get("alpha", 0.35)),
        )
    )
    ckpt_by_tp = hybrid_cfg.get("checkpoint_by_task_protocol") or {}
    for task, proto_map in ckpt_by_tp.items():
        for proto, path in proto_map.items():
            ckpt_by_tp[task][proto] = os.path.abspath(path)
    return HybridVLAPolicy(
        vla_checkpoint=os.path.abspath(hybrid_cfg["checkpoint"]),
        teacher_fn=teacher_fn,
        alpha=alpha,
        image_size=image_size,
        device=device,
        blend_mode=str(hybrid_cfg.get("blend_mode", "residual")),
        beta_default=float(hybrid_cfg.get("beta_default", 0.25)),
        beta_easy=float(hybrid_cfg.get("beta_easy", 0.0)),
        beta_by_protocol=hybrid_cfg.get("beta_by_protocol"),
        beta_by_task_protocol=hybrid_cfg.get("beta_by_task_protocol"),
        alpha_by_protocol=hybrid_cfg.get("alpha_by_protocol"),
        max_correction=hybrid_cfg.get("max_correction"),
        checkpoint_by_task_protocol=ckpt_by_tp,
        vla_gpu_port=hybrid_cfg.get("vla_gpu_port"),
    )


class HybridVLAPolicy:
    """
    Residual blend (default):
      action = teacher + beta * (vla - teacher)

    P0-P5: beta=0 -> pure teacher (matches baseline on easy protocols).
    P6-P11: small beta lets VLA nudge teacher without destroying good actions.
    """

    def __init__(
        self,
        vla_checkpoint: str,
        teacher_fn: Callable,
        alpha: float = 0.35,
        image_size: int = 128,
        device: Optional[str] = None,
        blend_mode: str = "residual",
        beta_default: float = 0.25,
        beta_easy: float = 0.0,
        beta_by_protocol: Optional[Dict[str, float]] = None,
        beta_by_task_protocol: Optional[Dict[str, Dict[str, float]]] = None,
        alpha_by_protocol: Optional[Dict[str, float]] = None,
        max_correction: Optional[float] = None,
        checkpoint_by_task_protocol: Optional[Dict[str, Dict[str, str]]] = None,
        vla_gpu_port: Optional[int] = None,
    ):
        self.teacher_fn = teacher_fn
        self.alpha = float(alpha)
        self.blend_mode = blend_mode
        self.beta_default = float(beta_default)
        self.beta_easy = float(beta_easy)
        self.beta_by_protocol = beta_by_protocol or {}
        self.beta_by_task_protocol = beta_by_task_protocol or {}
        self.alpha_by_protocol = alpha_by_protocol or {}
        self.max_correction = float(max_correction) if max_correction is not None else None
        self._remote_client = None
        port = vla_gpu_port
        if port is None:
            env_port = os.environ.get("VLA_GPU_PORT", "").strip()
            if env_port:
                port = int(env_port)
        if port is not None:
            from vla_pipeline.vla_gpu_client import VlaGpuClient

            self._remote_client = VlaGpuClient(port=int(port))
            self.device = torch.device("cpu")
        else:
            self.device = resolve_vla_device(device)
        self.default_checkpoint = vla_checkpoint
        self.checkpoint_by_task_protocol = checkpoint_by_task_protocol or {}

        self._model_cache: Dict[str, torch.nn.Module] = {}
        self._meta_cache: Dict[str, dict] = {}
        if self._remote_client is None:
            meta = self._load_checkpoint(vla_checkpoint)[1]
            self.image_size = meta.get("image_size", image_size)
            self.action_horizon = int(meta.get("action_horizon", 1))
        else:
            self.image_size = image_size
            self.action_horizon = 1

        self._action_queue: Deque[np.ndarray] = deque()
        self._task_name = None
        self._protocol = None
        self._env = None
        self._active_checkpoint: Optional[str] = None

    def _load_checkpoint(self, path: str):
        if path not in self._model_cache:
            model, meta = load_policy_from_checkpoint(path, self.device)
            self._model_cache[path] = model
            self._meta_cache[path] = meta
        return self._model_cache[path], self._meta_cache[path]

    def _resolve_checkpoint(self) -> str:
        if self._task_name and self._protocol:
            override = (self.checkpoint_by_task_protocol.get(self._task_name) or {}).get(
                self._protocol
            )
            if override:
                return override
        return self.default_checkpoint

    @property
    def model(self) -> torch.nn.Module:
        if self._remote_client is not None:
            raise RuntimeError("VLA runs on GPU sidecar; no local model")
        ckpt = self._resolve_checkpoint()
        if ckpt != self._active_checkpoint:
            self._action_queue.clear()
            self._active_checkpoint = ckpt
        return self._load_checkpoint(ckpt)[0]

    def _mix_weight(self) -> float:
        """Weight on VLA correction (residual) or (1-alpha) in linear blend."""
        if self._task_name and self._protocol:
            return effective_hybrid_beta(
                {
                    "beta_by_task_protocol": self.beta_by_task_protocol,
                    "beta_by_protocol": self.beta_by_protocol,
                    "alpha_by_protocol": self.alpha_by_protocol,
                    "beta_easy": self.beta_easy,
                    "beta_default": self.beta_default,
                    "blend_mode": self.blend_mode,
                    "alpha_by_task": {},
                    "alpha_default": self.alpha,
                },
                self._task_name,
                self._protocol,
            )
        if self._protocol in self.beta_by_protocol:
            return float(self.beta_by_protocol[self._protocol])
        if self._protocol in self.alpha_by_protocol:
            return 1.0 - float(self.alpha_by_protocol[self._protocol])
        if self._protocol in EASY_PROTOCOLS:
            return self.beta_easy
        return self.beta_default if self.blend_mode == "residual" else (1.0 - self.alpha)

    def bind_env(self, env, task_name: str, protocol_name: str):
        self._env = env
        self._task_name = task_name
        self._protocol = protocol_name
        self._action_queue.clear()

    def _views_tensor(self, robot) -> torch.Tensor:
        from vla_pipeline.camera_utils import capture_tool_camera_rgb_uint8

        ensure_tool_cameras(self._env)
        imgs = capture_tool_camera_rgb_uint8(self._env)
        tensors = []
        for img in imgs[:3]:
            arr = np.asarray(img, dtype=np.uint8)
            from PIL import Image

            im = Image.fromarray(arr).resize((self.image_size, self.image_size))
            t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        while len(tensors) < 3:
            tensors.append(tensors[-1])
        return torch.stack(tensors, dim=0).unsqueeze(0)

    def _plan_chunk(self) -> None:
        state = self._env.get_current_state_variables()
        instr = build_instruction(self._task_name, state, protocol=self._protocol)
        if self._remote_client is not None:
            views = self._views_tensor(self._env._robot).numpy()
            ckpt = self._resolve_checkpoint()
            chunk = self._remote_client.infer(ckpt, views, instr)
            if chunk.ndim == 3:
                chunk = chunk[0]
        else:
            views = self._views_tensor(self._env._robot).to(self.device)
            chunk = self.model.sample_action_chunk(views, [instr]).cpu().numpy()[0]
        chunk = np.clip(chunk, -1.0, 1.0)
        for h in range(chunk.shape[0]):
            self._action_queue.append(chunk[h].astype(np.float32))

    def __call__(self, obs_structured: np.ndarray) -> np.ndarray:
        teacher_action = np.asarray(self.teacher_fn(obs_structured), dtype=np.float32)
        if self._env is None:
            return teacher_action

        w_vla = self._mix_weight()
        if w_vla <= 1e-6:
            return np.clip(teacher_action, -1.0, 1.0).astype(np.float32)

        if not self._action_queue:
            self._plan_chunk()
        vla_action = self._action_queue.popleft()

        delta = vla_action - teacher_action
        if self.max_correction is not None and np.linalg.norm(delta) > self.max_correction:
            return np.clip(teacher_action, -1.0, 1.0).astype(np.float32)

        if self.blend_mode == "residual":
            blended = teacher_action + w_vla * delta
        else:
            blended = w_vla * vla_action + (1.0 - w_vla) * teacher_action
        return np.clip(blended, -1.0, 1.0).astype(np.float32)
