"""PyTorch dataset for CausalWorld VLA (single-step and action-chunk modes)."""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CausalWorldVLADataset(Dataset):
    def __init__(
        self,
        data_root: str,
        json_file: str,
        image_size: int = 128,
        action_horizon: int = 1,
        augment: bool = False,
        dagger_sample_weight: float = 1.0,
        protocol_sample_weights: Optional[dict] = None,
    ):
        with open(os.path.join(data_root, json_file), encoding="utf-8") as f:
            grouped = json.load(f)
        self.image_size = image_size
        self.action_horizon = max(1, int(action_horizon))
        self.augment = augment
        self.dagger_sample_weight = float(dagger_sample_weight)
        self.protocol_sample_weights = protocol_sample_weights or {}
        self._index: List[tuple] = []  # (traj_idx, start_t)

        for ti, traj in enumerate(grouped):
            if not traj:
                continue
            for start in range(len(traj)):
                self._index.append((ti, start))

        self._trajectories = grouped
        if grouped and grouped[0]:
            self.action_dim = len(grouped[0][0]["action"])
        else:
            self.action_dim = 9

    def __len__(self) -> int:
        return len(self._index)

    def _load_img(self, path: str) -> torch.Tensor:
        if path.endswith(".npy"):
            arr = np.load(path)
        else:
            arr = np.array(Image.open(path).convert("RGB"))
        img = Image.fromarray(arr.astype(np.uint8))
        if self.augment:
            arr = np.array(img).astype(np.float32)
            arr *= np.random.uniform(0.85, 1.15)
            arr += np.random.uniform(-0.08, 0.08)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        img = img.resize((self.image_size, self.image_size))
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def _get_action_chunk(self, traj: list, start: int) -> torch.Tensor:
        h = self.action_horizon
        chunk = []
        for k in range(h):
            idx = min(start + k, len(traj) - 1)
            chunk.append(traj[idx]["action"])
        return torch.tensor(chunk, dtype=torch.float32)

    def _protocol_weight(self, task: str, protocol: str) -> float:
        wmap = self.protocol_sample_weights
        if not wmap:
            return 1.0
        task_map = wmap.get(task)
        if isinstance(task_map, dict):
            return float(task_map.get(protocol, task_map.get("*", 1.0)))
        key = f"{task}/{protocol}"
        if key in wmap:
            return float(wmap[key])
        return float(wmap.get("*", 1.0))

    def __getitem__(self, idx: int):
        ti, start = self._index[idx]
        traj = self._trajectories[ti]
        s = traj[start]

        views = torch.stack(
            [self._load_img(s["left"]), self._load_img(s["center"]), self._load_img(s["right"])],
            dim=0,
        )
        action_chunk = self._get_action_chunk(traj, start)
        weight = float(s.get("fractional_success", 1.0))
        weight = max(weight, 0.05)
        weight *= self._protocol_weight(s["task"], s.get("protocol", ""))
        if s.get("source") == "dagger" and self.dagger_sample_weight > 1.0:
            weight *= self.dagger_sample_weight
        protocol = s.get("protocol", "")
        return views, s["instruction"], action_chunk, s["task"], weight, protocol
