"""Protocol feature augmentation for CPPPO v2 (privileged teacher)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from gym.spaces import Box


NUM_PROTOCOLS = 12
PROTOCOL_NAMES = tuple(f"P{i}" for i in range(NUM_PROTOCOLS))


def protocol_index(name: str) -> int:
    if name in PROTOCOL_NAMES:
        return PROTOCOL_NAMES.index(name)
    if name.startswith("P") and name[1:].isdigit():
        idx = int(name[1:])
        if 0 <= idx < NUM_PROTOCOLS:
            return idx
    return 0


def protocol_one_hot(name: str, n: int = NUM_PROTOCOLS) -> np.ndarray:
    vec = np.zeros(n, dtype=np.float32)
    idx = protocol_index(name)
    if 0 <= idx < n:
        vec[idx] = 1.0
    return vec


def augment_observation(obs: np.ndarray, protocol_name: str) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    return np.concatenate([obs, protocol_one_hot(protocol_name)], axis=0)


def augmented_observation_space(base_space: Box, n_protocols: int = NUM_PROTOCOLS) -> Box:
    base_dim = int(np.prod(base_space.shape))
    low = np.concatenate([base_space.low.flatten(), np.zeros(n_protocols, dtype=base_space.dtype)])
    high = np.concatenate([base_space.high.flatten(), np.ones(n_protocols, dtype=base_space.dtype)])
    return Box(low=low, high=high, dtype=base_space.dtype)


def resolve_protocol_name(env) -> str:
    cur = env
    for _ in range(8):
        if hasattr(cur, "get_active_protocol"):
            return cur.get_active_protocol()
        if hasattr(cur, "protocol") and hasattr(cur.protocol, "get_name"):
            return cur.protocol.get_name()
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    return "P0"
