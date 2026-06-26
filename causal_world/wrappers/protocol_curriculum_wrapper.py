"""Weighted / adaptive evaluation-protocol curriculum for CPPPO training."""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence

import gym
import numpy as np

from causal_world.wrappers.protocol_obs_utils import (
    augment_observation,
    augmented_observation_space,
    protocol_index,
)


class ProtocolCurriculumWrapper(gym.Wrapper):
    """
    Sample benchmark protocols on reset (static or adaptive weighted).

    v1: fixed protocol_weights
    v2: adaptive reweighting from rolling episode success + obs augmentation
    """

    def __init__(
        self,
        env,
        evaluation_protocols: Sequence,
        protocol_weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
        adaptive_sampling: bool = False,
        adaptive_alpha: float = 1.5,
        adaptive_momentum: float = 0.92,
        hard_protocol_min_weight: float = 0.08,
        stratified_static_ratio: float = 0.30,
        protocol_allowlist: Optional[Sequence[str]] = None,
        anchor_protocols: Optional[Sequence[str]] = None,
        anchor_min_sampling_ratio: float = 0.0,
        augment_obs: bool = False,
        success_key: str = "fractional_success",
    ):
        super().__init__(env)
        self._protocols = list(evaluation_protocols)
        if not self._protocols:
            raise ValueError("evaluation_protocols must be non-empty")
        self._names = [p.get_name() for p in self._protocols]
        self._rng = random.Random(int(seed))
        self._adaptive = bool(adaptive_sampling)
        self._adaptive_alpha = float(adaptive_alpha)
        self._adaptive_momentum = float(adaptive_momentum)
        self._hard_min = float(hard_protocol_min_weight)
        self._static_ratio = float(np.clip(stratified_static_ratio, 0.0, 1.0))
        self._anchor_names = set(anchor_protocols or [])
        self._anchor_min = float(np.clip(anchor_min_sampling_ratio, 0.0, 1.0))
        self._augment_obs = bool(augment_obs)
        self._success_key = success_key

        allow = set(protocol_allowlist or [])
        if allow:
            filtered = [
                (p, n)
                for p, n in zip(self._protocols, self._names)
                if n in allow
            ]
            if not filtered:
                raise ValueError(f"protocol_allowlist {allow} matches no evaluation protocols")
            self._protocols, self._names = zip(*filtered)
            self._protocols = list(self._protocols)
            self._names = list(self._names)

        self._static_weights = self._build_static_weights(protocol_weights)
        self._weights = np.array(self._static_weights, dtype=np.float64)
        self._ema_success = np.full(len(self._protocols), 0.35, dtype=np.float64)
        self._last_episode_return: Deque[float] = deque(maxlen=32)

        self.protocol = self._protocols[0]
        self._active_idx = 0
        self._elapsed_episodes = 0
        self._elapsed_timesteps = 0
        self._episode_return = 0.0

        if self._augment_obs and isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = augmented_observation_space(env.observation_space)

        self.env.add_wrapper_info(
            {
                "cppo_protocol_curriculum": {
                    "protocols": self._names,
                    "adaptive": self._adaptive,
                    "augment_obs": self._augment_obs,
                }
            }
        )

    def _build_static_weights(self, protocol_weights: Optional[Dict[str, float]]) -> List[float]:
        weights = []
        for i, name in enumerate(self._names):
            w = 1.0
            if protocol_weights:
                w = float(protocol_weights.get(name, protocol_weights.get("*", 1.0)))
            # Hard protocols P6+ get a sampling floor in v2.
            if i >= 6 and (self._adaptive or self._augment_obs):
                w = max(w, self._hard_min * len(self._names))
            weights.append(max(w, 1e-6))
        total = sum(weights)
        return [w / total for w in weights]

    def _update_adaptive_weights(self, protocol_idx: int, episode_score: float):
        self._ema_success[protocol_idx] = (
            self._adaptive_momentum * self._ema_success[protocol_idx]
            + (1.0 - self._adaptive_momentum) * float(episode_score)
        )
        difficulty = np.power(np.clip(1.0 - self._ema_success, 0.05, 1.0), self._adaptive_alpha)
        weights = np.array(self._static_weights, dtype=np.float64) * difficulty
        # Keep hard protocols from starving.
        for i in range(6, len(weights)):
            weights[i] = max(weights[i], self._hard_min)
        weights /= weights.sum()
        self._weights = weights

    def _sample_protocol_index(self) -> int:
        if self._anchor_min > 0 and self._anchor_names:
            if self._rng.random() < self._anchor_min:
                anchor_idx = [
                    i for i, name in enumerate(self._names) if name in self._anchor_names
                ]
                if anchor_idx:
                    anchor_w = [self._static_weights[i] for i in anchor_idx]
                    s = sum(anchor_w)
                    anchor_w = [w / s for w in anchor_w]
                    return int(self._rng.choices(anchor_idx, weights=anchor_w, k=1)[0])

        if not self._adaptive:
            return int(
                self._rng.choices(range(len(self._protocols)), weights=self._weights.tolist(), k=1)[0]
            )
        if self._rng.random() < self._static_ratio:
            return int(
                self._rng.choices(range(len(self._protocols)), weights=self._static_weights, k=1)[0]
            )
        return int(
            self._rng.choices(range(len(self._protocols)), weights=self._weights.tolist(), k=1)[0]
        )

    def _maybe_augment(self, observation, protocol_name: str):
        if not self._augment_obs:
            return observation
        return augment_observation(observation, protocol_name)

    def _activate_protocol(self, protocol_idx: int):
        self._active_idx = int(protocol_idx)
        self.protocol = self._protocols[self._active_idx]
        self.protocol.init_protocol(
            env=self.env,
            tracker=self.env.get_tracker(),
            fraction=1.0,
        )
        self.env.add_wrapper_info(
            {"evaluation_environment": self.protocol.get_name()}
        )

    def _apply_reset_intervention(self, observation):
        invalid_interventions = 0
        interventions_dict = self.protocol.get_intervention(
            episode=self._elapsed_episodes, timestep=0
        )
        if interventions_dict is None:
            return observation
        success_signal, observation = self.env.do_intervention(interventions_dict)
        while not success_signal and invalid_interventions < 5:
            invalid_interventions += 1
            interventions_dict = self.protocol.get_intervention(
                episode=self._elapsed_episodes, timestep=0
            )
            if interventions_dict is not None:
                success_signal, observation = self.env.do_intervention(
                    interventions_dict
                )
            else:
                break
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_timesteps += 1
        self._episode_return += float(reward)
        invalid_interventions = 0
        interventions_dict = self.protocol.get_intervention(
            episode=self._elapsed_episodes, timestep=self._elapsed_episodes
        )
        if interventions_dict is not None:
            success_signal, observation = self.env.do_intervention(
                interventions_dict=interventions_dict
            )
            while not success_signal and invalid_interventions < 5:
                invalid_interventions += 1
                interventions_dict = self.protocol.get_intervention(
                    episode=self._elapsed_episodes,
                    timestep=self._elapsed_episodes,
                )
                if interventions_dict is not None:
                    success_signal, observation = self.env.do_intervention(
                        interventions_dict=interventions_dict
                    )
                else:
                    break
        info = dict(info or {})
        info["cppo_protocol"] = self.protocol.get_name()
        if self._success_key in info:
            self._last_episode_return.append(float(info[self._success_key]))
        observation = self._maybe_augment(observation, self.protocol.get_name())
        if done and self._adaptive:
            score = float(info.get(self._success_key, self._episode_return))
            self._update_adaptive_weights(self._active_idx, score)
        return observation, reward, done, info

    def reset(self):
        if self._adaptive and self._elapsed_episodes > 0:
            score = float(np.mean(self._last_episode_return)) if self._last_episode_return else 0.0
            self._update_adaptive_weights(self._active_idx, score)
            self._last_episode_return.clear()

        self._elapsed_episodes += 1
        self._elapsed_timesteps = 0
        self._episode_return = 0.0
        pidx = self._sample_protocol_index()
        self._activate_protocol(pidx)
        observation = self.env.reset()
        observation = self._apply_reset_intervention(observation)
        observation = self._maybe_augment(observation, self.protocol.get_name())
        return observation

    def get_active_protocol(self) -> str:
        return self.protocol.get_name()

    def get_sampling_stats(self) -> dict:
        return {
            "ema_success": {
                self._names[i]: float(self._ema_success[i]) for i in range(len(self._names))
            },
            "weights": {
                self._names[i]: float(self._weights[i]) for i in range(len(self._names))
            },
        }
