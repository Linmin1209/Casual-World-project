"""Append protocol one-hot to observations for CPPPO v2 inference/training."""
from __future__ import annotations

import gym
import numpy as np

from causal_world.wrappers.protocol_obs_utils import (
    augment_observation,
    augmented_observation_space,
    resolve_protocol_name,
)


class ProtocolObsWrapper(gym.ObservationWrapper):
    """Concatenate active evaluation-protocol one-hot to structured observations."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.observation_space = augmented_observation_space(env.observation_space)

    def observation(self, observation):
        pname = resolve_protocol_name(self.env)
        return augment_observation(observation, pname)
