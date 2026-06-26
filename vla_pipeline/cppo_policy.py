"""CPPPO v2: protocol-conditioned fusion MLP for PPO2 (TensorFlow 1.x)."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, mlp_extractor
from stable_baselines.common.tf_layers import linear

from causal_world.wrappers.protocol_obs_utils import NUM_PROTOCOLS


class CpppoFusionPolicy(ActorCriticPolicy):
    """
    Privileged actor-critic with protocol embedding fusion.

    Observation layout: [base_structured_obs | protocol_one_hot (12)]
    """

    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        reuse=False,
        net_arch=None,
        act_fun=tf.tanh,
        embed_dim=64,
        num_protocols=NUM_PROTOCOLS,
        **kwargs,
    ):
        self._cppo_num_protocols = int(num_protocols)
        self._cppo_embed_dim = int(embed_dim)
        self._cppo_base_dim = int(np.prod(ob_space.shape)) - self._cppo_num_protocols
        if self._cppo_base_dim <= 0:
            raise ValueError("CPPPO fusion policy expects obs = base + protocol one-hot")
        if net_arch is None:
            net_arch = [dict(pi=[512, 512], vf=[512, 256])]

        super(CpppoFusionPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            reuse=reuse,
            scale=False,
        )

        with tf.variable_scope("model", reuse=reuse):
            flat = tf.layers.flatten(self.processed_obs)
            base = flat[:, : self._cppo_base_dim]
            proto = flat[:, self._cppo_base_dim :]
            proto_h = tf.nn.relu(
                linear(proto, "cppo_proto_embed", self._cppo_embed_dim, init_scale=np.sqrt(2))
            )
            fused = tf.concat([base, proto_h], axis=1)
            pi_latent, vf_latent = mlp_extractor(fused, net_arch, act_fun)
            self._value_fn = linear(vf_latent, "vf", 1)
            self._proba_distribution, self._policy, self.q_value = (
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
            )
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp],
                {self.obs_ph: obs},
            )
        else:
            action, value, neglogp = self.sess.run(
                [self.action, self.value_flat, self.neglogp],
                {self.obs_ph: obs},
            )
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
