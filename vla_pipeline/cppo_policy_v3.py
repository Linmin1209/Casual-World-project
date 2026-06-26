"""CPPPO v3: baseline MLP + zero-init protocol residual (dual-path)."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, mlp_extractor
from stable_baselines.common.tf_layers import linear

from causal_world.wrappers.protocol_obs_utils import NUM_PROTOCOLS


class CpppoResidualPolicy(ActorCriticPolicy):
    """
    Privileged actor-critic with protocol residual on top of baseline trunk.

    Observation layout: [base_structured_obs (56) | protocol_one_hot (12)]

    Forward:
      base_pi = MLP_pi(base_obs)          # warm-started from official baseline
      proto_h = relu(embed(proto_onehot)) # zero-init -> no initial protocol effect
      pi_latent = base_pi + proto_scale * proto_pi(proto_h)
      (same structure for vf)

    At init, behaviour matches baseline on every protocol (proto path outputs 0).
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
        proto_scale=1.0,
        **kwargs,
    ):
        self._cppo_num_protocols = int(num_protocols)
        self._cppo_embed_dim = int(embed_dim)
        self._cppo_proto_scale = float(proto_scale)
        self._cppo_base_dim = int(np.prod(ob_space.shape)) - self._cppo_num_protocols
        if self._cppo_base_dim <= 0:
            raise ValueError("CPPPO v3 expects obs = base + protocol one-hot")
        if net_arch is None:
            net_arch = [dict(pi=[256, 256], vf=[256, 256])]

        super(CpppoResidualPolicy, self).__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            reuse=reuse,
            scale=False,
        )

        arch0 = net_arch[0]
        pi_layers = list(arch0.get("pi", [256, 256]))
        vf_layers = list(arch0.get("vf", [256, 256]))
        if not pi_layers or not vf_layers:
            raise ValueError("CPPPO v3 net_arch must define pi and vf layer sizes")

        with tf.variable_scope("model", reuse=reuse):
            flat = tf.layers.flatten(self.processed_obs)
            base = flat[:, : self._cppo_base_dim]
            proto = flat[:, self._cppo_base_dim :]

            proto_h = tf.nn.relu(
                linear(proto, "cppo_proto_embed", self._cppo_embed_dim, init_scale=0.0)
            )
            proto_pi = act_fun(
                linear(proto_h, "cppo_proto_pi", pi_layers[-1], init_scale=0.0)
            )
            proto_vf = act_fun(
                linear(proto_h, "cppo_proto_vf", vf_layers[-1], init_scale=0.0)
            )

            base_pi, _ = mlp_extractor(base, [{"pi": pi_layers, "vf": []}], act_fun)
            _, base_vf = mlp_extractor(base, [{"pi": [], "vf": vf_layers}], act_fun)

            scale = self._cppo_proto_scale
            pi_latent = base_pi + scale * proto_pi
            vf_latent = base_vf + scale * proto_vf

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
