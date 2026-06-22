"""Load CausalWorld baseline PPO teachers (privileged structured-obs policies)."""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

TEACHER_PIP_HINT = (
    "Install teacher deps in causal_world (Python 3.7 only):\n"
    "  pip install -r vla_pipeline/requirements_teacher.txt\n"
    "Expected: stable-baselines==2.10.2 + tensorflow==1.15.5 (TF1, not TF2)."
)

ACTOR_BY_TASK = {
    "pushing": "causal_world.actors.pushing_policy.PushingActorPolicy",
    "picking": "causal_world.actors.picking_policy.PickingActorPolicy",
    "pick_and_place": "causal_world.actors.pick_and_place_policy.PickAndPlaceActorPolicy",
}

DEFAULT_CKPT = {
    "pushing": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../causal_world/assets/baseline_actors/pushing_ppo_curr1.zip",
    ),
    "picking": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../causal_world/assets/baseline_actors/picking_ppo_curr1.zip",
    ),
    "pick_and_place": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../causal_world/assets/baseline_actors/pick_and_place_ppo_curr0.zip",
    ),
}


def _import_ppo2():
    try:
        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        from stable_baselines import PPO2
    except ImportError as exc:
        raise ImportError(TEACHER_PIP_HINT) from exc
    if tf.__version__.split(".")[0] != "1":
        raise ImportError(
            f"Teacher checkpoints need TensorFlow 1.x, got {tf.__version__}.\n"
            + TEACHER_PIP_HINT
        )
    return PPO2


def _load_actor(task_name: str):
    module_path, cls_name = ACTOR_BY_TASK[task_name].rsplit(".", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


def load_teacher(task_name: str, checkpoint: Optional[str] = None) -> Callable:
    """Return policy_fn(obs_structured) -> action."""
    PPO2 = _import_ppo2()

    ckpt = checkpoint or DEFAULT_CKPT[task_name]
    ckpt = os.path.abspath(ckpt)
    if os.path.isfile(ckpt):
        model = PPO2.load(ckpt)

        def policy_fn(obs):
            return model.predict(obs, deterministic=True)[0]

        return policy_fn

    # Fallback: bundled ActorPolicy (loads default zip next to actors)
    return _load_actor(task_name).act
