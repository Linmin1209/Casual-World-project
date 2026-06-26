"""Load CausalWorld baseline PPO teachers (privileged structured-obs policies)."""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict, Optional

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

CPPPO_CKPT = {
    "pushing": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/cppo_checkpoints/pushing/pushing_cppo.zip",
    ),
    "picking": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/cppo_checkpoints/picking/picking_cppo.zip",
    ),
    "pick_and_place": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/cppo_checkpoints/pick_and_place/pick_and_place_cppo.zip",
    ),
}

CPPPO_V2_CKPT = {
    "pushing": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/cppo_checkpoints_v2/pushing/pushing_cppo_v2.zip",
    ),
    "picking": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/cppo_checkpoints_v2/picking/picking_cppo_v2.zip",
    ),
    "pick_and_place": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../data/cppo_checkpoints_v2/pick_and_place/pick_and_place_cppo_v2.zip",
    ),
}


def is_cppo_v2_checkpoint(checkpoint: Optional[str]) -> bool:
    if not checkpoint:
        return False
    base = os.path.basename(checkpoint)
    return "cppo_v2" in base


def resolve_teacher_checkpoint(task_name: str, prefer_cppo: bool = True) -> str:
    if not prefer_cppo:
        return os.path.abspath(DEFAULT_CKPT[task_name])
    use_cppo = os.environ.get("CPPPO_TEACHER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    prefer_v2 = os.environ.get("CPPPO_VERSION", "v2").strip().lower() in (
        "v2",
        "2",
        "auto",
    )
    if use_cppo and prefer_v2:
        v2 = CPPPO_V2_CKPT.get(task_name)
        if v2 and os.path.isfile(os.path.abspath(v2)):
            return os.path.abspath(v2)
    if use_cppo:
        v1 = CPPPO_CKPT.get(task_name)
        if v1 and os.path.isfile(os.path.abspath(v1)):
            return os.path.abspath(v1)
    return os.path.abspath(DEFAULT_CKPT[task_name])


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


def load_teacher(
    task_name: str,
    checkpoint: Optional[str] = None,
    prefer_cppo: bool = True,
) -> Callable:
    """Return policy_fn(obs_structured) -> action."""
    PPO2 = _import_ppo2()

    if checkpoint is None and prefer_cppo:
        checkpoint = resolve_teacher_checkpoint(task_name, prefer_cppo=True)

    ckpt = checkpoint or DEFAULT_CKPT[task_name]
    ckpt = os.path.abspath(ckpt)
    if os.path.isfile(ckpt):
        model = PPO2.load(ckpt)

        def policy_fn(obs):
            return model.predict(obs, deterministic=True)[0]

        return policy_fn

    # Fallback: bundled ActorPolicy (loads default zip next to actors)
    return _load_actor(task_name).act


def _is_cppo_checkpoint(checkpoint: str) -> bool:
    base = os.path.basename(checkpoint).lower()
    return "cppo" in base


def _resolve_router_path(router_config: Optional[str]) -> str:
    if router_config:
        return os.path.abspath(router_config)
    env_path = os.environ.get("CPPPO_ROUTER_CONFIG", "").strip()
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)
    default = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "config_cppo_v4_teacher_router.json",
    )
    return os.path.abspath(default)


def _router_checkpoint_for_protocol(task_name: str, protocol: str, router: dict) -> str:
    spec = router["tasks"][task_name]
    override = (spec.get("protocol_overrides") or {}).get(protocol)
    if override:
        ckpt = override["checkpoint"]
    else:
        ckpt = spec["default_checkpoint"]
    if not os.path.isabs(ckpt):
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        ckpt = os.path.join(root, ckpt)
    return os.path.abspath(ckpt)


def load_routed_teacher(
    task_name: str,
    router_config: Optional[str] = None,
) -> Callable:
    """Return policy_fn(obs) routing to per-protocol expert (CPPPO or baseline)."""
    from causal_world.wrappers.protocol_obs_utils import NUM_PROTOCOLS, PROTOCOL_NAMES

    router_path = _resolve_router_path(router_config)
    router = json.loads(open(router_path, encoding="utf-8").read())
    if task_name not in router["tasks"]:
        return load_teacher(task_name)

    PPO2 = _import_ppo2()
    models: Dict[str, object] = {}

    def _get_model(protocol: str):
        ckpt = _router_checkpoint_for_protocol(task_name, protocol, router)
        if ckpt not in models:
            models[ckpt] = PPO2.load(ckpt)
        return models[ckpt], ckpt

    def _protocol_from_obs(obs) -> str:
        obs = obs.reshape(-1)
        if obs.size >= NUM_PROTOCOLS:
            tail = obs[-NUM_PROTOCOLS:]
            if tail.max() > 0.5 and abs(tail.sum() - 1.0) < 0.01:
                return PROTOCOL_NAMES[int(tail.argmax())]
        return "P0"

    def policy_fn(obs, protocol: Optional[str] = None):
        proto = protocol or _protocol_from_obs(obs)
        model, ckpt = _get_model(proto)
        obs_in = obs
        if not _is_cppo_checkpoint(ckpt) and obs.reshape(-1).size > NUM_PROTOCOLS:
            obs_in = obs.reshape(-1)[:-NUM_PROTOCOLS]
        return model.predict(obs_in, deterministic=True)[0]

    return policy_fn
