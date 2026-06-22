"""On-policy rollouts for VLA RL fine-tuning on CausalWorld hard protocols."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from vla_pipeline.camera_utils import capture_tool_camera_rgb_uint8, ensure_tool_cameras
from vla_pipeline.label_utils import build_instruction


@dataclass
class RLTransition:
    views: torch.Tensor  # 3,H,W float
    instruction: str
    action: np.ndarray
    teacher_action: np.ndarray
    step_fs: float


@dataclass
class RLEpisode:
    task: str
    protocol: str
    score: float
    transitions: List[RLTransition] = field(default_factory=list)


def _views_tensor(env, image_size: int, device: torch.device) -> torch.Tensor:
    ensure_tool_cameras(env)
    imgs = capture_tool_camera_rgb_uint8(env)
    tensors = []
    from PIL import Image

    for img in imgs[:3]:
        im = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((image_size, image_size))
        t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    while len(tensors) < 3:
        tensors.append(tensors[-1])
    return torch.stack(tensors, dim=0).to(device)


def make_env(task_name: str, benchmark: dict, protocol_idx: int, skip_frame: int, seed: int):
    from causal_world.envs.causalworld import CausalWorld
    from causal_world.task_generators.task import generate_task
    from causal_world.wrappers.protocol_wrapper import ProtocolWrapper

    protocol = benchmark["evaluation_protocols"][protocol_idx]
    task = generate_task(task_generator_id=task_name, variables_space="space_a_b")
    env = CausalWorld(
        task=task,
        enable_visualization=False,
        seed=seed,
        skip_frame=skip_frame,
        observation_mode="structured",
        normalize_observations=True,
        normalize_actions=True,
    )
    wrapped = ProtocolWrapper(env, protocol)
    protocol.init_protocol(env=env, tracker=env.get_tracker(), fraction=1.0)
    return env, wrapped, protocol


def rollout_episode(
    task_name: str,
    benchmark: dict,
    protocol_idx: int,
    model: torch.nn.Module,
    teacher_fn: Callable,
    device: torch.device,
    image_size: int,
    skip_frame: int = 10,
    seed: int = 0,
    explore_std: float = 0.08,
) -> RLEpisode:
    """Stochastic Gaussian exploration around VLA mean action."""
    env, wrapped, protocol = make_env(
        task_name, benchmark, protocol_idx, skip_frame, seed
    )
    pname = protocol.get_name()
    transitions: List[RLTransition] = []
    fs_vals: List[float] = []

    obs = wrapped.reset()
    done = False
    model.eval()
    with torch.no_grad():
        while not done:
            views = _views_tensor(env, image_size, device).unsqueeze(0)
            instr = build_instruction(
                task_name, env.get_current_state_variables(), protocol=pname
            )
            mu = model.predict_action(views, [instr])
            if explore_std > 0:
                noise = torch.randn_like(mu) * explore_std
                action_t = torch.clamp(mu + noise, -1.0, 1.0)
            else:
                action_t = mu
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            teacher_a = np.asarray(teacher_fn(obs), dtype=np.float32)
            obs, _rew, done, info = wrapped.step(action)
            fs = float(info.get("fractional_success", 0.0))
            fs_vals.append(fs)
            transitions.append(
                RLTransition(
                    views=views.squeeze(0).cpu(),
                    instruction=instr,
                    action=action,
                    teacher_action=teacher_a,
                    step_fs=fs,
                )
            )

    score = float(np.mean(fs_vals)) if fs_vals else 0.0
    try:
        wrapped.close()
    except Exception:
        pass
    return RLEpisode(task=task_name, protocol=pname, score=score, transitions=transitions)


def sample_protocol_jobs(
    tasks: List[str],
    protocols_by_task: Dict[str, List[str]],
    benchmarks: Dict[str, dict],
    episodes_per_iter: int,
    rng: np.random.RandomState,
) -> List[Tuple[str, int]]:
    jobs: List[Tuple[str, int]] = []
    pool: List[Tuple[str, int]] = []
    for task in tasks:
        bench = benchmarks[task]
        names = protocols_by_task.get(task, [])
        for pname in names:
            idx = next(
                i
                for i, p in enumerate(bench["evaluation_protocols"])
                if p.get_name() == pname
            )
            pool.append((task, idx))
    if not pool:
        raise ValueError("No RL protocol jobs configured")
    for _ in range(episodes_per_iter):
        jobs.append(pool[int(rng.randint(len(pool)))])
    return jobs
