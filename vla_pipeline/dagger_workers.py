"""Multiprocessing workers for DAgger data collection."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vla_pipeline.quiet_utils import setup_quiet_env, suppress_stdout_stderr


def run_dagger_episode_range(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect DAgger episodes: rollout hybrid, label with teacher."""
    gpu_id = payload.get("gpu_id")
    if gpu_id is not None and int(gpu_id) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    setup_quiet_env()
    if payload.get("quiet", True):
        logging.getLogger().setLevel(logging.ERROR)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from causal_world.benchmark import (
        PICKING_BENCHMARK,
        PICK_AND_PLACE_BENCHMARK,
        PUSHING_BENCHMARK,
    )
    from causal_world.envs.causalworld import CausalWorld
    from causal_world.task_generators.task import generate_task
    from causal_world.wrappers.protocol_wrapper import ProtocolWrapper

    from vla_pipeline.collect_workers import _save_rgb
    from vla_pipeline.hybrid_policy import build_hybrid_policy
    from vla_pipeline.label_utils import build_instruction
    from vla_pipeline.teacher import load_teacher

    benchmarks = {
        "pushing": PUSHING_BENCHMARK,
        "picking": PICKING_BENCHMARK,
        "pick_and_place": PICK_AND_PLACE_BENCHMARK,
    }

    task_name = payload["task_name"]
    benchmark = benchmarks[task_name]
    protocol = benchmark["evaluation_protocols"][payload["protocol_idx"]]
    pname = protocol.get_name()

    round_id = int(payload["round_id"])
    ep_start = int(payload["episode_start"])
    ep_count = int(payload["episode_count"])
    skip_frame = int(payload.get("skip_frame", 3))
    seed = int(payload.get("seed", 0))
    out_root = Path(payload["out_root"]) / f"dagger/r{round_id:02d}" / task_name / pname

    teacher_ckpt = payload.get("teacher_ckpt")
    with suppress_stdout_stderr():
        teacher_fn = load_teacher(task_name, checkpoint=teacher_ckpt)
        hybrid = build_hybrid_policy(
            payload["hybrid_cfg"],
            teacher_fn,
            task_name,
            int(payload["image_size"]),
            device="cuda" if gpu_id is not None and int(gpu_id) >= 0 else "cpu",
        )

    task = generate_task(task_generator_id=task_name, variables_space="space_a_b")
    with suppress_stdout_stderr():
        env = CausalWorld(
            task=task,
            enable_visualization=False,
            seed=seed,
            skip_frame=skip_frame,
            observation_mode="structured",
            normalize_observations=True,
            normalize_actions=True,
        )
        protocol.init_protocol(env=env, tracker=env.get_tracker(), fraction=1.0)
        wrapped = ProtocolWrapper(env, protocol)
        hybrid.bind_env(env, task_name, pname)

    manifest: List[dict] = []
    n_done = n_skipped = 0
    log_every = int(payload.get("log_every_episodes", 5))

    for ep in range(ep_start, ep_start + ep_count):
        ep_dir = out_root / f"ep_{ep:04d}"
        if payload.get("skip_existing") and (ep_dir / "episode.json").is_file():
            n_skipped += 1
            continue

        wrapped._elapsed_episodes = ep
        t0 = time.time()
        obs = wrapped.reset()
        frames, actions, instructions, infos = [], [], [], []
        done = False
        step_idx = 0
        while not done:
            rollout_action = np.asarray(hybrid(obs), dtype=np.float32)
            teacher_action = np.asarray(teacher_fn(obs), dtype=np.float32)
            cam_paths = _save_rgb(env, ep_dir / "rgb", step_idx)
            instr = build_instruction(
                task_name, env.get_current_state_variables(), protocol=pname
            )
            obs, rew, done, info = wrapped.step(rollout_action)
            frames.append(cam_paths)
            actions.append(teacher_action.tolist())
            instructions.append(instr)
            infos.append(
                {
                    "reward": float(rew),
                    "fractional_success": float(info.get("fractional_success", 0.0)),
                    "success": bool(info.get("success", False)),
                    "rollout_action": rollout_action.tolist(),
                }
            )
            step_idx += 1

        final_fs = infos[-1]["fractional_success"] if infos else 0.0
        record = {
            "task": task_name,
            "protocol": pname,
            "episode": ep,
            "steps": step_idx,
            "final_fractional_success": final_fs,
            "frames": frames,
            "actions": actions,
            "instructions": instructions,
            "infos": infos,
            "source": "dagger",
            "dagger_round": round_id,
            "episode_dir": str(ep_dir),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        ep_dir.mkdir(parents=True, exist_ok=True)
        with open(ep_dir / "episode.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        manifest.append(record)
        n_done += 1
        if log_every > 0 and (n_done % log_every == 0 or ep + 1 == ep_start + ep_count):
            print(
                f"  [dagger r{round_id:02d}] {task_name}/{pname} "
                f"ep {ep + 1}/{ep_start + ep_count} steps={step_idx} fs={final_fs:.3f}",
                flush=True,
            )

    try:
        env.close()
    except Exception:
        pass

    return {
        "task_name": task_name,
        "protocol": pname,
        "round_id": round_id,
        "episode_start": ep_start,
        "episode_count": ep_count,
        "n_done": n_done,
        "n_skipped": n_skipped,
        "manifest": manifest,
    }
