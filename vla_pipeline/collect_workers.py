"""Multiprocessing workers for parallel teacher demo collection."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from vla_pipeline.quiet_utils import setup_quiet_env, suppress_stdout_stderr


def _save_rgb(env, out_dir: Path, step_idx: int) -> list:
    from vla_pipeline.camera_utils import capture_tool_camera_rgb_uint8

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = capture_tool_camera_rgb_uint8(env)
    paths = []
    for cam_i, img in enumerate(imgs):
        path = out_dir / f"step_{step_idx:05d}_cam{cam_i}.png"
        arr = np.asarray(img, dtype=np.uint8)
        try:
            from PIL import Image

            Image.fromarray(arr).save(path)
        except ImportError:
            np.save(path.with_suffix(".npy"), arr)
            path = path.with_suffix(".npy")
        paths.append(str(path))
    return paths


def collect_episode_range(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Collect a contiguous range of episodes in an isolated process."""
    gpu_id = payload.get("gpu_id")
    if gpu_id is not None and int(gpu_id) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    setup_quiet_env()
    if payload.get("quiet_intervention_logs", True):
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

    from vla_pipeline.camera_utils import ensure_tool_cameras
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

    out_root = Path(payload["out_root"])
    min_success = float(payload["min_success"])
    skip_existing = bool(payload.get("skip_existing", False))
    log_every = int(payload.get("log_every_steps", 0))
    ep_start = int(payload["episode_start"])
    ep_count = int(payload["episode_count"])
    seed = int(payload.get("seed", 0))

    teacher_ckpt = payload.get("teacher_ckpt")
    with suppress_stdout_stderr():
        teacher_fn = load_teacher(task_name, checkpoint=teacher_ckpt)

    task = generate_task(task_generator_id=task_name, variables_space="space_a_b")
    env = CausalWorld(
        task=task,
        enable_visualization=False,
        seed=seed,
        observation_mode="structured",
        normalize_observations=True,
        normalize_actions=True,
    )
    protocol.init_protocol(env=env, tracker=env.get_tracker(), fraction=1.0)
    ensure_tool_cameras(env)
    wrapped_env = ProtocolWrapper(env, protocol)

    manifest: List[dict] = []
    n_skipped = n_done = n_usable = 0
    min_usable_fs = float(payload.get("min_usable_episode_max_fs", 0.05))

    for ep in range(ep_start, ep_start + ep_count):
        ep_dir = out_root / task_name / pname / f"ep_{ep:04d}"
        if skip_existing and (
            (ep_dir / "episode.json").is_file()
            or (ep_dir / "episode_rejected.json").is_file()
        ):
            n_skipped += 1
            continue

        t0 = time.time()
        obs = wrapped_env.reset()
        frames, actions, instructions, infos = [], [], [], []
        done = False
        step_idx = 0
        while not done:
            state_vars = env.get_current_state_variables()
            action = teacher_fn(obs)
            cam_paths = _save_rgb(env, ep_dir / "rgb", step_idx)
            instr = build_instruction(task_name, state_vars, protocol=pname)
            frames.append(cam_paths)
            actions.append(np.asarray(action, dtype=np.float32).tolist())
            instructions.append(instr)
            obs, rew, done, info = wrapped_env.step(action)
            infos.append(
                {
                    "reward": float(rew),
                    "fractional_success": float(info.get("fractional_success", 0.0)),
                    "success": bool(info.get("success", False)),
                }
            )
            step_idx += 1

        final_fs = infos[-1]["fractional_success"] if infos else 0.0
        ep_max_fs = max((float(i.get("fractional_success", 0.0)) for i in infos), default=0.0)
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
            "episode_dir": str(ep_dir),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        if final_fs >= min_success:
            manifest.append(record)
            with open(ep_dir / "episode.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        else:
            record["rejected_for_bc"] = True
            with open(ep_dir / "episode_rejected.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        if ep_max_fs >= min_usable_fs:
            n_usable += 1
        n_done += 1

    try:
        env.close()
    except Exception:
        pass

    return {
        "task_name": task_name,
        "protocol": pname,
        "episode_start": ep_start,
        "episode_count": ep_count,
        "n_done": n_done,
        "n_skipped": n_skipped,
        "n_accepted": len(manifest),
        "n_usable": n_usable,
        "manifest": manifest,
    }


def split_episode_jobs(n_episodes: int, num_workers: int) -> List[tuple]:
    """Return [(episode_start, count), ...]."""
    if n_episodes <= 0:
        return []
    num_workers = max(1, min(num_workers, n_episodes))
    base = n_episodes // num_workers
    rem = n_episodes % num_workers
    jobs = []
    start = 0
    for i in range(num_workers):
        count = base + (1 if i < rem else 0)
        if count > 0:
            jobs.append((start, count))
            start += count
    return jobs
