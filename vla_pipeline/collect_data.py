#!/usr/bin/env python3
"""
Collect teacher demonstrations for VLA BC pretraining.

Supports parallel collection via data.num_workers (>1): each worker owns a
PyBullet env and collects a slice of episodes.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import yaml

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from causal_world.benchmark import (
    PICKING_BENCHMARK,
    PICK_AND_PLACE_BENCHMARK,
    PUSHING_BENCHMARK,
)
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
from causal_world.wrappers.protocol_wrapper import ProtocolWrapper

from vla_pipeline.camera_utils import ensure_tool_cameras
from vla_pipeline.collect_workers import collect_episode_range, split_episode_jobs
from vla_pipeline.label_utils import build_instruction
from vla_pipeline.teacher import load_teacher

BENCHMARK_BY_TASK = {
    "pushing": PUSHING_BENCHMARK,
    "picking": PICKING_BENCHMARK,
    "pick_and_place": PICK_AND_PLACE_BENCHMARK,
}


def _protocol_by_name(benchmark, names):
    all_p = benchmark["evaluation_protocols"]
    wanted = set(names)
    return [p for p in all_p if p.get_name() in wanted]


def _protocol_index(benchmark, protocol) -> int:
    for i, p in enumerate(benchmark["evaluation_protocols"]):
        if p.get_name() == protocol.get_name():
            return i
    raise ValueError(f"protocol {protocol.get_name()} not in benchmark")


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


def _quiet_intervention_warnings(quiet: bool) -> None:
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)


def _is_hard_protocol(name: str) -> bool:
    return name in {"P6", "P7", "P8", "P9", "P10", "P11"}


def _resolve_gpu_ids(num_workers: int, gpu_ids_cfg) -> List[Optional[int]]:
    if gpu_ids_cfg is not None and len(gpu_ids_cfg) == 0:
        return [None] * num_workers
    if gpu_ids_cfg:
        ids = [int(g) for g in gpu_ids_cfg]
    else:
        ids = [None]
    return [ids[i % len(ids)] for i in range(num_workers)]


def collect_for_protocol_serial(
    task_name,
    protocol,
    teacher_fn,
    out_root,
    episodes,
    min_success,
    seed,
    log_every_steps=25,
    skip_existing=False,
):
    import numpy as np

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

    manifest = []
    ep_iter = range(episodes)
    if tqdm is not None:
        ep_iter = tqdm(ep_iter, desc=f"{task_name}/{protocol.get_name()}", unit="ep")

    for ep in ep_iter:
        ep_dir = out_root / task_name / protocol.get_name() / f"ep_{ep:04d}"
        if skip_existing and (
            (ep_dir / "episode.json").is_file()
            or (ep_dir / "episode_rejected.json").is_file()
        ):
            continue
        obs = wrapped_env.reset()
        frames, actions, instructions, infos = [], [], [], []
        done = False
        step_idx = 0
        while not done:
            state_vars = env.get_current_state_variables()
            action = teacher_fn(obs)
            cam_paths = _save_rgb(env, ep_dir / "rgb", step_idx)
            instr = build_instruction(
                task_name, state_vars, protocol=protocol.get_name()
            )
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
        record = {
            "task": task_name,
            "protocol": protocol.get_name(),
            "episode": ep,
            "steps": step_idx,
            "final_fractional_success": final_fs,
            "frames": frames,
            "actions": actions,
            "instructions": instructions,
            "infos": infos,
            "episode_dir": str(ep_dir),
        }
        if final_fs >= min_success:
            manifest.append(record)
            with open(ep_dir / "episode.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        else:
            record["rejected_for_bc"] = True
            with open(ep_dir / "episode_rejected.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

    env.close()
    return manifest


def collect_parallel(
    tasks,
    protocol_names,
    out_root,
    episodes,
    min_success,
    num_workers,
    gpu_ids_cfg,
    cfg,
    log_every_steps,
    skip_existing,
    quiet_intervention_logs,
    collection_seed_offset=0,
):
    jobs = []
    seed = int(collection_seed_offset)
    for task_name in tasks:
        benchmark = BENCHMARK_BY_TASK[task_name]
        protocols = _protocol_by_name(benchmark, protocol_names)
        ckpt = (cfg.get("teacher", {}).get("checkpoints", {}) or {}).get(task_name)
        if ckpt:
            ckpt = os.path.abspath(ckpt)
        for protocol in protocols:
            pidx = _protocol_index(benchmark, protocol)
            for ep_start, ep_count in split_episode_jobs(episodes, num_workers):
                jobs.append(
                    {
                        "task_name": task_name,
                        "protocol_idx": pidx,
                        "episode_start": ep_start,
                        "episode_count": ep_count,
                        "out_root": str(out_root),
                        "min_success": min_success,
                        "seed": seed + ep_start,
                        "log_every_steps": log_every_steps,
                        "skip_existing": skip_existing,
                        "quiet_intervention_logs": quiet_intervention_logs,
                        "teacher_ckpt": ckpt,
                    }
                )
            seed += 1

    gpu_ids = _resolve_gpu_ids(max(1, num_workers), gpu_ids_cfg)
    for i, job in enumerate(jobs):
        job["gpu_id"] = gpu_ids[i % len(gpu_ids)]

    total_jobs = len(jobs)
    print(
        f"[collect] parallel: {total_jobs} jobs, "
        f"{num_workers} concurrent envs",
        flush=True,
    )

    all_manifest = []
    done_jobs = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(collect_episode_range, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            res = fut.result()
            all_manifest.extend(res["manifest"])
            done_jobs += 1
            print(
                f"[collect] {done_jobs}/{total_jobs} | "
                f"{res['task_name']}/{res['protocol']} "
                f"ep[{res['episode_start']}:{res['episode_start'] + res['episode_count']}) "
                f"done={res['n_done']} skip={res['n_skipped']} "
                f"accept={res['n_accepted']} usable={res.get('n_usable', 0)} | "
                f"{(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
    return all_manifest


def main():
    parser = argparse.ArgumentParser(description="Collect CausalWorld teacher demos")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    parser.add_argument("--task", default=None, choices=["pushing", "picking", "pick_and_place"])
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel env workers")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    out_root = Path(data_cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = [args.task] if args.task else data_cfg["tasks"]
    protocol_names = data_cfg["protocols"]
    episodes = int(data_cfg["episodes_per_protocol"])
    min_success = float(data_cfg["min_fractional_success"])
    log_every_steps = int(data_cfg.get("log_every_steps", 25))
    quiet_intervention_logs = bool(data_cfg.get("quiet_intervention_logs", True))
    skip_existing = bool(data_cfg.get("skip_existing", False))
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else data_cfg.get("num_workers", 1)
    )
    gpu_ids = data_cfg.get("gpu_ids")
    collection_seed_offset = int(data_cfg.get("collection_seed_offset", 0))
    _quiet_intervention_warnings(quiet_intervention_logs)

    total_eps = len(tasks) * len(protocol_names) * episodes
    print(
        f"[collect] {len(tasks)} tasks x {len(protocol_names)} protocols x "
        f"{episodes} episodes = {total_eps} total | num_workers={num_workers} | "
        f"seed_offset={collection_seed_offset}",
        flush=True,
    )

    if num_workers <= 1:
        all_manifest = []
        seed = collection_seed_offset
        for task_name in tasks:
            benchmark = BENCHMARK_BY_TASK[task_name]
            protocols = _protocol_by_name(benchmark, protocol_names)
            ckpt = (cfg.get("teacher", {}).get("checkpoints", {}) or {}).get(task_name)
            teacher_fn = load_teacher(task_name, checkpoint=ckpt)
            for protocol in protocols:
                pname = protocol.get_name()
                print(f"[collect] start {task_name}/{pname} x{episodes}", flush=True)
                manifest = collect_for_protocol_serial(
                    task_name,
                    protocol,
                    teacher_fn,
                    out_root,
                    episodes,
                    min_success,
                    seed,
                    log_every_steps=log_every_steps,
                    skip_existing=skip_existing,
                )
                print(
                    f"[collect] finished {task_name}/{pname}: "
                    f"{len(manifest)}/{episodes} accepted",
                    flush=True,
                )
                all_manifest.extend(manifest)
                seed += 1
    else:
        all_manifest = collect_parallel(
            tasks,
            protocol_names,
            out_root,
            episodes,
            min_success,
            num_workers,
            gpu_ids,
            cfg,
            log_every_steps,
            skip_existing,
            quiet_intervention_logs,
            collection_seed_offset=collection_seed_offset,
        )

    summary_path = out_root / "manifest.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, indent=2)
    print(f"Saved {len(all_manifest)} accepted episodes -> {summary_path}")


if __name__ == "__main__":
    main()
