#!/usr/bin/env python3
"""Re-render RGB frames for existing demos by replaying stored actions."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vla_pipeline.collect_data import BENCHMARK_BY_TASK, _protocol_by_name, _protocol_index
from vla_pipeline.collect_workers import _save_rgb
from vla_pipeline.quiet_utils import setup_quiet_env, suppress_stdout_stderr


def _episode_json_paths(demo_root: Path) -> List[Path]:
    paths = sorted(demo_root.rglob("episode.json"))
    paths += sorted(demo_root.rglob("episode_rejected.json"))
    return paths


def _rgb_is_black(ep_dir: Path, sample_step: int = 0) -> bool:
    p = ep_dir / "rgb" / f"step_{sample_step:05d}_cam0.png"
    if not p.is_file():
        return True
    try:
        from PIL import Image

        arr = np.array(Image.open(p).convert("RGB"))
        return float(arr.mean()) < 1.0
    except Exception:
        return True


def _protocol_seed(task_name: str, protocol_name: str, collection_seed_offset: int = 0) -> int:
    benchmark = BENCHMARK_BY_TASK[task_name]
    names = [p.get_name() for p in benchmark["evaluation_protocols"]]
    if protocol_name in names:
        return collection_seed_offset + names.index(protocol_name)
    return collection_seed_offset


def repair_episode(payload: Dict[str, Any]) -> Dict[str, Any]:
    ep_path = Path(payload["episode_path"])
    force = bool(payload.get("force", False))
    ep_dir = ep_path.parent
    if not force and not _rgb_is_black(ep_dir):
        return {"episode_path": str(ep_path), "status": "skip_ok", "steps": 0}

    with open(ep_path, encoding="utf-8") as f:
        record = json.load(f)

    task_name = record["task"]
    protocol_name = record["protocol"]
    episode_idx = int(record.get("episode", 0))
    actions = record.get("actions") or []
    if not actions:
        return {"episode_path": str(ep_path), "status": "skip_no_actions", "steps": 0}

    setup_quiet_env()
    if payload.get("quiet", True):
        logging.getLogger().setLevel(logging.ERROR)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from causal_world.envs.causalworld import CausalWorld
    from causal_world.task_generators.task import generate_task
    from causal_world.wrappers.protocol_wrapper import ProtocolWrapper

    benchmark = BENCHMARK_BY_TASK[task_name]
    protocol = _protocol_by_name(benchmark, [protocol_name])[0]
    seed = _protocol_seed(task_name, protocol_name, int(payload.get("collection_seed_offset", 0)))

    task = generate_task(task_generator_id=task_name, variables_space="space_a_b")
    with suppress_stdout_stderr():
        env = CausalWorld(
            task=task,
            enable_visualization=False,
            seed=seed,
            observation_mode="structured",
            normalize_observations=True,
            normalize_actions=True,
        )
        protocol.init_protocol(env=env, tracker=env.get_tracker(), fraction=1.0)
        wrapped = ProtocolWrapper(env, protocol)
        wrapped._elapsed_episodes = episode_idx
        wrapped.reset()

        rgb_dir = ep_dir / "rgb"
        if force and rgb_dir.is_dir():
            for old in rgb_dir.glob("*.png"):
                old.unlink()

        new_frames = []
        for step_idx, action in enumerate(actions):
            cam_paths = _save_rgb(env, rgb_dir, step_idx)
            new_frames.append(cam_paths)
            wrapped.step(np.asarray(action, dtype=np.float32))

        env.close()

    record["frames"] = new_frames
    with open(ep_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    mean0 = 0.0
    p0 = rgb_dir / "step_00000_cam0.png"
    if p0.is_file():
        from PIL import Image

        mean0 = float(np.array(Image.open(p0).convert("RGB")).mean())

    return {
        "episode_path": str(ep_path),
        "status": "repaired",
        "steps": len(actions),
        "cam0_mean": mean0,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-render black demo RGB by action replay")
    parser.add_argument(
        "--demo_root",
        default="/data1/linmin/CausalWorld/data/vla_demos",
    )
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Re-render even if RGB looks OK")
    parser.add_argument("--limit", type=int, default=0, help="Max episodes (0=all)")
    parser.add_argument("--task", default=None)
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args()

    demo_root = Path(args.demo_root)
    ep_paths = _episode_json_paths(demo_root)
    if args.task:
        ep_paths = [p for p in ep_paths if args.task in str(p)]
    if args.protocol:
        ep_paths = [p for p in ep_paths if f"/{args.protocol}/" in str(p)]
    if args.limit > 0:
        ep_paths = ep_paths[: args.limit]

    jobs = [
        {
            "episode_path": str(p),
            "force": args.force,
            "quiet": True,
            "collection_seed_offset": 0,
        }
        for p in ep_paths
    ]
    print(f"[repair_rgb] {len(jobs)} episodes | workers={args.num_workers} force={args.force}")

    repaired = skipped = failed = 0
    t0 = time.time()
    if args.num_workers <= 1:
        for job in jobs:
            try:
                res = repair_episode(job)
            except Exception as exc:
                failed += 1
                print(f"FAIL {job['episode_path']}: {exc}", flush=True)
                continue
            if res["status"] == "repaired":
                repaired += 1
            else:
                skipped += 1
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futs = {pool.submit(repair_episode, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                job = futs[fut]
                try:
                    res = fut.result()
                    if res["status"] == "repaired":
                        repaired += 1
                        if done <= 3 or done % 50 == 0:
                            print(
                                f"[repair_rgb] {done}/{len(jobs)} repaired "
                                f"cam0_mean={res.get('cam0_mean', 0):.1f} "
                                f"{Path(job['episode_path']).parent}",
                                flush=True,
                            )
                    else:
                        skipped += 1
                except Exception as exc:
                    failed += 1
                    print(f"FAIL {job['episode_path']}: {exc}", flush=True)

    print(
        f"[repair_rgb] done in {(time.time()-t0)/60:.1f} min | "
        f"repaired={repaired} skipped={skipped} failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
