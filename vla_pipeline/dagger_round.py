#!/usr/bin/env python3
"""
DAgger round(s): rollout hybrid on hard protocols, save teacher-labeled (s,a) pairs.

  python vla_pipeline/dagger_round.py --config vla_pipeline/config_v4_dagger.yaml --round 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vla_pipeline.collect_data import (
    BENCHMARK_BY_TASK,
    _protocol_by_name,
    _protocol_index,
    _resolve_gpu_ids,
)
from vla_pipeline.collect_workers import split_episode_jobs
from vla_pipeline.dagger_workers import run_dagger_episode_range


def _quiet(quiet: bool) -> None:
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)


def build_jobs(cfg: dict, round_id: int, num_workers: int) -> list:
    dagger = cfg["dagger"]
    data = cfg["data"]
    tasks = list(dagger.get("tasks", data["tasks"]))
    protocols_by_task = dagger.get("protocols_by_task") or {}
    protocol_names = list(dagger.get("protocols") or [])
    episodes = int(dagger["episodes_per_protocol"])
    skip_frame = int(dagger.get("skip_frame", cfg.get("evaluation", {}).get("skip_frame", 3)))
    skip_existing = bool(dagger.get("skip_existing", True))
    out_root = Path(data["output_dir"])
    seed_offset = int(dagger.get("seed_offset", 5000)) + round_id * 100

    hybrid_cfg = dict(cfg.get("hybrid", {}))
    if hybrid_cfg.get("checkpoint"):
        hybrid_cfg["checkpoint"] = os.path.abspath(hybrid_cfg["checkpoint"])

    jobs = []
    proto_seed = seed_offset
    for task_name in tasks:
        benchmark = BENCHMARK_BY_TASK[task_name]
        if task_name in protocols_by_task:
            protocol_names_task = list(protocols_by_task[task_name])
        elif protocol_names:
            protocol_names_task = protocol_names
        else:
            raise ValueError(f"dagger: no protocols for task {task_name}")
        protocols = _protocol_by_name(benchmark, protocol_names_task)
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
                        "round_id": round_id,
                        "episode_start": ep_start,
                        "episode_count": ep_count,
                        "out_root": str(out_root),
                        "skip_frame": skip_frame,
                        "seed": proto_seed + ep_start,
                        "hybrid_cfg": hybrid_cfg,
                        "image_size": int(cfg["training"]["image_size"]),
                        "teacher_ckpt": ckpt,
                        "skip_existing": skip_existing,
                        "log_every_episodes": int(dagger.get("log_every_episodes", 5)),
                        "quiet": bool(dagger.get("quiet", True)),
                    }
                )
            proto_seed += 1

    gpu_ids = _resolve_gpu_ids(max(1, num_workers), dagger.get("gpu_ids"))
    for i, job in enumerate(jobs):
        job["gpu_id"] = gpu_ids[i % len(gpu_ids)]
    return jobs


def run_round(cfg: dict, round_id: int, num_workers: int) -> int:
    jobs = build_jobs(cfg, round_id, num_workers)
    total_eps = sum(j["episode_count"] for j in jobs) // max(1, len(set(
        (j["task_name"], j["protocol_idx"]) for j in jobs
    )))  # approximate
    print(
        f"[dagger] round {round_id}: {len(jobs)} jobs, "
        f"~{int(cfg['dagger']['episodes_per_protocol'])} ep/protocol, "
        f"workers={num_workers}, ckpt={cfg['hybrid']['checkpoint']}",
        flush=True,
    )
    print("[dagger] workers loading TF+VLA+PyBullet (~2-5 min before first log)...", flush=True)

    manifest = []
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(run_dagger_episode_range, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            res = fut.result()
            manifest.extend(res["manifest"])
            done += 1
            print(
                f"[dagger] round {round_id} job {done}/{len(jobs)} | "
                f"{res['task_name']}/{res['protocol']} "
                f"ep[{res['episode_start']}:{res['episode_start'] + res['episode_count']}) "
                f"new={res['n_done']} skip={res['n_skipped']} | "
                f"{(time.time() - t0) / 60:.1f} min",
                flush=True,
            )

    summary = Path(cfg["data"]["output_dir"]) / f"dagger_manifest_r{round_id:02d}.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[dagger] round {round_id} done: {len(manifest)} episodes -> {summary}",
        flush=True,
    )
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description="DAgger collection (hybrid rollout, teacher labels)")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_v4_dagger.yaml")),
    )
    parser.add_argument("--round", type=int, default=None, help="Round index (1-based)")
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dagger = cfg["dagger"]
    num_workers = int(
        args.num_workers if args.num_workers is not None else dagger.get("num_workers", 4)
    )
    _quiet(bool(dagger.get("quiet", True)))

    rounds = [args.round] if args.round is not None else list(range(1, int(dagger.get("rounds", 1)) + 1))
    for r in rounds:
        run_round(cfg, r, num_workers)


if __name__ == "__main__":
    main()
