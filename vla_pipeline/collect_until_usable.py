#!/usr/bin/env python3
"""
Keep collecting teacher demos until each (task, protocol) has enough *usable*
episodes for BC export (episode max step fs >= min_usable_episode_max_fs).
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
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from vla_pipeline.collect_data import (
    BENCHMARK_BY_TASK,
    _protocol_by_name,
    _protocol_index,
    _resolve_gpu_ids,
)
from vla_pipeline.collect_workers import collect_episode_range, split_episode_jobs


def _quiet_intervention_warnings(quiet: bool) -> None:
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)


def _episode_max_fs(ep: dict) -> float:
    infos = ep.get("infos") or []
    if infos:
        return max(float(i.get("fractional_success", 0.0)) for i in infos)
    return float(ep.get("final_fractional_success", 0.0))


def scan_usable(
    demo_root: Path,
    tasks: List[str],
    protocol_names: List[str],
    min_usable_fs: float,
) -> Tuple[Counter, Dict[Tuple[str, str], int]]:
    wanted = {(t, p) for t in tasks for p in protocol_names}
    usable = Counter()
    max_ep: Dict[Tuple[str, str], int] = {k: -1 for k in wanted}
    for ep_json in demo_root.rglob("episode*.json"):
        with open(ep_json, encoding="utf-8") as f:
            ep = json.load(f)
        key = (ep["task"], ep["protocol"])
        if key not in wanted:
            continue
        max_ep[key] = max(max_ep[key], int(ep.get("episode", 0)))
        if _episode_max_fs(ep) >= min_usable_fs:
            usable[key] += 1
    return usable, max_ep


def build_round_jobs(
    tasks: List[str],
    protocol_names: List[str],
    deficits: Dict[Tuple[str, str], int],
    max_ep: Dict[Tuple[str, str], int],
    out_root: Path,
    batch_episodes: int,
    max_episodes_per_protocol: int,
    min_success: float,
    min_usable_fs: float,
    skip_existing: bool,
    log_every_steps: int,
    quiet_intervention_logs: bool,
    collection_seed_offset: int,
    cfg: dict,
    num_workers: int,
) -> List[dict]:
    jobs = []
    proto_seed = int(collection_seed_offset)
    for task_name in tasks:
        benchmark = BENCHMARK_BY_TASK[task_name]
        protocols = _protocol_by_name(benchmark, protocol_names)
        ckpt = (cfg.get("teacher", {}).get("checkpoints", {}) or {}).get(task_name)
        if ckpt:
            ckpt = str(Path(ckpt).resolve())
        for protocol in protocols:
            pname = protocol.get_name()
            key = (task_name, pname)
            need = deficits.get(key, 0)
            if need <= 0:
                continue
            start = max_ep[key] + 1
            if start >= max_episodes_per_protocol:
                continue
            count = min(batch_episodes, max_episodes_per_protocol - start)
            if count <= 0:
                continue
            pidx = _protocol_index(benchmark, protocol)
            for ep_start, ep_count in split_episode_jobs(count, num_workers):
                jobs.append(
                    {
                        "task_name": task_name,
                        "protocol_idx": pidx,
                        "episode_start": start + ep_start,
                        "episode_count": ep_count,
                        "out_root": str(out_root),
                        "min_success": min_success,
                        "min_usable_episode_max_fs": min_usable_fs,
                        "seed": proto_seed + ep_start,
                        "log_every_steps": log_every_steps,
                        "skip_existing": skip_existing,
                        "quiet_intervention_logs": quiet_intervention_logs,
                        "teacher_ckpt": ckpt,
                    }
                )
            proto_seed += 1
    gpu_ids = _resolve_gpu_ids(max(1, num_workers), cfg["data"].get("gpu_ids"))
    for i, job in enumerate(jobs):
        job["gpu_id"] = gpu_ids[i % len(gpu_ids)]
    return jobs


def run_jobs(jobs: List[dict], num_workers: int) -> None:
    if not jobs:
        return
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(collect_episode_range, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            res = fut.result()
            done += 1
            print(
                f"[round] {done}/{len(jobs)} | "
                f"{res['task_name']}/{res['protocol']} "
                f"ep[{res['episode_start']}:{res['episode_start'] + res['episode_count']}) "
                f"done={res['n_done']} skip={res['n_skipped']} "
                f"accept={res['n_accepted']} usable={res['n_usable']} | "
                f"{(time.time() - t0) / 60:.1f} min",
                flush=True,
            )


def print_status(
    tasks: List[str],
    protocol_names: List[str],
    usable: Counter,
    max_ep: Dict[Tuple[str, str], int],
    target: int,
) -> int:
    deficit = 0
    print("[status] usable episodes (ep_max >= min_usable_fs):", flush=True)
    for task in tasks:
        for pname in protocol_names:
            key = (task, pname)
            u = usable[key]
            need = max(0, target - u)
            deficit += need
            flag = "OK" if need == 0 else f"NEED {need}"
            print(
                f"  {task}/{pname}: {u}/{target} (max_ep={max_ep[key]}) {flag}",
                flush=True,
            )
    return deficit


def main():
    parser = argparse.ArgumentParser(description="Collect until usable demo quota met")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_v3_collect_hard.yaml")),
    )
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    out_root = Path(data["output_dir"])
    tasks = list(data["tasks"])
    protocol_names = list(data["protocols"])
    num_workers = int(
        args.num_workers if args.num_workers is not None else data.get("num_workers", 6)
    )
    target = int(data.get("target_usable_episodes_per_protocol", 30))
    min_usable_fs = float(
        data.get(
            "min_usable_episode_max_fs",
            data.get("min_step_fractional_success", 0.05),
        )
    )
    max_episodes = int(data.get("max_episodes_per_protocol", 300))
    batch_episodes = int(data.get("batch_episodes_per_round", 20))
    seed_offset = int(data.get("collection_seed_offset", 1000))
    seed_step = int(data.get("collection_seed_step", 200))
    min_success = float(data.get("min_fractional_success", 0.0))
    skip_existing = bool(data.get("skip_existing", True))
    log_every = int(data.get("log_every_steps", 25))
    quiet = bool(data.get("quiet_intervention_logs", True))
    _quiet_intervention_warnings(quiet)

    print(
        f"[until-usable] target={target}/protocol, min_ep_max_fs={min_usable_fs}, "
        f"max_ep={max_episodes}, batch={batch_episodes}, workers={num_workers}",
        flush=True,
    )

    round_idx = 0
    while True:
        usable, max_ep = scan_usable(out_root, tasks, protocol_names, min_usable_fs)
        deficit = print_status(tasks, protocol_names, usable, max_ep, target)
        if deficit == 0:
            print("[until-usable] All quotas met. Done.", flush=True)
            break

        deficits = {
            (t, p): target - usable[(t, p)]
            for t in tasks
            for p in protocol_names
            if usable[(t, p)] < target
        }
        stuck = [
            k
            for k, need in deficits.items()
            if max_ep[k] + 1 >= max_episodes
        ]
        if stuck and len(stuck) == len(deficits):
            print(
                "[until-usable] STOP: max_ep reached with deficits remaining:",
                stuck,
                flush=True,
            )
            break

        jobs = build_round_jobs(
            tasks,
            protocol_names,
            deficits,
            max_ep,
            out_root,
            batch_episodes,
            max_episodes,
            min_success,
            min_usable_fs,
            skip_existing,
            log_every,
            quiet,
            seed_offset + round_idx * seed_step,
            cfg,
            num_workers,
        )
        if not jobs:
            print("[until-usable] No jobs left (max_ep cap). Stopping.", flush=True)
            break

        round_idx += 1
        print(
            f"\n=== Round {round_idx}: {len(jobs)} jobs, "
            f"deficit={deficit} usable episodes ===",
            flush=True,
        )
        run_jobs(jobs, num_workers)


if __name__ == "__main__":
    main()
