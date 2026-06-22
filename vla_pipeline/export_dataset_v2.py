#!/usr/bin/env python3
"""
Re-export demos from existing vla_demos (including rejected) with step-level filtering.

Usage (no re-simulation):
  python vla_pipeline/export_dataset_v2.py --config vla_pipeline/config_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml


def stratified_train_val_split(samples, val_ratio: float, seed: int = 42):
    """Split by (task, protocol) so val matches train distribution."""
    if not samples:
        return [], []
    rng = random.Random(seed)
    by_key = defaultdict(list)
    for i, traj in enumerate(samples):
        by_key[(traj[0]["task"], traj[0]["protocol"])].append(i)

    val_idx = []
    for idxs in by_key.values():
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            continue
        n_val = max(1, round(n * val_ratio))
        n_val = min(n_val, n - 1)
        val_idx.extend(idxs[:n_val])

    target = max(1, int(len(samples) * val_ratio))
    pool = [i for i in range(len(samples)) if i not in val_idx]
    rng.shuffle(pool)
    while len(val_idx) < target and pool:
        val_idx.append(pool.pop())

    val_set = set(val_idx)
    train_split = [samples[i] for i in range(len(samples)) if i not in val_set]
    val_data = [samples[i] for i in val_idx]
    return train_split, val_data


def iter_episodes(demo_root: Path, include_rejected: bool):
    for ep_json in demo_root.rglob("episode.json"):
        with open(ep_json, encoding="utf-8") as f:
            yield json.load(f)
    if include_rejected:
        for ep_json in demo_root.rglob("episode_rejected.json"):
            with open(ep_json, encoding="utf-8") as f:
                yield json.load(f)


def _dagger_allowed(task: str, protocol: str, data: dict) -> bool:
    include = data.get("dagger_include")
    if not include:
        return True
    allowed = include.get(task)
    if allowed is None:
        return False
    return protocol in allowed


def export(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    demo_root = Path(data["output_dir"])
    export_dir = Path(data["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    include_rejected = bool(data.get("include_rejected", True))
    min_step_fs = float(data.get("min_step_fractional_success", 0.05))
    min_ep_max = float(data.get("min_episode_max_fractional_success", 0.0))
    min_dagger_ep_max = float(data.get("min_dagger_episode_max_fractional_success", 0.0))
    subsample = max(1, int(data.get("subsample_every_n_steps", 1)))

    samples = []
    ep_kept = ep_skip = ep_dagger = ep_teacher = 0
    for ep in iter_episodes(demo_root, include_rejected):
        infos = ep.get("infos") or []
        if infos:
            ep_max = max(float(i.get("fractional_success", 0.0)) for i in infos)
        else:
            ep_max = float(ep.get("final_fractional_success", 0.0))
        if ep_max < min_ep_max:
            ep_skip += 1
            continue
        src = ep.get("source", "teacher")
        if src == "dagger":
            if not _dagger_allowed(ep["task"], ep["protocol"], data):
                ep_skip += 1
                continue
            if min_dagger_ep_max > 0 and ep_max < min_dagger_ep_max:
                ep_skip += 1
                continue
        ep_kept += 1
        if src == "dagger":
            ep_dagger += 1
        else:
            ep_teacher += 1
        traj = []
        for t, (cams, action, instr) in enumerate(
            zip(ep["frames"], ep["actions"], ep["instructions"])
        ):
            if t % subsample != 0:
                continue
            fs = float(infos[t]["fractional_success"]) if t < len(infos) else 0.0
            if fs < min_step_fs:
                continue
            if len(cams) < 3:
                continue
            traj.append(
                {
                    "task": ep["task"],
                    "protocol": ep["protocol"],
                    "episode": ep["episode"],
                    "timestep": t,
                    "left": cams[0],
                    "center": cams[1],
                    "right": cams[2],
                    "instruction": instr,
                    "action": action,
                    "fractional_success": fs,
                    "source": src,
                }
            )
        if traj:
            samples.append(traj)

    val_ratio = float(cfg["training"]["val_ratio"])
    split_seed = int(cfg["training"].get("split_seed", 42))
    train_split, val_data = stratified_train_val_split(samples, val_ratio, seed=split_seed)
    n_steps = sum(len(t) for t in samples)

    with open(export_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_split, f)
    with open(export_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f)
    meta = {
        "episodes_scanned_kept": ep_kept,
        "episodes_skipped": ep_skip,
        "num_trajectories": len(samples),
        "num_steps": n_steps,
        "train_trajectories": len(train_split),
        "val_trajectories": len(val_data),
        "min_step_fractional_success": min_step_fs,
        "include_rejected": include_rejected,
        "episodes_teacher": ep_teacher,
        "episodes_dagger": ep_dagger,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
    }
    with open(export_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"V2 export: {n_steps} steps / {len(samples)} trajs -> {export_dir}")
    print(f"  episodes kept={ep_kept} skipped={ep_skip}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_v2.yaml"))
    export(p.parse_args().config)


if __name__ == "__main__":
    main()
