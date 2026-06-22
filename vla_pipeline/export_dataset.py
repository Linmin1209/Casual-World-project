#!/usr/bin/env python3
"""Export collected episodes to flat JSON for BC training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def load_episodes(demo_root: Path):
    for ep_json in demo_root.rglob("episode.json"):
        with open(ep_json, encoding="utf-8") as f:
            yield json.load(f)


def export(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    demo_root = Path(cfg["data"]["output_dir"])
    export_dir = Path(cfg["data"]["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for ep in load_episodes(demo_root):
        for t, (cams, action, instr) in enumerate(
            zip(ep["frames"], ep["actions"], ep["instructions"])
        ):
            if len(cams) < 3:
                continue
            samples.append(
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
                }
            )

    # Group by episode for compatibility with list-of-trajectories format
    by_ep = {}
    for s in samples:
        key = (s["task"], s["protocol"], s["episode"])
        by_ep.setdefault(key, []).append(s)
    train_data = list(by_ep.values())

    n_val = max(1, int(len(train_data) * cfg["training"]["val_ratio"]))
    val_data = train_data[:n_val]
    train_split = train_data[n_val:]

    with open(export_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)
    with open(export_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    meta = {
        "num_trajectories": len(train_data),
        "num_steps": len(samples),
        "train_trajectories": len(train_split),
        "val_trajectories": len(val_data),
    }
    with open(export_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported {meta['num_steps']} steps / {meta['num_trajectories']} trajs -> {export_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    export(p.parse_args().config)


if __name__ == "__main__":
    main()
