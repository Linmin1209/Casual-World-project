#!/usr/bin/env python3
"""Sweep multiple CPPPO checkpoints (quick eval by default)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vla_pipeline.train_cppo import _quick_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep CPPPO checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default="pick_and_place")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = raw["cppo"]
    cfg["evaluation"] = dict(raw.get("evaluation") or {})
    if args.fraction is not None:
        cfg["evaluation"]["fraction"] = float(args.fraction)

    labels = args.labels or [Path(p).stem for p in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit("labels count must match checkpoints")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    metric = "mean_full_integrated_fractional_success"
    fraction = float(cfg["evaluation"].get("fraction", 0.1))
    n_ep = max(1, int(200 * fraction))

    for label, ckpt_str in zip(labels, args.checkpoints):
        ckpt = Path(ckpt_str)
        if not ckpt.is_file():
            print(f"SKIP missing: {ckpt}", flush=True)
            continue
        print(f"\n=== {label} ===", flush=True)
        print(f"ckpt: {ckpt}", flush=True)
        t0 = time.time()
        scores = _quick_eval(args.task, str(ckpt), cfg)
        proto = {p: float(scores[p].get(metric, 0.0)) for p in sorted(scores)}
        macro = sum(proto.values()) / max(1, len(proto))
        row = {
            "label": label,
            "checkpoint": str(ckpt),
            "fraction": fraction,
            "episodes_per_protocol": n_ep,
            "task_macro": macro,
            "protocols": proto,
            "elapsed_min": (time.time() - t0) / 60,
        }
        results.append(row)
        print(f"task_macro={macro:.4f} elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    best = max(results, key=lambda r: r["task_macro"]) if results else None
    summary = {
        "task": args.task,
        "fraction": fraction,
        "episodes_per_protocol": n_ep,
        "results": results,
        "best": best,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)
    if best:
        print(f"BEST: {best['label']} task_macro={best['task_macro']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
