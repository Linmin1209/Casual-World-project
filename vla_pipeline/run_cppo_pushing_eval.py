#!/usr/bin/env python3
"""Full-protocol eval for CPPPO v2 official pushing checkpoint (200 ep/protocol)."""
from __future__ import annotations

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
    cfg_path = ROOT / "vla_pipeline/config_cppo_v2_local.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = raw["cppo"]
    cfg["evaluation"] = dict(raw.get("evaluation") or {})
    cfg["evaluation"]["fraction"] = 1.0

    ckpt = ROOT / "data/cppo_checkpoints_v2_official/pushing/pushing_cppo_v2.zip"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    print(f"Eval pushing: {ckpt}", flush=True)
    print("fraction=1.0 (200 ep/protocol), ProtocolObsWrapper=v2", flush=True)
    t0 = time.time()
    scores = _quick_eval("pushing", str(ckpt), cfg)
    metric = "mean_full_integrated_fractional_success"
    proto = {p: float(scores[p].get(metric, 0.0)) for p in sorted(scores)}
    macro = sum(proto.values()) / max(1, len(proto))

    out_dir = ROOT / "data/cppo_eval_v2_official"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task": "pushing",
        "checkpoint": str(ckpt),
        "fraction": 1.0,
        "episodes_per_protocol": 200,
        "metric": metric,
        "protocols": {
            p: {"mean_full_integrated_fractional_success": proto[p], "episodes": 200}
            for p in proto
        },
        "task_macro": macro,
        "elapsed_min": (time.time() - t0) / 60,
    }
    out_path = out_dir / "pushing_full_eval.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"task_macro={macro:.4f}", flush=True)
    for p in sorted(proto):
        print(f"  {p}: {proto[p]:.4f}", flush=True)
    print(f"saved -> {out_path}", flush=True)
    print(f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    base_path = ROOT / "data/baseline_eval/summary_all_tasks.json"
    if base_path.is_file():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        bpush = next(x for x in base if x["task"] == "pushing")
        bproto = {
            p: bpush["protocols"][p]["mean_full_integrated_fractional_success"]
            for p in bpush["protocols"]
        }
        bmacro = sum(bproto.values()) / len(bproto)
        print(f"baseline pushing macro={bmacro:.4f}", flush=True)
        print(
            f"delta macro={macro - bmacro:+.4f} ({(macro - bmacro) / bmacro * 100:+.1f}%)",
            flush=True,
        )
        print("per-protocol delta (cppo - baseline):", flush=True)
        for p in sorted(proto):
            d = proto[p] - bproto.get(p, 0.0)
            print(f"  {p}: {bproto.get(p,0):.4f} -> {proto[p]:.4f} ({d:+.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
