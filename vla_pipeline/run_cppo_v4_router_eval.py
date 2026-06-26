#!/usr/bin/env python3
"""Evaluate protocol-routed CPPPO teacher (v4 assembly)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vla_pipeline.train_cppo import _quick_eval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--router",
        default=str(ROOT / "vla_pipeline/config_cppo_v4_teacher_router.json"),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "vla_pipeline/config_cppo_v4_pap_p45_local.yaml"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    router = json.loads(Path(args.router).read_text(encoding="utf-8"))
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = raw["cppo"]
    cfg["evaluation"] = dict(raw.get("evaluation") or {})

    results = {}
    for task, spec in router["tasks"].items():
        overrides = spec.get("protocol_overrides") or {}
        default_ckpt = spec["default"]
        eval_cfg = dict(cfg)
        if task in cfg.get("task_configs", {}):
            pass
        # per-task eval fraction override
        task_eval = dict(cfg.get("evaluation") or {})
        task_eval.update(spec.get("eval") or {})
        eval_cfg["evaluation"] = task_eval

        proto_scores = {}
        # group protocols by checkpoint
        by_ckpt = {}
        for p in [f"P{i}" for i in range(12)]:
            ck = overrides.get(p, default_ckpt)
            by_ckpt.setdefault(ck, []).append(p)

        for ckpt, protos in by_ckpt.items():
            print(f"[v4] {task} eval {ckpt} for {protos}", flush=True)
            scores = _quick_eval(task, ckpt, eval_cfg)
            metric = "mean_full_integrated_fractional_success"
            for p in protos:
                proto_scores[p] = float(scores[p].get(metric, 0.0))

        macro = sum(proto_scores.values()) / 12
        results[task] = {
            "task_macro": macro,
            "protocols": proto_scores,
            "routing": {p: overrides.get(p, default_ckpt) for p in proto_scores},
        }
        print(f"[v4] {task} task_macro={macro:.4f}", flush=True)

    out = {"tasks": results}
    if len(results) == 3:
        g = sum(results[t]["task_macro"] for t in results) / 3
        out["avg_task_macro"] = g
        # proper 36-proto macro
        scores36 = []
        for t in results:
            scores36.extend(results[t]["protocols"].values())
        out["global_macro_36proto"] = sum(scores36) / len(scores36)

    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
