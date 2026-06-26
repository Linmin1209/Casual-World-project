#!/usr/bin/env python3
"""Full-protocol eval for CPPPO checkpoint (200 ep/protocol)."""
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

from vla_pipeline.train_cppo import _checkpoint_suffix, _quick_eval


def _default_ckpt(cfg: dict, task: str) -> Path:
    ckpt_dir = Path(cfg["checkpoint_dir"])
    suffix = _checkpoint_suffix(cfg)
    return ckpt_dir / task / f"{task}_{suffix}.zip"


def eval_task(task: str, cfg: dict, ckpt: Path, out_dir: Path) -> dict:
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    version = str(cfg.get("version", "v2"))
    print(f"Eval {task} ({version}): {ckpt}", flush=True)
    print("fraction=1.0 (200 ep/protocol), ProtocolObsWrapper=on", flush=True)
    t0 = time.time()
    scores = _quick_eval(task, str(ckpt), cfg)
    metric = "mean_full_integrated_fractional_success"
    proto = {p: float(scores[p].get(metric, 0.0)) for p in sorted(scores)}
    macro = sum(proto.values()) / max(1, len(proto))

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task": task,
        "checkpoint": str(ckpt),
        "version": version,
        "stage": cfg.get("stage"),
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
    out_path = out_dir / f"{task}_full_eval.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"task_macro={macro:.4f}", flush=True)
    for p in sorted(proto):
        print(f"  {p}: {proto[p]:.4f}", flush=True)
    print(f"saved -> {out_path}", flush=True)
    print(f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    base_path = ROOT / "data/baseline_eval/summary_all_tasks.json"
    if base_path.is_file():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        btask = next(x for x in base if x["task"] == task)
        bproto = {
            p: btask["protocols"][p]["mean_full_integrated_fractional_success"]
            for p in btask["protocols"]
        }
        bmacro = sum(bproto.values()) / len(bproto)
        print(f"baseline {task} macro={bmacro:.4f}", flush=True)
        print(
            f"delta macro={macro - bmacro:+.4f} ({(macro - bmacro) / bmacro * 100:+.1f}%)",
            flush=True,
        )
        print("per-protocol delta (cppo - baseline):", flush=True)
        for p in sorted(proto):
            d = proto[p] - bproto.get(p, 0.0)
            print(f"  {p}: {bproto.get(p,0):.4f} -> {proto[p]:.4f} ({d:+.4f})", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CPPPO full eval")
    parser.add_argument(
        "--config",
        default=str(ROOT / "vla_pipeline/config_cppo_v2_local.yaml"),
    )
    parser.add_argument(
        "--task",
        choices=["pushing", "picking", "pick_and_place"],
        default=None,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["pushing", "picking", "pick_and_place"],
        default=None,
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = raw["cppo"]
    cfg["evaluation"] = dict(raw.get("evaluation") or {})
    cfg["evaluation"]["fraction"] = 1.0

    out_dir = Path(args.output_dir) if args.output_dir else Path(
        raw.get("evaluation", {}).get(
            "output_dir",
            str(ROOT / "data/cppo_eval_v2_official"),
        )
    )

    tasks = args.tasks or ([args.task] if args.task else ["pushing", "picking", "pick_and_place"])
    results = {}
    for task in tasks:
        ckpt = Path(args.checkpoint) if args.checkpoint else _default_ckpt(cfg, task)
        results[task] = eval_task(task, cfg, ckpt, out_dir)
        print("", flush=True)

    if len(tasks) > 1:
        macro = sum(
            results[t]["protocols"][p]["mean_full_integrated_fractional_success"]
            for t in tasks
            for p in results[t]["protocols"]
        ) / (len(tasks) * 12)
        base_path = ROOT / "data/baseline_eval/summary_all_tasks.json"
        base_macro = None
        if base_path.is_file():
            base = json.loads(base_path.read_text(encoding="utf-8"))
            base_macro = sum(
                d["mean_full_integrated_fractional_success"]
                for t in base
                for d in t["protocols"].values()
            ) / 36
        print("=" * 60, flush=True)
        print(f"Combined task_macro avg ({len(tasks)} tasks): {macro:.4f}", flush=True)
        if base_macro is not None:
            print(
                f"Global macro ({len(tasks)*12} protocols): {macro:.4f} "
                f"({(macro/base_macro-1)*100:+.1f}% vs baseline {base_macro:.4f})",
                flush=True,
            )
            print(f"Target +15%: {base_macro*1.15:.4f}", flush=True)
        summary_path = out_dir / "summary_all_tasks.json"
        summary_path.write_text(
            json.dumps(
                [
                    {
                        "task": t,
                        "model": f"{t}_cppo_{cfg.get('version','v2')}",
                        "fraction": 1.0,
                        "metric": "mean_full_integrated_fractional_success",
                        "protocols": results[t]["protocols"],
                    }
                    for t in tasks
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
