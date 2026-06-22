#!/usr/bin/env python3
"""Grid-search hybrid beta per (task, protocol) on a small eval slice."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from vla_pipeline.evaluate import BENCHMARKS, evaluate_task, macro_average


def load_baseline_macro(baseline_dir: Path, tasks: list, protocols: list) -> float:
    scores = []
    for task in tasks:
        path = baseline_dir / f"{task}_all_protocols.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for p in protocols:
            if p in data["protocols"]:
                scores.append(
                    data["protocols"][p]["mean_full_integrated_fractional_success"]
                )
    return sum(scores) / max(1, len(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vla_pipeline/config_v5.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--candidates",
        nargs="*",
        type=float,
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    hybrid = cfg.setdefault("hybrid", {})
    beta_map = copy.deepcopy(hybrid.get("beta_by_task_protocol") or {})
    tasks = cfg.get("dagger", {}).get("tasks", cfg["data"]["tasks"])
    protocols = ["P6", "P7", "P8", "P9", "P10", "P11"]
    baseline_dir = Path(cfg["evaluation"]["baseline_dir"])
    base_macro = load_baseline_macro(baseline_dir, tasks, protocols)
    print(f"Baseline hard macro ({tasks}, {protocols}): {base_macro:.4f}")

    best_map = copy.deepcopy(beta_map)
    out_dir = Path(cfg["evaluation"]["output_dir"]).parent / "vla_beta_tune"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_map = best_map.setdefault(task, {})
        for proto in protocols:
            if task == "pick_and_place" and proto in {"P6", "P7", "P8", "P9"}:
                task_map[proto] = 0.0
                continue
            if task == "picking" and proto in {"P6", "P7", "P8", "P9"}:
                task_map[proto] = 0.0
                continue

            best_beta = float(task_map.get(proto, hybrid.get("beta_default", 0.0)))
            best_score = -1.0
            bb_path = baseline_dir / f"{task}_all_protocols.json"
            base_p = 0.0
            if bb_path.is_file():
                bb = json.loads(bb_path.read_text(encoding="utf-8"))
                base_p = float(bb["protocols"].get(proto, {}).get(
                    "mean_full_integrated_fractional_success", 0.0
                ))

            for beta in args.candidates:
                trial_cfg = copy.deepcopy(cfg)
                tmap = trial_cfg["hybrid"].setdefault("beta_by_task_protocol", {})
                tmap.setdefault(task, {})[proto] = float(beta)
                trial_cfg["evaluation"] = dict(trial_cfg["evaluation"])
                trial_cfg["evaluation"]["output_dir"] = str(out_dir / f"{task}_{proto}")

                res = evaluate_task(
                    task,
                    trial_cfg,
                    use_hybrid=True,
                    num_workers=args.num_workers,
                    gpu_ids=trial_cfg["evaluation"].get("gpu_ids"),
                    protocol_filter=[proto],
                    episodes_override=args.episodes,
                )
                score = res["protocols"][proto]["mean_full_integrated_fractional_success"]
                print(
                    f"  {task}/{proto} beta={beta:.2f} -> {score:.4f} "
                    f"(baseline {base_p:.4f}, delta {score - base_p:+.4f})"
                )
                if score >= best_score:
                    best_score = score
                    best_beta = float(beta)

            task_map[proto] = best_beta
            print(f"  -> pick {task}/{proto} beta={best_beta:.2f} score={best_score:.4f}")

    hybrid["beta_by_task_protocol"] = best_map
    cfg["hybrid"] = hybrid
    out_cfg = cfg_path.with_name(cfg_path.stem + "_tuned.yaml")
    out_cfg.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Wrote tuned config -> {out_cfg}")


if __name__ == "__main__":
    main()
