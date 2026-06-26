#!/usr/bin/env python3
"""Assemble optimal CPPPO full eval (200 ep/protocol) from per-task results."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PATHS = {
    "pushing": ROOT / "data/cppo_eval_v3_stage1/pushing_full_eval.json",
    "picking": ROOT / "data/cppo_eval_v3_stage1/picking_full_eval.json",
    "pick_and_place": ROOT / "data/cppo_eval_v4/pick_and_place_full_eval.json",
}


def assemble(paths: dict[str, Path], out_dir: Path, update_latest: bool) -> dict:
    base_all = json.loads((ROOT / "data/baseline_eval/summary_all_tasks.json").read_text())
    base_by_task = {t["task"]: t for t in base_all}

    ckpts = {}
    tasks_out = []
    all_scores = []
    summary_rows = []

    for task, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing full eval: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        ckpts[task] = data["checkpoint"]
        protos = []
        task_scores = []
        for i in range(12):
            p = f"P{i}"
            score = float(data["protocols"][p]["mean_full_integrated_fractional_success"])
            task_scores.append(score)
            all_scores.append(score)
            baseline = float(
                base_by_task[task]["protocols"][p]["mean_full_integrated_fractional_success"]
            )
            protos.append(
                {
                    "protocol": p,
                    "baseline": baseline,
                    "latest": score,
                    "delta": score - baseline,
                    "relative_pct": ((score / baseline - 1) * 100) if baseline else 0.0,
                }
            )
        task_macro = sum(task_scores) / 12
        base_macro = sum(
            base_by_task[task]["protocols"][p]["mean_full_integrated_fractional_success"]
            for p in base_by_task[task]["protocols"]
        ) / 12
        tasks_out.append(
            {
                "task": task,
                "baseline_task_macro": base_macro,
                "latest_task_macro": task_macro,
                "latest_checkpoint": data["checkpoint"],
                "latest_eval": {
                    "fraction": 1.0,
                    "episodes_per_protocol": 200,
                    "source": path.name,
                },
                "protocols": protos,
            }
        )
        summary_rows.append(
            {
                "task": task,
                "model": f"{task}_cppo_v3_best",
                "fraction": 1.0,
                "metric": "mean_full_integrated_fractional_success",
                "protocols": {
                    p: {"mean_full_integrated_fractional_success": s, "episodes": 200}
                    for p, s in zip([f"P{i}" for i in range(12)], task_scores)
                },
            }
        )

    base_macro = sum(
        d["mean_full_integrated_fractional_success"]
        for t in base_all
        for d in t["protocols"].values()
    ) / 36
    latest_macro = sum(all_scores) / 36
    target = base_macro * 1.15

    result = {
        "title": "CPPPO optimal full eval (200 ep/protocol)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "mean_full_integrated_fractional_success",
        "macro_definition": "equal-weight average over 36 protocols (12 per task)",
        "composition": {
            "pushing": "v3 stage-1",
            "picking": "v3 stage-1",
            "pick_and_place": "v3 stage-2 @ 12M steps (not v4a; v4a regressed P4/P5)",
        },
        "summary": {
            "baseline_macro": base_macro,
            "latest_best_macro": latest_macro,
            "latest_best_vs_baseline_pct": (latest_macro / base_macro - 1) * 100,
            "target_plus_15pct_macro": target,
            "gap_to_target": latest_macro - target,
            "target_met": latest_macro >= target - 1e-4,
        },
        "checkpoints": ckpts,
        "tasks": tasks_out,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "best_combined_full_eval.json"
    combined_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "summary_all_tasks.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    if update_latest:
        (ROOT / "data/cppo_eval_latest_vs_baseline.json").write_text(
            json.dumps(
                {
                    **result,
                    "title": "CPPPO latest best vs baseline comparison",
                    "latest_best_composition": result["composition"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"global_macro={latest_macro:.6f} target={target:.6f} gap={latest_macro-target:+.6f}")
    print(f"saved -> {combined_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "data/cppo_eval_v4"))
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args()
    assemble(DEFAULT_PATHS, Path(args.output_dir), update_latest=not args.no_update_latest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
