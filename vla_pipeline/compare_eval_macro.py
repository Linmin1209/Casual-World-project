#!/usr/bin/env python3
"""Compare eval summary JSON against baseline_eval and print per-protocol deltas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def macro_average(results: list) -> float:
    scores = []
    for task in results:
        for p in task["protocols"].values():
            scores.append(p["mean_full_integrated_fractional_success"])
    return sum(scores) / max(1, len(scores))


def load_summary(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def index_by_task_proto(summary: list) -> dict:
    out = {}
    for task in summary:
        tname = task["task"]
        for pname, pdata in task["protocols"].items():
            out[(tname, pname)] = float(pdata["mean_full_integrated_fractional_success"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="/home/work/l30083605/Casual-World-project/data/baseline_eval/summary_all_tasks.json",
    )
    parser.add_argument("--candidate", required=True, help="Eval summary_all_tasks.json")
    parser.add_argument("--target_pct", type=float, default=15.0)
    args = parser.parse_args()

    base = load_summary(Path(args.baseline))
    cand_path = Path(args.candidate)
    if not cand_path.is_file():
        print(f"Missing candidate summary: {cand_path}")
        return 1
    cand = load_summary(cand_path)

    base_macro = macro_average(base)
    cand_macro = macro_average(cand)
    target = base_macro * (1.0 + args.target_pct / 100.0)
    rel = (cand_macro - base_macro) / base_macro * 100 if base_macro > 0 else 0.0

    print(f"Baseline macro : {base_macro:.4f}")
    print(f"Candidate macro: {cand_macro:.4f}  ({rel:+.1f}% vs baseline)")
    print(f"Target (+{args.target_pct:.0f}%): {target:.4f}  gap={cand_macro - target:+.4f}")
    print()

    bidx = index_by_task_proto(base)
    cidx = index_by_task_proto(cand)
    rows = []
    for key in sorted(set(bidx) | set(cidx)):
        b = bidx.get(key, 0.0)
        c = cidx.get(key, float("nan"))
        if key not in cidx:
            continue
        rows.append((c - b, key[0], key[1], b, c))

    print("Top gains:")
    for d, task, proto, b, c in sorted(rows, reverse=True)[:12]:
        print(f"  {task}/{proto}: {b:.4f} -> {c:.4f}  ({d:+.4f})")
    print("Top regressions:")
    for d, task, proto, b, c in sorted(rows)[:12]:
        print(f"  {task}/{proto}: {b:.4f} -> {c:.4f}  ({d:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
