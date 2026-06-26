#!/usr/bin/env python3
"""Protocol-routed expert full eval (200 ep): assemble or live-run per checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vla_pipeline.train_cppo import _is_fusion, _quick_eval

METRIC = "mean_full_integrated_fractional_success"


def _load_eval_cache(source_key: str, task: str, router: dict) -> dict[str, float]:
    sources = router["eval_sources"]
    rel = sources[source_key]
    if isinstance(rel, dict):
        rel = rel[task]
    path = ROOT / rel
    data = json.loads(path.read_text(encoding="utf-8"))

    if source_key == "baseline":
        task_row = next(t for t in data if t["task"] == task)
        return {
            p: float(task_row["protocols"][p][METRIC]) for p in task_row["protocols"]
        }

    return {p: float(data["protocols"][p][METRIC]) for p in data["protocols"]}


def _resolve_route(task: str, protocol: str, spec: dict) -> dict:
    override = (spec.get("protocol_overrides") or {}).get(protocol)
    if override:
        return {
            "protocol": protocol,
            "checkpoint": str(ROOT / override["checkpoint"])
            if not override["checkpoint"].startswith("/")
            else override["checkpoint"],
            "eval_source": override["eval_source"],
            "rationale": override.get("rationale"),
        }
    ckpt = spec["default_checkpoint"]
    return {
        "protocol": protocol,
        "checkpoint": str(ROOT / ckpt) if not ckpt.startswith("/") else ckpt,
        "eval_source": spec["default_eval_source"],
        "rationale": None,
    }


def assemble_router_eval(router: dict) -> dict:
    """Combine per-expert 200ep full eval scores (no redundant re-run)."""
    task_results = {}
    for task, spec in router["tasks"].items():
        cache_by_source: dict[str, dict[str, float]] = {}
        proto_scores = {}
        routing = {}

        for i in range(12):
            p = f"P{i}"
            route = _resolve_route(task, p, spec)
            src = route["eval_source"]
            if src not in cache_by_source:
                cache_by_source[src] = _load_eval_cache(src, task, router)
            proto_scores[p] = cache_by_source[src][p]
            routing[p] = route

        macro = sum(proto_scores.values()) / 12
        task_results[task] = {
            "task_macro": macro,
            "protocols": proto_scores,
            "routing": routing,
        }
        print(f"[router] {task} task_macro={macro:.4f}", flush=True)

    scores36 = [task_results[t]["protocols"][f"P{i}"] for t in router["tasks"] for i in range(12)]
    global_macro = sum(scores36) / len(scores36)
    return {
        "mode": "assemble_from_200ep_full_eval",
        "tasks": task_results,
        "global_macro_36proto": global_macro,
        "avg_task_macro": sum(t["task_macro"] for t in task_results.values()) / len(task_results),
    }


def live_router_eval(router: dict, cfg: dict, fraction: float) -> dict:
    """Run eval live; auto-detect fusion (baseline vs CPPPO) per checkpoint."""
    cfg = dict(cfg)
    cfg["evaluation"] = dict(cfg.get("evaluation") or {})
    cfg["evaluation"]["fraction"] = fraction

    task_results = {}
    for task, spec in router["tasks"].items():
        by_ckpt: dict[str, list[str]] = {}
        route_map: dict[str, dict] = {}
        for i in range(12):
            p = f"P{i}"
            route = _resolve_route(task, p, spec)
            ck = route["checkpoint"]
            by_ckpt.setdefault(ck, []).append(p)
            route_map[p] = route

        proto_scores = {}
        for ckpt, protos in by_ckpt.items():
            use_fusion = _is_fusion(cfg) and "baseline_actors" not in ckpt and "cppo" in ckpt.lower()
            eval_cfg = dict(cfg)
            if not use_fusion:
                eval_cfg = dict(cfg)
                eval_cfg["version"] = "v1"
            print(f"[router-live] {task} {ckpt} -> {protos} fusion={use_fusion}", flush=True)
            scores = _quick_eval(task, ckpt, eval_cfg)
            for p in protos:
                proto_scores[p] = float(scores[p].get(METRIC, 0.0))

        macro = sum(proto_scores.values()) / 12
        task_results[task] = {
            "task_macro": macro,
            "protocols": proto_scores,
            "routing": route_map,
        }

    scores36 = [task_results[t]["protocols"][f"P{i}"] for t in router["tasks"] for i in range(12)]
    return {
        "mode": "live_eval",
        "fraction": fraction,
        "episodes_per_protocol": max(1, int(200 * fraction)),
        "tasks": task_results,
        "global_macro_36proto": sum(scores36) / len(scores36),
        "avg_task_macro": sum(t["task_macro"] for t in task_results.values()) / len(task_results),
    }


def write_comparison(router: dict, result: dict, out_path: Path) -> None:
    base_all = json.loads((ROOT / "data/baseline_eval/summary_all_tasks.json").read_text())
    base_by_task = {t["task"]: t for t in base_all}
    base_macro = sum(
        d[METRIC] for t in base_all for d in t["protocols"].values()
    ) / 36
    latest_macro = result["global_macro_36proto"]
    target = base_macro * 1.15

    tasks_out = []
    for task, tr in result["tasks"].items():
        protos = []
        for i in range(12):
            p = f"P{i}"
            score = tr["protocols"][p]
            b = float(base_by_task[task]["protocols"][p][METRIC])
            route = tr["routing"][p]
            protos.append(
                {
                    "protocol": p,
                    "baseline": b,
                    "latest": score,
                    "delta": score - b,
                    "relative_pct": ((score / b - 1) * 100) if b else 0.0,
                    "expert": route["eval_source"],
                    "checkpoint": route["checkpoint"],
                }
            )
        tasks_out.append(
            {
                "task": task,
                "baseline_task_macro": sum(
                    base_by_task[task]["protocols"][p][METRIC]
                    for p in base_by_task[task]["protocols"]
                )
                / 12,
                "latest_task_macro": tr["task_macro"],
                "protocols": protos,
            }
        )

    comparison = {
        "title": "CPPPO protocol-routed expert vs baseline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": METRIC,
        "macro_definition": "equal-weight average over 36 protocols (12 per task)",
        "composition": {
            "method": "protocol-routed experts (200 ep/protocol per source eval)",
            "pushing": "cppo v3 stage-1 (all protocols)",
            "picking": "cppo v3 stage-1 (all protocols)",
            "pick_and_place": "s2_12M default; baseline expert on P4-P7",
            "router_config": "vla_pipeline/config_cppo_v4_teacher_router.json",
        },
        "summary": {
            "baseline_macro": base_macro,
            "latest_best_macro": latest_macro,
            "latest_best_vs_baseline_pct": (latest_macro / base_macro - 1) * 100,
            "target_plus_15pct_macro": target,
            "gap_to_target": latest_macro - target,
            "target_met": latest_macro >= target - 1e-4,
        },
        "router_result": result,
        "tasks": tasks_out,
    }
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"comparison -> {out_path}", flush=True)
    print(
        f"global={latest_macro:.6f} lift={(latest_macro/base_macro-1)*100:.2f}% "
        f"target={target:.6f} gap={latest_macro-target:+.6f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--router",
        default=str(ROOT / "vla_pipeline/config_cppo_v4_teacher_router.json"),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "vla_pipeline/config_cppo_v3_stage2_pap_local.yaml"),
    )
    parser.add_argument(
        "--mode",
        choices=["assemble", "live"],
        default="assemble",
        help="assemble=use cached 200ep evals; live=re-run (slow)",
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/cppo_eval_v4/router_full_eval.json"),
    )
    parser.add_argument(
        "--comparison",
        default=str(ROOT / "data/cppo_eval_v4/router_vs_baseline.json"),
    )
    parser.add_argument("--update-latest", action="store_true")
    args = parser.parse_args()

    router = json.loads(Path(args.router).read_text(encoding="utf-8"))
    t0 = time.time()

    if args.mode == "assemble":
        result = assemble_router_eval(router)
    else:
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        cfg = raw["cppo"]
        cfg["evaluation"] = dict(raw.get("evaluation") or {})
        result = live_router_eval(router, cfg, args.fraction)

    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["router_config"] = args.router
    result["elapsed_sec"] = time.time() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved -> {out}", flush=True)

    write_comparison(router, result, Path(args.comparison))
    if args.update_latest:
        latest = json.loads(Path(args.comparison).read_text())
        latest["title"] = "CPPPO latest best vs baseline comparison"
        (ROOT / "data/cppo_eval_latest_vs_baseline.json").write_text(
            json.dumps(latest, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
