#!/usr/bin/env python3
"""
Evaluate hybrid VLA+teacher vs baseline on all 12 protocols x 3 tasks.

Supports parallel evaluation via evaluation.num_workers (>1).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from vla_pipeline.quiet_utils import setup_quiet_env

from causal_world.benchmark import (
    PICKING_BENCHMARK,
    PICK_AND_PLACE_BENCHMARK,
    PUSHING_BENCHMARK,
)
from causal_world.evaluation.evaluation import EvaluationPipeline

from vla_pipeline.eval_workers import (
    aggregate_scores,
    run_eval_worker_batch,
    run_protocol_episode_chunk,
    split_episode_chunks,
)
from vla_pipeline.hybrid_policy import build_hybrid_policy, HybridVLAPolicy
from vla_pipeline.teacher import load_teacher

BENCHMARKS = {
    "pushing": ("pushing_ppo_curr1", PUSHING_BENCHMARK),
    "picking": ("picking_ppo_curr1", PICKING_BENCHMARK),
    "pick_and_place": ("pick_and_place_ppo_curr0", PICK_AND_PLACE_BENCHMARK),
}


def _resolve_gpu_ids(num_workers: int, gpu_ids_cfg) -> List[Optional[int]]:
    if gpu_ids_cfg:
        ids = [int(g) for g in gpu_ids_cfg]
    else:
        try:
            import torch

            n_gpu = torch.cuda.device_count()
        except Exception:
            n_gpu = 0
        ids = list(range(n_gpu)) if n_gpu > 0 else [None]
    if not ids:
        ids = [None]
    return [ids[i % len(ids)] for i in range(num_workers)]


def macro_average(results: list) -> float:
    scores = []
    for task in results:
        for p in task["protocols"].values():
            scores.append(p["mean_full_integrated_fractional_success"])
    return sum(scores) / max(1, len(scores))


def _resolve_n_episodes(
    eval_cfg: dict, task_name: str, fraction: float, episodes_override: Optional[int]
) -> int:
    if episodes_override is not None:
        return int(episodes_override)
    by_task = eval_cfg.get("episodes_per_protocol_by_task") or {}
    if task_name in by_task:
        return int(by_task[task_name])
    if eval_cfg.get("fast_eval"):
        return int(eval_cfg.get("episodes_per_protocol_fast", 50))
    return int(eval_cfg.get("episodes_per_protocol", 200) * fraction)


def _evaluate_task_serial(task_name, cfg, use_hybrid: bool):
    model_tag, benchmark = BENCHMARKS[task_name]
    teacher_ckpt = (cfg.get("teacher", {}).get("checkpoints", {}) or {}).get(task_name)
    teacher_fn = load_teacher(task_name, checkpoint=teacher_ckpt)

    world_params = {
        "observation_mode": "structured",
        "normalize_observations": True,
        "normalize_actions": True,
        "skip_frame": int(cfg.get("evaluation", {}).get("skip_frame", 3)),
    }
    task_params = {"task_generator_id": benchmark["task_generator_id"]}

    evaluator = EvaluationPipeline(
        evaluation_protocols=benchmark["evaluation_protocols"],
        world_params=world_params,
        task_params=task_params,
        initial_seed=0,
    )

    fraction = float(cfg["evaluation"]["fraction"])

    if use_hybrid:
        hybrid_cfg = cfg["hybrid"]
        policy = build_hybrid_policy(
            hybrid_cfg,
            teacher_fn,
            task_name,
            int(cfg["training"]["image_size"]),
        )

        def policy_fn(obs):
            return policy(obs)

        pipeline_scores = {}
        for evaluation_protocol in evaluator.evaluation_protocols:
            from causal_world.wrappers.protocol_wrapper import ProtocolWrapper

            evaluator.evaluation_env = ProtocolWrapper(evaluator.env, evaluation_protocol)
            evaluation_protocol.init_protocol(
                env=evaluator.env,
                tracker=evaluator.env.get_tracker(),
                fraction=fraction,
            )
            policy.bind_env(evaluator.env, task_name, evaluation_protocol.get_name())
            for _ in range(evaluation_protocol.get_num_episodes()):
                ep = evaluator.run_episode(policy_fn)
                evaluator.process_metrics(ep)
                evaluator.data_recorder.clear_recorder()
            pipeline_scores[evaluation_protocol.get_name()] = evaluator.get_metric_scores()
            evaluator.reset_metric_scores()
        evaluator.evaluation_env.close()
    else:
        pipeline_scores = evaluator.evaluate_policy(teacher_fn, fraction=fraction)

    return _format_task_result(task_name, model_tag, cfg, pipeline_scores, use_hybrid)


def _evaluate_task_parallel(
    task_name,
    cfg,
    use_hybrid: bool,
    num_workers: int,
    gpu_ids_cfg,
    progress: Optional[dict] = None,
    protocol_filter: Optional[List[str]] = None,
    episodes_override: Optional[int] = None,
):
    model_tag, benchmark = BENCHMARKS[task_name]
    eval_cfg = cfg["evaluation"]
    fraction = float(eval_cfg["fraction"])
    n_episodes = _resolve_n_episodes(eval_cfg, task_name, fraction, episodes_override)
    gpu_ids = _resolve_gpu_ids(num_workers, gpu_ids_cfg)

    world_params = {
        "observation_mode": "structured",
        "normalize_observations": True,
        "normalize_actions": True,
        "skip_frame": int(cfg.get("evaluation", {}).get("skip_frame", 3)),
    }
    teacher_ckpt = (cfg.get("teacher", {}).get("checkpoints", {}) or {}).get(task_name)
    if teacher_ckpt:
        teacher_ckpt = os.path.abspath(teacher_ckpt)

    hybrid_cfg = dict(cfg.get("hybrid", {}))
    if hybrid_cfg.get("checkpoint"):
        hybrid_cfg["checkpoint"] = os.path.abspath(hybrid_cfg["checkpoint"])

    pipeline_scores = {}
    all_protocols = benchmark["evaluation_protocols"]
    protocols = all_protocols
    if protocol_filter:
        pf = set(protocol_filter)
        protocols = [p for p in all_protocols if p.get_name() in pf]

    task_t0 = time.time()
    use_batch_workers = bool(eval_cfg.get("batch_workers", True))

    if use_batch_workers and num_workers > 1:
        worker_batches: List[List[dict]] = [[] for _ in range(num_workers)]
        for protocol in protocols:
            protocol_idx = all_protocols.index(protocol)
            pname = protocol.get_name()
            chunks = split_episode_chunks(n_episodes, num_workers)
            for worker_idx, (episode_start, count) in enumerate(chunks):
                worker_batches[worker_idx].append(
                    {
                        "protocol_idx": protocol_idx,
                        "episode_start": episode_start,
                        "num_episodes": count,
                    }
                )

        for wi, sub_jobs in enumerate(worker_batches):
            random.Random(42 + wi).shuffle(sub_jobs)

        batch_jobs = []
        for worker_idx, sub_jobs in enumerate(worker_batches):
            if not sub_jobs:
                continue
            batch_jobs.append(
                {
                    "task_name": task_name,
                    "jobs": sub_jobs,
                    "initial_seed": 0,
                    "fraction": fraction,
                    "world_params": world_params,
                    "use_hybrid": use_hybrid,
                    "hybrid_cfg": hybrid_cfg,
                    "image_size": int(cfg["training"]["image_size"]),
                    "teacher_ckpt": teacher_ckpt,
                    "gpu_id": gpu_ids[worker_idx % len(gpu_ids)],
                    "log_every_episodes": int(eval_cfg.get("log_every_episodes", 5)),
                }
            )

        if progress is not None:
            progress["protocol_idx"] += len(protocols)
            print(
                f"[batch] {task_name}: {len(protocols)} protocols x {n_episodes} eps | "
                f"{len(batch_jobs)} persistent workers",
                flush=True,
            )

        protocol_scores: dict = {p.get_name(): [] for p in protocols}
        workers_done = 0
        with ProcessPoolExecutor(max_workers=len(batch_jobs)) as pool:
            futures = [pool.submit(run_eval_worker_batch, job) for job in batch_jobs]
            for fut in as_completed(futures):
                workers_done += 1
                for block in fut.result():
                    pidx = block["protocol_idx"]
                    pname = benchmark["evaluation_protocols"][pidx].get_name()
                    protocol_scores[pname].extend(block["scores"])
                ep_done = sum(len(v) for v in protocol_scores.values())
                ep_total = len(protocols) * n_episodes
                elapsed = time.time() - task_t0
                print(
                    f"[batch] {task_name}: worker {workers_done}/{len(batch_jobs)} done | "
                    f"eps {ep_done}/{ep_total} | {elapsed / 60:.1f} min",
                    flush=True,
                )

        for protocol in protocols:
            pname = protocol.get_name()
            all_scores = protocol_scores.get(pname, [])
            pipeline_scores[pname] = aggregate_scores(all_scores)
            mean = pipeline_scores[pname]["mean_full_integrated_fractional_success"]
            std = pipeline_scores[pname]["std_full_integrated_fractional_success"]
            if progress is not None:
                print(
                    f"  {task_name}/{pname} mean={mean:.4f} std={std:.4f} "
                    f"({len(all_scores)} eps)",
                    flush=True,
                )
    else:
        for protocol in protocols:
            protocol_idx = all_protocols.index(protocol)
            pname = protocol.get_name()
            chunks = split_episode_chunks(n_episodes, num_workers)
            if progress is not None:
                progress["protocol_idx"] += 1
                pi = progress["protocol_idx"]
                pt = progress["protocol_total"]
                print(
                    f"[{pi}/{pt}] {task_name}/{pname} start | "
                    f"{n_episodes} eps x {len(chunks)} workers",
                    flush=True,
                )
            else:
                print(
                    f"  [{task_name}/{pname}] {n_episodes} eps / "
                    f"{len(chunks)} workers (parallel)",
                    flush=True,
                )

            jobs = []
            for worker_idx, (episode_start, count) in enumerate(chunks):
                jobs.append(
                    {
                        "task_name": task_name,
                        "protocol_idx": protocol_idx,
                        "episode_start": episode_start,
                        "num_episodes": count,
                        "initial_seed": 0,
                        "fraction": fraction,
                        "world_params": world_params,
                        "use_hybrid": use_hybrid,
                        "hybrid_cfg": hybrid_cfg,
                        "image_size": int(cfg["training"]["image_size"]),
                        "teacher_ckpt": teacher_ckpt,
                        "gpu_id": gpu_ids[worker_idx % len(gpu_ids)],
                        "log_every_episodes": int(eval_cfg.get("log_every_episodes", 5)),
                    }
                )

            all_scores: List[float] = []
            proto_t0 = time.time()
            workers_done = 0
            with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
                futures = {
                    pool.submit(run_protocol_episode_chunk, job): worker_idx
                    for worker_idx, job in enumerate(jobs)
                }
                for fut in as_completed(futures):
                    worker_idx = futures[fut]
                    scores = fut.result()
                    all_scores.extend(scores)
                    workers_done += 1
                    running_mean = float(np.mean(all_scores)) if all_scores else 0.0
                    ep_done = len(all_scores)
                    elapsed = time.time() - proto_t0
                    eps_per_sec = ep_done / max(elapsed, 1e-6)
                    eta = (n_episodes - ep_done) / max(eps_per_sec, 1e-6)
                    if progress is not None:
                        pi = progress["protocol_idx"]
                        pt = progress["protocol_total"]
                        print(
                            f"[{pi}/{pt}] {task_name}/{pname} | "
                            f"workers {workers_done}/{len(jobs)} | "
                            f"eps {ep_done}/{n_episodes} | "
                            f"run_mean={running_mean:.4f} | "
                            f"ETA {eta:.0f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"    worker {worker_idx} done ({len(scores)} eps)",
                            flush=True,
                        )

            pipeline_scores[pname] = aggregate_scores(all_scores)
            mean = pipeline_scores[pname]["mean_full_integrated_fractional_success"]
            std = pipeline_scores[pname]["std_full_integrated_fractional_success"]
            proto_elapsed = time.time() - proto_t0
            if progress is not None:
                pi = progress["protocol_idx"]
                pt = progress["protocol_total"]
                print(
                    f"[{pi}/{pt}] {task_name}/{pname} done | "
                    f"mean={mean:.4f} std={std:.4f} | {proto_elapsed:.0f}s",
                    flush=True,
                )
            else:
                print(f"  [{task_name}/{pname}] mean={mean:.4f}", flush=True)

    task_elapsed = time.time() - task_t0
    if progress is not None:
        print(f"=== {task_name} finished in {task_elapsed / 60:.1f} min ===", flush=True)

    return _format_task_result(
        task_name, model_tag, cfg, pipeline_scores, use_hybrid, n_episodes
    )


def _format_task_result(
    task_name, model_tag, cfg, pipeline_scores, use_hybrid, n_episodes: Optional[int] = None
):
    fraction = float(cfg["evaluation"]["fraction"])
    if n_episodes is None:
        n_episodes = int(cfg["evaluation"]["episodes_per_protocol"] * fraction)
    out = {
        "task": task_name,
        "model": f"hybrid_vla_{model_tag}" if use_hybrid else model_tag,
        "fraction": fraction,
        "metric": "mean_full_integrated_fractional_success",
        "protocols": {},
    }
    for pname, scores in pipeline_scores.items():
        out["protocols"][pname] = {
            "mean_full_integrated_fractional_success": scores[
                "mean_full_integrated_fractional_success"
            ],
            "std_full_integrated_fractional_success": scores[
                "std_full_integrated_fractional_success"
            ],
            "episodes": n_episodes,
        }
    return out


def evaluate_task(
    task_name,
    cfg,
    use_hybrid: bool,
    num_workers: int = 1,
    gpu_ids=None,
    progress: Optional[dict] = None,
    protocol_filter: Optional[List[str]] = None,
    episodes_override: Optional[int] = None,
):
    if num_workers <= 1:
        return _evaluate_task_serial(task_name, cfg, use_hybrid)
    return _evaluate_task_parallel(
        task_name,
        cfg,
        use_hybrid,
        num_workers,
        gpu_ids,
        progress=progress,
        protocol_filter=protocol_filter,
        episodes_override=episodes_override,
    )


def main():
    setup_quiet_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_v2.yaml"))
    parser.add_argument("--teacher_only", action="store_true", help="Sanity-check teacher (no VLA)")
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel env workers (overrides config)")
    parser.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks to evaluate")
    parser.add_argument(
        "--protocols",
        nargs="*",
        default=None,
        help="Subset of protocols e.g. P6 P7 (default: all 12)",
    )
    parser.add_argument(
        "--episodes_per_protocol",
        type=int,
        default=None,
        help="Override evaluation.episodes_per_protocol",
    )
    parser.add_argument(
        "--no-batch-workers",
        action="store_true",
        help="Evaluate protocol-by-protocol (more log output, slower startup)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = cfg["evaluation"]
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else eval_cfg.get("num_workers", 1)
    )
    gpu_ids = eval_cfg.get("gpu_ids")
    tasks = args.tasks or cfg["data"]["tasks"]
    protocol_filter = args.protocols
    if protocol_filter:
        protocol_total = len(protocol_filter) * len(tasks)
    else:
        protocol_total = sum(len(BENCHMARKS[t][1]["evaluation_protocols"]) for t in tasks)
    episodes_override = args.episodes_per_protocol
    if args.no_batch_workers:
        cfg = dict(cfg)
        cfg["evaluation"] = dict(cfg["evaluation"])
        cfg["evaluation"]["batch_workers"] = False
    progress = {"protocol_idx": 0, "protocol_total": protocol_total}
    mode = "teacher" if args.teacher_only else "hybrid"
    sample_eps = _resolve_n_episodes(eval_cfg, tasks[0], float(eval_cfg["fraction"]), episodes_override)

    print(
        f"Evaluation ({mode}): num_workers={num_workers}, "
        f"episodes_per_protocol~{sample_eps} (per-task overrides in config), "
        f"batch_workers={cfg['evaluation'].get('batch_workers', True)}, "
        f"tasks={tasks}, protocols={protocol_filter or 'all'}, "
        f"protocol_slots={protocol_total}",
        flush=True,
    )

    eval_t0 = time.time()
    results = []
    for task_name in tasks:
        print(f"=== Task {task_name} ===", flush=True)
        res = evaluate_task(
            task_name,
            cfg,
            use_hybrid=not args.teacher_only,
            num_workers=num_workers,
            gpu_ids=gpu_ids,
            progress=progress,
            protocol_filter=protocol_filter,
            episodes_override=episodes_override,
        )
        path = out_dir / f"{task_name}_all_protocols.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        results.append(res)
        print(f"[done] {path}", flush=True)

    summary_path = out_dir / "summary_all_tasks.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    macro = macro_average(results)
    target = float(cfg["evaluation"]["target_macro_avg"])
    total_min = (time.time() - eval_t0) / 60.0
    print(
        f"Macro avg = {macro:.4f} (target {target:.4f}) | "
        f"total {total_min:.1f} min",
        flush=True,
    )

    baseline_path = Path(cfg["evaluation"]["baseline_dir"]) / "summary_all_tasks.json"
    if baseline_path.is_file():
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
        base_macro = macro_average(baseline)
        rel = (macro - base_macro) / base_macro * 100 if base_macro > 0 else 0.0
        print(f"Baseline macro avg = {base_macro:.4f} | delta = {rel:+.1f}%")


if __name__ == "__main__":
    main()
