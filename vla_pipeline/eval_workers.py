"""Multiprocessing workers for parallel CausalWorld evaluation."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from vla_pipeline.hybrid_policy import EASY_PROTOCOLS, effective_hybrid_beta
from vla_pipeline.quiet_utils import setup_quiet_env, suppress_stdout_stderr


def episode_integrated_success(episode_obj) -> float:
    if not episode_obj.infos:
        return 0.0
    total = sum(float(i.get("fractional_success", 0.0)) for i in episode_obj.infos)
    return total / len(episode_obj.infos)


def _teacher_only_for_protocol(payload: Dict[str, Any], protocol_name: str) -> bool:
    """Skip VLA load/inference when effective beta is zero (pure teacher)."""
    if not payload.get("use_hybrid"):
        return True
    hybrid_cfg = payload.get("hybrid_cfg") or {}
    task_name = payload.get("task_name", "")
    beta = effective_hybrid_beta(hybrid_cfg, task_name, protocol_name)
    return beta <= 1e-6


def run_protocol_episode_chunk(payload: Dict[str, Any]) -> List[float]:
    """
    Run a contiguous block of evaluation episodes in an isolated process.

    Each worker owns one PyBullet env + policy stack (teacher or hybrid).
    """
    gpu_id: Optional[int] = payload.get("gpu_id")
    if gpu_id is not None and gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    setup_quiet_env()
    logging.getLogger("root").setLevel(logging.ERROR)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from causal_world.envs.causalworld import CausalWorld
    from causal_world.loggers.data_recorder import DataRecorder
    from causal_world.task_generators.task import generate_task
    from causal_world.wrappers.protocol_wrapper import ProtocolWrapper

    from causal_world.benchmark import (
        PICKING_BENCHMARK,
        PICK_AND_PLACE_BENCHMARK,
        PUSHING_BENCHMARK,
    )
    from vla_pipeline.hybrid_policy import build_hybrid_policy
    from vla_pipeline.teacher import load_teacher

    benchmarks = {
        "pushing": PUSHING_BENCHMARK,
        "picking": PICKING_BENCHMARK,
        "pick_and_place": PICK_AND_PLACE_BENCHMARK,
    }

    task_name = payload["task_name"]
    benchmark = benchmarks[task_name]
    protocol = benchmark["evaluation_protocols"][payload["protocol_idx"]]

    task = generate_task(
        task_generator_id=benchmark["task_generator_id"],
        variables_space="space_a_b",
    )
    recorder = DataRecorder(output_directory=None)
    env = CausalWorld(
        task,
        **payload["world_params"],
        seed=int(payload["initial_seed"]),
        data_recorder=recorder,
        enable_visualization=False,
    )
    protocol.init_protocol(
        env=env,
        tracker=env.get_tracker(),
        fraction=float(payload["fraction"]),
    )
    wrapped = ProtocolWrapper(env, protocol)

    teacher_ckpt = payload.get("teacher_ckpt")
    with suppress_stdout_stderr():
        teacher_fn = load_teacher(task_name, checkpoint=teacher_ckpt)

    pname = protocol.get_name()
    teacher_only = _teacher_only_for_protocol(payload, pname)
    if payload["use_hybrid"] and not teacher_only:
        hybrid_cfg = payload["hybrid_cfg"]
        device = "cuda" if gpu_id is not None and gpu_id >= 0 else "cpu"
        with suppress_stdout_stderr():
            policy = build_hybrid_policy(
                hybrid_cfg,
                teacher_fn,
                task_name,
                int(payload["image_size"]),
                device=device,
            )
        policy.bind_env(env, task_name, pname)
        policy_fn = policy
    else:
        policy_fn = teacher_fn

    episode_start = int(payload["episode_start"])
    num_episodes = int(payload["num_episodes"])
    log_every = int(payload.get("log_every_episodes", 5))
    wrapped._elapsed_episodes = episode_start

    scores: List[float] = []
    for ep_i in range(num_episodes):
        obs = wrapped.reset()
        done = False
        while not done:
            action = policy_fn(obs)
            obs, _, done, _ = wrapped.step(action)
        scores.append(episode_integrated_success(recorder.get_current_episode()))
        recorder.clear_recorder()
        ep_done = ep_i + 1
        if log_every > 0 and (
            ep_done % log_every == 0 or ep_done == num_episodes
        ):
            run_mean = float(np.mean(scores)) if scores else 0.0
            print(
                f"  worker ep {episode_start + ep_done}/{episode_start + num_episodes} "
                f"{task_name}/{protocol.get_name()} run_mean={run_mean:.4f}",
                flush=True,
            )

    try:
        wrapped.close()
    except Exception:
        pass
    return scores


def run_eval_worker_batch(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run multiple protocol chunks in one process (teacher/VLA loaded once).

    Returns list of {protocol_idx, episode_start, scores}.
    """
    gpu_id: Optional[int] = payload.get("gpu_id")
    if gpu_id is not None and gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    setup_quiet_env()
    logging.getLogger("root").setLevel(logging.ERROR)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from causal_world.envs.causalworld import CausalWorld
    from causal_world.loggers.data_recorder import DataRecorder
    from causal_world.task_generators.task import generate_task
    from causal_world.wrappers.protocol_wrapper import ProtocolWrapper

    from causal_world.benchmark import (
        PICKING_BENCHMARK,
        PICK_AND_PLACE_BENCHMARK,
        PUSHING_BENCHMARK,
    )
    from vla_pipeline.hybrid_policy import build_hybrid_policy
    from vla_pipeline.teacher import load_teacher

    benchmarks = {
        "pushing": PUSHING_BENCHMARK,
        "picking": PICKING_BENCHMARK,
        "pick_and_place": PICK_AND_PLACE_BENCHMARK,
    }

    task_name = payload["task_name"]
    benchmark = benchmarks[task_name]
    teacher_ckpt = payload.get("teacher_ckpt")
    with suppress_stdout_stderr():
        teacher_fn = load_teacher(task_name, checkpoint=teacher_ckpt)

    hybrid_policy = None
    results: List[Dict[str, Any]] = []

    for sub in payload["jobs"]:
        protocol_idx = int(sub["protocol_idx"])
        protocol = benchmark["evaluation_protocols"][protocol_idx]
        pname = protocol.get_name()
        teacher_only = _teacher_only_for_protocol(payload, pname)

        if not teacher_only and hybrid_policy is None:
            device = "cuda" if gpu_id is not None and gpu_id >= 0 else "cpu"
            with suppress_stdout_stderr():
                hybrid_policy = build_hybrid_policy(
                    payload["hybrid_cfg"],
                    teacher_fn,
                    task_name,
                    int(payload["image_size"]),
                    device=device,
                )
            hybrid_policy.eval() if hasattr(hybrid_policy, "eval") else None
            if hasattr(hybrid_policy, "model"):
                hybrid_policy.model.eval()

        task = generate_task(
            task_generator_id=benchmark["task_generator_id"],
            variables_space="space_a_b",
        )
        recorder = DataRecorder(output_directory=None)
        env = CausalWorld(
            task,
            **payload["world_params"],
            seed=int(payload["initial_seed"]),
            data_recorder=recorder,
            enable_visualization=False,
        )
        wrapped = ProtocolWrapper(env, protocol)
        protocol.init_protocol(
            env=env,
            tracker=env.get_tracker(),
            fraction=float(payload["fraction"]),
        )

        if not teacher_only and hybrid_policy is not None:
            hybrid_policy.bind_env(env, task_name, pname)
            policy_fn = hybrid_policy
        else:
            policy_fn = teacher_fn

        episode_start = int(sub["episode_start"])
        num_episodes = int(sub["num_episodes"])
        log_every = int(payload.get("log_every_episodes", 5))
        wrapped._elapsed_episodes = episode_start

        scores: List[float] = []
        for ep_i in range(num_episodes):
            obs = wrapped.reset()
            done = False
            while not done:
                action = policy_fn(obs)
                obs, _, done, _ = wrapped.step(action)
            scores.append(episode_integrated_success(recorder.get_current_episode()))
            recorder.clear_recorder()
            ep_done = ep_i + 1
            if log_every > 0 and (
                ep_done % log_every == 0 or ep_done == num_episodes
            ):
                run_mean = float(np.mean(scores)) if scores else 0.0
                print(
                    f"  worker ep {episode_start + ep_done}/"
                    f"{episode_start + num_episodes} "
                    f"{task_name}/{pname} run_mean={run_mean:.4f}",
                    flush=True,
                )

        try:
            wrapped.close()
        except Exception:
            pass

        results.append(
            {
                "protocol_idx": protocol_idx,
                "episode_start": episode_start,
                "scores": scores,
            }
        )
        if scores:
            run_mean = float(np.mean(scores))
            print(
                f"  [eval] {task_name}/{pname} "
                f"ep[{episode_start}:{episode_start + num_episodes}) "
                f"mean={run_mean:.4f} (n={len(scores)})",
                flush=True,
            )

    return results


def split_episode_chunks(n_episodes: int, num_workers: int) -> List[tuple]:
    """Return [(episode_start, count), ...] for up to num_workers chunks."""
    if n_episodes <= 0:
        return []
    num_workers = max(1, min(num_workers, n_episodes))
    base = n_episodes // num_workers
    rem = n_episodes % num_workers
    chunks = []
    start = 0
    for i in range(num_workers):
        count = base + (1 if i < rem else 0)
        if count > 0:
            chunks.append((start, count))
            start += count
    return chunks


def aggregate_scores(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {
            "mean_full_integrated_fractional_success": 0.0,
            "std_full_integrated_fractional_success": 0.0,
        }
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean_full_integrated_fractional_success": float(arr.mean()),
        "std_full_integrated_fractional_success": float(arr.std()),
    }
