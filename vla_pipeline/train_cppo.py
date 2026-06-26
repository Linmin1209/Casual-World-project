#!/usr/bin/env python3
"""
CPPPO: finetune privileged PPO2 teachers on evaluation protocol distribution.

Uses ProtocolCurriculumWrapper (weighted P0–P11) instead of random curriculum,
matching the benchmark eval intervention structure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BENCHMARKS = {
    "pushing": "causal_world.benchmark.benchmarks.PUSHING_BENCHMARK",
    "picking": "causal_world.benchmark.benchmarks.PICKING_BENCHMARK",
    "pick_and_place": "causal_world.benchmark.benchmarks.PICK_AND_PLACE_BENCHMARK",
}


def _import_benchmark(task_name: str):
    import importlib

    mod_path, attr = {
        "pushing": ("causal_world.benchmark.benchmarks", "PUSHING_BENCHMARK"),
        "picking": ("causal_world.benchmark.benchmarks", "PICKING_BENCHMARK"),
        "pick_and_place": ("causal_world.benchmark.benchmarks", "PICK_AND_PLACE_BENCHMARK"),
    }[task_name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _resolve_num_envs(cppo: dict, task_name: str, cli_override: int | None = None) -> int:
    if cli_override is not None:
        return max(1, int(cli_override))
    by_task = cppo.get("num_envs_by_task") or {}
    if task_name in by_task:
        return max(1, int(by_task[task_name]))
    return max(1, int(cppo.get("num_envs", 8)))


def _resolve_nminibatches(ppo_cfg: dict, num_envs: int) -> int:
    nb = ppo_cfg.get("nminibatches")
    if nb is not None and int(nb) > 0:
        return max(1, int(nb))
    # Match official ratio (~2x num_envs) for stable PPO2 updates.
    return max(8, min(int(num_envs) * 2, 64))


def _protocol_weights(cfg: dict, task_name: str) -> dict:
    pw = (cfg.get("protocol_weights") or {}).get(task_name) or {}
    return dict(pw)


def _is_v2(cfg: dict) -> bool:
    return str(cfg.get("version", "v1")).lower() in ("v2", "2", "cppo_v2")


def _is_v3(cfg: dict) -> bool:
    return str(cfg.get("version", "v1")).lower() in ("v3", "3", "cppo_v3")


def _is_fusion(cfg: dict) -> bool:
    return _is_v2(cfg) or _is_v3(cfg)


def _checkpoint_suffix(cfg: dict) -> str:
    if _is_v3(cfg):
        return "cppo_v3"
    return "cppo_v2" if _is_v2(cfg) else "cppo"


def _read_progress_timesteps(log_dir: Path, out_dir: Path) -> int:
    """Best-effort cumulative env steps (SB2 zips often omit num_timesteps)."""
    best = 0
    jsonl = log_dir / "ppo_rollouts.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            best = max(
                best,
                int(rec.get("total_timesteps") or rec.get("timesteps") or 0),
            )
    meta_path = out_dir / "cppo_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            best = max(best, int(meta.get("timesteps_done") or 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return best


def _resolve_timesteps_done(model, log_dir: Path, out_dir: Path) -> int:
    done = int(getattr(model, "num_timesteps", 0) or 0)
    logged = _read_progress_timesteps(log_dir, out_dir)
    if logged > done:
        model.num_timesteps = logged
        done = logged
    return done


def _make_env_fn(task_name: str, cfg: dict, rank: int, log_dir: Path):
    def _init():
        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        from stable_baselines.bench.monitor import Monitor

        from causal_world.envs.causalworld import CausalWorld
        from causal_world.task_generators.task import generate_task
        from causal_world.wrappers.protocol_curriculum_wrapper import (
            ProtocolCurriculumWrapper,
        )

        benchmark = _import_benchmark(task_name)
        task_cfg = dict(cfg["task_configs"][task_name])
        task = generate_task(
            task_generator_id=benchmark["task_generator_id"],
            variables_space=str(cfg.get("variables_space", "space_a_b")),
            **task_cfg,
        )
        world_params = {
            "observation_mode": "structured",
            "normalize_observations": True,
            "normalize_actions": True,
            "action_mode": "joint_positions",
            "skip_frame": int(cfg["skip_frame"]),
            "enable_visualization": False,
        }
        seed = int(cfg["seed"]) + rank
        env = CausalWorld(task, **world_params, seed=seed)

        sampling = cfg.get("sampling") or {}
        fusion = _is_fusion(cfg)
        allow_map = cfg.get("protocol_allowlist_by_task") or {}
        allowlist = allow_map.get(task_name) or cfg.get("protocol_allowlist")
        env = ProtocolCurriculumWrapper(
            env,
            evaluation_protocols=benchmark["evaluation_protocols"],
            protocol_weights=_protocol_weights(cfg, task_name),
            seed=seed + 1000,
            adaptive_sampling=fusion and bool(sampling.get("adaptive", True)),
            adaptive_alpha=float(sampling.get("adaptive_alpha", 1.5)),
            adaptive_momentum=float(sampling.get("adaptive_momentum", 0.92)),
            hard_protocol_min_weight=float(sampling.get("hard_protocol_min_weight", 0.08)),
            stratified_static_ratio=float(sampling.get("stratified_static_ratio", 0.30)),
            protocol_allowlist=allowlist,
            anchor_protocols=cfg.get("anchor_protocols"),
            anchor_min_sampling_ratio=float(cfg.get("anchor_min_sampling_ratio", 0.0)),
            augment_obs=fusion,
        )
        monitor_path = log_dir / f"monitor_{rank}"
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=("fractional_success", "cppo_protocol"),
        )
        return env

    return _init


def _train_task(task_name: str, cfg: dict, cfg_path: str, num_envs_cli: int | None = None) -> str:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines import PPO2
    from stable_baselines.common import set_global_seeds
    from stable_baselines.common.callbacks import CheckpointCallback, CallbackList
    from stable_baselines.common.vec_env import SubprocVecEnv

    from vla_pipeline.cppo_curves import CppoCurveCallback, export_training_curves
    from vla_pipeline.cppo_policy import CpppoFusionPolicy
    from vla_pipeline.cppo_transfer import build_cppo_fusion_model

    cppo = cfg
    ppo_cfg = cppo["ppo"]
    fusion = _is_fusion(cppo)
    v3 = _is_v3(cppo)
    suffix = _checkpoint_suffix(cppo)
    out_dir = Path(cppo["checkpoint_dir"]) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = str(out_dir / "tensorboard")
    ppo_jsonl = log_dir / "ppo_rollouts.jsonl"

    init_ckpt = cppo["init_checkpoints"][task_name]
    init_ckpt = os.path.abspath(init_ckpt)
    if not os.path.isfile(init_ckpt):
        raise FileNotFoundError(f"Missing init checkpoint for {task_name}: {init_ckpt}")

    num_envs = _resolve_num_envs(cppo, task_name, num_envs_cli)
    nminibatches = _resolve_nminibatches(ppo_cfg, num_envs)
    set_global_seeds(int(cppo["seed"]))

    env = SubprocVecEnv([_make_env_fn(task_name, cppo, rank, log_dir) for rank in range(num_envs)])

    policy_cfg = cppo.get("policy") or {}
    policy_kwargs = dict(
        act_fun=tf.nn.tanh,
        embed_dim=int(policy_cfg.get("embed_dim", 64)),
    )
    if fusion:
        arch0 = (policy_cfg.get("net_arch") or [{"pi": [512, 512], "vf": [512, 256]}])[0]
        policy_kwargs["net_arch"] = [
            {"pi": list(arch0.get("pi", [512, 512])), "vf": list(arch0.get("vf", [512, 256]))}
        ]
        if v3:
            from vla_pipeline.cppo_policy_v3 import CpppoResidualPolicy

            policy_kwargs["proto_scale"] = float(policy_cfg.get("proto_scale", 1.0))
            policy_cls = CpppoResidualPolicy
        else:
            policy_cls = CpppoFusionPolicy
    else:
        policy_kwargs["net_arch"] = list(ppo_cfg.get("net_arch", [256, 256]))
        from stable_baselines.common.policies import MlpPolicy

        policy_cls = MlpPolicy
    train_kw = {
        "gamma": float(ppo_cfg["gamma"]),
        "n_steps": int(ppo_cfg["n_steps"]),
        "ent_coef": float(ppo_cfg["ent_coef"]),
        "learning_rate": float(ppo_cfg["learning_rate"]),
        "vf_coef": float(ppo_cfg["vf_coef"]),
        "max_grad_norm": float(ppo_cfg["max_grad_norm"]),
        "nminibatches": nminibatches,
        "noptepochs": int(ppo_cfg["noptepochs"]),
    }

    print(
        f"[CPPPO] Building {task_name} ({'v3' if v3 else ('v2' if _is_v2(cppo) else 'v1')})",
        flush=True,
    )
    total_ts = int(cppo["total_timesteps"])
    final_path = out_dir / f"{task_name}_{suffix}.zip"
    resume = bool(cppo.get("resume", True))
    warm_stats = {}
    train_steps = total_ts

    if resume and final_path.is_file():
        print(f"[CPPPO] Resuming {task_name} from {final_path}", flush=True)
        load_kw = dict(
            env=env,
            _init_setup_model=True,
            verbose=1,
            tensorboard_log=tb_dir,
            **train_kw,
        )
        if fusion:
            # Resume must match stored policy architecture.
            pass
        model = PPO2.load(str(final_path.with_suffix("")), **load_kw)
        done = _resolve_timesteps_done(model, log_dir, out_dir)
        train_steps = max(0, total_ts - done)
        warm_stats = {"mode": "resume", "from": str(final_path), "timesteps_done": done}
        print(
            f"[CPPPO] {task_name}: {done}/{total_ts} done, training {train_steps} more steps",
            flush=True,
        )
    elif fusion and not bool(cppo.get("warm_start_baseline", True)) and os.path.isfile(init_ckpt):
        print(f"[CPPPO] Loading {task_name} from prior CPPPO checkpoint {init_ckpt}", flush=True)
        load_kw = dict(
            env=env,
            _init_setup_model=True,
            verbose=1,
            tensorboard_log=tb_dir,
            **train_kw,
        )
        # Use architecture stored in checkpoint (avoid policy_kwargs mismatch on stage-2).
        model = PPO2.load(str(Path(init_ckpt).with_suffix("")), **load_kw)
        warm_stats = {"mode": "cppo_checkpoint", "from": init_ckpt}
    elif fusion:
        model, warm_stats = build_cppo_fusion_model(
            init_ckpt,
            env,
            policy_cls,
            policy_kwargs,
            train_kw,
            warm_start=bool(cppo.get("warm_start_baseline", True)),
            tensorboard_log=tb_dir,
            residual=v3,
        )
    else:
        print(f"[CPPPO] Loading {task_name} from {init_ckpt}", flush=True)
        model = PPO2.load(
            init_ckpt,
            env=env,
            _init_setup_model=True,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tb_dir,
            **train_kw,
        )
        warm_stats = {}
    validate_every = int(cppo.get("validate_every_timesteps", 200000))
    ckpt_freq = max(1, int(validate_every / num_envs))
    checkpoint_cb = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=str(out_dir / "intermediate"),
        name_prefix=f"{task_name}_{suffix}",
    )
    curve_cb = CppoCurveCallback(str(ppo_jsonl))
    callbacks = CallbackList([checkpoint_cb, curve_cb])

    meta = {
        "task": task_name,
        "version": "v3" if v3 else ("v2" if _is_v2(cppo) else "v1"),
        "stage": int(cppo.get("stage", 1)),
        "init_checkpoint": init_ckpt,
        "config": cfg_path,
        "protocol_weights": _protocol_weights(cppo, task_name),
        "total_timesteps": total_ts,
        "train_steps_this_run": train_steps,
        "num_envs": num_envs,
        "skip_frame": int(cppo["skip_frame"]),
        "warm_start": warm_stats,
        "augment_obs": fusion,
        "tensorboard_dir": tb_dir,
        "ppo_rollouts_jsonl": str(ppo_jsonl),
        "ppo_n_steps": int(ppo_cfg["n_steps"]),
    }
    (out_dir / "cppo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rollout_batch = int(ppo_cfg["n_steps"]) * num_envs
    n_updates_est = train_steps // max(1, rollout_batch)
    print(
        f"[CPPPO] Training {task_name}: {train_steps} steps this run "
        f"(target {total_ts}), {num_envs} envs, ~{n_updates_est} PPO updates, "
        f"n_steps={ppo_cfg['n_steps']}, nminibatches={nminibatches}, "
        f"skip_frame={cppo['skip_frame']}, version={'v3' if v3 else ('v2' if _is_v2(cppo) else 'v1')}",
        flush=True,
    )
    if train_steps <= 0:
        print(f"[CPPPO] {task_name} already at {total_ts} timesteps, skipping train", flush=True)
    else:
        model.learn(train_steps, callback=callbacks, reset_num_timesteps=False)

    meta["timesteps_done"] = int(getattr(model, "num_timesteps", 0) or 0)
    (out_dir / "cppo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    final_path = out_dir / f"{task_name}_{suffix}.zip"
    model.save(str(final_path.with_suffix("")))  # stable_baselines adds .zip
    env.close()

    curves_path = export_training_curves(out_dir, task_name, log_dir, ppo_jsonl)
    print(f"[CPPPO] Training curves -> {curves_path}", flush=True)

    if not final_path.is_file():
        # PPO2.save may write without double extension
        alt = out_dir / f"{task_name}_{suffix}.zip"
        if not alt.is_file():
            candidates = sorted(out_dir.glob(f"{task_name}_{suffix}*.zip"))
            if candidates:
                final_path = candidates[-1]
    print(f"[CPPPO] Saved {task_name} -> {final_path}", flush=True)
    return str(final_path)


def _quick_eval(task_name: str, ckpt_path: str, cfg: dict) -> dict:
    """Short eval on all protocols (fraction of default 200 ep)."""
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from stable_baselines import PPO2

    from causal_world.envs.causalworld import CausalWorld
    from causal_world.loggers.data_recorder import DataRecorder
    from causal_world.task_generators.task import generate_task
    from causal_world.wrappers.protocol_wrapper import ProtocolWrapper
    from causal_world.wrappers.protocol_obs_wrapper import ProtocolObsWrapper

    benchmark = _import_benchmark(task_name)
    eval_cfg = cfg.get("evaluation", {})
    fraction = float(eval_cfg.get("fraction", 0.1))
    v_fusion = _is_fusion(cfg)
    world_params = {
        "observation_mode": "structured",
        "normalize_observations": True,
        "normalize_actions": True,
        "skip_frame": int(eval_cfg.get("skip_frame", cfg["skip_frame"])),
    }
    task_cfg = dict(cfg["task_configs"][task_name])
    model = PPO2.load(ckpt_path)

    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    pipeline_scores = {}
    n_episodes = max(1, int(200 * fraction))
    for protocol in benchmark["evaluation_protocols"]:
        task = generate_task(
            benchmark["task_generator_id"],
            variables_space="space_a_b",
            **task_cfg,
        )
        recorder = DataRecorder(output_directory=None)
        env = CausalWorld(task, **world_params, seed=0, data_recorder=recorder, enable_visualization=False)
        wrapped = ProtocolWrapper(env, protocol)
        if v_fusion:
            wrapped = ProtocolObsWrapper(wrapped)
        protocol.init_protocol(env=env, tracker=env.get_tracker(), fraction=1.0)
        ep_scores = []
        for _ in range(n_episodes):
            obs = wrapped.reset()
            done = False
            while not done:
                action = policy_fn(obs)
                obs, _, done, _ = wrapped.step(action)
            if recorder.get_current_episode().infos:
                ep_scores.append(
                    sum(float(i.get("fractional_success", 0.0)) for i in recorder.get_current_episode().infos)
                    / len(recorder.get_current_episode().infos)
                )
            recorder.clear_recorder()
        pipeline_scores[protocol.get_name()] = {
            "mean_full_integrated_fractional_success": float(np.mean(ep_scores)) if ep_scores else 0.0,
        }
        wrapped.close()
    return pipeline_scores


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train CPPPO teachers")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "vla_pipeline/config_cppo_local.yaml"),
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None, help="Override parallel env count")
    parser.add_argument("--no_resume", action="store_true", help="Ignore existing task checkpoints")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = raw["cppo"]
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = int(args.total_timesteps)
    if args.no_resume:
        cfg["resume"] = False
    eval_cfg = raw.get("evaluation", {})
    cfg["evaluation"] = eval_cfg

    tasks = args.tasks or cfg.get("tasks", ["pushing", "picking", "pick_and_place"])
    results = {}
    for task_name in tasks:
        ckpt = _train_task(task_name, cfg, args.config, num_envs_cli=args.num_envs)
        results[task_name] = {"checkpoint": ckpt}
        if not args.no_eval:
            print(f"[CPPPO] Quick eval {task_name} (fraction={eval_cfg.get('fraction', 0.1)})", flush=True)
            try:
                scores = _quick_eval(task_name, ckpt, cfg)
                metric = "mean_full_integrated_fractional_success"
                proto_scores = {
                    p: float(scores[p].get(metric, 0.0)) for p in sorted(scores.keys())
                }
                task_macro = sum(proto_scores.values()) / max(1, len(proto_scores))
                results[task_name]["eval"] = proto_scores
                results[task_name]["task_macro"] = task_macro
                print(f"[CPPPO] {task_name} task_macro={task_macro:.4f}", flush=True)
            except Exception as exc:
                print(f"[CPPPO] eval failed for {task_name}: {exc}", flush=True)

    summary_path = Path(cfg["checkpoint_dir"]) / "cppo_train_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[CPPPO] Done. Summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
