#!/usr/bin/env python3
"""
Expert-Anchored Advantage Weighted Regression (EA-AWR) for VLA fine-tuning.

  Phase A: warm-start from BC checkpoint (expert demos)
  Phase B: online rollouts on hard causal protocols (P6-P11)
  Phase C: weighted BC on expert + high-advantage online data, with teacher KL anchor

Reference: Nair et al. AWRC; practical hybrid of BC pretrain + RL-style online improvement.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from causal_world.benchmark import (
    PICKING_BENCHMARK,
    PICK_AND_PLACE_BENCHMARK,
    PUSHING_BENCHMARK,
)

from vla_pipeline.dataset import CausalWorldVLADataset
from vla_pipeline.model_resnet_film_flow import build_policy_from_config, set_backbone_trainable
from vla_pipeline.rl_rollout import RLEpisode, rollout_episode, sample_protocol_jobs
from vla_pipeline.teacher import load_teacher
from vla_pipeline.train_bc import build_optimizer


BENCHMARKS = {
    "pushing": PUSHING_BENCHMARK,
    "picking": PICKING_BENCHMARK,
    "pick_and_place": PICK_AND_PLACE_BENCHMARK,
}


class OnlineBuffer:
    def __init__(self, max_episodes: int = 200):
        self.max_episodes = max_episodes
        self.episodes: deque = deque()

    def add(self, ep: RLEpisode, advantage: float, temperature: float) -> None:
        w = float(np.exp(advantage / max(temperature, 1e-4)))
        self.episodes.append((ep, w))
        while len(self.episodes) > self.max_episodes:
            self.episodes.popleft()

    def sample_batch(self, batch_size: int, rng: random.Random):
        if not self.episodes:
            return []
        weights = np.array([w for _, w in self.episodes], dtype=np.float64)
        weights /= weights.sum()
        idxs = rng.choices(range(len(self.episodes)), weights=weights, k=batch_size)
        batch = []
        for idx in idxs:
            ep, _ = self.episodes[idx]
            t = ep.transitions[rng.randrange(len(ep.transitions))]
            batch.append(t)
        return batch


def _load_init_model(tcfg: dict, action_dim: int, init_ckpt: str, device: torch.device):
    model = build_policy_from_config(tcfg, action_dim).to(device)
    if init_ckpt and os.path.isfile(init_ckpt):
        state = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        print(f"Loaded init checkpoint: {init_ckpt}")
    return model


def train(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rl = cfg["rl"]
    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = cfg["data"]["export_dir"]
    expert_ds = CausalWorldVLADataset(
        data_root,
        "train.json",
        tcfg["image_size"],
        action_horizon=int(tcfg.get("action_horizon", 1)),
        augment=False,
    )
    expert_loader = DataLoader(
        expert_ds,
        batch_size=int(rl.get("expert_batch_size", tcfg["batch_size"])),
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    expert_iter = iter(expert_loader)

    model = _load_init_model(tcfg, expert_ds.action_dim, rl["init_checkpoint"], device)
    set_backbone_trainable(model, bool(rl.get("train_backbone", False)))
    opt = build_optimizer(model, {**tcfg, "lr_head": rl.get("lr_head", 3e-5), "lr_backbone": rl.get("lr_backbone", 5e-6)})

    tasks = list(rl.get("tasks", cfg["data"]["tasks"]))
    protocols_by_task = rl["protocols_by_task"]
    skip_frame = int(rl.get("skip_frame", cfg["evaluation"]["skip_frame"]))
    image_size = int(tcfg["image_size"])

    teachers = {}
    for task in tasks:
        ckpt = (cfg.get("teacher", {}).get("checkpoints", {}) or {}).get(task)
        teachers[task] = load_teacher(task, checkpoint=ckpt)

    online_buf = OnlineBuffer(max_episodes=int(rl.get("online_buffer_episodes", 150)))
    score_baseline = 0.0
    baseline_momentum = float(rl.get("baseline_momentum", 0.9))
    rng = np.random.RandomState(int(rl.get("seed", 42)))
    py_rng = random.Random(int(rl.get("seed", 42)))

    ckpt_dir = Path(rl["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    n_iters = int(rl["iters"])
    episodes_per_iter = int(rl["episodes_per_iter"])
    grad_steps = int(rl.get("grad_steps_per_iter", 4))
    explore_std = float(rl["explore_std"])
    explore_decay = float(rl.get("explore_decay", 0.995))
    temp = float(rl.get("advantage_temperature", 0.15))
    w_expert = float(rl.get("expert_bc_weight", 1.0))
    w_online = float(rl.get("online_bc_weight", 2.0))
    w_teacher = float(rl.get("teacher_kl_weight", 0.3))
    min_adv = float(rl.get("min_advantage_to_store", -0.05))

    print(
        f"EA-AWR: tasks={tasks} skip_frame={skip_frame} iters={n_iters} "
        f"ep/iter={episodes_per_iter} explore_std={explore_std}",
        flush=True,
    )

    for it in range(1, n_iters + 1):
        t0 = time.time()
        jobs = sample_protocol_jobs(
            tasks, protocols_by_task, BENCHMARKS, episodes_per_iter, rng
        )
        iter_scores = []
        for j, (task_name, pidx) in enumerate(jobs):
            ep = rollout_episode(
                task_name,
                BENCHMARKS[task_name],
                pidx,
                model,
                teachers[task_name],
                device,
                image_size,
                skip_frame=skip_frame,
                seed=int(rl.get("seed", 42)) + it * 1000 + j,
                explore_std=explore_std,
            )
            adv = ep.score - score_baseline
            score_baseline = (
                baseline_momentum * score_baseline + (1 - baseline_momentum) * ep.score
            )
            iter_scores.append(ep.score)
            if adv >= min_adv:
                online_buf.add(ep, adv, temp)
            print(
                f"  [rl iter {it}] roll {j+1}/{len(jobs)} "
                f"{task_name}/{ep.protocol} score={ep.score:.4f} adv={adv:+.4f}",
                flush=True,
            )

        model.train()
        losses = []
        for _ in range(grad_steps):
            try:
                views_e, texts_e, actions_e, _, weights_e, _ = next(expert_iter)
            except StopIteration:
                expert_iter = iter(expert_loader)
                views_e, texts_e, actions_e, _, weights_e, _ = next(expert_iter)

            views_e = views_e.to(device)
            actions_e = actions_e[:, 0, :].to(device)
            weights_e = weights_e.to(device).float()
            pred_e = model.predict_action(views_e, list(texts_e))
            loss_expert = (weights_e * ((pred_e - actions_e) ** 2).mean(dim=-1)).mean()

            loss_online = torch.tensor(0.0, device=device)
            loss_teacher = torch.tensor(0.0, device=device)
            batch = online_buf.sample_batch(int(rl.get("online_batch_size", 16)), py_rng)
            if batch:
                views_o = torch.stack([t.views for t in batch], dim=0).to(device)
                texts_o = [t.instruction for t in batch]
                act_o = torch.tensor(np.stack([t.action for t in batch]), device=device, dtype=torch.float32)
                teach_o = torch.tensor(
                    np.stack([t.teacher_action for t in batch]), device=device, dtype=torch.float32
                )
                pred_o = model.predict_action(views_o, texts_o)
                loss_online = ((pred_o - act_o) ** 2).mean()
                loss_teacher = ((pred_o - teach_o) ** 2).mean()

            loss = w_expert * loss_expert + w_online * loss_online + w_teacher * loss_teacher
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 1.0)))
            opt.step()
            losses.append(float(loss.item()))

        explore_std *= explore_decay
        mean_score = float(np.mean(iter_scores)) if iter_scores else 0.0
        print(
            f"[rl iter {it}/{n_iters}] mean_score={mean_score:.4f} "
            f"baseline={score_baseline:.4f} loss={np.mean(losses):.5f} "
            f"online_eps={len(online_buf.episodes)} explore={explore_std:.4f} "
            f"elapsed={(time.time()-t0)/60:.1f}min",
            flush=True,
        )

        meta = {"rl_iter": it, "mean_score": mean_score, "baseline": score_baseline}
        torch.save(
            {
                "model": model.state_dict(),
                "model_type": tcfg.get("model_type", "resnet_film_mse"),
                "action_dim": expert_ds.action_dim,
                "action_horizon": int(tcfg.get("action_horizon", 1)),
                "image_size": image_size,
                "rl_meta": meta,
            },
            ckpt_dir / "rl_last.pt",
        )
        if mean_score > best_score:
            best_score = mean_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": tcfg.get("model_type", "resnet_film_mse"),
                    "action_dim": expert_ds.action_dim,
                    "action_horizon": int(tcfg.get("action_horizon", 1)),
                    "image_size": image_size,
                    "rl_meta": meta,
                },
                ckpt_dir / "rl_best.pt",
            )
            print(f"  -> new best mean_score={best_score:.4f}", flush=True)

    print(f"Done. Best online mean_score={best_score:.4f} -> {ckpt_dir}/rl_best.pt")


def main():
    p = argparse.ArgumentParser(description="EA-AWR RL fine-tune for VLA")
    p.add_argument("--config", default=str(Path(__file__).with_name("config_v5_rl.yaml")))
    train(p.parse_args().config)


if __name__ == "__main__":
    main()
