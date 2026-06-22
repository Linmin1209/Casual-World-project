#!/usr/bin/env python3
"""Train ResNet+FiLM Flow Matching policy (multi-step action chunks)."""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from vla_pipeline.dataset import CausalWorldVLADataset
from vla_pipeline.model_resnet_film_flow import (
    build_policy_from_config,
    set_backbone_trainable,
)


def build_optimizer(model, tcfg):
    wd = tcfg.get("weight_decay", 1e-4)
    lr_head = float(tcfg.get("lr_head", tcfg.get("lr", 1e-4)))
    lr_backbone = float(tcfg.get("lr_backbone", lr_head * 0.3))

    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("vision_enc.stem") or name.startswith("vision_enc.layer"):
            backbone.append(p)
        else:
            head.append(p)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": wd})
    if head:
        groups.append({"params": head, "lr": lr_head, "weight_decay": wd})
    if not groups:
        groups = [{"params": list(model.parameters()), "lr": lr_head, "weight_decay": wd}]
    return torch.optim.AdamW(groups)


def _batch_loss(model, views, texts, action_chunk, weights, aux_weight: float):
    if hasattr(model, "bc_loss"):
        return model.bc_loss(views, texts, action_chunk, weights=weights)
    if hasattr(model, "training_loss"):
        return model.training_loss(
            views, texts, action_chunk, weights=weights, aux_weight=aux_weight
        )
    return model.flow_matching_loss(views, texts, action_chunk, weights=weights)


def train(cfg_path: str, round_id: int | None = None):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_root = cfg["data"]["export_dir"]
    tcfg = cfg["training"]
    ckpt_dir = tcfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    if round_id is None:
        round_id = int(tcfg.get("round", 0))

    action_horizon = int(tcfg.get("action_horizon", 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dagger_w = float(tcfg.get("dagger_sample_weight", 1.0))
    proto_w = tcfg.get("protocol_sample_weights")
    train_ds = CausalWorldVLADataset(
        data_root,
        "train.json",
        tcfg["image_size"],
        action_horizon=action_horizon,
        augment=bool(tcfg.get("augment_train", False)),
        dagger_sample_weight=dagger_w,
        protocol_sample_weights=proto_w,
    )
    val_ds = CausalWorldVLADataset(
        data_root,
        "val.json",
        tcfg["image_size"],
        action_horizon=action_horizon,
        protocol_sample_weights=proto_w,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples in {data_root}/train.json")

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=tcfg.get("num_workers", 4) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg.get("num_workers", 4),
    )

    model = build_policy_from_config(tcfg, train_ds.action_dim).to(device)

    init_ckpt = tcfg.get("init_checkpoint")
    if init_ckpt and os.path.isfile(init_ckpt):
        state = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        print(f"Loaded init checkpoint: {init_ckpt}")

    use_fs_weight = bool(tcfg.get("loss_weight_by_fs", False))
    aux_weight = float(tcfg.get("aux_denoise_loss_weight", 0.0))
    freeze_epochs = int(tcfg.get("freeze_backbone_epochs", 0))
    patience = int(tcfg.get("early_stop_patience", 0))
    mse_eval_batches = int(tcfg.get("action_mse_eval_batches", 0))
    mse_eval_every = int(tcfg.get("action_mse_eval_every", 1))
    best_on = tcfg.get("best_checkpoint_metric", "val_cfm_w")
    is_mse = tcfg.get("model_type") == "resnet_film_mse"
    if is_mse and best_on.startswith("val_cfm"):
        best_on = "val_mse_w" if use_fs_weight else "val_mse"

    hard_protocols = set(tcfg.get("hard_protocols", ["P6", "P7", "P8", "P9", "P10", "P11"]))
    round_ckpt = os.path.join(ckpt_dir, f"bc_best_r{round_id:02d}.pt")

    if freeze_epochs > 0:
        set_backbone_trainable(model, False)
    opt = build_optimizer(model, tcfg)
    scheduler = None
    if tcfg.get("cosine_lr", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(tcfg["epochs"]), eta_min=float(tcfg.get("lr_min", 1e-6))
        )

    def save_ckpt(path, metric):
        torch.save(
            {
                "model": model.state_dict(),
                "model_type": tcfg.get("model_type", "resnet_film_flow"),
                "action_dim": train_ds.action_dim,
                "action_horizon": action_horizon,
                "image_size": tcfg["image_size"],
                "fm_sample_steps": int(tcfg.get("fm_sample_steps", 10)),
                "val_loss": metric,
            },
            path,
        )

    def eval_action_mse(loader, max_batches: int) -> float:
        total = 0.0
        n = 0
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            views, texts, action_chunk, _, _weights, _protocol = batch
            views = views.to(device)
            action_chunk = action_chunk.to(device)
            pred = model.sample_action_chunk(views, list(texts))
            total += ((pred - action_chunk) ** 2).mean().item()
            n += 1
        return total / max(1, n)

    def eval_hard_val_loss_w() -> float:
        if not use_fs_weight or not hard_protocols:
            return float("inf")
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for views, texts, action_chunk, _task, weights, protocols in val_loader:
                hard_idx = [i for i, p in enumerate(protocols) if p in hard_protocols]
                if not hard_idx:
                    continue
                idx = torch.tensor(hard_idx)
                v = views.index_select(0, idx).to(device)
                ac = action_chunk.index_select(0, idx).to(device)
                w = weights.index_select(0, idx).to(device).float()
                texts_h = [texts[i] for i in hard_idx]
                total += _batch_loss(model, v, texts_h, ac, w, 0.0).item()
                n_batches += 1
        return total / max(1, n_batches)

    best_val = float("inf")
    stale = 0
    for epoch in range(tcfg["epochs"]):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            set_backbone_trainable(model, True)
            opt = build_optimizer(model, tcfg)
            if tcfg.get("cosine_lr", True):
                remaining = int(tcfg["epochs"]) - epoch
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=remaining, eta_min=float(tcfg.get("lr_min", 1e-6))
                )
        elif freeze_epochs == 0 or epoch >= freeze_epochs:
            set_backbone_trainable(model, True)
        else:
            set_backbone_trainable(model, False)

        model.train()
        train_loss = 0.0
        for views, texts, action_chunk, _, weights, _protocol in train_loader:
            views = views.to(device)
            action_chunk = action_chunk.to(device)
            weights = weights.to(device).float()
            w = weights if use_fs_weight else None
            loss = _batch_loss(
                model, views, list(texts), action_chunk, w, aux_weight
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
            opt.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss_w = val_loss = 0.0
        val_loss_w_hard = None
        with torch.no_grad():
            for views, texts, action_chunk, _, weights, _protocol in val_loader:
                views = views.to(device)
                action_chunk = action_chunk.to(device)
                weights = weights.to(device).float()
                val_loss += _batch_loss(
                    model, views, list(texts), action_chunk, None, 0.0
                ).item()
                if use_fs_weight:
                    val_loss_w += _batch_loss(
                        model, views, list(texts), action_chunk, weights, 0.0
                    ).item()
        val_loss /= max(1, len(val_loader))
        val_loss_w /= max(1, len(val_loader))

        if best_on == "val_cfm_w" and use_fs_weight:
            pick_metric = val_loss_w
        elif best_on in ("val_mse_w", "val_mse") and use_fs_weight:
            pick_metric = val_loss_w if best_on == "val_mse_w" else val_loss
        elif best_on == "val_act_mse":
            pick_metric = float("inf")
        elif best_on == "val_mse_w_hard" and use_fs_weight:
            val_loss_w_hard = eval_hard_val_loss_w()
            pick_metric = val_loss_w_hard
        else:
            pick_metric = val_loss

        val_label = "val_mse" if is_mse else "val_cfm"
        msg = (
            f"epoch {epoch + 1}/{tcfg['epochs']} "
            f"train_loss={train_loss:.5f} {val_label}={val_loss:.5f}"
        )
        if use_fs_weight:
            msg += f" {val_label}_w={val_loss_w:.5f}"
            if val_loss_w_hard is not None:
                msg += f" {val_label}_w_hard={val_loss_w_hard:.5f}"
        act_mse = None
        if (
            mse_eval_batches > 0
            and len(val_loader) > 0
            and (epoch + 1) % mse_eval_every == 0
        ):
            act_mse = eval_action_mse(val_loader, mse_eval_batches)
            msg += f" val_act_mse={act_mse:.5f}"
            if best_on == "val_act_mse":
                pick_metric = act_mse
        msg += f" H={action_horizon}"
        if freeze_epochs > 0 and epoch < freeze_epochs:
            msg += " [backbone frozen]"
        lrs = [g["lr"] for g in opt.param_groups]
        msg += f" lr={lrs[0]:.1e}" + (f"/{lrs[1]:.1e}" if len(lrs) > 1 else "")
        print(msg)

        if scheduler is not None:
            scheduler.step()

        save_ckpt(os.path.join(ckpt_dir, "bc_last.pt"), pick_metric)
        if pick_metric < best_val:
            best_val = pick_metric
            stale = 0
            save_ckpt(os.path.join(ckpt_dir, "bc_best.pt"), pick_metric)
            save_ckpt(round_ckpt, pick_metric)
        else:
            stale += 1

        if patience > 0 and stale >= patience:
            print(f"Early stop: {best_on} plateau for {patience} epochs (best={best_val:.5f})")
            break

    print(f"Best {best_on} {best_val:.5f} -> {ckpt_dir}/bc_best.pt + {round_ckpt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_v2.yaml"))
    p.add_argument("--round", type=int, default=None, help="Training round index for bc_best_rNN.pt")
    args = p.parse_args()
    train(args.config, round_id=args.round)


if __name__ == "__main__":
    main()
