"""Warm-start CPPPO v2 fusion policy from official PPO2 teacher checkpoints."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


def _extract_baseline_params(baseline_zip: str) -> Dict[str, np.ndarray]:
    """Load baseline PPO2 checkpoint params (name -> ndarray)."""
    from stable_baselines import PPO2

    baseline = PPO2.load(os.path.abspath(baseline_zip))
    try:
        params = dict(baseline.get_parameters())
        if not params:
            raise RuntimeError(f"No parameters found in {baseline_zip}")
        return params
    finally:
        try:
            baseline.sess.close()
        except Exception:
            pass
        tf.reset_default_graph()


def _fusion_vars(model) -> Dict[str, tf.Variable]:
    return {p.name: p for p in model.get_parameter_list()}


def _assign_exact(sess: tf.Session, var: tf.Variable, arr: np.ndarray) -> bool:
    dst = sess.run(var)
    if tuple(dst.shape) != tuple(arr.shape):
        return False
    sess.run(var.assign(arr))
    return True


def _assign_rows_cols(
    sess: tf.Session,
    var: tf.Variable,
    arr: np.ndarray,
    row_end: Optional[int] = None,
    col_end: Optional[int] = None,
) -> bool:
    dst = sess.run(var)
    if dst.ndim != 2 or arr.ndim != 2:
        return False
    r = row_end if row_end is not None else arr.shape[0]
    c = col_end if col_end is not None else arr.shape[1]
    if r > dst.shape[0] or c > dst.shape[1] or r > arr.shape[0] or c > arr.shape[1]:
        return False
    block = dst.copy()
    block[:r, :c] = arr[:r, :c]
    sess.run(var.assign(block))
    return True


def _assign_prefix(sess: tf.Session, var: tf.Variable, arr: np.ndarray) -> bool:
    dst = sess.run(var)
    if dst.ndim == 1:
        n = min(len(dst), len(arr))
        block = dst.copy()
        block[:n] = arr[:n]
        sess.run(var.assign(block))
        return n > 0
    if dst.ndim == 2 and arr.ndim == 2:
        return _assign_rows_cols(sess, var, arr, row_end=arr.shape[0], col_end=arr.shape[1])
    return False


def _transfer_shared_mlp_baseline(
    sess: tf.Session,
    baseline: Dict[str, np.ndarray],
    fusion: Dict[str, tf.Variable],
    *,
    exact_input: bool = False,
) -> Tuple[int, int, List[str]]:
    """Map official shared-MLP baseline (shared_fc*) into fusion pi_fc*/vf_fc*."""
    copied = 0
    skipped = 0
    notes: List[str] = []

    def b(key: str) -> Optional[np.ndarray]:
        return baseline.get(key)

    def f(key: str) -> Optional[tf.Variable]:
        return fusion.get(key)

    def try_copy(label: str, fusion_key: str, baseline_key: str, mode: str = "exact") -> None:
        nonlocal copied, skipped
        arr = b(baseline_key)
        var = f(fusion_key)
        if arr is None or var is None:
            skipped += 1
            return
        ok = False
        if mode == "exact":
            ok = _assign_exact(sess, var, arr)
        elif mode == "prefix":
            ok = _assign_prefix(sess, var, arr)
        elif mode == "input_rows":
            ok = _assign_rows_cols(sess, var, arr, row_end=arr.shape[0], col_end=arr.shape[1])
        elif mode == "pi_head_rows":
            ok = _assign_rows_cols(sess, var, arr, row_end=arr.shape[0], col_end=arr.shape[1])
        if ok:
            copied += 1
            notes.append(f"{baseline_key} -> {fusion_key} ({mode})")
        else:
            skipped += 1

    if b("model/shared_fc0/w:0") is None:
        return copied, skipped, notes

    fc0_mode = "exact" if exact_input else "input_rows"

    try_copy("fc0", "model/pi_fc0/w:0", "model/shared_fc0/w:0", fc0_mode)
    try_copy("fc0", "model/vf_fc0/w:0", "model/shared_fc0/w:0", fc0_mode)
    try_copy("fc0b", "model/pi_fc0/b:0", "model/shared_fc0/b:0", "prefix")
    try_copy("fc0b", "model/vf_fc0/b:0", "model/shared_fc0/b:0", "prefix")

    try_copy("fc1", "model/pi_fc1/w:0", "model/shared_fc1/w:0", "exact")
    try_copy("fc1", "model/vf_fc1/w:0", "model/shared_fc1/w:0", "exact")
    try_copy("fc1b", "model/pi_fc1/b:0", "model/shared_fc1/b:0", "prefix")
    try_copy("fc1b", "model/vf_fc1/b:0", "model/shared_fc1/b:0", "prefix")

    try_copy("pi", "model/pi/w:0", "model/pi/w:0", "pi_head_rows")
    for key in ("model/pi/b:0", "model/pi/logstd:0"):
        try_copy("pi", key, key, "exact")
    for key in ("model/vf/w:0", "model/vf/b:0", "model/q/w:0", "model/q/b:0"):
        try_copy("head", key, key, "exact")

    return copied, skipped, notes


def _transfer_exact_matches(
    sess: tf.Session,
    baseline: Dict[str, np.ndarray],
    fusion: Dict[str, tf.Variable],
    skip_substrings: Tuple[str, ...] = ("cppo_proto_embed",),
) -> Tuple[int, int]:
    copied = 0
    skipped = 0
    for name, var in fusion.items():
        if any(s in name for s in skip_substrings):
            skipped += 1
            continue
        if name not in baseline:
            skipped += 1
            continue
        if _assign_exact(sess, var, baseline[name]):
            copied += 1
        else:
            skipped += 1
    return copied, skipped


def transfer_baseline_params(
    model,
    baseline_params: Dict[str, np.ndarray],
    *,
    residual: bool = False,
) -> Dict[str, int]:
    fusion = _fusion_vars(model)
    sess = model.sess
    if not fusion:
        return {"copied": 0, "skipped": 0, "mode": "no_fusion_vars"}

    if "model/shared_fc0/w:0" in baseline_params:
        copied, skipped, notes = _transfer_shared_mlp_baseline(
            sess, baseline_params, fusion, exact_input=residual
        )
        tag = "v3-residual" if residual else "v2-fusion"
        print(
            f"[CPPPO {tag}] warm-start shared-MLP: copied={copied} skipped={skipped}",
            flush=True,
        )
        if notes:
            print(f"[CPPPO {tag}] warm-start detail: {', '.join(notes[:6])}"
                  f"{'' if len(notes) <= 6 else f' (+{len(notes)-6} more)'}",
                  flush=True)
        return {"copied": copied, "skipped": skipped, "mode": "shared_mlp_residual" if residual else "shared_mlp"}

    copied, skipped = _transfer_exact_matches(sess, baseline_params, fusion)
    print(f"[CPPPO] warm-start exact-match: copied={copied} skipped={skipped}", flush=True)
    return {"copied": copied, "skipped": skipped, "mode": "exact"}


def build_cppo_fusion_model(
    baseline_zip: str,
    env,
    policy_cls,
    policy_kwargs: dict,
    train_kw: dict,
    warm_start: bool = True,
    tensorboard_log: str = None,
    residual: bool = False,
) -> Tuple[object, Dict[str, int]]:
    from stable_baselines import PPO2

    baseline_params = {}
    if warm_start and os.path.isfile(baseline_zip):
        tag = "v3" if residual else "v2"
        print(f"[CPPPO {tag}] extracting baseline params from {baseline_zip}", flush=True)
        baseline_params = _extract_baseline_params(baseline_zip)
        print(f"[CPPPO {tag}] extracted {len(baseline_params)} baseline tensors", flush=True)

    cppo = PPO2(
        policy_cls,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log,
        **train_kw,
    )
    stats: Dict[str, int] = {"copied": 0, "skipped": 0}
    if baseline_params:
        stats = transfer_baseline_params(cppo, baseline_params, residual=residual)
        print(f"[CPPPO] warm-start stats: {stats}", flush=True)
    return cppo, stats


# Backward-compatible alias
build_cppo_v2_model = build_cppo_fusion_model
