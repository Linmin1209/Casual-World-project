"""Offline training-curve logging for CPPPO (PPO2 + Monitor)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from stable_baselines.common.callbacks import BaseCallback
except ImportError:
    BaseCallback = object  # type: ignore


class CppoCurveCallback(BaseCallback):
    """Append PPO rollout metrics to JSONL when stable-baselines dumps logs."""

    def __init__(self, jsonl_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._orig_dumpkvs = None

    def _init_callback(self) -> None:
        if self.logger is None:
            return
        self._orig_dumpkvs = self.logger.dumpkvs
        jsonl_path = self.jsonl_path
        orig_dumpkvs = self._orig_dumpkvs

        def dumpkvs_and_log():
            ntv = _logger_kv(self.logger)
            if ntv:
                rec = _record_from_logger(ntv, int(self.num_timesteps))
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            orig_dumpkvs()

        self.logger.dumpkvs = dumpkvs_and_log

    def _on_training_end(self) -> None:
        if self.logger is not None and self._orig_dumpkvs is not None:
            self.logger.dumpkvs = self._orig_dumpkvs


def _logger_kv(logger_obj) -> dict:
    """Read current logger values (SB2: name2val, SB3-style: name_to_value)."""
    if hasattr(logger_obj, "name2val"):
        return dict(logger_obj.name2val)
    if hasattr(logger_obj, "name_to_value"):
        return dict(logger_obj.name_to_value)
    return {}


def _record_from_logger(ntv: dict, timesteps: int) -> dict:
    return {
        "timesteps": timesteps,
        "total_timesteps": _f(ntv, "total_timesteps"),
        "n_updates": _f(ntv, "n_updates"),
        "ep_rew_mean": _first(ntv, ("ep_reward_mean", "rollout/ep_rew_mean")),
        "ep_len_mean": _first(ntv, ("ep_len_mean", "rollout/ep_len_mean")),
        "policy_loss": _first(ntv, ("policy_loss", "train/policy_loss")),
        "value_loss": _first(ntv, ("value_loss", "train/value_loss")),
        "entropy": _first(ntv, ("policy_entropy", "train/entropy")),
        "approx_kl": _first(ntv, ("approxkl", "train/approx_kl")),
        "clip_fraction": _first(ntv, ("clipfrac", "train/clip_fraction")),
        "explained_variance": _first(ntv, ("explained_variance", "train/explained_variance")),
        "fps": _f(ntv, "fps"),
        "time_elapsed": _f(ntv, "time_elapsed"),
    }


def _first(ntv: dict, keys: tuple) -> Optional[float]:
    for key in keys:
        val = _f(ntv, key)
        if val is not None:
            return val
    return None


def _f(ntv: dict, key: str) -> Optional[float]:
    if key not in ntv:
        return None
    try:
        return float(ntv[key])
    except (TypeError, ValueError):
        return None


def _read_monitor_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "t": float(row.get("t", 0) or 0),
                        "episode_reward": float(row.get("r", 0) or 0),
                        "episode_length": float(row.get("l", 0) or 0),
                        "fractional_success": _optional_float(row.get("fractional_success")),
                        "cppo_protocol": row.get("cppo_protocol"),
                    }
                )
            except (TypeError, ValueError):
                continue
    return rows


def _optional_float(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def aggregate_monitor_logs(log_dir: Path) -> Dict[str, Any]:
    """Parse Monitor CSVs from parallel env workers."""
    episodes: List[dict] = []
    for rank in range(256):
        for suffix in (".monitor.csv", ""):
            path = log_dir / f"monitor_{rank}{suffix}"
            if path.is_file():
                for row in _read_monitor_csv(path):
                    row["worker"] = rank
                    episodes.append(row)
    episodes.sort(key=lambda r: (r.get("t", 0), r.get("worker", 0)))
    rewards = [e["episode_reward"] for e in episodes]
    fs = [e["fractional_success"] for e in episodes if e.get("fractional_success") is not None]
    summary = {
        "num_episodes": len(episodes),
        "reward_mean": float(np.mean(rewards)) if rewards else None,
        "reward_std": float(np.std(rewards)) if rewards else None,
        "fractional_success_mean": float(np.mean(fs)) if fs else None,
    }
    return {"episodes": episodes, "summary": summary}


def load_ppo_jsonl(jsonl_path: Path) -> List[dict]:
    if not jsonl_path.is_file():
        return []
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def export_training_curves(
    out_dir: Path,
    task_name: str,
    log_dir: Path,
    ppo_jsonl: Path,
) -> Path:
    """Write combined offline curve bundle for one task."""
    bundle = {
        "task": task_name,
        "ppo_rollouts": load_ppo_jsonl(ppo_jsonl),
        "monitor": aggregate_monitor_logs(log_dir),
    }
    out_path = out_dir / "training_curves.json"
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    # Compact CSV for quick plotting
    csv_path = out_dir / "ppo_rollouts.csv"
    if bundle["ppo_rollouts"]:
        keys = sorted({k for r in bundle["ppo_rollouts"] for k in r.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in bundle["ppo_rollouts"]:
                w.writerow(row)
    return out_path
