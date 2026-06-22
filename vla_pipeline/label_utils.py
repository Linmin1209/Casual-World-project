"""Text instruction builders from CausalWorld task / state variables."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _fmt_vec(v) -> str:
    if v is None:
        return "unknown"
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(f"{float(x):.3f}" for x in v) + ")"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _block_phrase(prefix: str, state: Dict[str, Any]) -> str:
    parts = []
    for key in ("type", "size", "cartesian_position", "orientation", "mass"):
        k = f"{prefix}_{key}" if not key.startswith(prefix) else key
        if k in state:
            parts.append(f"{key}={_fmt_vec(state[k])}")
        elif key in state.get(prefix, {}):
            parts.append(f"{key}={_fmt_vec(state[prefix][key])}")
    return f"{prefix}[" + "; ".join(parts) + "]" if parts else prefix


def build_instruction(
    task_name: str,
    state_vars: Dict[str, Any],
    protocol: Optional[str] = None,
    info: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a natural-language goal + context string for VLA conditioning."""
    task_desc = {
        "pushing": "Push the tool block to match the goal block silhouette on the table.",
        "picking": "Pick up the tool block and align it with the goal block.",
        "pick_and_place": "Pick the tool block and place it onto the goal region.",
    }.get(task_name, f"Complete the {task_name} manipulation task.")

    tool = _block_phrase("tool_block", state_vars)
    goal = _block_phrase("goal_block", state_vars)

    extra = []
    if "floor_friction" in state_vars:
        extra.append(f"floor_friction={_fmt_vec(state_vars['floor_friction'])}")
    if info and info.get("fractional_success") is not None:
        extra.append(f"current_overlap={float(info['fractional_success']):.3f}")

    proto = f" Evaluation protocol {protocol}." if protocol else ""
    ctx = " Context: " + ", ".join(extra) + "." if extra else ""
    return f"{task_desc}{proto} Tool: {tool}. Goal: {goal}.{ctx}"


def instruction_to_json(instruction: str) -> str:
    return json.dumps(instruction, ensure_ascii=False)
