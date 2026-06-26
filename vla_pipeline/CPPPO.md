# CPPPO — Protocol-Conditioned PPO Teacher

CPPPO (Concat / residual **P**rotocol-conditioned **PPO**) trains privileged structured-observation teachers that beat the CausalWorld baseline on the **global macro** metric: equal-weight average of `mean_full_integrated_fractional_success` over **36 protocols** (12 per task × 3 tasks).

## Latest results (200 ep/protocol, full eval)

| Scheme | Global macro | vs baseline (0.4994) | Notes |
|--------|-------------|----------------------|-------|
| Baseline PPO | 0.4994 | — | `causal_world/assets/baseline_actors/` |
| CPPPO v2 (concat fusion) | 0.5467 | +9.5% | Single checkpoint per task |
| CPPPO v3 stage-1 (push + pick) | 0.5598 | +12.1% | Residual dual-path policy |
| **Single-checkpoint best** | **0.5743** | **+15.0%** | v3 push/pick + pap **s2_12M** |
| **Protocol-routed experts** | **0.6018** | **+20.5%** | pap P4–P7 → baseline; rest → CPPPO |

Metric definition matches `data/baseline_eval/summary_all_tasks.json` (regenerate locally; `data/` is gitignored).

### Recommended checkpoints (train locally or copy from your runs)

| Task | Single-checkpoint | Router (pap) |
|------|-------------------|--------------|
| pushing | `data/cppo_checkpoints_v3_stage1/pushing/pushing_cppo_v3.zip` | same |
| picking | `data/cppo_checkpoints_v3_stage1/picking/picking_cppo_v3.zip` | same |
| pick_and_place | `data/cppo_checkpoints_v3_stage2_pap/pick_and_place/intermediate/pick_and_place_cppo_v3_12000000_steps.zip` | s2_12M for P0–P3,P8–P11; **baseline** `pick_and_place_ppo_curr0.zip` for **P4–P7** |

Router config: `vla_pipeline/config_cppo_v4_teacher_router.json`.

---

## Environment

CPPPO training and teacher eval use the **causal_world** conda env (Python 3.7, TF1, stable-baselines 2.x):

```bash
conda activate causal_world
export PYTHONPATH=/path/to/Casual-World-project
export PYTHONUNBUFFERED=1
```

VLA / hybrid pipelines may use a separate torch env; teachers are always loaded via `vla_pipeline/teacher.py` in the causal_world env when possible.

---

## Architecture versions

| Version | Policy | Warm-start |
|---------|--------|------------|
| v2 | `CpppoFusionPolicy` — concat `[base_obs \| proto_one_hot]` → MLP | First layers from baseline MLP |
| v3 | `CpppoResidualPolicy` — baseline MLP + zero-init protocol residual | Exact copy of baseline MLP |

Code: `vla_pipeline/cppo_policy.py`, `cppo_policy_v3.py`, `cppo_transfer.py`.

Observations use `ProtocolObsWrapper` (+12-dim protocol one-hot). Training uses `ProtocolCurriculumWrapper` for weighted / anchor protocol sampling.

---

## Training

### Stage 1 — all three tasks (v3, 15M steps each)

```bash
bash vla_pipeline/run_cppo_v3_stage1_train.sh
# config: config_cppo_v3_stage1_local.yaml
# output: data/cppo_checkpoints_v3_stage1/
```

### Stage 2 — pick_and_place only (15M from stage-1 pap ckpt)

Best pap checkpoint is usually **12M steps**, not the final 15M (overtraining on easy protocols).

```bash
bash vla_pipeline/run_cppo_v3_stage2a_train.sh
# config: config_cppo_v3_stage2_pap_local.yaml
# output: data/cppo_checkpoints_v3_stage2_pap/
# best: .../intermediate/pick_and_place_cppo_v3_12000000_steps.zip
```

### v2 baseline (concat fusion, optional)

```bash
bash vla_pipeline/run_cppo_v2_train.sh
```

Generic entry point:

```bash
python vla_pipeline/train_cppo.py --config vla_pipeline/config_cppo_v3_stage1_local.yaml
```

---

## Evaluation

### Per-task full eval (200 ep/protocol)

```bash
python vla_pipeline/run_cppo_task_eval.py \
  --config vla_pipeline/config_cppo_v3_stage2_pap_local.yaml \
  --task pick_and_place \
  --checkpoint data/cppo_checkpoints_v3_stage2_pap/pick_and_place/intermediate/pick_and_place_cppo_v3_12000000_steps.zip \
  --output-dir data/cppo_eval_v4
```

### Single-checkpoint best — assemble global macro

Combines existing 200 ep results (no re-run):

```bash
python vla_pipeline/run_cppo_v4_best_full_eval.py
# writes: data/cppo_eval_v4/best_combined_full_eval.json
```

### Protocol-routed experts (recommended for max score)

Uses per-protocol 200 ep scores from each expert’s own full eval (baseline for pap P4–P7, CPPPO elsewhere):

```bash
bash vla_pipeline/run_cppo_v4_router_full_eval.sh
# or:
python vla_pipeline/run_cppo_v4_router_full_eval.py --update-latest
# writes: data/cppo_eval_v4/router_vs_baseline.json
#         data/cppo_eval_latest_vs_baseline.json
```

Live re-eval (slow, ~hours):

```bash
python vla_pipeline/run_cppo_v4_router_full_eval.py --mode live --fraction 1.0
```

### Checkpoint sweep (quick 20 ep)

```bash
bash vla_pipeline/run_cppo_v3_stage2_pap_sweep.sh
```

---

## Using routed teachers in code

### Single checkpoint

```python
from vla_pipeline.teacher import load_teacher

teacher_fn = load_teacher(
    "pick_and_place",
    checkpoint="data/cppo_checkpoints_v3_stage2_pap/pick_and_place/intermediate/pick_and_place_cppo_v3_12000000_steps.zip",
)
action = teacher_fn(obs)  # obs must include protocol one-hot if CPPPO ckpt
```

### Protocol-routed experts (privileged teacher)

```python
from vla_pipeline.teacher import load_routed_teacher

teacher_fn = load_routed_teacher("pick_and_place")
# optional: export CPPPO_ROUTER_CONFIG=/path/to/config_cppo_v4_teacher_router.json
action = teacher_fn(obs)
```

Routing table is in `config_cppo_v4_teacher_router.json`. For pap, P4–P7 load the baseline actor (56-dim obs); CPPPO checkpoints use augmented obs (68-dim). `load_routed_teacher` strips the protocol tail automatically for baseline experts.

### Hybrid / VLA pipeline

Point teacher checkpoints in your hybrid config, or set router config for pap:

```yaml
# config_v4_hybrid_local.yaml (example)
teacher:
  checkpoints:
    pushing: data/cppo_checkpoints_v3_stage1/pushing/pushing_cppo_v3.zip
    picking: data/cppo_checkpoints_v3_stage1/picking/picking_cppo_v3.zip
    pick_and_place: data/cppo_checkpoints_v3_stage2_pap/.../pick_and_place_cppo_v3_12000000_steps.zip
  router_config: vla_pipeline/config_cppo_v4_teacher_router.json
```

---

## File map

| Path | Role |
|------|------|
| `train_cppo.py` | Training entry |
| `config_cppo_v3_stage1_local.yaml` | Stage-1 v3 |
| `config_cppo_v3_stage2_pap_local.yaml` | Stage-2 pap |
| `config_cppo_v4_teacher_router.json` | Router expert table |
| `run_cppo_v4_router_full_eval.py` | Router full eval / assembly |
| `run_cppo_v4_best_full_eval.py` | Single-ckpt global assembly |
| `teacher.py` | `load_teacher`, `load_routed_teacher` |
| `causal_world/wrappers/protocol_*` | Protocol obs + curriculum |

---

## Reproducing reported numbers locally

1. Train stage-1 (3 tasks) and stage-2 pap (or use your saved checkpoints).
2. Full-eval push, pick (stage-1 ckpt) and pap s2_12M at `--fraction 1.0`.
3. Run `run_cppo_v4_best_full_eval.py` → expect global ≈ **0.574** (+15%).
4. Run `run_cppo_v4_router_full_eval.sh` → expect global ≈ **0.602** (+20.5%).

Eval artifacts live under `data/cppo_eval_*` (not in git). Compare against `data/baseline_eval/summary_all_tasks.json`.
