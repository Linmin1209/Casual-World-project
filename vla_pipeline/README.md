# VLA Hybrid Pipeline (CausalWorld)

Vision-language-action policy training for CausalWorld: teacher demos → BC → DAgger → hybrid eval → optional EA-AWR RL fine-tuning.

**CPPPO privileged teachers** (protocol-conditioned PPO, +15% single-ckpt / +20.5% routed): see **[CPPPO.md](CPPPO.md)**.

## Quick start

```bash
conda activate causal_world
export PYTHONPATH=/path/to/CausalWorld

# Phase A: collect demos + export + BC
bash vla_pipeline/run_phase_a.sh

# V5 full pipeline
bash vla_pipeline/run_phase_v5.sh

# Optional RL fine-tune (expert BC + hard-protocol EA-AWR)
bash vla_pipeline/run_train_v5_rl.sh
```

Large datasets and checkpoints are excluded from git; see `config_v5.yaml` for output paths under `data/`.
