#!/usr/bin/env bash
# Expert BC warm-start -> EA-AWR RL fine-tune -> gate eval with RL checkpoint
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
RL_CFG="${ROOT}/vla_pipeline/config_v5_rl.yaml"
EVAL_CFG="${ROOT}/vla_pipeline/config_v5.yaml"
LOG="${ROOT}/data/vla_rl_train.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1

exec > >(tee -a "${LOG}") 2>&1
echo "=== V5 RL pipeline $(date -Iseconds) ==="
mkdir -p "${ROOT}/data/vla_checkpoints_v5_rl" "${ROOT}/data/vla_eval_v5_rl"

echo "=== [RL-1] EA-AWR fine-tune (expert + hard protocol rollouts) ==="
python "${ROOT}/vla_pipeline/train_rl_finetune.py" --config "${RL_CFG}"

echo "=== [RL-2] Patch hybrid checkpoint path ==="
python - <<PY
import yaml
from pathlib import Path
p = Path("${EVAL_CFG}")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
ck = "${ROOT}/data/vla_checkpoints_v5_rl/rl_best.pt"
cfg["hybrid"]["checkpoint"] = ck
p.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
print(f"[config] hybrid.checkpoint -> {ck}")
PY

echo "=== [RL-3] Gate eval (20 ep hard, skip_frame=10) ==="
python - <<PY
import yaml
from pathlib import Path
p = Path("${EVAL_CFG}")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg["evaluation"]["output_dir"] = "${ROOT}/data/vla_eval_v5_rl"
out = Path("${ROOT}/data/vla_eval_v5_rl")
out.mkdir(parents=True, exist_ok=True)
(out / "config_gate.yaml").write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
PY

python "${ROOT}/vla_pipeline/evaluate.py" \
  --config "${ROOT}/data/vla_eval_v5_rl/config_gate.yaml" \
  --protocols P6 P7 P8 P9 P10 P11 \
  --episodes_per_protocol 20 \
  --num_workers 2

echo "=== V5 RL pipeline done $(date -Iseconds) ==="
