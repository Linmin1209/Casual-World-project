#!/usr/bin/env bash
# Continue V5 after BC round 0 finishes: DAgger -> export -> train r1 -> gate eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v5.yaml"
LOG="${ROOT}/data/vla_pipeline_v5_continue.log"
NUM_WORKERS="${NUM_WORKERS:-2}"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1

exec > >(tee -a "${LOG}") 2>&1
echo "=== V5 continue start $(date -Iseconds) | workers=${NUM_WORKERS} ==="

sync_v5_config_ckpt() {
  local CKPT="$1"
  python - <<PY
import yaml
from pathlib import Path
p = Path("${CFG}")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
ck = "${CKPT}"
cfg["hybrid"]["checkpoint"] = ck
cfg["training"]["init_checkpoint"] = ck
p.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True), encoding="utf-8")
print(f"[config] hybrid + init_checkpoint -> {ck}")
PY
}

R0_CKPT="${ROOT}/data/vla_checkpoints_v5/bc_best_r00.pt"
if [[ ! -f "${R0_CKPT}" ]]; then
  echo "Missing ${R0_CKPT} — wait for train round 0 to finish."
  exit 1
fi
sync_v5_config_ckpt "${R0_CKPT}"

echo "=== [V5-2] Selective DAgger round 1 (skip_frame=10) ==="
python "${ROOT}/vla_pipeline/dagger_round.py" --config "${CFG}" --round 1 --num_workers "${NUM_WORKERS}"

echo "=== [V5-3] Re-export ==="
python "${ROOT}/vla_pipeline/export_dataset_v2.py" --config "${CFG}"

echo "=== [V5-4] BC round 1 ==="
python "${ROOT}/vla_pipeline/train_bc.py" --config "${CFG}" --round 1

R1_CKPT="${ROOT}/data/vla_checkpoints_v5/bc_best_r01.pt"
if [[ -f "${R1_CKPT}" ]]; then
  sync_v5_config_ckpt "${R1_CKPT}"
else
  sync_v5_config_ckpt "${ROOT}/data/vla_checkpoints_v5/bc_best.pt"
fi

echo "=== [V5-5] Eval gate (20 ep hard) ==="
bash "${ROOT}/vla_pipeline/run_eval_v5_gate.sh"

echo "=== V5 continue done $(date -Iseconds) ==="
