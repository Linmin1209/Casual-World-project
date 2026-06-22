#!/usr/bin/env bash
# V5 pipeline: export -> BC (v3 init) -> selective DAgger -> BC -> fast eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v5.yaml"
LOG="${ROOT}/data/vla_pipeline_v5.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1
mkdir -p "${ROOT}/data"

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

exec > >(tee -a "${LOG}") 2>&1
echo "=== V5 pipeline start $(date -Iseconds) ==="

echo "=== [V5-0] Export teacher + selective V4 dagger demos ==="
python "${ROOT}/vla_pipeline/export_dataset_v2.py" --config "${CFG}"

echo "=== [V5-1] BC round 0 (init from V3) ==="
python "${ROOT}/vla_pipeline/train_bc.py" --config "${CFG}" --round 0

R0_CKPT="${ROOT}/data/vla_checkpoints_v5/bc_best_r00.pt"
sync_v5_config_ckpt "${R0_CKPT}"

echo "=== [V5-2] Selective DAgger round 1 (pushing P7/P10/P11, picking P11) ==="
python "${ROOT}/vla_pipeline/dagger_round.py" --config "${CFG}" --round 1 --num_workers 6

echo "=== [V5-3] Re-export with new dagger data ==="
python "${ROOT}/vla_pipeline/export_dataset_v2.py" --config "${CFG}"

echo "=== [V5-4] BC round 1 (init from r00 best) ==="
python "${ROOT}/vla_pipeline/train_bc.py" --config "${CFG}" --round 1

R1_CKPT="${ROOT}/data/vla_checkpoints_v5/bc_best_r01.pt"
if [[ -f "${R1_CKPT}" ]]; then
  sync_v5_config_ckpt "${R1_CKPT}"
else
  sync_v5_config_ckpt "${ROOT}/data/vla_checkpoints_v5/bc_best.pt"
fi

echo "=== [V5-5] Fast eval (50 eps x 12 protocols) ==="
bash "${ROOT}/vla_pipeline/run_eval_v5_fast.sh"

echo "=== V5 pipeline done $(date -Iseconds) ==="
echo "Results -> data/vla_eval_v5/summary_all_tasks.json"
