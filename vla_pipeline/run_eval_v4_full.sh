#!/usr/bin/env bash
# Full V4 hybrid eval: 12 protocols x 200 eps x 3 tasks (matches baseline protocol).
# Config: P6 beta=0 (pure teacher), P0-P5 beta_easy=0, batch_workers for speed.
# ETA: ~8-20h depending on pick_and_place (1667 steps/ep on hard protocols).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v4_dagger.yaml"
OUT="${ROOT}/data/vla_eval_v4_full"
LOG="${OUT}/eval_full.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1
mkdir -p "${OUT}"

echo "=== V4 full eval $(date -Iseconds) ===" | tee "${LOG}"
echo "36 protocols x 200 eps | P6 beta=0 | batch_workers | ckpt=v4_dagger/bc_best.pt" | tee -a "${LOG}"

python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}" \
  --episodes_per_protocol 200 \
  --num_workers 6 \
  2>&1 | tee -a "${LOG}"

echo "Done -> ${OUT}/summary_all_tasks.json" | tee -a "${LOG}"
