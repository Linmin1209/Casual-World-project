#!/usr/bin/env bash
# V5 sanity: V3 checkpoint + V5 selective beta (no retrain). Validates picking recovery.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v5_sanity.yaml"
OUT="${ROOT}/data/vla_eval_v5_sanity"
LOG="${OUT}/eval_sanity.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1
mkdir -p "${OUT}"

echo "=== V5 sanity eval (V3 ckpt + V5 beta) $(date -Iseconds) ===" | tee "${LOG}"

python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}" \
  --protocols P6 P7 P8 P9 P10 P11 \
  --episodes_per_protocol 50 \
  --num_workers 6 \
  2>&1 | tee -a "${LOG}"

echo "Done -> ${OUT}/summary_all_tasks.json" | tee -a "${LOG}"
