#!/usr/bin/env bash
# Ultra-fast V4 hybrid eval (hard P6-P11 only).
# - batch_workers: 6 GPU workers load TF+VLA once per task (~3x faster than per-protocol)
# - 10 eps for pushing/picking, 5 eps for pick_and_place (1667 steps/ep)
# ETA: pushing ~25min, picking ~25min, pick_and_place ~45min -> ~1.5h total
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v4_dagger.yaml"
OUT="${ROOT}/data/vla_eval_v4_dagger"
LOG="${OUT}/eval_fast.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1
mkdir -p "${OUT}"

echo "=== V4 ultra-fast eval $(date -Iseconds) ===" | tee "${LOG}"
echo "hard P6-P11 | batch_workers | 10/10/5 eps | results per task json" | tee -a "${LOG}"

python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}" \
  --protocols P6 P7 P8 P9 P10 P11 \
  --num_workers 6 \
  2>&1 | tee -a "${LOG}"

echo "Done -> ${OUT}/summary_all_tasks.json" | tee -a "${LOG}"
