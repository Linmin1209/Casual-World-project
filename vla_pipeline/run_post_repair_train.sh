#!/usr/bin/env bash
# Wait for RGB repair, then export dataset and train v3 BC.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${ROOT}/vla_pipeline/config_v3.yaml"
LOG="${ROOT}/data/vla_demos/post_repair_train.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p "${ROOT}/data/vla_checkpoints_v3"

REPAIR_PID="${1:-}"
if [[ -n "${REPAIR_PID}" ]] && kill -0 "${REPAIR_PID}" 2>/dev/null; then
  echo "=== Waiting for repair_rgb PID ${REPAIR_PID} ===" | tee "${LOG}"
  while kill -0 "${REPAIR_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "Repair process finished." | tee -a "${LOG}"
else
  echo "=== No repair PID (or already done); starting export/train ===" | tee "${LOG}"
fi

echo "=== Export v3 dataset ===" | tee -a "${LOG}"
python "${ROOT}/vla_pipeline/export_dataset_v2.py" --config "${CFG}" 2>&1 | tee -a "${LOG}"

echo "=== Train v3 MSE BC ===" | tee -a "${LOG}"
python "${ROOT}/vla_pipeline/train_bc.py" --config "${CFG}" 2>&1 | tee -a "${LOG}"

echo "Done -> ${ROOT}/data/vla_checkpoints_v3/bc_best.pt" | tee -a "${LOG}"
