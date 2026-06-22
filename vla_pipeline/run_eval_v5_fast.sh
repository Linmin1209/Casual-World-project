#!/usr/bin/env bash
# V5 fast eval: all 12 protocols x 50 eps (batch workers)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v5.yaml"
OUT="${ROOT}/data/vla_eval_v5"
LOG="${OUT}/eval_fast.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1
mkdir -p "${OUT}"

echo "=== V5 fast eval $(date -Iseconds) ===" | tee "${LOG}"
echo "12 protocols x 50 eps | selective task×protocol beta" | tee -a "${LOG}"

python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}" \
  --episodes_per_protocol 50 \
  --num_workers 6 \
  2>&1 | tee -a "${LOG}"

echo "Done -> ${OUT}/summary_all_tasks.json" | tee -a "${LOG}"
