#!/usr/bin/env bash
# V5 optional: collect extra picking P6-P9 teacher demos (run in parallel with training)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v5_collect_hard_picking.yaml"
LOG="${ROOT}/data/vla_demos/v5_collect_hard_picking.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1

echo "=== V5 hard picking collection $(date -Iseconds) ===" | tee -a "${LOG}"
python "${ROOT}/vla_pipeline/collect_until_usable.py" --config "${CFG}" --num_workers 6 \
  2>&1 | tee -a "${LOG}"

echo "Done. Re-export: python vla_pipeline/export_dataset_v2.py --config vla_pipeline/config_v5.yaml"
