#!/usr/bin/env bash
# Phase C+: keep collecting P6-P11 until 30 usable episodes per task×protocol
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v3_collect_hard.yaml"
LOG="${ROOT}/data/vla_demos/phase_c_collect.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world

echo "=== Phase C+: until-usable P6-P11 (ep_max>=0.05) ===" | tee -a "${LOG}"
python "${ROOT}/vla_pipeline/collect_until_usable.py" --config "${CFG}" --num_workers 6 \
  2>&1 | tee -a "${LOG}"

echo "Done. Re-export: python vla_pipeline/export_dataset_v2.py --config vla_pipeline/config_v3.yaml"
