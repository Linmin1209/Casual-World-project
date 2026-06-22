#!/usr/bin/env bash
# V3 pipeline: collect P0-P5 -> export all demos -> train MSE -> eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v3.yaml"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world

echo "=== [B1] Collect teacher demos P0-P5 (skip existing) ==="
python "${ROOT}/vla_pipeline/collect_data.py" --config "${CFG}"

echo "=== [B2] Export combined dataset (900 hard + new easy episodes) ==="
python "${ROOT}/vla_pipeline/export_dataset_v2.py" --config "${CFG}"

echo "=== [B3] Train ResNet+FiLM direct MSE ==="
python "${ROOT}/vla_pipeline/train_bc.py" --config "${CFG}"

echo "=== [B4] Teacher sanity (should match baseline on P0-P5) ==="
python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}" \
  --teacher_only --tasks pushing --num_workers 6

echo "=== [B5] Hybrid eval (protocol-gated residual) ==="
python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}" --num_workers 6

echo "Done -> data/vla_eval_v3/summary_all_tasks.json"
