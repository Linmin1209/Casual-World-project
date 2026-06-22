#!/usr/bin/env bash
# Phase A: reuse 900 episodes on disk -> v2 export -> train -> eval (no re-simulation)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v2.yaml"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world

echo "=== [A1] Re-export (step filter + rejected) ==="
python "${ROOT}/vla_pipeline/export_dataset_v2.py" --config "${CFG}"

echo "=== [A2] Flow Matching train (ResNet+FiLM, multi-step) ==="
python "${ROOT}/vla_pipeline/train_bc.py" --config "${CFG}"

echo "=== [A3] Evaluate hybrid (12 protocols x 3 tasks) ==="
python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}"

echo "Done. See data/vla_eval_v2/summary_all_tasks.json"
