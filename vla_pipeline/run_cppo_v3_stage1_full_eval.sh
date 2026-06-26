#!/usr/bin/env bash
# Full eval (200 ep/protocol) for CPPPO v3 stage-1 checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${ROOT}/vla_pipeline/config_cppo_v3_stage1_local.yaml"
LOG="${ROOT}/data/cppo_eval_v3_stage1/full_eval.log"
OUT="${ROOT}/data/cppo_eval_v3_stage1"

mkdir -p "${OUT}"
exec > >(tee -a "${LOG}") 2>&1

echo "=== CPPPO v3 stage-1 full eval start $(date -Iseconds) ==="

source "${CONDA_BASE:-/home/work/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-causal_world}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 "${ROOT}/vla_pipeline/run_cppo_task_eval.py" \
  --config "${CFG}" \
  --tasks pushing picking pick_and_place \
  --output-dir "${OUT}"

echo "=== CPPPO v3 stage-1 full eval done $(date -Iseconds) ==="
