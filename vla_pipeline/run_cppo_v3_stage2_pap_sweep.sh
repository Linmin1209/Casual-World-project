#!/usr/bin/env bash
# Quick-eval sweep of stage-2a pap intermediate checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${ROOT}/vla_pipeline/config_cppo_v3_stage2_pap_local.yaml"
OUT="${ROOT}/data/cppo_eval_v3_stage2_pap/sweep_quick.json"
LOG="${ROOT}/data/cppo_eval_v3_stage2_pap/sweep_quick.log"
INT="${ROOT}/data/cppo_checkpoints_v3_stage2_pap/pick_and_place/intermediate"
FINAL="${ROOT}/data/cppo_checkpoints_v3_stage2_pap/pick_and_place/pick_and_place_cppo_v3.zip"
S1="${ROOT}/data/cppo_checkpoints_v3_stage1/pick_and_place/pick_and_place_cppo_v3.zip"

mkdir -p "${ROOT}/data/cppo_eval_v3_stage2_pap"
exec > >(tee -a "${LOG}") 2>&1

echo "=== stage-2a pap checkpoint sweep $(date -Iseconds) ==="

source "${CONDA_BASE:-/home/work/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-causal_world}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 "${ROOT}/vla_pipeline/run_cppo_checkpoint_sweep.py" \
  --config "${CFG}" \
  --task pick_and_place \
  --fraction 0.1 \
  --output "${OUT}" \
  --labels s1_final s2_3M s2_6M s2_9M s2_12M s2_15M_final \
  --checkpoints \
    "${S1}" \
    "${INT}/pick_and_place_cppo_v3_3000000_steps.zip" \
    "${INT}/pick_and_place_cppo_v3_6000000_steps.zip" \
    "${INT}/pick_and_place_cppo_v3_9000000_steps.zip" \
    "${INT}/pick_and_place_cppo_v3_12000000_steps.zip" \
    "${FINAL}"

echo "=== sweep done $(date -Iseconds) ==="
