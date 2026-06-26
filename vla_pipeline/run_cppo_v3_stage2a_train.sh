#!/usr/bin/env bash
# CPPPO v3 Stage-2a — pick_and_place only (15M steps).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${ROOT}/vla_pipeline/config_cppo_v3_stage2_pap_local.yaml"
LOG="${ROOT}/data/cppo_train_v3_stage2a.log"
TASKS="${CPPPO_TASKS:-pick_and_place}"

mkdir -p "${ROOT}/data/cppo_checkpoints_v3_stage2_pap"
exec > >(tee -a "${LOG}") 2>&1

echo "=== CPPPO v3 stage-2a start $(date -Iseconds) tasks=${TASKS} ==="
echo "pap-only finetune from stage-1 ckpt, 15M steps, n_steps=2048, lr=4e-5"
echo "Checkpoints: data/cppo_checkpoints_v3_stage2_pap/pick_and_place/"

source "${CONDA_BASE:-/home/work/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-causal_world}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ARGS=(--config "${CFG}" --tasks ${TASKS} --no_resume)
if [[ -n "${CPPPO_NUM_ENVS:-}" ]]; then
  ARGS+=(--num_envs "${CPPPO_NUM_ENVS}")
fi
python "${ROOT}/vla_pipeline/train_cppo.py" "${ARGS[@]}"

echo "=== CPPPO v3 stage-2a done $(date -Iseconds) ==="
