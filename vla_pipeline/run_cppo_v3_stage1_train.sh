#!/usr/bin/env bash
# Train CPPPO v3 Stage-1 (residual policy + anchor sampling).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CPPPO_VERSION=v3
export CPPPO_STAGE=1
CFG="${ROOT}/vla_pipeline/config_cppo_v3_stage1_local.yaml"
LOG="${ROOT}/data/cppo_train_v3_stage1.log"
TASKS="${CPPPO_TASKS:-pushing picking pick_and_place}"
NUM_ENVS="${CPPPO_NUM_ENVS:-}"

mkdir -p "${ROOT}/data/cppo_checkpoints_v3_stage1"
exec > >(tee -a "${LOG}") 2>&1

echo "=== CPPPO v3 stage-1 start $(date -Iseconds) tasks=${TASKS} ==="
echo "Residual policy, space_a_b, 15M steps/task, n_steps=4096, lr=1.5e-4"
echo "Checkpoints: data/cppo_checkpoints_v3_stage1/<task>/"

source "${CONDA_BASE:-/home/work/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-causal_world}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ARGS=(--config "${CFG}" --tasks ${TASKS} --no_resume)
if [[ -n "${NUM_ENVS}" ]]; then
  ARGS+=(--num_envs "${NUM_ENVS}")
fi
python "${ROOT}/vla_pipeline/train_cppo.py" "${ARGS[@]}"

echo "=== CPPPO v3 stage-1 done $(date -Iseconds) ==="
