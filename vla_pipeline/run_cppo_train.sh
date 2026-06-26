#!/usr/bin/env bash
# Train CPPPO teachers (protocol-weighted PPO finetune) for all tasks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/work/miniconda3}"
ENV_NAME="${ENV_NAME:-causal_world}"
CFG="${ROOT}/vla_pipeline/config_cppo_local.yaml"
LOG="${ROOT}/data/cppo_train.log"
TASKS="${CPPPO_TASKS:-pushing picking pick_and_place}"
NUM_ENVS="${CPPPO_NUM_ENVS:-}"

mkdir -p "${ROOT}/data/cppo_checkpoints"
exec > >(tee -a "${LOG}") 2>&1

echo "=== CPPPO train start $(date -Iseconds) tasks=${TASKS} num_envs=${NUM_ENVS:-config} ==="

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ARGS=(--config "${CFG}" --tasks ${TASKS})
if [[ -n "${NUM_ENVS}" ]]; then
  ARGS+=(--num_envs "${NUM_ENVS}")
fi
python "${ROOT}/vla_pipeline/train_cppo.py" "${ARGS[@]}"

echo "=== CPPPO train done $(date -Iseconds) ==="
echo "Checkpoints: ${ROOT}/data/cppo_checkpoints/{task}/{task}_cppo.zip"
echo "Log: ${LOG}"
