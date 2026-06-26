#!/usr/bin/env bash
# Train CPPPO v2 teachers (fusion MLP + adaptive protocol sampling).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CPPPO_VERSION=v2
export CPPPO_TEACHER=1
CFG="${ROOT}/vla_pipeline/config_cppo_v2_local.yaml"
LOG="${ROOT}/data/cppo_train_v2.log"
TASKS="${CPPPO_TASKS:-pushing picking pick_and_place}"
NUM_ENVS="${CPPPO_NUM_ENVS:-}"

mkdir -p "${ROOT}/data/cppo_checkpoints_v2_official"
exec > >(tee -a "${LOG}") 2>&1

echo "=== CPPPO v2 train start $(date -Iseconds) tasks=${TASKS} ==="
echo "Official-aligned finetune: 10M steps, n_steps=6000, 20 envs, lr=2.5e-4, net=[256,256]"
echo "Checkpoints: data/cppo_checkpoints_v2_official/<task>/"
echo "TensorBoard: data/cppo_checkpoints_v2/<task>/tensorboard/"

# shellcheck disable=SC1091
source "${CONDA_BASE:-/home/work/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-causal_world}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ARGS=(--config "${CFG}" --tasks ${TASKS})
if [[ -n "${NUM_ENVS}" ]]; then
  ARGS+=(--num_envs "${NUM_ENVS}")
fi
python "${ROOT}/vla_pipeline/train_cppo.py" "${ARGS[@]}"

echo "=== CPPPO v2 done $(date -Iseconds) ==="
