#!/usr/bin/env bash
# CPPPO v4a — pap P4/P5 recovery from s2_12M checkpoint (2M steps).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${ROOT}/vla_pipeline/config_cppo_v4_pap_p45_local.yaml"
LOG="${ROOT}/data/cppo_train_v4a.log"
mkdir -p "${ROOT}/data/cppo_checkpoints_v4_pap_p45"
exec > >(tee -a "${LOG}") 2>&1
echo "=== CPPPO v4a P4/P5 recovery $(date -Iseconds) ==="
source "${CONDA_BASE:-/home/work/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-causal_world}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
python "${ROOT}/vla_pipeline/train_cppo.py" --config "${CFG}" --tasks pick_and_place --no_resume
echo "=== CPPPO v4a done $(date -Iseconds) ==="
