#!/usr/bin/env bash
# CPPPO v4 Phase 0 — s2_12M pap full eval (200 ep/protocol).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/data/cppo_eval_v4"
LOG="${OUT}/phase0_pap_s2_12M_full.log"
CKPT="${ROOT}/data/cppo_checkpoints_v3_stage2_pap/pick_and_place/intermediate/pick_and_place_cppo_v3_12000000_steps.zip"
mkdir -p "${OUT}"
exec > >(tee -a "${LOG}") 2>&1
echo "=== v4 Phase 0: pap s2_12M full eval $(date -Iseconds) ==="
source "${CONDA_BASE:-/home/work/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-causal_world}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
python "${ROOT}/vla_pipeline/run_cppo_task_eval.py" \
  --config "${ROOT}/vla_pipeline/config_cppo_v3_stage2_pap_local.yaml" \
  --task pick_and_place \
  --checkpoint "${CKPT}" \
  --output-dir "${OUT}"
echo "=== v4 Phase 0 done $(date -Iseconds) ==="
