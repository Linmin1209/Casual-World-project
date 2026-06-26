#!/usr/bin/env bash
# Protocol-routed expert full eval (200 ep): pap P4-P7 use baseline, rest CPPPO.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${CONDA_BASE:-/home/work/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-causal_world}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python "${ROOT}/vla_pipeline/run_cppo_v4_router_full_eval.py" \
  --mode assemble \
  --output "${ROOT}/data/cppo_eval_v4/router_full_eval.json" \
  --comparison "${ROOT}/data/cppo_eval_v4/router_vs_baseline.json" \
  --update-latest
