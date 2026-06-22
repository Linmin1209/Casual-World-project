#!/usr/bin/env bash
# Re-render black RGB PNGs by replaying stored actions (no teacher re-run).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/data/vla_demos/repair_rgb.log"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world

echo "=== Repair demo RGB (skip non-black unless --force) ===" | tee "${LOG}"
python "${ROOT}/vla_pipeline/repair_rgb.py" --num_workers 6 "$@" 2>&1 | tee -a "${LOG}"
