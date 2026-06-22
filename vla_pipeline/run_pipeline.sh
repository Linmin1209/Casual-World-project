#!/usr/bin/env bash
# End-to-end hybrid VLA pipeline for CausalWorld
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v2.yaml"
CONDA_ENV="${CAUSALWORLD_CONDA_ENV:-causal_world}"

# Activate conda env (baseline PPO needs Python 3.7 + TF1)
if [ -f "${CONDA_SH:-/data1/linmin/miniconda3/etc/profile.d/conda.sh}" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_SH:-/data1/linmin/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV}"
fi

python - <<'PY' || {
import sys
print("Checking teacher dependencies (stable-baselines + TF1)...", file=sys.stderr)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from stable_baselines import PPO2  # noqa: F401
assert tf.__version__.startswith("1."), tf.__version__
print("tf", tf.__version__, "ok")
PY
  echo "Missing teacher deps. Run:" >&2
  echo "  pip install -r ${ROOT}/vla_pipeline/requirements_teacher.txt" >&2
  exit 1
}

echo "=== [1/4] Collect teacher demos (P6-P11) ==="
python "${ROOT}/vla_pipeline/collect_data.py" --config "${CFG}"

echo "=== [2/4] Export VLA dataset (v2 step filter) ==="
python "${ROOT}/vla_pipeline/export_dataset_v2.py" --config "${CFG}"

echo "=== [3/4] BC pretrain ==="
python "${ROOT}/vla_pipeline/train_bc.py" --config "${CFG}"

echo "=== [4/4] Evaluate hybrid vs baseline ==="
python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}"

echo "Done. See data/vla_eval_v2/summary_all_tasks.json"
