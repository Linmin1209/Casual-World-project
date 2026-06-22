#!/usr/bin/env bash
# V4 DAgger: 2 rounds of on-policy hard-protocol collection + BC fine-tune + eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v4_dagger.yaml"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1

# Inline config sync (avoid shell function name / encoding issues)
sync_v4_config_ckpt() {
  python - <<'PY'
import yaml
from pathlib import Path
p = Path("/data1/linmin/CausalWorld/vla_pipeline/config_v4_dagger.yaml")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
ck = str(Path(cfg["training"]["checkpoint_dir"]) / "bc_best.pt")
cfg["hybrid"]["checkpoint"] = ck
cfg["training"]["init_checkpoint"] = ck
p.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True), encoding="utf-8")
print(f"[config] hybrid + init_checkpoint -> {ck}")
PY
}

for ROUND in 1 2; do
  echo "=== [D${ROUND}a] DAgger collection round ${ROUND} (P6-P11 hybrid rollout) ==="
  python "${ROOT}/vla_pipeline/dagger_round.py" --config "${CFG}" --round "${ROUND}" --num_workers 6

  echo "=== [D${ROUND}b] Export teacher + dagger demos ==="
  python "${ROOT}/vla_pipeline/export_dataset_v2.py" --config "${CFG}"

  echo "=== [D${ROUND}c] BC fine-tune (init from previous best) ==="
  python "${ROOT}/vla_pipeline/train_bc.py" --config "${CFG}"

  if [[ "${ROUND}" -lt 2 ]]; then
    sync_v4_config_ckpt
  fi
done

echo "=== [D5] Hybrid eval (fast_eval=50 eps, batch workers) ==="
mkdir -p "${ROOT}/data/vla_eval_v4_dagger"
python "${ROOT}/vla_pipeline/evaluate.py" --config "${CFG}" --num_workers 6 \
  2>&1 | tee "${ROOT}/data/vla_eval_v4_dagger/eval.log"

echo "Done -> data/vla_eval_v4_dagger/summary_all_tasks.json"
