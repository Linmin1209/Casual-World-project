#!/usr/bin/env bash
# Quick gate: hard P6-P11 x 20 eps, compare macro to baseline before full eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CFG="${ROOT}/vla_pipeline/config_v5.yaml"
OUT="${ROOT}/data/vla_eval_v5_gate"
LOG="${OUT}/gate.log"
TMP_CFG="${OUT}/config_gate.yaml"

source /data1/linmin/miniconda3/etc/profile.d/conda.sh
conda activate causal_world
export PYTHONUNBUFFERED=1
mkdir -p "${OUT}"

python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${CFG}").read_text(encoding="utf-8"))
cfg["evaluation"]["output_dir"] = "${OUT}"
Path("${TMP_CFG}").write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
PY

echo "=== V5 eval gate $(date -Iseconds) ===" | tee "${LOG}"

python "${ROOT}/vla_pipeline/evaluate.py" --config "${TMP_CFG}" \
  --protocols P6 P7 P8 P9 P10 P11 \
  --episodes_per_protocol 20 \
  --num_workers 2 \
  2>&1 | tee -a "${LOG}"

python3 - <<'PY' | tee -a "${LOG}"
import json
from pathlib import Path

gate = Path("/data1/linmin/CausalWorld/data/vla_eval_v5_gate")
bb = Path("/data1/linmin/CausalWorld/baseline_eval")
tasks = ["pushing", "picking", "pick_and_place"]
protos = ["P6","P7","P8","P9","P10","P11"]

def hard_macro(folder):
    scores = []
    for t in tasks:
        p = folder / f"{t}_all_protocols.json"
        if not p.is_file():
            continue
        d = json.load(open(p))
        for pr in protos:
            if pr in d["protocols"]:
                scores.append(d["protocols"][pr]["mean_full_integrated_fractional_success"])
    return sum(scores)/max(1, len(scores))

hy = hard_macro(gate)
ba = hard_macro(bb)
print(f"Hard macro: hybrid={hy:.4f} baseline={ba:.4f} delta={(hy-ba)/max(ba,1e-6)*100:+.1f}%")
print("PASS gate" if hy >= ba * 0.98 else "FAIL gate — tune beta or checkpoint before full eval")
PY
