#!/usr/bin/env bash
# Minimal conda (python only) + pip (everything else). Much faster than conda env create.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/"
PYPI="https://pypi.tuna.tsinghua.edu.cn/simple"
ENV_NAME="causal_world"

echo "[1/2] conda: python 3.7.12 + pip only (skip global defaults channels)..."
conda create -n "${ENV_NAME}" -y \
  --override-channels -c "${MIRROR}" \
  --repodata-fn current_repodata.json \
  python=3.7.12 pip

echo "[2/2] pip: install remaining deps..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
pip install -i "${PYPI}" \
  scipy sphinx jupyter \
  pybullet==3.2.5 gym==0.17.2 catkin_pkg sphinx_rtd_theme pytest psutil sphinxcontrib-bibtex

echo "[3/3] pip: teacher (TF1 + stable-baselines) + VLA (torch)..."
pip install -i "${PYPI}" -r "${ROOT}/vla_pipeline/requirements_teacher.txt"
pip install -i "${PYPI}" -r "${ROOT}/vla_pipeline/requirements_vla.txt"

echo "Done: conda activate ${ENV_NAME}"
