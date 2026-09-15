#!/usr/bin/env bash
set -euo pipefail

VIDEOPHY_DIR="${VIDEOPHY_DIR:-/data/datasets/gagi/videophy}"
VENV_DIR="${VENV_DIR:-/data/datasets/gagi/envs/videophy_venv}"
HF_HOME="${HF_HOME:-/data/datasets/gagi/.hf_home}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/datasets/gagi/models/videocon_physics}"

mkdir -p "$(dirname "${VIDEOPHY_DIR}")" "$(dirname "${VENV_DIR}")" "${HF_HOME}" "$(dirname "${CHECKPOINT_DIR}")"

if [[ ! -d "${VIDEOPHY_DIR}/.git" ]]; then
  git clone https://github.com/Hritikbansal/videophy.git "${VIDEOPHY_DIR}"
else
  git -C "${VIDEOPHY_DIR}" pull --ff-only
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  /home/jovyan/miniconda/envs/giga_models/bin/python -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${VIDEOPHY_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install huggingface_hub pandas

echo "VideoPhy env ready:"
echo "  VIDEOPHY_DIR=${VIDEOPHY_DIR}"
echo "  VENV_DIR=${VENV_DIR}"
echo "  HF_HOME=${HF_HOME}"
echo
echo "Large checkpoint download command:"
echo "  HF_HOME=${HF_HOME} ${VENV_DIR}/bin/huggingface-cli download videophysics/videocon_physics --local-dir ${CHECKPOINT_DIR} --local-dir-use-symlinks False"
echo
echo "Use this checkpoint in PA-II kjob:"
echo "  CHECKPOINT=${CHECKPOINT_DIR}"
