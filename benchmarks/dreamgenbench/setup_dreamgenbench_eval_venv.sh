#!/usr/bin/env bash
set -euo pipefail

# Create an isolated Python venv for official DreamGenBench Qwen2.5-VL eval.
# This avoids changing the existing EVEWorld conda environment, which is
# also used by VBench and GigaWorld inference.

BASE_PYTHON="${BASE_PYTHON:-/home/jovyan/miniconda/envs/EVEWorld/bin/python}"
VENV_DIR="${VENV_DIR:-/data/datasets/gagi/envs/dreamgenbench_eval_venv}"

if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "Missing BASE_PYTHON: ${BASE_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating DreamGenBench eval venv: ${VENV_DIR}"
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
else
  echo "Using existing DreamGenBench eval venv: ${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install --no-deps --ignore-installed \
  "transformers==5.11.0" \
  "tokenizers==0.22.2" \
  "huggingface_hub==1.18.0"

"${PYTHON_BIN}" -m pip install \
  "qwen-vl-utils" \
  "decord" \
  "opencv-python-headless" \
  "tqdm"

echo
"${PYTHON_BIN}" - <<'PY'
import cv2
import decord
import qwen_vl_utils
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401

print("DreamGenBench eval venv import checks:")
print("  transformers:", transformers.__version__)
print("  qwen_vl_utils:", qwen_vl_utils.__file__)
print("  decord:", decord.__version__)
print("  cv2:", cv2.__version__)
print("  Qwen2_5_VLForConditionalGeneration: ok")
PY

echo
echo "DreamGenBench eval venv is ready: ${VENV_DIR}"

