#!/usr/bin/env bash
set -euo pipefail

# Create a lightweight training venv that reuses the existing EVEWorld conda
# env for heavy packages such as torch/natten/deepspeed, while overriding the
# Python packages that VBench commonly downgrades.
#
# This keeps VBench on conda env EVEWorld and uses this venv only for
# GigaWorld training/inference.

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-${EVEWORLD_ROOT}:${EVEWORLD_ROOT}/giga-world-0:${EVEWORLD_ROOT}/giga-models}"
export TRAIN_VENV

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

mkdir -p "$(dirname "${TRAIN_VENV}")"

if [[ ! -x "${TRAIN_VENV}/bin/python" ]]; then
  echo "Creating training venv: ${TRAIN_VENV}"
  python -m venv --system-site-packages "${TRAIN_VENV}"
fi

# shellcheck disable=SC1091
source "${TRAIN_VENV}/bin/activate"

echo "Training python: $(python -c 'import sys; print(sys.executable)')"
echo "Installing/refreshing lightweight training dependency overrides..."
python -m pip install --no-deps --ignore-installed -U \
  "accelerate==1.13.0" \
  "transformers==5.11.0" \
  "tokenizers==0.22.2" \
  "huggingface_hub==1.18.0" \
  "peft==0.19.1" \
  "timm==1.0.27" \
  "numpy==2.4.4" \
  "Pillow==11.3.0"

export PYTHONPATH="${PYTHONPATH_ROOT}:${PYTHONPATH:-}"

echo
echo "Training venv import checks:"
python - <<'PY'
import sys
import os

import accelerate
import diffusers
import torch
import transformers
import tokenizers
import huggingface_hub
import peft
import timm
import numpy
from PIL import Image
import shutil

print("  python:", sys.executable)
print("  accelerate bin:", shutil.which("accelerate"))
print("  torch:", torch.__version__)
print("  cuda compiled:", torch.version.cuda)
print("  accelerate:", accelerate.__version__)
print("  diffusers:", diffusers.__version__)
print("  transformers:", transformers.__version__)
print("  tokenizers:", tokenizers.__version__)
print("  huggingface_hub:", huggingface_hub.__version__)
print("  peft:", peft.__version__)
print("  timm:", timm.__version__)
print("  numpy:", numpy.__version__)
print("  pillow:", Image.__version__)

from transformers import Dinov2WithRegistersConfig  # noqa: F401
from transformers.models import Qwen2_5_VLProcessor, Qwen2_5_VLTextModel  # noqa: F401
from giga_world_0 import GigaWorld0Trainer  # noqa: F401

venv_accelerate = shutil.which("accelerate", path=os.path.join(os.environ["TRAIN_VENV"], "bin"))
assert venv_accelerate is not None, "accelerate console script is missing from training venv"
print("  GigaWorld training imports: ok")
PY

echo
echo "Training venv is ready: ${TRAIN_VENV}"
