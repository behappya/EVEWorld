#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/giga-models}"
TRAIN_VENV="${TRAIN_VENV:-/data/datasets/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

echo "Repo:        ${REPO_DIR}"
echo "GigaModels:  ${GIGA_MODELS_DIR}"
echo "Train venv:  ${TRAIN_VENV}"

"${REPO_DIR}/scripts/setup_gigaworld_train_venv.sh"

export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

echo
echo "PhysLatent import checks:"
"${TRAIN_PYTHON}" - <<'PY'
import sys

import diffusers
import torch
import transformers
from transformers import Dinov2WithRegistersConfig  # noqa: F401
from transformers.models import Qwen2_5_VLProcessor, Qwen2_5_VLTextModel  # noqa: F401

from physlatent_gigaworld import PhysicsLatentEncoder, PhysicsLatentGigaWorld0Trainer

print("  python:", sys.executable)
print("  torch:", torch.__version__)
print("  diffusers:", diffusers.__version__)
print("  transformers:", transformers.__version__)
print("  encoder:", PhysicsLatentEncoder.__name__)
print("  trainer:", PhysicsLatentGigaWorld0Trainer.__name__)
print("  PhysLatent training imports: ok")
PY

echo
echo "PhysLatent training env is ready."
