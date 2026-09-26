#!/usr/bin/env bash
set -euo pipefail

# Fix the GigaWorld-0 conda env for clusters whose NVIDIA driver supports
# CUDA 12.8 but not CUDA 13.x. The current PyPI default can install torch
# cu130, which fails on those nodes.

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
NATTEN_FIND_LINKS="${NATTEN_FIND_LINKS:-https://whl.natten.org}"

TORCH_VERSION="${TORCH_VERSION:-2.11.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0+cu128}"
NATTEN_VERSION="${NATTEN_VERSION:-0.21.6+torch2110cu128}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

echo "Using python: $(command -v python)"
python --version

echo
echo "Installing CUDA 12.8-compatible PyTorch stack:"
echo "  torch==${TORCH_VERSION}"
echo "  torchvision==${TORCHVISION_VERSION}"
echo "  natten==${NATTEN_VERSION}"
echo

python -m pip install --upgrade --force-reinstall \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

python -m pip install --upgrade --force-reinstall --no-deps \
  -f "${NATTEN_FIND_LINKS}" \
  "natten==${NATTEN_VERSION}"

python - <<'PY'
import torch
import torchvision
import natten

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("natten:", getattr(natten, "__version__", "unknown"))
print("cuda available:", torch.cuda.is_available())
PY

