#!/usr/bin/env bash
set -euo pipefail

# Restore dependency versions needed by GigaWorld-0 training/inference after
# VBench-style evaluation dependencies have downgraded transformers/peft/etc.
#
# This does not install torch/torchvision/natten and does not download model
# weights. Run it only when you are about to run GigaWorld training/inference;
# VBench may require its own older dependency set.

CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

echo "Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Before:"
"${PYTHON_BIN}" - <<'PY' || true
mods = ["transformers", "tokenizers", "huggingface_hub", "peft", "timm", "numpy", "PIL"]
for mod in mods:
    try:
        m = __import__(mod)
        version = getattr(m, "__version__", getattr(m, "PILLOW_VERSION", "unknown"))
        print(f"  {mod}: {version}")
    except Exception as exc:
        print(f"  {mod}: import failed: {exc}")
PY

"${PYTHON_BIN}" -m pip install -U \
  "transformers==5.11.0" \
  "tokenizers==0.22.2" \
  "huggingface_hub==1.18.0" \
  "peft==0.19.1" \
  "timm==1.0.27" \
  "numpy==2.4.4" \
  "Pillow==11.3.0"

echo
echo "After / import checks:"
"${PYTHON_BIN}" - <<'PY'
import transformers
import tokenizers
import huggingface_hub
import peft
import timm
import numpy
from PIL import Image

print("  transformers:", transformers.__version__)
print("  tokenizers:", tokenizers.__version__)
print("  huggingface_hub:", huggingface_hub.__version__)
print("  peft:", peft.__version__)
print("  timm:", timm.__version__)
print("  numpy:", numpy.__version__)
print("  pillow:", Image.__version__)

from transformers import Dinov2WithRegistersConfig
from transformers.models import Qwen2_5_VLProcessor, Qwen2_5_VLTextModel
from giga_world_0 import GigaWorld0Trainer

print("  GigaWorld training imports: ok")
PY
