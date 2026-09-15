#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${SOURCE_REPO:-/home/jovyan/gagibench/WorldArena}"
WORLDARENA_ROOT="${WORLDARENA_ROOT:-/home/jovyan/gagibench/WorldArena_WA1_EVAL}"
WORLDARENA_COMMIT="${WORLDARENA_COMMIT:-2da2ae253b8637ba9de3afc7bea4e087f778ee4d}"
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/giga-world-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${PATCH_FILE:-${SCRIPT_DIR}/worldarena_eval.patch}"
CONDA_SH="${CONDA_SH:-/home/jovyan/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-WorldArena}"

if [[ ! -e "${WORLDARENA_ROOT}/.git" ]]; then
  git -C "${SOURCE_REPO}" worktree add --detach "${WORLDARENA_ROOT}" "${WORLDARENA_COMMIT}"
fi

actual_commit=$(git -C "${WORLDARENA_ROOT}" rev-parse HEAD)
if [[ "${actual_commit}" != "${WORLDARENA_COMMIT}" ]]; then
  echo "Unexpected WorldArena commit: ${actual_commit}" >&2
  exit 1
fi

if git -C "${WORLDARENA_ROOT}" apply --check "${PATCH_FILE}" 2>/dev/null; then
  git -C "${WORLDARENA_ROOT}" apply "${PATCH_FILE}"
elif git -C "${WORLDARENA_ROOT}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  echo "WorldArena evaluator patch already applied"
else
  echo "WorldArena WA1 evaluator worktree has incompatible changes" >&2
  exit 1
fi

printf '%s  %s\n' \
  cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102 \
  /data/datasets/gagi/worldarena_evaluator/checkpoints/sea_raft/model.safetensors | sha256sum --check
printf '%s  %s\n' \
  c1dac5b08f4c41e95452f7a41b35347409fc58b0e25b3ba8de899144ff28a350 \
  /data/datasets/gagi/worldarena_evaluator/checkpoints/vfimamba/model.pkl | sha256sum --check
printf '%s  %s\n' \
  2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba \
  /data/datasets/gagi/worldarena_evaluator/checkpoints/aesthetic/sa_0_4_vit_l_14_linear.pth | sha256sum --check

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
WORLDARENA_ROOT="${WORLDARENA_ROOT}" python - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["WORLDARENA_ROOT"])
sys.path.insert(0, str(root / "video_quality"))
import WorldArena

if WorldArena.compute_motion_smoothness is None:
    raise SystemExit("motion_smoothness is unavailable")
print(f"WorldArena import OK: {WorldArena.__file__}")
PY

echo "WA1 evaluator ready: ${WORLDARENA_ROOT}"
