#!/usr/bin/env bash
# 下载 Wan2.2-I2V-A14B (阿里, MoE 总~27B/激活14B, 原生 I2V 旗舰)。
# 免 gated。体积较大(~55-65GB)。高端 Wan 旗舰，用于证明先进性。
# 可与其它 dl_*.sh 在不同终端并行执行。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

REPO_ID="${REPO_ID:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
DST="${DST:-${XMODEL_ROOT}/wan22_i2v_a14b}"

activate_env
check_login_if_gated 0

echo "==== [Wan2.2-I2V-A14B] 开始下载 (体积大，耐心等) ===="
dl_repo "${REPO_ID}" "${DST}" \
  --exclude "*.pth" \
  --exclude "*optim*" \
  --exclude "*.onnx"

verify_paths "${DST}" \
  "model_index.json" \
  "transformer" \
  "vae"
