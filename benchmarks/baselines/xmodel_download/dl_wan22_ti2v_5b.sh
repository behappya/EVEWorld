#!/usr/bin/env bash
# 下载 Wan2.2-TI2V-5B (阿里, dense 5B, 原生 文+图→视频)。
# 免 gated，体积最小(~10-12GB)，建议第一个下、第一个跑通。
# 可与其它 dl_*.sh 在不同终端并行执行。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

REPO_ID="${REPO_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
DST="${DST:-${XMODEL_ROOT}/wan22_ti2v_5b}"

activate_env
check_login_if_gated 0

echo "==== [Wan2.2-TI2V-5B] 开始下载 ===="
# 只要 diffusers 权重，排除原始 .pth 训练态/多余大文件（若有）。
dl_repo "${REPO_ID}" "${DST}" \
  --exclude "*.pth" \
  --exclude "*optim*" \
  --exclude "*.onnx"

# diffusers 版结构：model_index.json + transformer/ + vae/ + text_encoder/
verify_paths "${DST}" \
  "model_index.json" \
  "transformer" \
  "vae"
