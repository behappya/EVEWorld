#!/usr/bin/env bash
# 下载 CogVideoX1.5-5B-I2V (智谱, DiT 5B, 原生 I2V)。
# GATED: 需先在网页同意 CogVideoX License 并 hf auth login（或传 HF_TOKEN）。
# 体积 ~10-11GB。可与其它 dl_*.sh 在不同终端并行执行。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

REPO_ID="${REPO_ID:-zai-org/CogVideoX1.5-5B-I2V}"
DST="${DST:-${XMODEL_ROOT}/cogvideox15_5b_i2v}"

activate_env
check_login_if_gated 1

echo "==== [CogVideoX1.5-5B-I2V] 开始下载 (gated) ===="
dl_repo "${REPO_ID}" "${DST}" \
  --exclude "*.onnx"

verify_paths "${DST}" \
  "model_index.json" \
  "transformer" \
  "vae"
