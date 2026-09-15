#!/usr/bin/env bash
# EVE 共享环境变量。所有脚本 `source` 本文件。
# ==== 动手前请按你的真实路径核对/修改以下 3 处 ====

# 1) 集群侧数据/输出根(kjob 范式沿用现有约定)
export GAGI_ROOT="${GAGI_ROOT:-/data/datasets/gagi}"
export GW0_MODEL_DIR="${GW0_MODEL_DIR:-${GAGI_ROOT}/giga_world_0_video_pretrain}"
export GR1_DATA_ROOT="${GR1_DATA_ROOT:-${GAGI_ROOT}/gr1_finetune_data}"

# 2) 已生成的评测视频目录(P0 度量的输入)。
#    指向你 baseline / SFT / adapter 生成的 92 条 mp4 所在目录。
#    P0 会遍历其下的 *.mp4。若你的产物在别处,改这里或用 --video-dir 覆盖。
export EVE_VIDEO_ROOT="${EVE_VIDEO_ROOT:-${GAGI_ROOT}/giga_world_0_outputs}"

# 3) EVE 自己的输出根(大产出统一放 /data/datasets/gagi 下)
export EVE_OUT="${EVE_OUT:-${GAGI_ROOT}/eve_outputs}"
# 训练/生成的大输出(checkpoint/视频)约定目录:
#   ${GAGI_ROOT}/giga_world_0_outputs/eve/...   (沿用现有 outputs 树)

# ---- 以下一般不用改 ----
export EVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EVE_P0="${EVE_DIR}/diagnosis"
export PYBIN="${PYBIN:-python3}"

mkdir -p "${EVE_OUT}" 2>/dev/null || true

eve_log() { echo -e "\033[1;36m[EVE]\033[0m $*"; }
eve_warn() { echo -e "\033[1;33m[EVE WARN]\033[0m $*"; }
eve_err() { echo -e "\033[1;31m[EVE ERR]\033[0m $*" >&2; }
