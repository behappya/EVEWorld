#!/usr/bin/env bash
# EVE 数据策展。部分步骤需网络(HF 下载)。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../common/env.sh"
DC="${HERE}/.."
OUT="${EVE_OUT}/data"; mkdir -p "${OUT}"

eve_log "数据 1/4:held-out 切分(把 92 条切成 train/eval,消除重叠)"
${PYBIN} "${DC}/scripts/make_holdout.py" \
  --meta "${GR1_DATA_ROOT}/raw_hf/metadata.csv" \
  --out-dir "${OUT}/splits" --eval-frac 0.3 --seed 0

eve_log "数据 2/4:补齐 DreamGen Bench 官方 126 split(需网络/HF token)"
eve_warn "  # TODO(net): 确认 DreamGen 官方 split 清单来源后填入下载命令"
echo "  参考: https://arxiv.org/abs/2505.12705 (DreamGen), HF: nvidia/PhysicalAI-Robotics-GR00T-GR1"
echo "  期望产物: ${OUT}/dreamgen126/{env29,object50,behavior47}/ 清单"

eve_log "数据 3/4:拉无标注操作视频子集供 LAM 预训(OXE / BridgeData)"
eve_warn "  # TODO(net): 选定 OXE 子集并填 tfds/HF 下载。目标: ${OUT}/lam_pretrain_videos/"
echo "  建议: Open X-Embodiment 的若干 manipulation 子集(自监督,不需要动作标注)"

eve_log "数据 4/4:跨本体/过程性评测集(Something-Something V2 等,可选)"
eve_warn "  # TODO(net): SthSthV2 需申请下载;先占位 ${OUT}/crossdomain/"
eve_log "数据策展骨架完成。已实跑: held-out 切分。其余为需网络的 TODO 填空点。"
