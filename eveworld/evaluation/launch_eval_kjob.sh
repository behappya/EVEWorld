#!/usr/bin/env bash
# EVE · 跨模型生成 + 过程度量(集群 kjob)。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${HERE}/../../common/env.sh"
eve_log "跨模型评测编排"
eve_warn "# TODO(cluster): 对每个模型(pretrain/sft/eve/公开模型)在 held-out eval 上生成视频"
cat <<TIP
  流程: 1) 各模型生成 -> \${EVE_OUT}/eval/videos/<model>/
        2) 每个跑 P0 事件抽取 + 过程度量(复用 p0_diagnosis 脚本,纯 CPU 可在登录节点跑)
        3) 汇总进 \${EVE_OUT}/eval/metrics/<model>.json
        4) run_stats.sh 做统计与画图
  公开模型: 能拿到输出的 I2V/世界模型都纳入,证明 laziness 领域共性(§四)
TIP
