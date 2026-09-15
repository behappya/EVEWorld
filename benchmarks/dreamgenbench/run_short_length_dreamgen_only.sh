#!/usr/bin/env bash
set -euo pipefail

# 只跑 DreamGen 两条线（Pretrain + SFT）的短时长档纯推理，各补 3.8s / 7.8s。
# 单节点 8 卡；串行；断点续跑。等价于：
#   TASKS="dreamgen_pretrain dreamgen_sft" ./run_short_length_generation_only.sh
#
# 顺序（4 个作业，依次跑完一个再跑下一个）：
#   1. DreamGen Pretrain 3.8s (61帧)
#   2. DreamGen Pretrain 7.8s (125帧)
#   3. DreamGen SFT      3.8s (61帧)
#   4. DreamGen SFT      7.8s (125帧)
#
# DRY_RUN=1 可先空跑确认。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="dreamgen_pretrain dreamgen_sft" exec "${SCRIPT_DIR}/run_short_length_generation_only.sh" "$@"
