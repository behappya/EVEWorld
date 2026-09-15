#!/bin/bash
# L0: 7.8s 长档基线候选池 —— laziness 主战场(34 号 §2.6)的基线数据, 不依赖任何 gate, 可立即上卡。
# 内容: 旧全参 EMA 基线(gr1 SFT)上生成 92 prompt x 8 seed x 125 帧(=7.8s@16fps, 与既有 f125 口径一致)。
# 预算: 单节点 8 卡整机 (8 seed 各占 1 卡并行, 每卡 92 条串行) ~2h 墙钟。
# 用法: bash l0_longhorizon_pool.sh            # 旧全参 EMA 臂 (默认)
#       ARM=pretrain bash l0_longhorizon_pool.sh   # 可选: pretrain 底座臂 (对照)
set -euo pipefail
REPO=giga-world-0
PRETRAIN=/data/datasets/gagi/giga_world_0_video_pretrain
EMA_CKPT=/data/datasets/gagi/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200/transformer_ema
ARM="${ARM:-gr1_sft_ema}"
OUT_BASE=/data/datasets/gagi/eve_v2_outputs/longpool_f125

if [[ "$ARM" == "gr1_sft_ema" ]]; then
  # bestofn 链路写死 MODEL_DIR/{transformer,text_encoder,vae} 结构, 用软链目录适配 EMA ckpt
  MODEL_DIR=/data/datasets/gagi/eve_v2_outputs/anchor_models/gr1_fullft_ema
  mkdir -p "$MODEL_DIR"
  [[ -e "$MODEL_DIR/transformer"   ]] || ln -s "$EMA_CKPT" "$MODEL_DIR/transformer"
  [[ -e "$MODEL_DIR/text_encoder" ]] || ln -s "$PRETRAIN/text_encoder" "$MODEL_DIR/text_encoder"
  [[ -e "$MODEL_DIR/vae"          ]] || ln -s "$PRETRAIN/vae" "$MODEL_DIR/vae"
  [[ -d "$EMA_CKPT" ]] || { echo "缺权威基线 EMA: $EMA_CKPT"; exit 1; }
elif [[ "$ARM" == "pretrain" ]]; then
  MODEL_DIR="$PRETRAIN"
else
  echo "未知 ARM=$ARM (可选 gr1_sft_ema | pretrain)"; exit 1
fi

OUT_ROOT="$OUT_BASE/$ARM"
mkdir -p "$OUT_ROOT"
echo "== 提交长档基线池: arm=$ARM  frames=125(7.8s)  out=$OUT_ROOT =="
# 注意: OUT_ROOT 不在 submit_gigaworld0_kjob.sh 的 maybe_export 白名单里,
# 必须作为 KEY=VALUE 直接追加给 payload ("$@" 通道), 否则落到默认老 bestofn 目录。
cd "$REPO"
export REPO_DIR="$REPO"
export RL_DIR="${RL_DIR:-/home/jovyan/new_rl/rl}"
export JOB_SCRIPT="$REPO/eveworld/method/scripts/kjob_bestofn_8gpu.sh"
export MODEL_DIR   # 白名单透传
bash "$REPO/scripts/submit_gigaworld0_kjob.sh" \
  "OUT_ROOT=$OUT_ROOT" \
  "SEEDS=6666 1234 2025 777 42 314 2718 999" \
  "NUM_FRAMES=125" \
  "SKIP_EXISTING=1" \
  "EAG_WEIGHT=0" \
  "LIMIT=0"

echo ""
echo "产物: $OUT_ROOT/seed{S}_f125/generated_only/*.mp4 (92 条/seed)"
echo "后续: qwen_28 判分 -> 长档 severity 分布 + 7.8s 档共识构对(与短档同规则)"
echo "注意: kjob 单节点 8 卡整机纪律; 若同时跑 T0, 用不同节点"
