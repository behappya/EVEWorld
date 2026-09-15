#!/bin/bash
# P0-probe: 通用"臂探针"—— 给任意 checkpoint 快速生成+判分, 与 Round-0 基线对比。
# 探针模式(默认): LIMIT=12 (前 12 条 prompt x 8 seed, 每卡 12 条, 生成约 10min) — 方向性检查
# 全量模式: LIMIT=0 (92 x 8) — Gate-3 正式验收用
# 用法: CKPT=/path/to/checkpoint_dir/transformer TAG=raft_s40 [LIMIT=12] bash p0_probe_arm.sh
set -euo pipefail
REPO=giga-world-0
PRETRAIN=/data/datasets/gagi/giga_world_0_video_pretrain
CKPT="${CKPT:?需要 CKPT=transformer 目录路径}"
TAG="${TAG:?需要 TAG=标签}"
LIMIT="${LIMIT:-12}"
MODEL_DIR=/data/datasets/gagi/eve_v2_outputs/anchor_models/probe_${TAG}
OUT_ROOT=/data/datasets/gagi/eve_v2_outputs/probe/${TAG}

[[ -d "$CKPT" ]] || { echo "缺 ckpt: $CKPT"; exit 1; }
mkdir -p "$MODEL_DIR" "$OUT_ROOT"
ln -sfn "$CKPT" "$MODEL_DIR/transformer"
ln -sfn "$PRETRAIN/text_encoder" "$MODEL_DIR/text_encoder"
ln -sfn "$PRETRAIN/vae" "$MODEL_DIR/vae"

echo "== 探针提交: $TAG (LIMIT=$LIMIT) =="
cd "$REPO"
export REPO_DIR="$REPO"
export RL_DIR="${RL_DIR:-/home/jovyan/new_rl/rl}"
export JOB_SCRIPT="$REPO/eveworld/method/scripts/kjob_bestofn_8gpu.sh"
export MODEL_DIR
bash "$REPO/scripts/submit_gigaworld0_kjob.sh" \
  "OUT_ROOT=$OUT_ROOT" \
  "SEEDS=6666 1234 2025 777 42 314 2718 999" \
  "NUM_FRAMES=93" "SKIP_EXISTING=1" "EAG_WEIGHT=0" "LIMIT=$LIMIT"
echo "生成完成后判分+对比: bash p0_probe_score.sh $TAG"
