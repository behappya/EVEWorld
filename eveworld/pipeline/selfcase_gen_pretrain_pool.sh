#!/bin/bash
# SELF-CASE · pretrain 底座 rollout 池生成: 92 prompt x 4 seed x 93f。
# 与 pool_round0_f93 同协议(kjob_bestofn_8gpu + generate_eag, EAG_WEIGHT=0, 480x768, steps=30),
# 仅换 MODEL_DIR 为 GigaWorld-0 video pretrain 底座(transformer/vae/text_encoder 同目录)。
# 输出: selfcase/pool_pretrain_f93/seed{S}_f93/generated_only/{idx}_{slug}.mp4
set -eu
REPO=giga-world-0
PRETRAIN=/data/datasets/gagi/giga_world_0_video_pretrain
OUT_ROOT=/data/datasets/gagi/eve_v2_outputs/selfcase/pool_pretrain_f93

[[ -d "$PRETRAIN/transformer" ]] || { echo "缺 pretrain transformer: $PRETRAIN/transformer"; exit 1; }
[[ -d "$PRETRAIN/vae" ]] || { echo "缺 pretrain vae: $PRETRAIN/vae"; exit 1; }
mkdir -p "$OUT_ROOT"

echo "== 提交 pretrain 池生成: 92 prompt x 4 seed x 93f -> $OUT_ROOT =="
cd "$REPO"
export REPO_DIR="$REPO"
export RL_DIR="${RL_DIR:-/home/jovyan/new_rl/rl}"
export JOB_SCRIPT="$REPO/eveworld/method/scripts/kjob_bestofn_8gpu.sh"
export MODEL_DIR="$PRETRAIN"   # 白名单透传
bash "$REPO/scripts/submit_gigaworld0_kjob.sh" \
  "OUT_ROOT=$OUT_ROOT" \
  "SEEDS=42 314 777 999" \
  "NUM_FRAMES=93" \
  "SKIP_EXISTING=1" \
  "EAG_WEIGHT=0" \
  "LIMIT=0"

echo ""
echo "产物: $OUT_ROOT/seed{S}_f93/generated_only/*.mp4 (92 条/seed, 共 368)"
