#!/bin/bash
# SELF-CASE · A1 训练终档 rollout 池生成: 92 prompt x 4 seed x 93f。
# 协议照抄 pool_pretrain_f93 (kjob_bestofn_8gpu + generate_eag, EAG_WEIGHT=0, 480x768, steps=30, fps=16),
# 仅换 transformer 为 A1 终档 EMA:
#   /data/datasets/gagi/eve_v2_outputs/t4g_selfcase/experiments/models/checkpoint_epoch_125_step_250/transformer_ema
# vae/text_encoder 沿用 pretrain 底座。三者经符号链接目录 a1_s250_infer_modeldir 合成
# (kjob_bestofn_8gpu.sh 只认 MODEL_DIR/{transformer,text_encoder,vae} 布局)。
# 输出: selfcase/pool_a1_s250_f93/seed{S}_f93/generated_only/{idx}_{slug}.mp4
set -eu
REPO=giga-world-0
MODEL_DIR=/data/datasets/gagi/eve_v2_outputs/t4g_selfcase/experiments/models/a1_s250_infer_modeldir
OUT_ROOT=/data/datasets/gagi/eve_v2_outputs/selfcase/pool_a1_s250_f93

[[ -f "$MODEL_DIR/transformer/config.json" ]] || { echo "缺 A1 transformer(EMA): $MODEL_DIR/transformer"; exit 1; }
[[ -f "$MODEL_DIR/vae/config.json" ]] || { echo "缺 vae: $MODEL_DIR/vae"; exit 1; }
[[ -d "$MODEL_DIR/text_encoder" ]] || { echo "缺 text_encoder: $MODEL_DIR/text_encoder"; exit 1; }
mkdir -p "$OUT_ROOT"

echo "== 提交 A1(s250 EMA) 池生成: 92 prompt x 4 seed x 93f -> $OUT_ROOT =="
# 直接 kjobctl 提交(等价于 submit_gigaworld0_kjob.sh 路径, 显式 --profile slurm-profile);
# KEY=VALUE 作尾随位置参数透传(kjob_bestofn_8gpu.sh 有 for-arg export)。
kjobctl create slurm --profile slurm-profile \
  --pod-template-label vllm-metrics=true \
  -- \
  "$REPO/eveworld/method/scripts/kjob_bestofn_8gpu.sh" \
  "REPO_DIR=$REPO" \
  "MODEL_DIR=$MODEL_DIR" \
  "OUT_ROOT=$OUT_ROOT" \
  "SEEDS=42 314 777 999" \
  "NUM_FRAMES=93" \
  "SKIP_EXISTING=1" \
  "EAG_WEIGHT=0" \
  "LIMIT=0"

echo ""
echo "产物: $OUT_ROOT/seed{S}_f93/generated_only/*.mp4 (92 条/seed, 共 368)"
