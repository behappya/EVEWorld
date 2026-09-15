"""ICH 训练 config · 臂 B (三层 block10+12+16, 12 维检测特征) —— 71 号 B1 G3 50 步探针。

底座/数据/transform 全抄 t4g_selfcase_config (70 号 A1), 仅三处差异:
  1. runner = T4GICHTrainer (A1 修复损失 + λ_det×BCE + block10+12+16 融合检测, block16 抑制子);
  2. case_dir = case_bank_all (四池汇总 34 例, 先跑 t4g_ich_case_merge.py);
  3. 预算 = 50 步, 每 25 步存档 (G3 探针, 对照臂 A 单层)。
launcher 尾随 KEY=VALUE 是最终事实来源。
"""

ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
ASSETS_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets'
CASE_DIR = '/data/datasets/gagi/eve_v2_outputs/selfcase/case_bank_all'
ROUND0_EMA_TRANSFORMER = '/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer'

config = dict(
    runners=['eveworld.pipeline.t4g_ich_trainer.T4GICHTrainer'],
    project_dir='/data/datasets/gagi/eve_v2_outputs/t4g_ich_3L/experiments',
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                '/data/datasets/gagi/gr1_finetune_data/packed_data',
            ],
            batch_size_per_gpu=1,
            num_workers=6,
            transform=dict(
                type='T4GSelfCaseTransform',
                num_frames=93,
                height=480,
                width=768,
                fps=16,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
                idx2vid_path=ANNO_DIR + '/_packidx2vid.json',
                assets_dir=ASSETS_DIR,
                case_dir=CASE_DIR,
                p_case=0.8,
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
        ),
    ),
    models=dict(
        vae_model_path='/data/datasets/gagi/giga_world_0_video_pretrain/vae',
        transformer_model_path=ROUND0_EMA_TRANSFORMER,
        t4g_w_paste=4.0,               # 病例区 loss 上权重 (env T4G_W_PASTE 可覆盖)
        t4g_ich_blocks='block10,block12,block16',  # 臂 B: 三层融合 (env T4G_ICH_BLOCKS 可覆盖)
        t4g_ich_hidden=96,
        t4g_ich_win=3,
        t4g_lambda_det=1.0,
        t4g_det_warmup=50,             # 50 步内线性 0->1
        t4g_det_pos_weight=1.0,
    ),
    optimizers=dict(
        type='CAME8Bit',
        lr=2 ** (-14.5),
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=True,
        max_steps=50,
        gradient_accumulation_steps=8,
        mixed_precision='bf16',
        checkpoint_interval=25,
        checkpoint_total_limit=8,
        checkpoint_start_step=0,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['TransformerBlock'],
    ),
)
