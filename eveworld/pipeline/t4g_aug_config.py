"""合成复制增广训练 config (44 号阶段1', E10 机制修复)。

底座 = Round-0 EMA; 数据 = 92 条 GT packed; 50 步探针先行。
transform=T4GAugTransform (投毒计划), runner=T4GAugTrainer (双编码+干净目标+上权重)。
launcher 尾随 KEY=VALUE 是最终事实来源 (同 t4g_corr_config 的约定)。
"""

ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
ASSETS_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets'
ROUND0_EMA_TRANSFORMER = '/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer'

config = dict(
    runners=['eveworld.pipeline.t4g_aug_trainer.T4GAugTrainer'],
    project_dir='/data/datasets/gagi/eve_v2_outputs/t4g_aug/experiments',
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
                type='T4GAugTransform',
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
                p_aug=0.5,
                zone_bias='0.4,0.2,0.4',
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
        t4g_w_paste=4.0,               # 贴入区 loss 上权重 (防小区域信号稀释; env T4G_W_PASTE 可覆盖)
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
        checkpoint_interval=50,
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
