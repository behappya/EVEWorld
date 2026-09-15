"""联合训练 config: L_id + 合成复制增广 (44 号阶段2, 92 条 GT, 200 步)。

runner=T4GJointTrainer (corr 的 L_id 机制 + aug 的投毒前向);
transform=T4GAugTransform (vid 解析 + 贴入计划; k_retry 已放宽 60/min_dur 3);
t4g_lambdas='0.5,0.0' —— L_id 照 corr 探针验证值, L_change 关闭只留诊断。
每 50 步存档 (4 档), 生成评测选档防背死 (92 条 x 200 步 ≈ 140 epoch)。
"""

ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
ASSETS_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/aug_assets'
ROUND0_EMA_TRANSFORMER = '/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer'

config = dict(
    runners=['eveworld.pipeline.t4g_joint_trainer.T4GJointTrainer'],
    project_dir='/data/datasets/gagi/eve_v2_outputs/t4g_joint/experiments',
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
        # ---- L_id (corr 探针验证值) ----
        t4g_id_block='block22',
        t4g_change_block='block25',
        t4g_sigma_lo=0.2,
        t4g_sigma_hi=0.5,
        t4g_warmup=20,
        t4g_lambdas='0.5,0.0',          # L_change 关闭 (E7 机制真但训练无罪可罚)
        t4g_tau=0.07,
        t4g_tol=0.2,
        t4g_anno_dir=ANNO_DIR,
        t4g_idx2vid=ANNO_DIR + '/_packidx2vid.json',
        # ---- 增广 (E11 验证值) ----
        t4g_w_paste=4.0,
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
        max_steps=200,
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
