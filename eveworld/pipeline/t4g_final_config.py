"""终局合体臂 config (72 号, 用户修正版) —— pretrain 底座, 400 步。

组件: GT 干净 SFT (T4GCorrTransform, 3/4 锚) + L_id (block22, λ='0.5,0.0' 关
L_change) + ICH-D 固化头 (block22 注入) + A2 在线滚雪球 (1/4, K 由 kprobe 定,
env T4G_A2_K 最终事实)。runner = T4GFinalTrainer。
launcher 尾随 KEY=VALUE 是最终事实来源。
"""

ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
IDX2VID = ANNO_DIR + '/_packidx2vid.json'
PRETRAIN_TRANSFORMER = '/data/datasets/gagi/giga_world_0_video_pretrain/transformer'
ICH_D_WEIGHTS = '/data/datasets/gagi/eve_v2_outputs/selfcase/cic_match/ich_d_frozen_lr.npz'

config = dict(
    runners=['eveworld.pipeline.t4g_final_trainer.T4GFinalTrainer'],
    project_dir='/data/datasets/gagi/eve_v2_outputs/t4g_final/experiments',
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
                type='T4GCorrTransform',
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
                idx2vid_path=IDX2VID,
                anno_dir=ANNO_DIR,
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
        ),
    ),
    models=dict(
        vae_model_path='/data/datasets/gagi/giga_world_0_video_pretrain/vae',
        transformer_model_path=PRETRAIN_TRANSFORMER,
        # ---- L_id (t4g_corr 组件; L_change 关闭) ----
        t4g_lambdas='0.5,0.0',
        t4g_id_block='block22',
        t4g_sigma_lo=0.2,
        t4g_sigma_hi=0.5,
        t4g_warmup=50,
        # ---- ICH-D 固化头 ----
        t4g_ich_d_weights=ICH_D_WEIGHTS,
        t4g_ich_win=3,
        # v2: 无擦除子/无结构改动; 裁判仅训练侧 (t4g_ich_d_weights)
        # ---- A2 在线滚雪球 (K 以 kprobe 判决 + env T4G_A2_K 为最终事实) ----
        t4g_a2_p=0.25,
        t4g_a2_k=4,
        t4g_a2_sigma_hi=3.0,
        t4g_a2_sigma_lo=0.3,
        t4g_lambda_a2=1.0,
        t4g_a2_warmup=50,
        t4g_a2_m_thr=0.7,           # §5 学习态: 指认过筛 M>0.7
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
        max_steps=400,
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
