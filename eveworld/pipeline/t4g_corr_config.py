"""Track4Gen 式对应监督训练 config (42 号 §3)。

底座 = Round-0 EMA (safetensors); 数据 = 92 条 GT packed (packed_data)。
全参短跑, batch 64 = 8 卡 × 1/卡 × GA8。50 步探针先行 (42 号 §4 gate)。

注意: scripts/write_train_runtime_config.py 会用 kjob 尾随/环境变量覆盖本文件的
transformer_model_path / vae_model_path / data_or_config / max_steps / batch /
checkpoint_interval / checkpoint_total_limit / num_workers / transform 尺寸 等字段;
t4g_corr_launch.sh 的尾随 KEY=VALUE 才是最终事实来源。本文件默认值与其保持一致,
保证「直接 import 本 config」与「经 launcher」两条路径语义相同。

不被 writer 覆盖、由本文件决定的关键项:
  - runners (T4GCorrTrainer)
  - transform.type=T4GCorrTransform + idx2vid_path/anno_dir (注入 vid 供取 anno)
  - models.t4g_* (时间档/λ/margin/anno 路径; get_models 里可再被 env 覆盖)
  - train.checkpoint_start_step (自定义键, 经 **kwargs 透传; 缺省 0 = 存全部档)
  - checkpoint_total_limit=8 (> 探针档数, 杜绝轮转删档)
"""

ANNO_DIR = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/t4g_anno'
IDX2VID = ANNO_DIR + '/_packidx2vid.json'
ROUND0_EMA_TRANSFORMER = '/data/datasets/gagi/eve_v2_outputs/anchor_models/round0_ema_st/transformer'

config = dict(
    runners=['eveworld.pipeline.t4g_corr_trainer.T4GCorrTrainer'],
    project_dir='/data/datasets/gagi/eve_v2_outputs/t4g_corr/experiments',
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
        transformer_model_path=ROUND0_EMA_TRANSFORMER,
        # ---- t4g 对应 loss 超参 (43 号实证版; env 可再覆盖) ----
        t4g_id_block='block22',      # L_id 甜点层 (E4 EPE0.55)
        t4g_change_block='block25',  # L_change 甜点层 (E7 gap+0.45)
        t4g_sigma_lo=0.2,
        t4g_sigma_hi=0.5,
        t4g_warmup=20,
        t4g_lambdas='0.5,0.4',       # (id, change) 两道
        t4g_tau=0.07,
        t4g_tol=0.2,                 # 静止-变化容忍带 (E7 静止地板0.80)
        t4g_anno_dir=ANNO_DIR,
        t4g_idx2vid=IDX2VID,
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
        max_steps=50,                 # 探针档 (launcher MAX_STEPS 覆盖)
        gradient_accumulation_steps=8,
        mixed_precision='bf16',
        checkpoint_interval=50,
        checkpoint_total_limit=8,
        checkpoint_start_step=0,      # 自定义键: step<该值不存档 (探针=0 存全部)
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['TransformerBlock'],
    ),
)
