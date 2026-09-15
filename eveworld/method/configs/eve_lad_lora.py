# EVE · LAD-LoRA 训练级反偷懒配置(方案27 §二十 正路)。
# 从 physlatent adapter 派生, 改:
#   - runner -> EveLadLoraTrainer
#   - train_mode='lora'(冻结 backbone, 只训 LoRA, target to_{q,k,v,out}.0, rank 64)
#   - physics_latent.enabled=False(本方法不用 physics token, 只用冻结 LAD 正则)
#   - 新增 models.lad_lora: 冻结 LAD ckpt + 正则权重
# 主损失仍是 EDM 去噪(保画质); lad_reg 是 sigma 门控的 soft-top-k transition_error 正则。
import copy
import os
from pathlib import Path
from eveworld.alternatives.physlatent.configs.gr1_physlatent_adapter import config as _base

config = copy.deepcopy(_base)
config['runners'] = ['eveworld.EveLadLoraTrainer']
config['launch']['deepspeed_config']['deepspeed_config_file'] = str(
    Path(__file__).with_name('deepspeed_zero2_clip.json').resolve()
)
# project_dir 决定 checkpoint 落盘路径(trainer.py: cls(project_dir=config.project_dir,...))。
# 串行消融 w=0.1/0.0/0.2 若共用同一 project_dir 会互相覆盖 ckpt -> 用 W_LAD+SEED 隔离到独立目录。
# (W_LAD/SEED 与 w_lad 同源, 均由 launch 尾随位置参数透传到 kjob 内 export, config 执行时可读。)
_w_lad_dir = os.environ.get('W_LAD', '0.1')
_seed_dir = os.environ.get('SEED', '6666')
config['project_dir'] = (
    f'/data/datasets/gagi/giga_world_0_outputs/eve/'
    f'experiments_lad_lora_w{_w_lad_dir}_seed{_seed_dir}'
)

# LoRA 微调(冻结 backbone)
config['models']['train_mode'] = 'lora'
config['models']['lora_rank'] = 64
config['optimizers']['lr'] = float(os.environ.get('LAD_LORA_LR', str(config['optimizers']['lr'])))

# 关闭 physics latent 条件路径(本方法只需冻结 LAD 作正则, 不引入 physics token)
config['models']['physics_latent']['enabled'] = False
config['models']['physics_latent']['train_backbone'] = False

# LAD-LoRA 反偷懒正则(EveLadLoraTrainer.get_models 从 model_config['lad_lora'] 读)
# w_lad 从环境变量读, 便于消融(W_LAD=0.0 = 纯 LoRA control, 排除"只是多训了 LoRA 参数")
config['models']['lad_lora'] = dict(
    w_lad=float(os.environ.get('W_LAD', '0.1')),  # 正则权重; 扫 0.05/0.1/0.2 看画质-忠实帕累托; 0=control
    topk=int(os.environ.get('LAD_TOPK', '3')),    # soft-top-k 聚合(盯前3大非法跳变尖峰, 与 MAX 聚合发现一致)
    tau=float(os.environ.get('LAD_TAU', '0.5')),  # soft-max 温度(越小越接近硬 max)
    sigma_max=float(os.environ.get('LAD_SIGMA_MAX', '0.0')),  # >0 硬门控; 默认软门控 1/(1+sigma)
    # w_lad 表示 LAD/EDM 在共享 x0 上的目标梯度比；此上限只防 LAD 梯度接近0时放大过头。
    balance_max_scale=float(os.environ.get('LAD_BALANCE_MAX_SCALE', '1.0')),
    lam_ckpt=os.environ.get('LAM_CKPT', '/data/datasets/gagi/eve_outputs/lam/lam_gr1.pt'),
)

# 诊断默认 50 步, 每 5 步存点。launcher/runtime writer 可覆盖, 但这里也保持安全默认。
config['train']['max_steps'] = 50
config['train']['checkpoint_interval'] = 5
config['train']['checkpoint_total_limit'] = -1
config['train']['with_ema'] = False
# DeepSpeed 下 max_grad_norm 仅作为期望值/启动断言；真实裁剪由上面的专用 DS JSON 执行。
config['train']['max_grad_norm'] = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
# 默认禁止恢复，避免不同权重或旧发散状态被静默续训。
config['train']['resume'] = os.environ.get('RESUME', '0') == '1'
