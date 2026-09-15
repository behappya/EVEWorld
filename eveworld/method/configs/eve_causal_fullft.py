# EVE 因果过程忠实 · 全参数微调配置。
# 从 physlatent adapter config 派生(复用其合法 schema),仅改:
#   - runner -> EveCausalTrainer
#   - physics_latent.train_backbone=True(全参微调,原型失败主因之一)
#   - physics aux loss 全 0(不用弱标签辅助头)
#   - 新增 models.causal(EVE 因果损失权重,EveCausalTrainer.get_models 读取)
import copy
from eveworld.alternatives.physlatent.configs.gr1_physlatent_adapter import config as _base

config = copy.deepcopy(_base)
config['runners'] = ['eveworld.EveCausalTrainer']
config['project_dir'] = '/data/datasets/gagi/giga_world_0_outputs/eve/experiments_causal_fullft'

# 全参微调 + 关闭 physics 辅助头
pl = config['models']['physics_latent']
pl['train_backbone'] = True
pl['loss_weights'] = dict(state=0.0, goal=0.0, contact=0.0, trajectory=0.0)

# EVE 因果损失(EveCausalTrainer 从 model_config['causal'] 读)
config['models']['causal'] = dict(
    w_cpc=0.5,                       # 反事实过程对比(C)
    neg_kinds=['teleport', 'excision', 'freeze_jump'],
    margin=0.1,
    use_random_neg=False,            # 消融时置 True
    # w_dyn/w_prog 暂留 0:先验证 CPC 主干跑通,LAM/进度门控在 P0 通过后再开
    w_dyn=0.0,
    w_prog=0.0,
    lam_ckpt='/data/datasets/gagi/eve_outputs/lam/lam_pretrained.pt',
)

# 训练步数等沿用 base;如需覆盖用 kjob 的 MAX_STEPS 环境变量
config['train']['max_steps'] = 2000
config['train']['checkpoint_interval'] = 200
