# 消融:用随机置换负样本替代因果负样本(证明是因果信号在起作用)
from eveworld.method.configs.eve_causal_fullft import config
import copy
config = copy.deepcopy(config)
config['models']['causal']['use_random_neg'] = True
config['project_dir'] = config['project_dir'].replace('causal_fullft','ablation_random_neg')
