# 消融:关闭 CPC(退化为普通全参微调 baseline)
from eveworld.method.configs.eve_causal_fullft import config
import copy
config = copy.deepcopy(config)
config['models']['causal']['w_cpc'] = 0.0
config['project_dir'] = config['project_dir'].replace('causal_fullft','ablation_no_cpc')
