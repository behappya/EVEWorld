"""AgiBot wmap-only 最小验证档 (方案 Phase 7 fail-fast gate)。

= agi_joint_config 去掉 Copy-Paste (p_aug=0) 与 L_id (lambdas='0,0'), 只留
motionauto 双臂 weightmap。s150 后生成+评 1 臂, gate: NDTW/scene >= vanilla s250
(0.20/0.825)。绿灯才加组件; 红灯说明仍过标 (降 coverage cap / 倍数再来)。
"""

from eveworld.agibot.agi_joint_config import config as _base

import copy

config = copy.deepcopy(_base)
config['project_dir'] = ('/data/datasets/gagi/eve_v2_outputs/'
                         'agibot_ewm_apre/experiments')
_t = config['dataloaders']['train']['transform']
_t['p_aug'] = 0.0
_t['p_aug_state'] = 0.0
config['models']['t4g_lambdas'] = '0.0,0.0'
config['train']['max_steps'] = 150
config['train']['checkpoint_total_limit'] = 4
