from .gr1_physlatent_phase_done_aux import config as base_config


config = dict(base_config)
config['project_dir'] = '/data/datasets/gagi/giga_world_0_outputs/physlatent_gigaworld/gr1_phase_done_lora'

config['models'] = dict(base_config['models'])
config['models']['train_mode'] = 'lora'
config['models']['lora_rank'] = 64
config['models']['physics_latent'] = dict(base_config['models']['physics_latent'])
config['models']['physics_latent']['train_backbone'] = False

config['train'] = dict(base_config['train'])
config['train']['max_steps'] = 800
config['train']['checkpoint_interval'] = 100
