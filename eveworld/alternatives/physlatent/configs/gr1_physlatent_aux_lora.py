from .gr1_physlatent_aux import config as base_config


config = dict(base_config)
config['project_dir'] = '/data/datasets/gagi/giga_world_0_outputs/physlatent_gigaworld/gr1_aux_lora'
config['models'] = dict(base_config['models'])
config['models']['train_mode'] = 'lora'
config['models']['lora_rank'] = 64
config['models']['physics_latent'] = dict(base_config['models']['physics_latent'])
config['models']['physics_latent']['train_backbone'] = False
