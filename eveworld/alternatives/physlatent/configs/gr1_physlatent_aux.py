from .gr1_physlatent_adapter import config as base_config


config = dict(base_config)
config['project_dir'] = '/data/datasets/gagi/giga_world_0_outputs/physlatent_gigaworld/gr1_aux_warmup'

config['dataloaders'] = dict(base_config['dataloaders'])
config['dataloaders']['train'] = dict(base_config['dataloaders']['train'])
config['dataloaders']['train']['transform'] = dict(base_config['dataloaders']['train']['transform'])
config['dataloaders']['train']['transform']['random_crop'] = False
config['dataloaders']['train']['transform']['phys_labels_path'] = (
    '/data/datasets/gagi/gr1_finetune_data/physlatent_pseudo_labels/gr1_physlabels_v1.json'
)

config['models'] = dict(base_config['models'])
config['models']['physics_latent'] = dict(base_config['models']['physics_latent'])
config['models']['physics_latent']['loss_weights'] = dict(
    state=0.05,
    goal=0.10,
    contact=0.03,
    trajectory=0.05,
)

config['train'] = dict(base_config['train'])
config['train']['max_steps'] = 500
config['train']['checkpoint_interval'] = 100
