from .gr1_physlatent_aux import config as base_config


config = dict(base_config)
config['project_dir'] = '/data/datasets/gagi/giga_world_0_outputs/physlatent_gigaworld/gr1_phase_done_aux_warmup'

config['dataloaders'] = dict(base_config['dataloaders'])
config['dataloaders']['train'] = dict(base_config['dataloaders']['train'])
config['dataloaders']['train']['transform'] = dict(base_config['dataloaders']['train']['transform'])
config['dataloaders']['train']['transform']['random_crop'] = False
config['dataloaders']['train']['transform']['phys_labels_path'] = (
    '/data/datasets/gagi/gr1_finetune_data/physlatent_pseudo_labels/gr1_physlabels_v3_phase_done.json'
)

config['models'] = dict(base_config['models'])
config['models']['physics_latent'] = dict(base_config['models']['physics_latent'])
config['models']['physics_latent'].update(
    num_phase_classes=8,
    use_stop_token=True,
    terminal_denoise_weight=0.5,
    terminal_quality_threshold=0.6,
    loss_weights=dict(
        state=0.03,
        goal=0.05,
        trajectory=0.03,
        progress_stage=0.01,
        phase=0.06,
        done=0.08,
        goal_reached=0.05,
        release=0.05,
        object_motion=0.04,
        terminal_stable=0.10,
        quality_threshold=0.6,
    ),
)

config['train'] = dict(base_config['train'])
config['train']['max_steps'] = 500
config['train']['checkpoint_interval'] = 100
