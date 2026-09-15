from __future__ import annotations

import copy
import importlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    (
        runtime_config,
        base_config_module,
        project_dir,
        packed_data_dir,
        transformer_model_path,
        vae_model_path,
        max_steps,
        batch_size_per_gpu,
        gradient_accumulation_steps,
        checkpoint_interval,
        checkpoint_total_limit,
        num_workers,
        num_frames,
        height,
        width,
        fps,
        seed,
        mixed_precision,
        with_ema,
        activation_checkpointing,
        train_accelerate,
        *gpu_ids,
    ) = sys.argv[1:]

    config = copy.deepcopy(importlib.import_module(base_config_module).config)
    config['project_dir'] = project_dir
    config['launch']['gpu_ids'] = [int(x) for x in gpu_ids]
    config['launch']['executable'] = train_accelerate
    config['dataloaders']['train']['data_or_config'] = [packed_data_dir]
    config['dataloaders']['train']['batch_size_per_gpu'] = int(batch_size_per_gpu)
    config['dataloaders']['train']['num_workers'] = int(num_workers)

    transform = config['dataloaders']['train']['transform']
    transform['num_frames'] = int(num_frames)
    transform['height'] = int(height)
    transform['width'] = int(width)
    transform['fps'] = int(fps)

    phys_labels_path = os.environ.get('PHYS_LABELS_PATH', '')
    if phys_labels_path:
        transform['phys_labels_path'] = phys_labels_path
        transform['random_crop'] = os.environ.get('PHYS_RANDOM_CROP', '0') == '1'

    physics_config = config.get('models', {}).get('physics_latent')
    if isinstance(physics_config, dict):
        loss_weights = dict(physics_config.get('loss_weights', {}))
        env_to_key = {
            'PHYS_LOSS_STATE': 'state',
            'PHYS_LOSS_GOAL': 'goal',
            'PHYS_LOSS_CONTACT': 'contact',
            'PHYS_LOSS_TRAJECTORY': 'trajectory',
            'PHYS_LOSS_PHASE': 'phase',
            'PHYS_LOSS_DONE': 'done',
            'PHYS_LOSS_GOAL_REACHED': 'goal_reached',
            'PHYS_LOSS_RELEASE': 'release',
            'PHYS_LOSS_OBJECT_MOTION': 'object_motion',
            'PHYS_LOSS_TERMINAL_STABLE': 'terminal_stable',
        }
        for env_name, key in env_to_key.items():
            env_value = os.environ.get(env_name, '')
            if env_value:
                loss_weights[key] = float(env_value)
        physics_config['loss_weights'] = loss_weights

    config['models']['transformer_model_path'] = transformer_model_path
    config['models']['vae_model_path'] = vae_model_path
    config['train']['max_steps'] = int(max_steps)
    config['train'].pop('max_epochs', None)
    config['train']['gradient_accumulation_steps'] = int(gradient_accumulation_steps)
    config['train']['checkpoint_interval'] = int(checkpoint_interval)
    config['train']['checkpoint_total_limit'] = int(checkpoint_total_limit)
    config['train']['seed'] = int(seed)
    config['train']['mixed_precision'] = mixed_precision
    config['train']['with_ema'] = with_ema == '1'
    config['train']['activation_checkpointing'] = activation_checkpointing == '1'

    path = Path(runtime_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
