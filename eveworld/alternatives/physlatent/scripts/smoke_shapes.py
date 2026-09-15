from __future__ import annotations

import argparse

import torch

from eveworld.alternatives.physlatent.losses import PhysicsAuxiliaryLoss
from eveworld.alternatives.physlatent.modules import PhysicsLatentEncoder, append_physics_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description='Check PhysLatent tensor shapes.')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--prompt-length', type=int, default=512)
    parser.add_argument('--prompt-dim', type=int, default=1024)
    parser.add_argument('--latent-channels', type=int, default=16)
    parser.add_argument('--latent-frames', type=int, default=24)
    parser.add_argument('--latent-height', type=int, default=60)
    parser.add_argument('--latent-width', type=int, default=96)
    parser.add_argument('--num-tokens', type=int, default=8)
    parser.add_argument('--use-stop-token', action='store_true')
    args = parser.parse_args()

    encoder = PhysicsLatentEncoder(
        num_tokens=args.num_tokens,
        prompt_dim=args.prompt_dim,
        latent_channels=args.latent_channels,
        token_dim=args.prompt_dim,
        use_stop_token=args.use_stop_token,
    )
    ref_latents = torch.randn(
        args.batch_size,
        args.latent_channels,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
    )
    ref_masks = torch.zeros(args.batch_size, 1, args.latent_frames, 1, 1)
    ref_masks[:, :, :1] = 1
    prompt_embeds = torch.randn(args.batch_size, args.prompt_length, args.prompt_dim)
    prompt_embeds[:, args.prompt_length // 2 :] = 0

    tokens, aux = encoder(ref_latents, prompt_embeds, ref_masks=ref_masks, return_aux=True)
    crossattn = append_physics_tokens(prompt_embeds, tokens)

    expected_tokens = args.num_tokens + int(args.use_stop_token)
    assert tokens.shape == (args.batch_size, expected_tokens, args.prompt_dim)
    assert crossattn.shape == (args.batch_size, args.prompt_length + expected_tokens, args.prompt_dim)
    assert aux['trajectory'].shape == (args.batch_size, args.num_tokens, 2)
    assert aux['contact_logits'].shape[:2] == (args.batch_size, args.num_tokens)
    assert aux['phase_logits'].shape == (args.batch_size, args.num_tokens, 8)
    assert aux['done_logits'].shape == (args.batch_size, args.num_tokens)
    assert aux['goal_reached_logits'].shape == (args.batch_size, args.num_tokens)
    assert aux['release_logits'].shape == (args.batch_size, args.num_tokens)
    assert aux['object_motion'].shape == (args.batch_size, args.num_tokens)

    loss_fn = PhysicsAuxiliaryLoss(
        dict(
            state=0.1,
            goal=0.1,
            contact=0.1,
            trajectory=0.1,
            phase=0.1,
            done=0.1,
            goal_reached=0.1,
            release=0.1,
            object_motion=0.1,
            terminal_stable=0.1,
            quality_threshold=0.0,
        )
    )
    labels = {
        'phys_state': torch.zeros(args.batch_size, 4),
        'phys_goal': torch.zeros(args.batch_size, 4),
        'phys_trajectory': torch.zeros(args.batch_size, args.num_tokens, 2),
        'phys_contact': torch.zeros(args.batch_size, args.num_tokens, dtype=torch.long),
        'phys_phase': torch.zeros(args.batch_size, args.num_tokens, dtype=torch.long),
        'phys_done_bin': torch.full((args.batch_size,), args.num_tokens - 2, dtype=torch.long),
        'phys_goal_reached': torch.ones(args.batch_size, args.num_tokens),
        'phys_release': torch.ones(args.batch_size, args.num_tokens),
        'phys_object_motion': torch.zeros(args.batch_size, args.num_tokens),
        'phys_state_mask': torch.ones(args.batch_size),
        'phys_goal_mask': torch.ones(args.batch_size),
        'phys_trajectory_mask': torch.ones(args.batch_size, args.num_tokens),
        'phys_contact_mask': torch.ones(args.batch_size, args.num_tokens),
        'phys_phase_mask': torch.ones(args.batch_size, args.num_tokens),
        'phys_done_mask': torch.ones(args.batch_size),
        'phys_goal_reached_mask': torch.ones(args.batch_size, args.num_tokens),
        'phys_release_mask': torch.ones(args.batch_size, args.num_tokens),
        'phys_object_motion_mask': torch.ones(args.batch_size, args.num_tokens),
        'phys_terminal_mask': torch.ones(args.batch_size, args.num_tokens),
        'phys_label_quality': torch.ones(args.batch_size),
    }
    losses = loss_fn(aux, labels)
    assert set(losses) == {
        'state',
        'goal',
        'trajectory',
        'contact',
        'phase',
        'done',
        'goal_reached',
        'release',
        'object_motion',
        'terminal_stable',
    }

    print(f'physics_tokens={tuple(tokens.shape)}')
    print(f'crossattn={tuple(crossattn.shape)}')
    print(f"state={tuple(aux['state'].shape)} goal={tuple(aux['goal'].shape)} contact={tuple(aux['contact_logits'].shape)}")
    print(f"phase={tuple(aux['phase_logits'].shape)} done={tuple(aux['done_logits'].shape)} release={tuple(aux['release_logits'].shape)}")
    print('aux_losses=' + ','.join(sorted(losses)))


if __name__ == '__main__':
    main()
