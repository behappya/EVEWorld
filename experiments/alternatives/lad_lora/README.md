# LAD-LoRA: the transition energy as a training regulariser

The EAG route moved the same energy to inference time, and this route moves it
into training instead: the frozen latent-action model scores the
clean latent the denoiser predicts, and that score is added to the denoising
loss with a per-step weight, backpropagating only into a rank-64 attention
LoRA.

## The idea

The energy is the same soft top-3 mean of the latent-action model's
frame-to-frame transition errors that EAG used, but it enters the objective
rather than the sampler:

```text
L = L_EDM + (w_tilde / (1 + sigma)) * E(z_hat_0, theta)
```

`L_EDM` is the ordinary denoising loss. `E` is differentiable with respect to
the predicted clean latent `z_hat_0`, which is the same tensor the denoising
term already differentiates: both gradients flow into the shared `z_hat_0`,
which is what keeps the two terms commensurate. The coefficient is recomputed
on every step from the two gradient norms rather than fixed in advance, so the
energy contributes a bounded share of the update:

1. Denoise with the current LoRA weights and predict `z_hat_0`.
2. Score `z_hat_0` with the frozen latent-action energy.
3. Measure the gradient norm of the energy and of the denoising loss with
   respect to `z_hat_0` and set `w_tilde` so their ratio hits the target
   fraction, with a ceiling of 1.0; multiply by the `1 / (1 + sigma)` gate.
4. Backpropagate the combined objective into the rank-64 attention adapters
   only; the backbone and the energy stay frozen.

The target fraction is 0.1 and the ceiling is 1.0, so a step whose energy
gradient is already large cannot dominate the update. The `1 / (1 + sigma)`
gate gives the same shape as the EAG sampler weight: the supervision is
strongest at high noise, where the sample is still deciding what moves.

## How it was run

The training data, the optimizer, the schedule and the batch composition are
the standard SFT recipe; the only differences from the control are the energy
term and the rank-64 attention LoRA that carries it. The control arm is the
same run with the energy weight set to zero, which reduces the objective to
`L_EDM` on the identical adapter, so the pair separates the energy from the
LoRA.

```bash
python scripts/train/train_gigaworld.py --config configs/paper/gigaworld/dreamgen/sft.yaml \
    --steps 250 --seed 42 --output-dir outputs/alternatives/lad_lora
```

Both arms were generated with the released sampler settings, at 30 inference
steps and the paper's guidance weight, over the same prompts and seeds, and
judged by both instruction-following judges.

## Reading the record

- The energy head, the gradient-ratio weight and the gate are the three parts
  of the objective above; the route keeps them together in one place.
- The matched control is the same trainer with the energy weight set to zero,
  which is why the pair isolates the regulariser rather than the adapter.
- The metric definitions and the eligible set are in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md); the route is
  kept here as the record of the study.
