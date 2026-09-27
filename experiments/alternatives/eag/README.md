# EAG: energy-guided sampling

A test-time route that leaves the weights alone and changes what the sampler
does. A frozen latent-action model scores the clean latent the sampler is
heading towards, and the sampler takes one step down that energy, which pushes
the rollout toward frame-to-frame transitions that the action model considers
executable.

## The idea

The diagnostic that motivated the route is that laziness is not a high average
transition error; it is a few very large illegal jumps, while most frames
repeat content with almost no motion. An energy built on the mean transition
error would therefore be diluted by the static frames. EAG scores the predicted
clean latent `z_hat_0` with the soft top-3 mean of the latent-action model's
transition errors, which follows the largest jumps rather than the average one,
and moves the latent along the normalised negative gradient:

```text
z_hat_0 <- z_hat_0 - w(sigma) * ||z_hat_0|| * grad E / ||grad E||
w(sigma) = w_0 / (1 + sigma)
```

The weight decays with the noise level, so the guidance is strongest where the
sample is still fluid and fades as the sampler commits to details. Contact
events are read off the same action model rather than from a separate detector,
and the energy is what the route gates on.

## How it was run

The latent-action model was pretrained self-supervised on cached latents of the
training clips and then frozen for the whole study; only the sampler changed.
The route and its control share the prompts, the seeds and every sampler
setting apart from the guidance weight, and the control is the identical
command with `--eag-weight 0`, which reproduces the unguided baseline at the
same seed. The relevant sampler flags are the guidance weight, the top-k of the
soft aggregation and its temperature, alongside the usual 30 inference steps at
93 frames.

```bash
python scripts/inference/infer_gigaworld.py --config configs/paper/gigaworld/dreamgen/sft.yaml \
    --checkpoint Video-Pretrain-2B --prompt-file data/splits/dreamgen/test.txt \
    --num-steps 30 --cfg-scale 7.0 --output-dir outputs/alternatives/eag/w0.03
```

The paired run needs both arms at the same seed, so the control is generated
with the weight set to zero rather than with a different sampler.

## Outcome

| Arm | MLR (%) | Gemini-IF (mean ± s.e.) |
|---|---|---|
| Control, guidance weight 0 | 15.38 | 41.67 ± 3.61 |
| EAG, guidance weight 0.03 | 38.46 | 45.83 ± 9.55 |
| Relative change | +150.0%, 95% CI [0.0, +500.0] | +10.0%, 95% CI [−15.8, +54.5] |

The comparison ran over 16 prompt-seed pairs, of which 13 were
detector-eligible, so the MLR pool is the smallest of the five routes.
Instruction following moved by a small positive amount that its interval does
not separate from zero, while MLR rose by 150%: the energy was steering the
sample toward transitions the action model found plausible without regard for
whether the requested object was still on screen, which is the opposite of what
the benchmark measures.

The route was dropped because the contact-event detector underneath the energy
was the bottleneck: the energy is only as good as the action model's notion of
a legal transition, and on clips where the detector missed the grasp the
guidance walked the sample away from the instruction. The released method keeps
the insight that motion has to be supervised — and applies it as a weight map
inside the training loss, where the supervision points at the region the
instruction refers to, instead of as an inference-time push.

## Where the record lives

The energy and the guidance step are the two functions the study consists of:
one wraps the frozen action model into a differentiable energy, the other
applies the single normalised step after each classifier-free-guidance update.
The sampler stays otherwise unchanged, which is what makes the weight-zero arm
a valid matched control. The metric definitions and the eligible set are in
[`../../../docs/evaluation.md`](../../../docs/evaluation.md).
