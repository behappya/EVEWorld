# PhysicsLatent

A dynamics proxy that carries an explicit physics state through the
conditioning path: eight process tokens of the text-context width are appended
to the prompt embeddings before the Transformer, and weak labels train those
tokens to describe how the manipulation evolves.

## The idea

The backbone is conditioned on a first frame and a text prompt, and nothing in
that conditioning says what the object should be doing between the first frame
and the end of the clip. PhysicsLatent adds that channel without touching the
backbone implementation:

```text
prompt_embeds + physics_tokens -> crossattn_emb
```

The tokens are appended to the conditioning sequence, so the VAE, the
scheduler, the Transformer blocks and the released checkpoints stay
compatible, and the only new parameters sit in a token encoder that maps the
prompt plus the first frame to `K = 8` tokens of width `1024`.

The tokens are supervised by weak labels rather than by a learned dynamics
model of the scene. Four heads share the token state:

- the start and end box of the manipulated object,
- contact events between gripper and target,
- the 2D trajectories of eight keyframes,
- the manipulation stage, one of approach, grasp, transport or release.

The auxiliary loss is a weighted sum over the four heads, and the total weight
sits between 0.03 and 0.08 so the denoising loss stays dominant. The labels are
generated offline from the training clips, and the heads exist only during
training; generation uses the tokens alone.

## How it was run

Training had three stages on top of the pretrained backbone:

1. Adapter warm-up, with the Transformer frozen, training only the token
   encoder so the tokens start in the range the cross-attention expects.
2. Auxiliary warm-up, which adds the four head losses and trains the heads
   against the pseudo labels.
3. Joint stage, which trains the token encoder together with attention LoRA
   adapters while the backbone stays frozen.

The matched control is the pretrained backbone at the same 5.8 s clip length,
run on the same prompts and the same generation seeds. Both arms were generated
with the released sampler settings, at 30 inference steps and the paper's
guidance weight, and judged by both instruction-following judges. The stage
configs sit in the route's own `configs/` record, and the route starts from the
standard SFT recipe of the released tree:

```bash
python scripts/train/train_gigaworld.py --config configs/paper/gigaworld/dreamgen/sft.yaml \
    --steps 250 --seed 42 --output-dir outputs/alternatives/physics_latent
```

The three stages differ in which parameters are unfrozen and which losses are
active, not in the data pipeline, so the same command with the stage flags is
what a re-run reproduces.

## Outcome

The route moved MLR in the wanted direction and lost instruction fidelity:

| Arm | MLR (%) | Gemini-IF (mean ± s.e.) |
|---|---|---|
| Pretrained backbone, 5.8 s (control) | 17.39 | 45.65 ± 3.92 |
| PhysicsLatent | 16.18 | 40.22 ± 3.26 |
| Relative change | −7.0%, 95% CI [−51.5, +69.1] | −11.9%, 95% CI [−29.8, +7.9] |

The comparison ran over 92 matched prompt-seed pairs. It is the only one of the
five alternative routes that lowered MLR at all, and the reason it was still
dropped is the coupling visible in the table: the process tokens compete with
the prompt embeddings for cross-attention, so the more the tokens carry the
physical state, the less the conditioning follows the wording of the
instruction. The MLR gain is also inside its own interval, while the
instruction-following loss is not recovered by any later stage of the route.

The released method keeps the part of the idea that survives: the disturbed
region of the clip is supervised directly, in reconstruction space, where the
instruction conditioning is not a competitor.

## Reading the record

- The auxiliary heads and their losses are described by the four label groups
  above; each label group maps to one head and one term of the auxiliary sum.
- The matched-control numbers come from the DreamGenBench pool used by the
  paper's comparison of alternative designs, and the MLR protocol, including
  the eligible set and the occlusion rule, is documented in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md).
- The route is a negative control: it is kept in this directory as the record
  of the study, not as a supported arm of the released model.
