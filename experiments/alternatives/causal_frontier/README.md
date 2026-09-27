# Causal frontier: block-sequential training and generation

A variant that removes the global denoising pass instead of reweighting it. A
clip is cut into four-latent blocks, every step trains the model to reconstruct
one block from the clean blocks committed before it, and generation commits
blocks in the same order while conditioned on the committed prefix.

## The idea

The motivation is causal: a diffusion model that denoises the whole clip at
once can use a late frame to reconstruct an early one, which trains it to copy
instead of to move. Restricting each step to a prefix and one active block
removes the shortcut, because the frames after the frontier are simply not in
the input:

- A clip of 24 latent frames is divided into blocks of four, anchored at frame
  zero, so the active ranges are `[1, 4)`, `[4, 8)`, ..., `[20, 24)`; frame zero
  of the first block is the image condition.
- Training samples one frontier uniformly over the blocks and, for that
  frontier, feeds the committed prefix clean and detached at a history noise
  level of `1e-4` with the mask channel set to one, noises only the active
  block, and concatenates prefix and active block. Future blocks never enter
  the input.
- The denoising loss is taken on the active block alone and normalised by the
  number of active tokens, so an early and a late frontier contribute the same
  loss scale.
- Generation walks the blocks in order: the next block is denoised from the
  committed prefix with 30 sampling steps at a guidance weight of 7.0, then
  detached and appended. There is no overlap between blocks and no correction
  pass, so an error made in one block is never revisited.

The masking and the loss live in [`loss.py`](loss.py): `frontier_weights` marks
the active block, `FrontierLoss` builds the prefix + active pair and applies the
EDM-weighted error on the active tokens.

## How it was run

Both arms are fine-tuned from the same Video-Pretrain-2B checkpoint with the
same optimizer, schedule and data, and differ only in the training objective
and the generation procedure. The matched control is the plain full-clip
objective, `control50`, at the same length and step; the route's generation
uses its own block loop.

```bash
python scripts/train/train_gigaworld.py --config configs/paper/gigaworld/dreamgen/sft.yaml \
    --steps 250 --seed 42 --output-dir outputs/alternatives/causal_frontier
python scripts/inference/infer_gigaworld.py --config configs/paper/gigaworld/dreamgen/sft.yaml \
    --prompt-file data/splits/dreamgen/test.txt --num-steps 30 --cfg-scale 7.0 \
    --output-dir outputs/alternatives/causal_frontier/generated_only
```

The route's generation is the block loop described above with `loss.py`'s
geometry; the sampling settings inside a block are the released ones, which is
what makes the comparison about the frontier rather than about the sampler.

## Outcome

| Arm | MLR (%) | Gemini-IF (mean ± s.e.) |
|---|---|---|
| Control, full-clip objective | 30.77 | 27.08 ± 3.61 |
| Causal frontier, block-sequential | 41.67 | 12.50 ± 6.25 |
| Relative change | +35.4%, 95% CI [−42.9, +300.0] | −53.8%, 95% CI [−100.0, +85.7] |

The comparison ran over 16 matched prompt-seed pairs. MLR rose by roughly a
third and the instruction-following score fell by more than half, and although
the 16-pair intervals are wide, the direction is the same on both metrics.

The failure has a mechanism rather than a story: with no future frames in the
input and no correction after a block is committed, the prediction error of one
block becomes part of the conditioning of the next, so error accumulates along
the rollout. That is also what the diagnostic in [`loss.py`](loss.py) shows when
it is run with `--demo`: scoring a synthetic reconstruction whose error grows
with the frontier index produces a loss that grows in the same order. The route
was therefore dropped in favour of keeping the whole-clip objective and
weighting the region the instruction refers to, which is what the released IGR
term does.

## Reading the record

- [`loss.py`](loss.py) is self-contained: the block geometry, the prefix
  handling and the active-only loss are the three pieces the variant is built
  from, and `python loss.py --demo` prints the per-frontier scores.
- The metric definitions and the eligible set are in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md); the route is a
  negative control and is kept here as the record of the study.
