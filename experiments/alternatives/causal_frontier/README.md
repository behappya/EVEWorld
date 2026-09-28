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

## Reading the record

- [`loss.py`](loss.py) is self-contained: the block geometry, the prefix
  handling and the active-only loss are the three pieces the variant is built
  from, and `python loss.py --demo` prints the per-frontier scores.
- The metric definitions and the eligible set are in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md); the route is
  kept here as the record of the study.
