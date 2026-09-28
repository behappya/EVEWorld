# ICH-D: a frozen duplicate probe gating a deletion residual

The last alternative attacked laziness inside the representation instead of the
objective or the sampler. A lazy rollout repeats content, so this route learns
to recognise repeated content and deletes it from the latent state while the
sampler runs.

## The idea

A frozen linear probe estimates, per latent cell, the probability that the cell
belongs to repeated content, from a 16-dimensional descriptor of novelty and
consistency statistics taken from several Transformer blocks: novelty
statistics from blocks 10, 12 and 16, and consistency statistics from block 22,
where the temporal adapter attaches. A gated, zero-initialised residual then
erases the flagged content from the block-22 representation:

```text
M = sigmoid(beta^T phi(h))
h_22 <- h_22 - sigmoid(g) * M * W_out(h_22)
```

`phi` collects the 16 statistics, `beta` is the probe, `g` is the gate and
`W_out` is a zero-initialised projection, so the correction starts as the
identity and has to be learned. The probe itself is fitted offline on cached
clip features and frozen before the study; the run never updates it. Fitting
merges the cached rows of every noise level into one data set, because at
inference the noise level varies from step to step and a probe fitted on a
single level would be scored off its training distribution. Each feature is
standardised with the mean and standard deviation of the merged rows, and a
logistic regression is fitted on top; the stored weights reproduce the fitted
probabilities exactly, which `fit.py` checks against `predict_proba` on every
run.

## How it was run

The probe was fitted from a cache of per-block clip features at the two
recorded noise levels, with the consistency block pinned to block 22 in the
cache metadata. The fit is a CPU job over a few thousand rows and is the whole
offline part of the route; the deletion residual is then applied at the
block-22 output of every denoising forward pass, in both CFG branches, and the
sampler continues normally afterwards, with no re-noising or repair pass.

```bash
python experiments/alternatives/ich_d/fit.py \
    --cache cic_cell_features.npz --out ich_d_frozen_lr.npz --loo
```

The matched control is the pretrained model itself, so the pair isolates the
deletion path rather than a training change. Generation used the released
sampler settings, 30 inference steps at the paper's guidance weight, and the
comparison ran over 368 matched prompt-seed pairs across four seeds.

## Reading the record

- [`fit.py`](fit.py) is the offline fitting job: it reads the per-block feature
  cache, verifies the consistency block, merges the noise levels, fits the
  standardised logistic probe and writes the frozen parameters together with
  their mean and scale. `--loo` adds the leave-one-out AUC.
- The probe features come from the same feature-probe family as the TIA probe
  scan; the per-block evidence behind that scan is in
  [`../../analysis/tia_layer_probe/`](../../analysis/tia_layer_probe/).
- The metric definitions and the eligible set are in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md); the route is
  kept here as the record of the study.
