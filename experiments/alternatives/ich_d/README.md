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

## Outcome

| Arm | MLR (%) | Gemini-IF |
|---|---|---|
| Control, pretrained, 4 seeds | 11.23 | 45.83 |
| ICH-D | 13.09 | 31.79 |
| Relative change | +16.6%, 95% CI [−19.4, +70.8] | −30.6%, 95% CI [−40.1, −20.2] |

Both metrics move the wrong way, and this time the instruction-following drop
is the one the interval separates from zero: the deletion path introduces its
own errors. The probe answers one question — does this cell look like a repeat
of earlier content — and the metric asks a different one, whether the object
the instruction names is on screen. A cell that repeats real background motion
is indistinguishable from a cell that repeats a deleted target, so the residual
erases content the clip needed. The probe's own separation is not the
bottleneck: a probe of the same family detects an injected duplicate in the
early blocks with an AUC of 0.79–0.80, and the leave-one-out read-out of the
16-dimensional fit is above 0.8. The route failed because detection accuracy on
the training corruption does not transfer into a deletion decision that is safe
in every frame of a free-running rollout.

## Reading the record

- [`fit.py`](fit.py) is the offline fitting job: it reads the per-block feature
  cache, verifies the consistency block, merges the noise levels, fits the
  standardised logistic probe and writes the frozen parameters together with
  their mean and scale. `--loo` adds the leave-one-out AUC next to the value
  recorded for the 16-dimensional probe at the higher noise level.
- The probe features come from the same feature-probe family as the TIA probe
  scan; the per-block evidence behind that scan is in
  [`../../analysis/tia_layer_probe/`](../../analysis/tia_layer_probe/).
- The metric definitions and the eligible set are in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md); the route is a
  negative result and is kept here as the record of the study.
