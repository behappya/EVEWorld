# IGR restoration probe: reading back the weight map

The IGR objective re-weights the reconstruction term with a map that marks the
spatial support of the disturbance, so two questions decide how the map is
built: which support and which relative weight to use, and what the emphasis
does to the reconstruction of the disturbed region itself. The first question
is answered by a development comparison over the design space; the second by a
controlled corruption experiment that inserts one extra target instance into a
clean demonstration and measures how much of the injected area the
reconstruction keeps.

## What the probe reads

For every region of the map the probe measures the output against the clean
clip and against the corrupted clip the sampler was conditioned on:

```text
restore   = mean |pred - corrupted| / mean |target - corrupted|
retention = mean |pred - target| / mean |target - corrupted|
direction = cos(pred - corrupted, target - corrupted)
```

`restore` is one when the output matches the clean target and zero when it
keeps the corrupted content; `retention` is the opposite reading, one when the
injected content survives and zero when the model has removed it. The two add
up to one when the output lies on the segment between the clean and corrupted
clips, and separate when it does not. The direction cosine says where the
residual points: one when it moves exactly along the restoration axis, zero
when it is orthogonal to it, and negative when the output drifts further along
the injected direction. `analyze.py` reports these three readings once per region of
the map and once over the whole labelled area.

## The weight-map design comparison

Every candidate assigns ordinary locations relative weight 1 and marks an
emphasized set with a larger weight; coverage is the fraction of
spatiotemporal latent locations that fall in the emphasized set. The designs
differ in how much of the manipulation scene the emphasized set covers.

The multi-level map hands different weights to the target path, destination,
vacated foreground, gripper, distractors and pasted instance; the binary
designs use a single emphasis factor. Restricting the support to the target
path and its source and destination regions leaves the restoration errors
near the gripper, along the arm corridor and around the two endpoint regions
unweighted, so the released map uses the wider interaction support: ordinary
locations get relative weight 1, the interaction support and the pasted
instance get relative weight 3, and the map is normalised to unit mean before
it enters the loss.

## The restoration reading

The corruption experiment starts from the Standard SFT checkpoint, inserts one
extra target instance, reconstructs the clip under six combinations of paste
opacity and diffusion noise level, and repeats the same measurement after 50
IGR fine-tuning steps. Retention is the fraction of the injected duplicate
area the reconstruction preserves; the direction cosine is the alignment of
the residual error with the injected duplicate.

## Running it

```bash
python experiments/analysis/igr_restoration/analyze.py \
    --pred outputs/igr/restored/with_weighting.npz \
    --pred outputs/igr/restored/without_weighting.npz \
    --target outputs/igr/clean/clip.npz \
    --corrupted outputs/igr/corrupted/clip.npz \
    --weight-map outputs/igr/weight_maps/clip.npz
```

The clips are `(T, H, W, C)` arrays under `pred`/`x0_hat`, `target`/`clean` and
`corrupted`/`input`; the map holds integer `zones` or float `weights` on the
frame grid, resampled with nearest neighbours when it is stored on the latent
grid. Dropping `--corrupted` leaves the perturbation and the three normalised
columns as `n/a`, which is the right reading when only the error against the
clean clip is available. Adding a second `--pred` prints an extra comparison
table that lines the arms up region by region.

## Reading the record

- [`analyze.py`](analyze.py) is the probe: it prints the per-region table and
  the arm comparison, and takes `--out` to write the same report to a file.
- The objective and the weight map are specified in
  [`../../../docs/training.md`](../../../docs/training.md); the metric side of
  the corruption experiment is in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md).
- The alternative weighting designs are kept under
  [`../../alternatives/`](../../alternatives/), and the probe that localises
  where the temporal adapter should sit is in
  [`../tia_layer_probe/`](../tia_layer_probe/).
