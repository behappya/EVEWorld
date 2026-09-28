# TIA layer probe: choosing the attachment block by adjacent-frame matching

The TIA temporal adapter attaches to one Transformer block, and the first
question the route has to answer is which block that should be. Fixing the
attachment point heuristically would confound the consistency result with the
layer choice, so the block is selected offline by a probe that scores every
candidate layer and is run once per setting.

## The probe

For a candidate block, the target feature of frame `t - 1` retrieves its best
local match inside the `7 x 7` window of frame `t`, and the probe averages the
distance to the annotated target location over the frame pairs in `V`:

```text
u_hat_t = arg max_{v in C_t} f_{t-1}(u_{t-1})^T f_t(v)
l_star  = arg min_l (1 / |V|) sum_t || u_hat_t - u_t ||_2
```

The endpoint error (EPE) of the matched position, in token cells, is the
selection criterion: the block with the lowest mean EPE wins. Because feature
reliability can differ across domains and backbone architectures, the probe
reports a setting-specific consistency-reliability score next to EPE, defined
per setting as object--background separation for GigaWorld-0/DreamGen,
static-feature stability for AgiBot, and moving-pair top-1 hit rate for
FlowWAM.

## How it was run

The sweep covers blocks 8 to 26 and reads the same annotations the corruption
builder uses. Each run writes one JSON per candidate block into a per-domain
directory, `blockNN.json`, carrying the block number, its mean EPE and the
reliability score:

```bash
python experiments/analysis/tia_layer_probe/summarize.py \
    --root experiments/analysis/tia_layer_probe --block-range 8 26
```

`summarize.py` discovers the domains under the root, prints one block-sorted
table per domain with the lowest EPE inside the range in bold and the blocks
the domain did not report, and closes with a selection table across domains.
The winner is chosen inside `--block-range` only.

## Reading the record

- [`summarize.py`](summarize.py) reads the per-block probe records, prints the
  per-domain tables and can write them to a file with `--out`; it checks the
  block number in each file name against the record and reports the blocks of
  the range that a domain did not report.
- The probe definition and the selected layers are documented in
  [`../../../docs/training.md`](../../../docs/training.md); the sweep config is
  [`../../../configs/ablations/tia/layer_sweep.yaml`](../../../configs/ablations/tia/layer_sweep.yaml).
- The analysis directories and what each one regenerates are listed in
  [`../../../docs/reproduction.md`](../../../docs/reproduction.md).
- The restoration side of the adapter is analysed in
  [`../igr_restoration/`](../igr_restoration/); the duplicate-probe alternative
  that uses this probe family as a frozen gate is in
  [`../../alternatives/ich_d/`](../../alternatives/ich_d/).
