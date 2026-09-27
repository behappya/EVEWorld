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

## Outcome

The probe selects block 23 for GigaWorld-0/DreamGen and AgiBot and block 12 for
FlowWAM.

| Block | DreamGen EPE | Reliab. | AgiBot EPE | Reliab. | FlowWAM EPE | Reliab. |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.80 | −0.09 | 9.24 | 0.72 | 3.50 | 0.172 |
| 9 | 0.80 | −0.07 | 8.88 | 0.74 | 3.56 | 0.171 |
| 10 | 0.93 | −0.09 | 8.00 | 0.73 | 3.46 | 0.178 |
| 11 | 0.93 | −0.09 | 7.52 | 0.74 | 3.45 | 0.176 |
| 12 | 0.98 | −0.08 | 6.82 | 0.74 | **3.42** | **0.182** |
| 13 | 0.68 | −0.07 | 6.56 | 0.75 | 3.46 | 0.177 |
| 14 | 0.98 | −0.05 | 6.48 | 0.78 | 3.51 | 0.174 |
| 15 | 0.98 | −0.10 | 6.39 | 0.81 | 3.49 | 0.172 |
| 16 | 0.73 | −0.06 | 6.44 | 0.81 | 3.59 | 0.165 |
| 17 | 0.78 | −0.06 | 6.62 | 0.85 | 3.64 | 0.158 |
| 18 | 0.98 | −0.07 | 6.54 | 0.83 | 3.51 | 0.158 |
| 19 | 0.91 | −0.07 | 6.64 | 0.83 | 3.56 | 0.155 |
| 20 | 0.78 | −0.07 | 6.22 | 0.83 | 3.60 | 0.154 |
| 21 | 0.75 | −0.10 | 6.44 | 0.80 | 3.65 | 0.146 |
| 22 | 0.63 | −0.16 | 6.98 | 0.83 | 3.69 | 0.135 |
| 23 | **0.55** | **−0.10** | **5.72** | **0.76** | 3.68 | 0.144 |
| 24 | 1.08 | −0.19 | 9.81 | 0.81 | 3.71 | 0.139 |
| 25 | 0.96 | −0.20 | 7.53 | 0.71 | 3.89 | 0.126 |
| 26 | 1.28 | −0.24 | 9.79 | 0.65 | 3.87 | 0.124 |

Bold marks the selected block of each setting. The margin is small but the
ranking is stable: the runner-up sits at 0.63 for DreamGen (block 22), 6.22
for AgiBot (block 20) and 3.45 for FlowWAM (block 11). The two orderings do
not coincide everywhere — on AgiBot the highest reliability, 0.85, sits at
block 17 while the lowest EPE, 5.72, sits at block 23 — and the sweep follows
EPE, with the reliability column kept as the record of how well the matched
features hold up under the setting's own consistency notion. The much earlier
winner on FlowWAM shows that the attachment point is a property of the
backbone, not a constant of the pipeline, which is why the layer is stored as
a per-setting value rather than fixed in code.

## Reading the record

- [`summarize.py`](summarize.py) reads the per-block probe records, prints the
  tables shown above and can write them to a file with `--out`; it checks the
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
