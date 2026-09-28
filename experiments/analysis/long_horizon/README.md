# Long horizon: the length sweep from 3.8 s to 29.8 s

The single-clip evaluations measure short generations; the horizon sweep asks
what happens when the same model has to hold a scene together for tens of
seconds. It runs PBench Robot at seven robot-length tiers from 3.8 s to 29.8 s
and records both the physical-commonsense accuracy and the generation cost of
every tier, so a long-horizon failure can be read as a quality problem or a
budget problem depending on which side of the table moved.

## The record

`lengths.py` is the tier arithmetic. The frame counts are the published ones,
61, 93, 157, 253, 317, 413 and 477; the script prints their durations at 16
FPS, the fit of every count to the 4-frame temporal stride of the video VAE
and the latent length that follows from the backbone's `latent_frames` helper
(`1 + (frames - 1) // 4`):

```bash
python experiments/analysis/long_horizon/lengths.py
```

```text
| Frames | Seconds | (frames - 1) % 4 | Latent frames |
|---:|---:|---:|---:|
| 61 | 3.8 | 0 | 16 |
| 93 | 5.8 | 0 | 24 |
| 157 | 9.8 | 0 | 40 |
| 253 | 15.8 | 0 | 64 |
| 317 | 19.8 | 0 | 80 |
| 413 | 25.8 | 0 | 104 |
| 477 | 29.8 | 0 | 120 |
```

Every tier is one frame plus whole 4-frame temporal cells, so the stride
remainder is zero for all seven counts and the sweep can reuse a single VAE
and a single denoising schedule across the grid. `--fps` re-derives the
durations for another frame rate and `--out` writes the table to a file.

## How it was run

Rollouts are generated at all seven tiers at 16 FPS, and each tier is scored
with the same 913 physical-commonsense questions over 174 videos, so the curve
isolates the horizon and not the prompt set. The reported table averages the
five durations from 3.8 s to 19.8 s and adds a mean over the three longest
tiers; cross-trial robustness replays the two shortest tiers with three
inference seeds under a matched budget. Generation goes through the PBench
adapter like any other evaluation:

```bash
python scripts/evaluate/eval_pbench.py \
    --config configs/eval/pbench.yaml \
    --pred-dir outputs/pbench_eveworld \
    --output outputs/pbench_eveworld/horizon.json
```

The length-scaling runs additionally log latency, throughput and peak memory
per tier. The sweep has no item-level directory, so a run writes its summary
under `outputs/`.

## Reading the record

- [`lengths.py`](lengths.py) is the whole tool: it holds the seven published
  frame counts and prints the duration, stride and latent-length table at 16
  FPS by default; `--fps` re-derives the durations for another frame rate.
- The sweep configuration is
  [`../../../configs/eval/pbench.yaml`](../../../configs/eval/pbench.yaml);
  the protocol, the metric definitions and the reference numbers are in
  [`../../../docs/evaluation.md`](../../../docs/evaluation.md), and the study
  is scheduled among the additional analyses in
  [`../../../docs/reproduction.md`](../../../docs/reproduction.md).
- The horizon sweep shares its generation path with the seed-repeat study in
  [`../seed_stability/`](../seed_stability/), and the clip-level MLR protocol
  behind the shorter evaluations is recomputed in
  [`../mlr_sensitivity/`](../mlr_sensitivity/).
