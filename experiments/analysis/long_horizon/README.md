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
per tier, which is where the inference-cost table comes from. The sweep has no
item-level directory, so a run keeps its summary in the scratch tree rather
than under `results/item_level/`.

## Outcome

**Accuracy falls with the horizon and temporal reasoning falls fastest.**
On the pretrained backbone the overall PBench score drops from 82.43 at 3.8 s
to 66.97 at 19.8 s while the temporal dimension drops from 82.18 to 50.18.
Post-training improves all four dimensions averaged over the five durations,
and its advantage widens at long horizons: over the three tiers of at least
9.8 s Standard SFT gains 1.65, 1.95 and 4.24 points on Domain, Phys. and Time
respectively.

GigaWorld-0 (pretrained):

| Dur. (s) | Domain | Phys. | Space | Time |
|---:|---:|---:|---:|---:|
| 3.8 | 82.43 | 85.71 | 84.24 | 82.18 |
| 5.8 | 79.87 | 84.42 | 82.12 | 74.18 |
| 9.8 | 74.32 | 83.77 | 76.06 | 65.09 |
| 15.8 | 67.54 | 79.22 | 73.03 | 49.82 |
| 19.8 | 66.97 | 76.95 | 74.55 | 50.18 |
| Mean | 74.23 | 82.01 | 78.00 | 64.29 |
| Mean (>= 9.8 s) | 69.61 | 79.98 | 74.55 | 55.03 |

Standard SFT:

| Dur. (s) | Domain | Phys. | Space | Time |
|---:|---:|---:|---:|---:|
| 3.8 | 80.98 | 86.69 | 83.33 | 79.64 |
| 5.8 | 79.89 | 85.71 | 83.94 | 74.91 |
| 9.8 | 78.90 | 87.99 | 79.70 | 74.55 |
| 15.8 | 68.12 | 79.55 | 70.61 | 53.82 |
| 19.8 | 66.76 | 78.25 | 73.94 | 49.45 |
| Mean | **74.93** | **83.64** | **78.30** | **66.47** |
| Mean (>= 9.8 s) | **71.26** | **81.93** | **74.75** | **59.27** |

The bold cells mark the paper's highlighted means. Cross-trial replays of the
two shortest tiers stay in EVEWorld's favour under every seed pair, by 1.54
points on the overall PBench score at 3.8 s and 1.56 at 5.8 s.

**Cost grows with the horizon.** The length-scaling runs price the sweep: p50
latency rises from 8.17 s at 61 frames to 41.74 s at 317 frames and the
GPU-hour cost per 1,000 videos from 18.40 to 95.34, while peak memory stays
between 38.2 and 41.5 GiB per GPU and throughput between 442 and 505 frames
per minute. Longer rollouts therefore cost time and compute almost linearly,
with very little extra memory.

| Frames | p50 (s) | p95 (s) | Frames/min | Peak/GPU (GiB) | GPU-h/1k |
|---:|---:|---:|---:|---:|---:|
| 61 | 8.166 | 8.403 | 442.1 | 38.19 | 18.40 |
| 93 | 10.932 | 11.352 | 502.2 | 38.33 | 24.69 |
| 157 | 18.323 | 19.249 | 504.8 | 38.99 | 41.46 |
| 253 | 31.503 | 33.523 | 469.5 | 41.47 | 71.85 |
| 317 | 41.737 | 45.007 | 443.3 | 40.70 | 95.34 |

**The process diagnostics degrade with the horizon.** On 92 matched prompts
both GigaWorld-0 variants increase their shortcut severity between
3.8 s and 7.8 s (0.94 to 1.79 for Standard SFT, 1.17 to 1.97 for the
pretrained backbone) and the share of generations containing a shortcut rises
from 0.32 to 0.63 and 0.39 to 0.67 respectively, while Wan2.2-TI2V-5B at the
short horizon carries more of both (2.30 severity, 0.82 any-shortcut).

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
