# Seed stability: repeats of generation and judgment

Two stochastic steps sit behind every reported number: the sampler draws a
video from an inference seed, and the instruction-following judge scores that
video with its own sampling. This study records both spreads — repeated
judgments of the same videos and matched inference seeds — plus a five-seed
replay of the transfer protocol, so a single run cannot be mistaken for a
trend.

## The record

`aggregate.py` consumes the per-seed replay records of one or more
checkpoints. Each checkpoint has a directory named after its training step,
holding one JSON file per inference seed:

```text
<root>/250/seed004.json
{"seed": 4, "mlr": 2.5, "qwen_if": 60.1, "gemini_if": 58.4}
```

`seed` is required and must agree with the file name; at least one of the
requested metrics must be present, and a metric that is missing or has a null
value in one record leaves that seed out of the mean for that metric alone.
For every step and metric the script prints the mean, the sample standard
deviation and the normal-approximation 95% confidence interval of the mean,
and it is strict about the layout: a directory that is not a step, a duplicate
seed or a seed that disagrees with its file name stops the run instead of
silently shifting the mean.

## How it was run

The checkpoint ablation replays the checkpoints at steps 150 and 250 over
seeds 1 to 70 at 30 inference steps, and the paper reports the step-250 raw
checkpoint. The summariser is a standard-library job over the recorded
numbers, so it runs without the package:

```bash
python experiments/analysis/seed_stability/aggregate.py \
    --root outputs/seed_sweep --steps 150 250
```

`--metrics` selects the keys to read, `--steps` restricts the summary to a
subset and `--out` writes the markdown table to a file. The companion repeat
protocol regenerates every clip three times, the repeat layout of the EWMBench
evaluation directories; its config is
[`seed_stability.yaml`](../../../configs/ablations/checkpoint/seed_stability.yaml).

## Reading the record

- [`aggregate.py`](aggregate.py) is the whole analysis: it groups the seed
  records by checkpoint step and metric, prints mean, sample SD and the 95%
  confidence interval of the mean as one long table, and appends the step and
  seed counts it read.
- The replay protocol and its checkpoints are in
  [`../../../docs/reproduction.md`](../../../docs/reproduction.md); the metric
  definitions are in [`../../../docs/evaluation.md`](../../../docs/evaluation.md).
- The eligible set behind the MLR columns is recomputed in
  [`../mlr_sensitivity/`](../mlr_sensitivity/), and the horizon tiers of the
  length sweep are laid out in [`../long_horizon/`](../long_horizon/).
