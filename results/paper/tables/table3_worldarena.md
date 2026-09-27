# Table 3 — WorldArena 1.0 transfer

WorldArena 1.0, 1,000 prompts, evaluated zero-shot from the step-250 DreamGenBench
checkpoint of each arm: no WorldArena training data enters either run, so the table measures
how the DreamGen post-training transfers to a different prompt distribution and a different
generation setting (480 x 768, 93 frames, 30 denoising steps, CFG 7.0).

| Model | Overall ↑ | MLR (%) ↓ |
|---|---|---|
| Standard SFT | 53.95 | 27.39 |
| **EVEWorld** | **56.76** | **13.38** |

## Metrics

- **Overall** — higher is better, the equal-weight mean of the eight local core metrics:
  Image Quality, Aesthetic Quality, Dynamic Degree, Flow Score, Motion Smoothness, Subject
  Consistency, Background Consistency and Photometric Consistency. The per-dimension columns
  and the zero-shot general I2V baselines sit beside this table in the paper.
- **MLR (%)** — Model Laziness Rate, lower is better, over the same occlusion-aware
  persistence rule as [Table 1](table1_dreamgen.md). 157 of the 1,000 prompts are eligible
  (coverage 19.22%), and the paired comparison uses that shared eligible set rather than the
  per-model subsets, so both arms are scored on 157 prompts. The denominator of the rate is
  the 157 eligible prompts, not the full manifest: 157 / 1,000 would read 15.70%.

## Reproduction

`bash scripts/reproduce/table3_worldarena.sh` runs both arms and writes these rows. The
input checkpoint, the sharding flags and the compute estimate are in
[`../../../docs/reproduction.md`](../../../docs/reproduction.md); the prompt manifest and
its eligible set are in
[`../../../data/splits/worldarena/eval.txt`](../../../data/splits/worldarena/eval.txt) and
[`../../../data/metadata/worldarena/eligible_ids.json`](../../../data/metadata/worldarena/eligible_ids.json).
