# Table 4 — EWMBench after AgiBot post-training

The cross-distribution arm: both AgiBot-trained models start from the GigaWorld-0
video-pretrain checkpoint, use the same AgiBot clips and the same optimization budget, and
are scored by the official EWMBench suite on 777 AgiBot clips generated three times each
(generation seeds 42, 43, 44).

| Model | Motion ↑ | Semantics ↑ | DYN ↑ | HSD ↑ | nDTW ↑ | Scene ↑ | Overall ↑ |
|---|---|---|---|---|---|---|---|
| GigaWorld-0 (pretrained) | 50.30 | 2.2497 | 7.19 | 23.29 | 19.83 | **89.58** | 3.6486 |
| Standard SFT | 61.51 | 2.2477 | 14.88 | **24.35** | **22.28** | 84.38 | 3.7066 |
| **EVEWorld** | **63.65** | **2.2524** | **17.55** | 24.12 | 21.97 | 86.37 | **3.7525** |

## Metrics

All seven columns are higher-is-better.

- **Motion**, **Semantics**, **Overall** — the official EWMBench aggregate scores: motion
  quality, semantic alignment with the source clip, and their combination.
- **DYN**, **HSD**, **nDTW**, **Scene** — component metrics of the suite: dynamic degree,
  hand-object interaction, trajectory similarity and scene consistency. The scorer and
  `configs/eval/ewmbench.yaml` key the fourth component as `hsr`; the paper's tables label it
  HSD and the two names denote the same score.

The gains concentrate in Motion, DYN and Overall, the signature expected from supervising
target evolution; the Semantics and Scene columns sit close together across the three models
and are best read next to the Motion column rather than on their own.

## Reproduction

`bash scripts/reproduce/table4_ewmbench.sh` generates the 2,331 clips, lays them out for the
official scorer and writes these rows. The `eval_layout` tree, the AgiBot data root and the
seeds are documented in [`../../../docs/evaluation.md`](../../../docs/evaluation.md) and
[`../../../docs/reproduction.md`](../../../docs/reproduction.md).
