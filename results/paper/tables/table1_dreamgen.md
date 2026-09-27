# Table 1 — DreamGenBench, overall comparison

DreamGenBench with the GigaWorld-0 backbone. The general video models are evaluated
zero-shot; Standard SFT and EVEWorld are post-trained on the 92 DreamGen clips with the same
initialization and the same 250-step budget. MLR is computed on the shared eligible set
`U_63` for every model.

| Method | MLR (%) ↓ | Qwen-IF (%) ↑ | Gemini-IF (%) ↑ |
|---|---|---|---|
| CogVideoX1.5-5B-I2V | 28.57 | 38.89 | 5.56 |
| Wan2.2-TI2V-5B | 17.46 | 38.89 | 10.32 |
| Wan2.2-I2V-A14B | 11.11 | 64.29 | 15.87 |
| Cosmos-Predict2-2B | 14.29 | 62.70 | 24.60 |
| GigaWorld-0 | 12.70 | 79.37 | 60.19 |
| Standard SFT | 11.11 | 73.81 | 53.57 |
| **EVEWorld** | **1.59** | **80.16** | **60.85** |

## Metrics

- **MLR (%)** — Model Laziness Rate, lower is better. A clip is marked when the adjusted
  instance count of the tracked target differs from its first-frame count at two consecutive
  sampled timestamps; MLR is the share of eligible clips marked this way, in percent. A
  missing instance is exempt at a timestamp when the target is occluded by the robot beyond
  `tau_occ`, and an exemption resets the consecutive counter. DreamGenBench has 126 clips, of
  which 63 are eligible (`U_63`), so the rate is computed on 63 items for every model.
- **Qwen-IF (%)** — instruction-following accuracy of a rollout, judged by Qwen2.5-VL-7B
  against the instruction and its parsed requirement, in percent, higher is better.
- **Gemini-IF (%)** — the same quantity judged independently by Gemini, in percent, higher is
  better, and per category in [Table 6](table6_ablation.md).

EVEWorld reduces MLR from 11.11% to 1.59% against Standard SFT, an 85.7% relative reduction,
while instruction following improves on both judges.

## Reproduction

`bash scripts/reproduce/table1_dreamgen.sh` writes these rows; the protocol is described in
[`../../../docs/evaluation.md`](../../../docs/evaluation.md) and the run settings are in
`configs/eval/mlr/dreamgen.yaml` and `configs/eval/instruction_following/`.
