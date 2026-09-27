# Table 5 — Cross-backbone evaluation and ablation on RoboTwin

The cross-backbone arm: the FlowWAM backbone, a rank-32 LoRA fine-tune over the 2,250
RoboTwin training episodes, and 250 held-out episodes from episodes 45-49 of the 50 tasks.
Every variant is generated at 40 denoising steps and CFG 5.0 with the robot-only flow
condition, so the row differences come from the supervision terms alone.

| Variant | IGR | TIA | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Flow-EPE ↓ | MLR (%) ↓ |
|---|---|---|---|---|---|---|---|
| FlowWAM | | | 12.218 | 0.748 | 0.383 | 3.033 | 52.05 |
| Standard SFT | | | 11.687 | 0.735 | 0.414 | 2.828 | 47.89 |
| + IGR | ✓ | | 12.544 | 0.770 | 0.380 | 2.487 | 40.03 |
| + TIA | | ✓ | **13.445** | **0.776** | **0.333** | **1.850** | **22.54** |
| **EVEWorld** | ✓ | ✓ | 12.765 | 0.769 | 0.365 | 2.207 | 35.21 |

## Metrics

- **PSNR (dB)**, **SSIM** — frame-level visual fidelity against the aligned RGB ground truth,
  higher is better.
- **LPIPS**, **Flow-EPE** — perceptual distance and optical-flow endpoint error, lower is
  better; Flow-EPE is computed against the flow of the ground-truth clip.
- **MLR (%)** — Model Laziness Rate, lower is better, over the same occlusion-aware
  persistence rule as [Table 1](table1_dreamgen.md), averaged over the 250 held-out episodes.

Bold marks the best value in each column, as in the paper. The two single-component arms trade
perceptual fidelity against instance consistency: TIA gives the sharpest frames and the lowest
MLR on this backbone, IGR improves both against the backbone and the SFT control, and the joint
model lands between the two arms while staying ahead of FlowWAM and Standard SFT everywhere.

## Reproduction

`bash scripts/reproduce/table5_robotwin.sh` runs the five variants and writes these rows; the
adapter settings, the episode accounting and the expected numbers are in
[`../../../docs/reproduction.md`](../../../docs/reproduction.md).
