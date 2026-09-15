# Cross-Model Baseline Downloads

Download scripts for the general image-to-video baselines compared against in
the paper (Table: overall DreamGenBench results). All are diffusers-native I2V
checkpoints from the Hugging Face Hub.

| Script | Model | HF repo id | Notes |
|---|---|---|---|
| `dl_wan22_ti2v_5b.sh` | Wan2.2-TI2V-5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | dense, ~10-12 GB |
| `dl_cogvideox15_5b_i2v.sh` | CogVideoX1.5-5B-I2V | `zai-org/CogVideoX1.5-5B-I2V` | DiT, ~10-11 GB |
| `dl_wan22_i2v_a14b.sh` | Wan2.2-I2V-A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | MoE (A14B), ~55-65 GB |
| `dl_cosmos_predict25_2b.sh` | Cosmos-Predict2.5-2B | `nvidia/Cosmos-Predict2.5-2B` | gated on HF (the paper's "Cosmos" row) |

Weights land under `${GAGI_ROOT}/xmodels/<model>/`. Generation and judging of
these baselines live in [`../xmodel_infer/`](../xmodel_infer/); see
[`../README.md`](../README.md) for the full pipeline.
