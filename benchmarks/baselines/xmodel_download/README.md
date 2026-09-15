# 跨模型下载脚本（EVE Model-Laziness 跨模型对比）

为跨模型验证「Model Laziness 是领域共性」下载**原生 image→video** 的视频生成模型权重，
与 GigaWorld-0 (2B) 做公平对比。选型覆盖 4 厂商 × 3 规模 × 4 架构，全部 diffusers 原生 I2V。

## 模型清单（repo id 已实测存在 + 连通）

| 脚本 | 模型 | repo id | 厂商/架构 | 规模 | gated | 体积(估) |
|---|---|---|---|---|---|---|
| `dl_wan22_ti2v_5b.sh` | Wan2.2-TI2V-5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 阿里/dense | 5B | 否 | ~10-12GB |
| `dl_cogvideox15_5b_i2v.sh` | CogVideoX1.5-5B-I2V | `zai-org/CogVideoX1.5-5B-I2V` | 智谱/DiT | 5B | 否(实测) | ~10-11GB |
| `dl_wan22_i2v_a14b.sh` | Wan2.2-I2V-A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 阿里/MoE | 27B/激活14B | 否 | ~55-65GB |
| `dl_cosmos_predict25_2b.sh` | Cosmos-Predict2.5-2B | `nvidia/Cosmos-Predict2.5-2B` | NVIDIA/世界模型 | 2B | **是(auto)** | 未核 |

- 权重落在 `/data/datasets/gagi/xmodels/<model>/`（`/data` 余量 100T+）。
- 用 `giga_world1` conda env 内的 `hf`（1.23.0）。`hf download` 断点续传，可重复执行。

## 并行执行（多开终端，每个终端一个）

```bash
cd giga-world-0/scripts/xmodel_download

# 终端 1（最小，建议先跑）
bash dl_wan22_ti2v_5b.sh

# 终端 2
bash dl_cogvideox15_5b_i2v.sh

# 终端 3（大，最慢）
bash dl_wan22_i2v_a14b.sh

# 终端 4（gated，见下）
bash dl_cosmos_predict25_2b.sh
```

三个免 gated 的可直接并行开跑。

## Cosmos 需要先授权（唯一 gated）

`nvidia/Cosmos-Predict2.5-2B` 是 gated（NVIDIA Open Model License），下载前二选一：

1. 浏览器打开 https://huggingface.co/nvidia/Cosmos-Predict2.5-2B 点同意，然后：
   ```bash
   source /home/jovyan/miniconda/etc/profile.d/conda.sh && conda activate giga_world1
   hf auth login   # 粘贴 https://huggingface.co/settings/tokens 的 token
   bash dl_cosmos_predict25_2b.sh
   ```
2. 或直接给脚本传 token（同样需已在网页点过同意）：
   ```bash
   HF_TOKEN=hf_xxxxx bash dl_cosmos_predict25_2b.sh
   ```

## 覆盖参数（可选）

每个脚本都可用环境变量覆盖：
- `DST=/some/path` 改下载目录
- `REPO_ID=...` 换 repo（一般不用）
- `HF_HUB_ENABLE_HF_TRANSFER=1` 开加速（部分环境不稳，默认关）
- `HF_TOKEN=hf_xxx` 传 token

## 下载完成后

各脚本末尾会自动 `verify_paths` 校验关键子目录（`model_index.json` / `transformer` / `vae`）并打印体积。
若中断，重跑同一脚本会断点续传。全部就绪后，下一步是写各模型的 DreamGen 92 条批量 I2V 推理脚本
（Wan/CogVideoX 走 diffusers pipeline；Cosmos 走其 Video2World 入口），输出对齐 GW-0 的
`generated-only` 目录布局以便统一做偷懒度量。

## 备选：加第 5 家（腾讯）

如需再补一个厂商挡先进性质疑，可加 `tencent/HunyuanVideo-I2V`（~13B）。
gating 状态未核，需自行上 HF 页面确认，确认后照上面任一脚本改 `REPO_ID`/`DST` 即可。
