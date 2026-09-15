#!/usr/bin/env python3
"""EVE · 离线把操作视频编码成 Wan VAE latent 缓存(.pt), 供 LAD 自监督预训/EAG 复用。

为什么离线: LAD 训练/调参会反复用同一批 latent; 一次性编码成缓存后, 训练不再过 VAE,
极快, 且 EAG/诊断阶段可复用同一缓存。latent 归一化与 GigaWorld-0 trainer 完全一致
(latents_mean/std), 保证 LAD 学到的流形与 backbone 采样时的 latent 同分布。

必须用 train venv 跑(transformers==5.11.0, 有新版 Wan VAE); 需 GPU -> 走 kjob 或交互节点。

用法:
  TRAIN_PYTHON=/data/.../giga_world_train_venv/bin/python
  $TRAIN_PYTHON eveworld/method/scripts/encode_latents.py \
    --video-dir /data/.../gr1_finetune_data/raw_hf/gr1 \
    --out /data/.../eve_outputs/latents/gr1_real.pt \
    --num-frames 49 --height 480 --width 768
"""
import argparse, glob, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae-path", default="/data/datasets/gagi/giga_world_0_video_pretrain/vae")
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="float16")
    a = ap.parse_args()

    import torch
    import cv2
    from diffusers.models import AutoencoderKLWan

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("[warn] 无 GPU, VAE 编码会很慢/可能 OOM。建议走 kjob。", file=sys.stderr)
    dtype = getattr(torch, a.dtype)

    print(f"[encode] loading VAE from {a.vae_path}", flush=True)
    vae = AutoencoderKLWan.from_pretrained(a.vae_path).to(dev, dtype).eval()
    z_dim = vae.config.z_dim
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(dev, dtype)
    lat_std_inv = (1.0 / torch.tensor(vae.config.latents_std)).view(1, z_dim, 1, 1, 1).to(dev, dtype)

    vids = sorted(glob.glob(os.path.join(a.video_dir, "**", "*.mp4"), recursive=True))
    if a.limit:
        vids = vids[:a.limit]
    print(f"[encode] {len(vids)} videos -> {a.out}", flush=True)

    def read_video(path):
        cap = cv2.VideoCapture(path)
        fs = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.resize(f, (a.width, a.height))
            fs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(fs) < 2:
            return None
        idx = np.linspace(0, len(fs) - 1, a.num_frames).astype(int)
        vid = np.stack([fs[i] for i in idx]).astype(np.float32) / 127.5 - 1.0  # (T,H,W,3) in [-1,1]
        return torch.from_numpy(vid).permute(3, 0, 1, 2)[None]                  # (1,3,T,H,W)

    latents, names = [], []
    for i, vpath in enumerate(vids):
        x = read_video(vpath)
        if x is None:
            print(f"  skip {os.path.basename(vpath)} (too short)", flush=True)
            continue
        x = x.to(dev, dtype)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample()          # (1,z,T',H',W')
            lat = (lat - lat_mean) * lat_std_inv               # 与 trainer 同归一化
        latents.append(lat.squeeze(0).cpu())
        names.append(os.path.basename(vpath))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(vids)}  latent shape={tuple(latents[-1].shape)}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"latents": latents, "names": names,
                "num_frames": a.num_frames, "height": a.height, "width": a.width,
                "z_dim": z_dim}, a.out)
    print(f"[encode] saved {len(latents)} latents -> {a.out}", flush=True)
    if latents:
        print(f"[encode] example latent shape = {tuple(latents[0].shape)}", flush=True)


if __name__ == "__main__":
    main()
