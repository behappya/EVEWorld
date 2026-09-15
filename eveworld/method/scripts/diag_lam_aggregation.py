#!/usr/bin/env python3
"""EVE · 诊断 LAD transition_error 的聚合方式(定位 go/no-go 失败根因)。

冒烟发现: teleport/freeze_jump 的 transition_error 均值反而 < 真实, 因为它们制造大量
"复制帧"(z_t->z_t 零运动转移, LAD 预测近乎零误差), 把均值稀释。偷懒的真实 signature 是
"少数几个巨大的非法跳变"(复制段边界那一跳), 应该用 max/高分位 抓, 而非均值。

本脚本: 加载已训 LAD + latent 缓存, 对真实 vs 各破坏, 对比 mean/max/p90/top3 多种聚合,
找出能把偷懒转移与真实分开的聚合。不重训, 秒级出结果。

用法(需GPU或CPU均可, latent已缓存):
  python eveworld/method/scripts/diag_lam_aggregation.py \
    --lam /data/.../eve_outputs/lam/lam_gr1.pt \
    --latents /data/.../eve_outputs/latents/gr1_real.pt
"""
import argparse, os, sys, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--val-frac", type=float, default=0.3)
    a = ap.parse_args()

    import torch
    from method.lam.latent_action_model import LatentActionModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.lam, map_location=dev)
    lam = LatentActionModel(latent_ch=ck["z_dim"], action_dim=ck["action_dim"], codebook=ck["codebook"]).to(dev)
    lam.load_state_dict(ck["state_dict"]); lam.eval()

    blob = torch.load(a.latents, map_location="cpu")
    lats = blob["latents"]
    n_val = max(2, int(len(lats) * a.val_frac))
    val = lats[-n_val:]        # 用后段做验证
    print(f"[diag] {len(val)} val latents, z_dim={ck['z_dim']}", flush=True)

    def corrupt(z, kind):
        B, C, T, H, W = z.shape
        out = z.clone()
        if kind == "teleport":
            k = max(1, int(T * 0.4)); out[:, :, :k] = z[:, :, -1:].expand(-1, -1, k, -1, -1)
        elif kind == "shuffle":
            out = z[:, :, torch.randperm(T)]
        elif kind == "freeze_jump":
            k = int(T * 0.55)
            out[:, :, :k] = z[:, :, :1].expand(-1, -1, k, -1, -1)
            out[:, :, k:] = z[:, :, -1:].expand(-1, -1, T - k, -1, -1)
        elif kind == "teleport_single":
            # 更贴近真实偷懒: 只在中间插一个瞬移(单帧跳到终态再跳回), 不制造大量复制段
            m = T // 2
            out[:, :, m] = z[:, :, -1]
        return out

    # 多种聚合: 每种破坏 vs 真实, 逐视频配对
    aggs = {
        "mean": lambda te: te.mean().item(),
        "max": lambda te: te.max().item(),
        "p90": lambda te: te.quantile(0.9).item(),
        "top3mean": lambda te: te.topk(min(3, te.numel())).values.mean().item(),
    }
    kinds = ["teleport", "shuffle", "freeze_jump", "teleport_single"]
    # 收集
    data = {ag: {"real": [], **{k: [] for k in kinds}} for ag in aggs}
    with torch.no_grad():
        for zi in val:
            z = zi.to(dev).float().unsqueeze(0)
            if z.shape[2] < 4:
                continue
            te_real = lam.transition_error(z).flatten()
            for ag, fn in aggs.items():
                data[ag]["real"].append(fn(te_real))
            for k in kinds:
                te = lam.transition_error(corrupt(z, k)).flatten()
                for ag, fn in aggs.items():
                    data[ag][k].append(fn(te))

    print(f"\n{'agg':10}{'kind':16}{'val':>10}{'x_real':>9}{'%higher':>9}")
    print("-" * 55)
    for ag in aggs:
        real_m = st.mean(data[ag]["real"])
        print(f"{ag:10}{'real':16}{real_m:>10.4f}")
        for k in kinds:
            m = st.mean(data[ag][k])
            higher = sum(1 for r, c in zip(data[ag]["real"], data[ag][k]) if c > r) / max(1, len(data[ag]["real"]))
            flag = " ✓" if (m / (real_m + 1e-8) >= 1.3 and higher >= 0.7) else ""
            print(f"{'':10}{k:16}{m:>10.4f}{m/(real_m+1e-8):>9.2f}{higher*100:>8.0f}%{flag}")
    print("\n[判据] 找 x_real>=1.3 且 %higher>=70 的聚合, 作为 LAD 偷懒分的聚合方式。")


if __name__ == "__main__":
    main()
