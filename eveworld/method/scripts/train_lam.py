#!/usr/bin/env python3
"""EVE · LAD 潜在动作动力学模型 自监督预训(方案27 §四 I1)。

从离线 latent 缓存(encode_latents.py 产出)读 Wan VAE latent 序列, 自监督训练
inverse/forward 对, 学"真实单步转移流形"。自监督目标:
    L = ||z_{t+1} - forward(z_t, inverse(z_t, z_{t+1}))||
无需任何动作标注。

训完内建 go/no-go 验证: transition_error 在【真实转移】应低, 在【构造偷懒转移
(teleport/shuffle/freeze_jump)】应显著高 —— 这是 LAD 能用于 EAG 引导的前提。

需 GPU(VAE latent 已离线, LAD 本身小, 单卡足够) -> 走 kjob 或交互节点。
CPU 也能跑(latent 已缓存, 不过 VAE), 适合小规模冒烟。

用法:
  python eveworld/method/scripts/train_lam.py \
    --latents /data/.../eve_outputs/latents/gr1_real.pt \
    --out /data/.../eve_outputs/lam/lam_gr1.pt \
    --steps 8000 --window 12 --batch 8
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True, help="encode_latents.py 产出的 .pt")
    ap.add_argument("--out", default="lam_pretrained.pt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--window", type=int, default=12, help="每个样本取的连续帧数")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--action-dim", type=int, default=32)
    ap.add_argument("--codebook", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15, help="留作 held-out 验证的视频比例")
    a = ap.parse_args()

    import torch, torch.nn.functional as F
    from method.lam.latent_action_model import LatentActionModel

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lam] device={dev}", flush=True)

    blob = torch.load(a.latents, map_location="cpu")
    lats = blob["latents"]                       # list of (C,T,H,W)
    z_dim = blob.get("z_dim", lats[0].shape[0])
    print(f"[lam] {len(lats)} latent seqs, z_dim={z_dim}, example={tuple(lats[0].shape)}", flush=True)

    # train/val 划分(held-out 视频, 防止 LAD 只是背下训练转移)
    n_val = max(1, int(len(lats) * a.val_frac))
    val_lats = lats[:n_val]
    train_lats = lats[n_val:]
    print(f"[lam] train={len(train_lats)} val={len(val_lats)}", flush=True)

    def sample_batch(pool, bs, win):
        xs = []
        for _ in range(bs):
            zi = pool[torch.randint(len(pool), (1,)).item()]   # (C,T,H,W)
            T = zi.shape[1]
            if T <= win:
                seq = zi
                if T < win:  # pad by repeat last
                    seq = torch.cat([seq, seq[:, -1:].expand(-1, win - T, -1, -1)], dim=1)
            else:
                s = torch.randint(0, T - win + 1, (1,)).item()
                seq = zi[:, s:s + win]
            xs.append(seq)
        return torch.stack(xs).to(dev).float()                 # (B,C,win,H,W)

    lam = LatentActionModel(latent_ch=z_dim, action_dim=a.action_dim, codebook=a.codebook).to(dev)
    print(f"[lam] params={sum(p.numel() for p in lam.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(lam.parameters(), lr=a.lr)

    lam.train()
    for step in range(a.steps):
        z = sample_batch(train_lats, a.batch, a.window)
        zt, ztp = z[:, :, :-1], z[:, :, 1:]
        pred = lam.forward_dyn(zt, lam.inverse(zt, ztp))
        loss = F.mse_loss(pred, ztp)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"[lam] step {step} loss {loss.item():.5f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state_dict": lam.state_dict(), "z_dim": z_dim,
                "action_dim": a.action_dim, "codebook": a.codebook}, a.out)
    print(f"[lam] saved -> {a.out}", flush=True)

    # ---------------- go/no-go: transition_error 区分真实 vs 偷懒 ----------------
    print("\n[lam] === go/no-go: transition_error on held-out val ===", flush=True)
    lam.eval()

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
        return out

    with torch.no_grad():
        # 用整段 val 视频(逐条, 避免不同长度)
        # 聚合用 MAX 而非 mean: 偷懒=少数巨大非法跳变(复制段边界那一跳), 均值会被大量
        # 零运动复制帧稀释(诊断实测: mean 下 teleport/freeze_jump 反而<真实; max 全部可分)。
        te = {"real": [], "teleport": [], "shuffle": [], "freeze_jump": []}
        for zi in val_lats:
            z = zi.to(dev).float().unsqueeze(0)          # (1,C,T,H,W)
            if z.shape[2] < 4:
                continue
            te["real"].append(lam.transition_error(z).max().item())
            for k in ["teleport", "shuffle", "freeze_jump"]:
                te[k].append(lam.transition_error(corrupt(z, k)).max().item())
        import statistics as st
        real_m = st.mean(te["real"])
        print(f"  real (max-agg)  = {real_m:.4f}", flush=True)
        ok = True
        for k in ["teleport", "shuffle", "freeze_jump"]:
            m = st.mean(te[k])
            ratio = m / (real_m + 1e-8)
            higher = sum(1 for r, c in zip(te["real"], te[k]) if c > r) / max(1, len(te["real"]))
            print(f"  {k:12} = {m:.4f}  (x{ratio:.2f} real, {higher*100:.0f}% higher)", flush=True)
            if ratio < 1.3 or higher < 0.7:
                ok = False
        print(f"\n[lam] go/no-go: {'PASS ✓ (偷懒转移 max transition_error 显著高于真实, LAD 可用于EAG)' if ok else 'FAIL ✗ (区分不足, 需调 window/steps/结构)'}", flush=True)


if __name__ == "__main__":
    main()
