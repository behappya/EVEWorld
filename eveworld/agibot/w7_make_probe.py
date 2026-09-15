#!/usr/bin/env python3
"""WMB 适配 E1 前置:把训练 checkpoint 组装成 probe 模型目录(symlink)。

probe = 裸 checkpoint transformer + pretrain 底座的 vae/text_encoder(53 号同款口径)。
用法: python w7_make_probe.py --line vanilla --step 100
产出: wmb_adapt/probes/wmb_<line>_s<step>/{transformer,vae,text_encoder}
"""
import argparse
import glob
import os

GAGI = "/data/datasets/gagi"
LINES = {
    "vanilla": f"{GAGI}/wmb_adapt/t4g_wmb_vanilla/experiments",
    "apre": f"{GAGI}/wmb_adapt/t4g_wmb_apre/experiments",
}
PRETRAIN = f"{GAGI}/giga_world_0_video_pretrain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", choices=list(LINES), required=True)
    ap.add_argument("--step", type=int, required=True)
    a = ap.parse_args()
    pats = glob.glob(f"{LINES[a.line]}/**/checkpoint_*_step_{a.step}", recursive=True)
    if not pats:
        raise SystemExit(f"未找到 step {a.step} 的 checkpoint({LINES[a.line]})")
    ckpt = sorted(pats)[-1]
    tf = os.path.join(ckpt, "transformer")
    if not os.path.isfile(os.path.join(tf, "config.json")):
        raise SystemExit(f"checkpoint 缺 transformer/config.json: {tf}")
    probe = f"{GAGI}/wmb_adapt/probes/wmb_{a.line}_s{a.step}"
    os.makedirs(probe, exist_ok=True)
    for name, src in (("transformer", tf),
                      ("vae", f"{PRETRAIN}/vae"),
                      ("text_encoder", f"{PRETRAIN}/text_encoder")):
        dst = os.path.join(probe, name)
        if os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f"probe OK: {probe} (transformer -> {tf})")


if __name__ == "__main__":
    main()
