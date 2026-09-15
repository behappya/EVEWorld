#!/usr/bin/env python3
"""预计算 92 条权重图 -> weightmap_cache/<vid>.npy (24,30,48) float32。
训练时 transform 直接 load, 零 GDINO/anno 解析开销。CPU, 秒级。
"""
import os
import numpy as np
import t4g_weightmap as W

OUT = '/data/datasets/gagi/eve_v2_outputs/track4gen_probe/weightmap_cache'


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    stats = []
    for f in sorted(os.listdir(W.ANNO_DIR)):
        if not f[0].isdigit():
            continue
        vid = f[:-5]
        wm, seg = W.build_weightmap(W.load_anno(vid))
        assert wm.shape == (W.T_LAT, W.H_LAT, W.W_LAT), (vid, wm.shape)
        assert wm.min() >= W.W_BG - 1e-6 and wm.max() <= W.W_TRANS + 1e-6, (vid, wm.min(), wm.max())
        np.save(os.path.join(OUT, f'{vid}.npy'), wm.astype(np.float32))
        stats.append(float(wm.mean()))
        n += 1
    print(f'预计算 {n} 条 -> {OUT}')
    print(f'均值(归一化前): 中位{np.median(stats):.3f} 范围[{min(stats):.3f},{max(stats):.3f}]')


if __name__ == '__main__':
    main()
