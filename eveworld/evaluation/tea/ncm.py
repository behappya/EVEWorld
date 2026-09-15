#!/usr/bin/env python3
"""EVE · TEA 第 1 层主指标: 运动动力学偷懒签名 (Motion-Dynamics Laziness Signatures)。

方案 27 §三。核心: 偷懒不是单一现象, 而是一组"运动动力学签名"的集合, 每种对应一类
偷懒失败。我们用一组【对全局画质退化鲁棒、由数据验证过可分】的光流运动统计量度量它们,
不做接触/抓取/成功等语义事件检测(避开检测死穴)。

设计依据(诊断实验实测, 真实 GR1 vs 确定性破坏):
  统计量        real    shuffle  teleport  freeze_jump
  TELE(峰值跳变) 0.053   0.002    0.220     0.059     -> 抓 teleport(瞬移)
  JUMP(能量突变) 0.034   0.000    0.160     0.021     -> 抓 teleport
  ROUGH(能量粗糙) 0.449  0.420    0.544     1.983     -> 抓 freeze_jump(终态突现)
  STILL(静止帧率) 0.004  0.039    0.383     0.000     -> 抓 teleport(冻结段)
结论: teleport / freeze_jump(最严重的两类偷懒:瞬移、终态突现)可被清楚检出;
      shuffle 反而更"平滑", 不是好的偷懒代理 -> 主构造类型用 teleport/freeze_jump。

指标语义(全部: 越高越偷懒):
  TELE  单帧运动峰值相对中位的极端跳变比例          -> 物体瞬移/不连续跳变
  JUMP  帧间运动能量相对中位的极端突变比例          -> 突然出现/消失
  STILL 运动能量近零帧比例(相对自身中位)            -> 大段冻结(终态提前+空挥前的静止)
  ROUGH 运动能量一阶差分的相对粗糙度                -> 过程不连续/跳变
  LAZY  上述归一后的加权综合分(主报指标)

纯 CPU, 仅依赖 opencv-python + numpy。

用法:
  python3 ncm.py score    --video-dir DIR --out s.json --tag baseline [--crop 0.5,1.0]
  python3 ncm.py validate --video-dir REAL_DIR --out v.json [--limit 30]   # 真实 vs 破坏可分性
"""
import argparse, glob, json, os, sys
import numpy as np

try:
    import cv2
except Exception:
    print("[tea] 需要 opencv-python: pip install opencv-python-headless", file=sys.stderr)
    raise


# ----------------------- 视频读取 -----------------------
def read_frames(path, max_frames=48, resize=192, crop=None):
    """读视频 -> 灰度帧列表。crop=(lo,hi) 取水平区间(右半生成用 0.5,1.0)。"""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if crop is not None:
            w0 = f.shape[1]
            f = f[:, int(crop[0] * w0):int(crop[1] * w0)]
        h, w = f.shape[:2]
        s = resize / max(h, w)
        f = cv2.resize(f, (max(8, int(w * s)), max(8, int(h * s))))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def flow_seq(frames):
    """相邻帧 Farneback 光流运动幅值图序列。"""
    out = []
    for t in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(frames[t], frames[t + 1], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        out.append(np.sqrt((flow ** 2).sum(-1)))
    return out


# ----------------------- 视频级偷懒签名 -----------------------
def score_frames(frames, tele_k=3.0, jump_k=4.0, rough_ref=0.45,
                 still_frac=0.1, weights=(0.4, 0.2, 0.2, 0.2)):
    """从帧序列算偷懒签名分。返回 dict, 主分 LAZY。"""
    T = len(frames)
    if T < 4:
        return None
    mags = flow_seq(frames)
    peaks = np.array([m.max() for m in mags], dtype=np.float32)
    energy = np.array([m.sum() for m in mags], dtype=np.float32)

    peak_med = float(np.median(peaks) + 1e-6)
    energy_med = float(np.median(energy) + 1e-6)

    TELE = float((peaks > tele_k * peak_med).mean())
    er = energy / energy_med
    JUMP = float((er > jump_k).mean())
    STILL = float((energy < still_frac * energy_med).mean())
    d = np.abs(np.diff(energy))
    ROUGH = float(d.mean() / (energy.mean() + 1e-6))

    # 归一(ROUGH 相对真实基线 rough_ref; 其余本就是比例[0,1])
    rough_n = max(0.0, (ROUGH - rough_ref) / rough_ref)   # 超出真实粗糙度的相对超额
    w = weights
    LAZY = float(w[0] * TELE + w[1] * JUMP + w[2] * STILL + w[3] * min(1.0, rough_n))
    return {"n_frames": T, "LAZY": LAZY,
            "TELE": TELE, "JUMP": JUMP, "STILL": STILL, "ROUGH": ROUGH,
            "peak_med": peak_med, "energy_med": energy_med}


def score_video(path, crop=None, **kw):
    frames = read_frames(path, crop=crop)
    r = score_frames(frames, **kw)
    if r is None:
        return None
    r = {"video": os.path.basename(path), **r}
    return r


# ----------------------- 构造破坏(效度自测) -----------------------
def corrupt(frames, kind):
    """确定性时序破坏, 逐帧画质与真实完全相同(同批帧只改顺序/替换)。"""
    T = len(frames)
    rng = np.random.RandomState(0)
    out = frames[:]
    if kind == "shuffle":
        mid = list(range(int(T * 0.25), int(T * 0.85)))
        perm = mid[:]; rng.shuffle(perm)
        for s, d in zip(mid, perm):
            out[d] = frames[s]
    elif kind == "reverse":
        out = frames[::-1]
    elif kind == "teleport":
        k = max(1, int(T * 0.4))
        for i in range(k):
            out[i] = frames[-1]
    elif kind == "freeze_jump":
        k = int(T * 0.55)
        for i in range(k):
            out[i] = frames[0]
        for i in range(k, T):
            out[i] = frames[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--video-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tag", default="model")
    s.add_argument("--crop", default=None, help="水平裁剪, 右半生成用 0.5,1.0")
    s.add_argument("--limit", type=int, default=0)

    v = sub.add_parser("validate")
    v.add_argument("--video-dir", required=True, help="真实视频目录(锚点)")
    v.add_argument("--out", required=True)
    v.add_argument("--crop", default=None)
    v.add_argument("--limit", type=int, default=30)

    a = ap.parse_args()
    crop = tuple(float(x) for x in a.crop.split(",")) if a.crop else None
    vids = sorted(glob.glob(os.path.join(a.video_dir, "**", "*.mp4"), recursive=True))

    if a.cmd == "score":
        if a.limit:
            vids = vids[:a.limit]
        rows = []
        for i, vpath in enumerate(vids):
            r = score_video(vpath, crop=crop)
            if r:
                rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(vids)}")
        def m(key):
            return float(np.mean([r[key] for r in rows])) if rows else 0.0
        agg = {"tag": a.tag, "n": len(rows), "LAZY_mean": m("LAZY"),
               "TELE_mean": m("TELE"), "JUMP_mean": m("JUMP"),
               "STILL_mean": m("STILL"), "ROUGH_mean": m("ROUGH")}
        json.dump({"summary": agg, "per_video": rows}, open(a.out, "w"), indent=2)
        print(json.dumps(agg, indent=2, ensure_ascii=False)); print("wrote", a.out)
        return

    # validate
    if a.limit:
        vids = vids[:a.limit]
    kinds = ["real", "shuffle", "reverse", "teleport", "freeze_jump"]
    keys = ["LAZY", "TELE", "JUMP", "STILL", "ROUGH"]
    scores = {k: {kk: [] for kk in keys} for k in kinds}
    for i, vpath in enumerate(vids):
        base = read_frames(vpath, crop=crop)
        if len(base) < 4:
            continue
        for k in kinds:
            frames = base if k == "real" else corrupt(base[:], k)
            r = score_frames(frames)
            if r:
                for kk in keys:
                    scores[k][kk].append(r[kk])
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(vids)}")
    summary = {}
    real_lazy = np.array(scores["real"]["LAZY"])
    for k in kinds:
        summary[k] = {kk: float(np.mean(scores[k][kk])) for kk in keys}
        summary[k]["n"] = len(scores[k]["LAZY"])
        if k != "real":
            arr = np.array(scores[k]["LAZY"])
            if len(arr) == len(real_lazy) and len(real_lazy) > 1:
                diff = arr - real_lazy
                summary[k]["LAZY_gain_over_real"] = float(diff.mean())
                summary[k]["frac_higher_than_real"] = float((diff > 0).mean())
    json.dump({"summary": summary, "raw": scores}, open(a.out, "w"), indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False)); print("wrote", a.out)
    print("\n[效度判据] teleport/freeze_jump 的 LAZY_mean 应显著 > real,"
          " frac_higher_than_real 越接近 1 越好。shuffle 更平滑属预期(非偷懒代理)。")


if __name__ == "__main__":
    main()
