#!/usr/bin/env python3
"""EVE P0 · 事件时间轴抽取(纯 CPU,依赖 opencv-python + numpy)。

从一段操作视频抽取归一化事件时间(0~1)与逐帧信号,产出可供 metrics.py 使用的 JSON。

设计原则(对齐方案 §三):优先输出【顺序鲁棒】的信号,不追求精确坐标。
后续可把 detector 换成在人工标注集上训练的专用检测器(见 annotation/)。

用法:
  python3 event_extract.py --video a.mp4 --prompt "..." --out a.events.json
  python3 event_extract.py --video-dir DIR --meta meta.csv --out-dir OUT
"""
import argparse, json, os, glob, csv, sys
import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    print("[event_extract] 需要 opencv-python: pip install opencv-python-headless", file=sys.stderr)
    raise

N_BINS = 16  # 事件时间轴分箱数


def read_frames(path, max_frames=256, resize=256, crop=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        # crop: 只保留指定水平区间(如并排 [输入图|生成video],用 "0.5,1.0" 取右半生成部分)
        if crop is not None:
            w0 = f.shape[1]
            f = f[:, int(crop[0] * w0):int(crop[1] * w0)]
        h, w = f.shape[:2]
        s = resize / max(h, w)
        f = cv2.resize(f, (int(w * s), int(h * s)))
        frames.append(f)
    cap.release()
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def foreground_motion(frames):
    """逐帧前景运动量(相对首帧 + 相对前帧的差分,归一化)。返回 (T,) 数组。"""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    T = len(grays)
    motion = np.zeros(T, dtype=np.float32)
    for t in range(T):
        d0 = cv2.absdiff(grays[t], grays[0])
        dp = cv2.absdiff(grays[t], grays[max(0, t - 1)])
        d = np.maximum(d0, dp)
        motion[t] = float((d > 18).mean())
    return motion


def object_track_cheap(frames, prompt=None):
    """廉价目标定位:颜色词(若有)+ 运动前景连通域中心。返回 (T,2) 归一化中心与有效 mask。
    注意:这是 P0 baseline detector,已知不精确;顺序类度量对它鲁棒。
    可替换为 CoTracker/SAM2(见 track_cotracker 可选实现)。"""
    T = len(frames)
    H, W = frames[0].shape[:2]
    centers = np.full((T, 2), np.nan, dtype=np.float32)
    prev = None
    for t in range(T):
        g = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(g, cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY))
        _, m = cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        # 选面积大且离上一中心近的连通域
        cand = np.argsort(areas)[::-1][:5] + 1
        best, bestcost = None, 1e9
        for c in cand:
            cx, cy = cent[c]
            cost = -areas[c - 1] / (H * W)
            if prev is not None:
                cost += 3.0 * np.hypot((cx - prev[0]) / W, (cy - prev[1]) / H)
            if cost < bestcost:
                bestcost, best = cost, (cx, cy)
        if best is not None:
            centers[t] = [best[0] / W, best[1] / H]
            prev = best
    return centers


def extract(video, prompt=None, crop=None):
    frames = read_frames(video, crop=crop)
    if len(frames) < 4:
        return None
    T = len(frames)
    motion = foreground_motion(frames)
    centers = object_track_cheap(frames, prompt)
    # 物体位移速度(顺序鲁棒的关键信号)
    disp = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        if not np.isnan(centers[t]).any() and not np.isnan(centers[t - 1]).any():
            disp[t] = float(np.hypot(*(centers[t] - centers[t - 1])))
    # 事件启发(归一化时间 0~1)。阈值可后续用标注集校准。
    def first_ge(arr, thr):
        idx = np.where(arr >= thr)[0]
        return float(idx[0] / (T - 1)) if len(idx) else None
    mo_thr = max(0.02, float(np.nanpercentile(disp[disp > 0], 60)) if (disp > 0).any() else 0.02)
    events = {
        "t_action_start": first_ge(motion, max(0.02, motion.max() * 0.3)),
        "t_object_motion": first_ge(disp, mo_thr),
        "t_first_contact": None,   # 近似:物体显著位移前的最后一次手-物运动上升沿
        "t_success": None,         # 近似:物体到达并稳定(位移回落)
        "t_terminal_stable": None,
    }
    # contact 近似:object_motion 之前 motion 的显著上升点
    if events["t_object_motion"] is not None:
        om = int(events["t_object_motion"] * (T - 1))
        pre = motion[:max(1, om)]
        if len(pre):
            c = int(np.argmax(np.diff(pre))) if len(pre) > 1 else 0
            events["t_first_contact"] = float(c / (T - 1))
    # success/terminal 近似:位移最后一次超过阈值之后即视为稳定
    moving = np.where(disp >= mo_thr)[0]
    if len(moving):
        last = int(moving[-1])
        events["t_success"] = float(last / (T - 1))
        events["t_terminal_stable"] = float(min(T - 1, last + 1) / (T - 1))
    # teleport 信号:单帧位移跳变
    jump = float(disp.max()) if T > 1 else 0.0
    return {
        "video": os.path.basename(video), "prompt": prompt, "n_frames": T, "n_bins": N_BINS,
        "events": events,
        "signals": {"motion": motion.tolist(), "obj_disp": disp.tolist(),
                     "centers": np.nan_to_num(centers, nan=-1).tolist()},
        "max_jump": jump,
        "detector": "cheap_cv_baseline_v1",
        "crop": crop,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--prompt")
    ap.add_argument("--video-dir"); ap.add_argument("--meta")
    ap.add_argument("--out"); ap.add_argument("--out-dir")
    ap.add_argument("--crop", default=None, help="水平裁剪区间,如 0.5,1.0 取右半")
    a = ap.parse_args()
    crop = tuple(float(x) for x in a.crop.split(',')) if a.crop else None
    if a.video:
        r = extract(a.video, a.prompt, crop=crop)
        out = a.out or (a.video + ".events.json")
        json.dump(r, open(out, "w"), indent=2)
        print("wrote", out); return
    assert a.video_dir and a.out_dir, "需要 --video-dir 与 --out-dir"
    os.makedirs(a.out_dir, exist_ok=True)
    prompts = {}
    if a.meta and os.path.exists(a.meta):
        for row in csv.DictReader(open(a.meta)):
            prompts[row.get("file_name", row.get("video", ""))] = row.get("text", row.get("prompt", ""))
    vids = sorted(glob.glob(os.path.join(a.video_dir, "**", "*.mp4"), recursive=True))
    print(f"发现 {len(vids)} 个视频")
    for i, v in enumerate(vids):
        key = os.path.basename(v)
        r = extract(v, prompts.get(key), crop=crop)
        if r is None:
            continue
        json.dump(r, open(os.path.join(a.out_dir, key + ".events.json"), "w"), indent=2)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(vids)}")
    print("done ->", a.out_dir)


if __name__ == "__main__":
    main()
