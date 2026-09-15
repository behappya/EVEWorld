#!/usr/bin/env python3
"""EVE P0 · 过程忠实性度量(第 1 层几何度量,纯 CPU)。

输入:event_extract.py 产出的 *.events.json 目录。
输出:每视频指标 + 汇总。这些是论文 C1 的主指标。

指标(对齐方案 §四):
  PCR   Premature Completion Rate:成功早于释放/必要过程完成
  MBC   Motion-Before-Contact:物体在接触前显著位移
  TELE  Teleport Rate:单帧位移跳变过大
  COMP  Process Completeness:必要事件齐全
  CBE   Cause-before-Effect Accuracy:因果事件对顺序正确

用法:
  python3 process_metrics.py --events-dir DIR --out summary.json --tag baseline
"""
import argparse, json, glob, os
import numpy as np

TELEPORT_THR = 0.15   # 单帧归一化位移超过则判跳变(可用标注集校准)


def per_video(ev):
    e = ev["events"]
    disp = np.array(ev["signals"]["obj_disp"], dtype=np.float32)
    res = {"video": ev["video"]}
    # Teleport:任意单帧位移超阈
    res["teleport"] = int(ev.get("max_jump", 0.0) >= TELEPORT_THR)
    # Motion-before-contact:物体运动早于接触
    tc, tm = e.get("t_first_contact"), e.get("t_object_motion")
    res["mbc"] = int(tc is not None and tm is not None and tm < tc - 1e-6)
    # Premature completion:成功早于终态稳定所需过程 —— 近似:success 与 motion 结束几乎同时
    #   偷懒典型:物体没经过完整搬运就"到位"。用 success 时间是否过早近似。
    ts = e.get("t_success")
    res["premature"] = int(ts is not None and ts < 0.5 and disp[: max(1, int(ts * (len(disp) - 1)))].sum() < 0.3 * (disp.sum() + 1e-6))
    # Completeness:必要事件是否齐全(action_start/object_motion/success 都检出)
    need = ["t_action_start", "t_object_motion", "t_success"]
    res["complete"] = int(all(e.get(k) is not None for k in need))
    # Cause-before-Effect:接触先于运动、运动先于成功
    order_ok = 1
    seq = [("t_first_contact", "t_object_motion"), ("t_object_motion", "t_success")]
    for a, b in seq:
        if e.get(a) is not None and e.get(b) is not None and e[a] > e[b] + 1e-6:
            order_ok = 0
    res["cbe"] = order_ok
    # 综合 laziness 标记(任一严重信号)
    res["lazy"] = int(res["teleport"] or res["mbc"] or res["premature"] or (not res["complete"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="model")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.events_dir, "*.events.json")))
    rows = []
    for f in files:
        try:
            rows.append(per_video(json.load(open(f))))
        except Exception as ex:
            print("skip", f, ex)
    n = max(1, len(rows))
    agg = {
        "tag": a.tag, "n_videos": len(rows),
        "PCR_premature": sum(r["premature"] for r in rows) / n,
        "MBC_motion_before_contact": sum(r["mbc"] for r in rows) / n,
        "TELE_teleport": sum(r["teleport"] for r in rows) / n,
        "COMP_completeness": sum(r["complete"] for r in rows) / n,
        "CBE_cause_before_effect": sum(r["cbe"] for r in rows) / n,
        "LAZINESS_RATE": sum(r["lazy"] for r in rows) / n,
    }
    json.dump({"summary": agg, "per_video": rows}, open(a.out, "w"), indent=2)
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
