#!/usr/bin/env python3
"""EVE P0 · 人工标注工具(CLI,纯 CPU)。

给一批视频做过程忠实性标注,产出 ground-truth,用于:
  (1) 校准自动度量(算相关性 -> 可测性判据)
  (2) 训练专用事件检测器(可选)

标注维度(逐视频,y/n/skip):
  premature 提前完成 | mbc 先动后触 | teleport 瞬移 |
  incomplete 缺必要阶段 | not_executable 整体不可执行
并可选记录关键事件帧号。

用法:
  python3 annotate.py --video-dir DIR --out labels.jsonl [--resume]
产物 labels.jsonl 每行一个视频的标注。双人各跑一次 -> 用 agreement.py 算 κ。
"""
import argparse, json, os, glob


QS = [("premature", "是否在必要操作完成前任务就已成功(提前完成)?"),
      ("mbc", "物体是否在被接触前就开始移动(先动后触)?"),
      ("teleport", "物体是否发生瞬移/不连续跳变?"),
      ("incomplete", "是否缺少必要阶段(接触/抓取/搬运/释放)?"),
      ("not_executable", "整体过程是否物理上不可执行?")]


def ask(q):
    while True:
        v = input(f"  {q} [y/n/s跳过]: ").strip().lower()
        if v in ("y", "n", "s"):
            return {"y": 1, "n": 0, "s": None}[v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    done = set()
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["video"])
            except Exception:
                pass
    vids = sorted(glob.glob(os.path.join(a.video_dir, "**", "*.mp4"), recursive=True))
    fout = open(a.out, "a")
    print(f"共 {len(vids)} 条,已标 {len(done)} 条。逐条播放请用外部播放器打开路径。\n")
    for v in vids:
        key = os.path.basename(v)
        if key in done:
            continue
        print(f"\n=== {key} ===\n  路径: {v}")
        rec = {"video": key, "path": v}
        for name, q in QS:
            rec[name] = ask(q)
        rec["lazy_any"] = int(any(rec[n] == 1 for n, _ in QS))
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        print("  已记录。")
    print("\n标注完成 ->", a.out)


if __name__ == "__main__":
    main()
