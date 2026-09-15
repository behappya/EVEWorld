#!/usr/bin/env python3
"""EVE 数据 · held-out 切分(消除训练=评测重叠,基准可信度底线)。
按 prompt 的任务模板分层,保证 eval 覆盖多种物体/动作,避免泄漏。
用法: python3 make_holdout.py --meta metadata.csv --out-dir splits --eval-frac 0.3 --seed 0
"""
import argparse, csv, json, os, re, random


def task_key(prompt):
    p = (prompt or "").lower()
    m = re.search(r"pick up (.*?) from (.*?) to (.*)", p)
    if m:
        return f"{m.group(2).strip()[:12]}->{m.group(3).strip()[:12]}"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not os.path.exists(a.meta):
        print(f"[warn] 找不到 {a.meta};请先下载 GR1 数据。跳过。")
        return
    rows = list(csv.DictReader(open(a.meta)))
    rnd = random.Random(a.seed)
    buckets = {}
    for r in rows:
        buckets.setdefault(task_key(r.get("text", "")), []).append(r)
    train, evl = [], []
    for k, items in buckets.items():
        rnd.shuffle(items)
        n_eval = max(1, int(len(items) * a.eval_frac)) if len(items) > 1 else 0
        evl += items[:n_eval]; train += items[n_eval:]
    os.makedirs(a.out_dir, exist_ok=True)
    for name, data in [("train", train), ("eval", evl)]:
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w") as f:
            for r in data:
                f.write(json.dumps({"file_name": r.get("file_name"), "text": r.get("text")}, ensure_ascii=False) + "\n")
    json.dump({"n_total": len(rows), "n_train": len(train), "n_eval": len(evl),
               "n_task_buckets": len(buckets), "eval_frac": a.eval_frac, "seed": a.seed},
              open(os.path.join(a.out_dir, "split_summary.json"), "w"), indent=2)
    print(f"切分完成: train={len(train)} eval={len(evl)} buckets={len(buckets)} -> {a.out_dir}")


if __name__ == "__main__":
    main()
