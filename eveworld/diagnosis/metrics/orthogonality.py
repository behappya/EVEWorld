#!/usr/bin/env python3
"""EVE P0 · 正交性分析(生死关卡的关键产物)。

把【过程分(1 - laziness 相关指标)】与【物理/画质分(PBench/VideoPhy/Qwen-IF)】做散点,
计算相关系数。目标:证明二者【弱相关】—— 即存在"物理分高但过程作弊"的视频,
说明 Model Laziness 与物理合理性正交,选题成立。

输入:
  --process   process_metrics.py 的 summary.json(取 per_video)
  --physics   一个 CSV/JSON,含 video -> 物理分(你已有的 PBench/PA/Qwen-IF 逐视频分)
输出:散点图 png + 相关系数 json。

用法:
  python3 orthogonality.py --process baseline.json --physics phys.csv \
      --phys-col pbench_domain --out-prefix outputs/ortho_baseline
"""
import argparse, json, csv, os
import numpy as np


def load_physics(path, col):
    d = {}
    if path.endswith(".json"):
        j = json.load(open(path))
        for k, v in (j.items() if isinstance(j, dict) else []):
            d[k] = float(v if not isinstance(v, dict) else v.get(col, np.nan))
    else:
        for row in csv.DictReader(open(path)):
            key = row.get("video") or row.get("file_name") or row.get("name")
            try:
                d[key] = float(row[col])
            except Exception:
                pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", required=True)
    ap.add_argument("--physics", required=True)
    ap.add_argument("--phys-col", default="score")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    proc = json.load(open(a.process))["per_video"]
    # 过程分:1 - lazy(逐视频),越高越忠实
    proc_score = {r["video"]: 1.0 - r["lazy"] for r in proc}
    phys = load_physics(a.physics, a.phys_col)

    keys = [k for k in proc_score if k in phys]
    if len(keys) < 5:
        print(f"[warn] 仅 {len(keys)} 条匹配,请确认 video 命名一致(过程 JSON 的 video 字段 vs physics 表的 key)")
    x = np.array([phys[k] for k in keys])          # 物理分
    y = np.array([proc_score[k] for k in keys])    # 过程分

    def corr(fn):
        try:
            from scipy import stats
            r = fn(x, y)
            return float(r[0]), float(r[1])
        except Exception:
            c = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
            return c, float("nan")
    try:
        from scipy import stats
        pear = corr(stats.pearsonr); spear = corr(stats.spearmanr)
    except Exception:
        pear = (float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan"), float("nan")); spear = pear

    # 关键子集:物理分高(前 50%)但过程作弊的比例
    if len(x):
        hi = x >= np.median(x)
        cheat_in_hi = float((y[hi] < 0.5).mean()) if hi.any() else float("nan")
    else:
        cheat_in_hi = float("nan")

    res = {"n": len(keys), "pearson_r": pear[0], "pearson_p": pear[1],
           "spearman_r": spear[0], "spearman_p": spear[1],
           "frac_cheat_among_high_physics": cheat_in_hi,
           "verdict_hint": "正交性成立(弱相关)" if abs(pear[0]) < 0.4 else "相关偏强,正交性存疑—需人工复核"}
    json.dump(res, open(a.out_prefix + ".json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, alpha=0.5, s=18)
        plt.xlabel(f"Physics/Quality score ({a.phys_col})")
        plt.ylabel("Process faithfulness (1 - lazy)")
        plt.title(f"Orthogonality: pearson r={pear[0]:.2f}")
        plt.tight_layout(); plt.savefig(a.out_prefix + ".png", dpi=140)
        print("wrote", a.out_prefix + ".png")
    except Exception as ex:
        print("[warn] 画图跳过(缺 matplotlib):", ex)


if __name__ == "__main__":
    main()
