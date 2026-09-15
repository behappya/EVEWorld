#!/usr/bin/env python3
"""EVE · 轨B: Qwen best-of-N 拒绝采样选优 + 【选优/评测分离】去循环(方案27 §二十 下一步)。

背景(方案27 §十八): 偷懒二维。运动型(瞬移/跳变)由 LAD/EAG 治(轨A); 语义型(缺阶段/
终态提前)只有 Qwen(VLM)能可靠判, 但 VLM 不可微、做不了训练/采样引导。轨B 出路 =
best-of-N 拒绝采样: 一个 prompt 生成 N 个候选(不同 seed), 用 Qwen 过程审计选偷懒分最低的。

★ 为什么必须分离(评审致命异议): 若用同一裁判 selseverity 既选优又汇报, 取 N 个带噪测量
的最小值天然低于均值(赢家诅咒/回归噪声下侧), 哪怕视频没真变好 -> "你在优化自己的指标"。
破法 = 用【裁判A(--judge a)】选优, 用【独立裁判B(--judge b, 不同问题分解+不同抽帧)】评测
被选中的视频。若 B 在被选集上也显著更低 = 真信号; 若回归到均值 = 噪声拟合。
同时报三条基线, 一图讲清:
  - 随机选期望(mean over N, 在裁判B下)  <- 诚实的单次基线(而非某个走运seed)
  - best-of-N (A选/B评)                  <- 分离后的真实改善
  - best-of-N (A选/A评)                  <- 自报(乐观上界), 与B评的差距 = 循环夸大量
  - oracle (B选/B评, min)                <- 若能直接用B选的理论上界

用法(分离评测):
  python best_of_n_select.py
    --cand-dirs .../seed6666/generated_only .../seed1234/generated_only ...
    --sel-csvs  .../tea_qwen/bon_seed6666_A.csv ...      # 裁判A(选优)
    --eval-csvs .../tea_qwen/bon_seed6666_B.csv ...      # 裁判B(独立评测)
    --seeds 6666 1234 2025 777 --out .../selection.json
如果只给 --sel-csvs(无 --eval-csvs), 退化为旧行为(A选A评, 仅供快速看, 不能作论文结论)。
"""
import argparse, csv, json, os
import statistics as st
from itertools import combinations


def load_qwen(csv_path):
    """video basename -> (severity, any_shortcut)。"""
    out = {}
    if not csv_path or not os.path.exists(csv_path):
        return out
    for r in csv.DictReader(open(csv_path)):
        name = os.path.basename(r.get("video_path", ""))
        try:
            sev = float(r.get("laziness_severity", 0))
        except Exception:
            sev = 0.0
        try:
            anysc = float(r.get("any_shortcut", 0))
        except Exception:
            anysc = 0.0
        out[name] = (sev, anysc)
    return out


def common_prompts(scores):
    prompts = set(scores[0].keys())
    for s in scores[1:]:
        prompts &= set(s.keys())
    return sorted(prompts)


def select_and_eval(sel, ev, prompts, N):
    """用 sel(裁判A)选每prompt最优候选, 用 ev(裁判B)读被选候选的分。
    返回该 N 下的多条聚合量。所有"降幅"都在【同一量表内】计算, 避免跨量表点值相减。"""
    rand_A, rand_B = [], []          # 随机选期望(A量表 / B量表)
    bon_ev_sev, bon_ev_any = [], []  # best-of-N(A选)在B量表的分
    self_ev = []                     # best-of-N(A选)在A量表的分(自报)
    oracle_ev = []                   # B选B评(理论上界)
    for p in prompts:
        sel_sev = [sel[i].get(p, (99, 1))[0] for i in range(N)]
        ev_sev = [ev[i].get(p, (99, 1))[0] for i in range(N)]
        ev_any = [ev[i].get(p, (99, 1))[1] for i in range(N)]
        rand_A.append(st.mean(sel_sev)); rand_B.append(st.mean(ev_sev))
        # best-of-N: 裁判A选severity最小(并列取A的any小)
        best_i = min(range(N), key=lambda i: (sel_sev[i], sel[i].get(p, (99, 1))[1]))
        bon_ev_sev.append(ev_sev[best_i]); bon_ev_any.append(ev_any[best_i])
        self_ev.append(sel_sev[best_i])          # 同一被选候选, 在A量表的分
        oracle_ev.append(min(ev_sev))
    rA, rB = st.mean(rand_A), st.mean(rand_B)
    selfA, evalB = st.mean(self_ev), st.mean(bon_ev_sev)
    # 同量表降幅: A量表内(自报/乐观) vs B量表内(独立)。二者差=循环夸大量(同尺度可比)。
    red_A = rA - selfA               # A量表: 随机A -> best-of-N(A评)
    red_B = rB - evalB               # B量表: 随机B -> best-of-N(B评, 独立确认)
    return {
        "random_sev_A": round(rA, 4), "random_sev_B": round(rB, 4),
        "bestofN_sev_selfA": round(selfA, 4),     # A选A评(自报, 乐观)
        "bestofN_sev_evalB": round(evalB, 4),     # A选B评(独立, 论文主报)
        "bestofN_any_evalB": round(st.mean(bon_ev_any), 4),
        "oracle_sev_evalB": round(st.mean(oracle_ev), 4),
        "reduction_A": round(red_A, 4),           # 自报降幅(同A量表)
        "reduction_B": round(red_B, 4),           # 独立降幅(同B量表)=论文主结论
        "circularity_gap": round(red_A - red_B, 4),  # 循环夸大量(同尺度差)
    }


def scaling_curve(sel, ev, prompts, Nmax, max_combos=35):
    """N=2..Nmax 的 scaling: 对每个 N, 枚举/采样 seed 子集组合, 取分离评测均值。
    组合数爆炸时截断到 max_combos(前 max_combos 个组合, 确定性可复现)。"""
    curve = []
    for n in range(2, Nmax + 1):
        combos = list(combinations(range(Nmax), n))
        if len(combos) > max_combos:
            combos = combos[:max_combos]
        rb, bs, rda = [], [], []
        for c in combos:
            sub_sel = [sel[i] for i in c]
            sub_ev = [ev[i] for i in c]
            r = select_and_eval(sub_sel, sub_ev, prompts, n)
            rb.append(r["random_sev_B"]); bs.append(r["bestofN_sev_evalB"]); rda.append(r["reduction_B"])
        curve.append({"N": n, "n_combos": len(combos),
                      "random_sev_B": round(st.mean(rb), 4),
                      "bestofN_sev_evalB": round(st.mean(bs), 4),
                      "reduction_B": round(st.mean(rda), 4)})
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dirs", nargs="+", required=True, help="N 个候选目录(各 generated_only)")
    ap.add_argument("--sel-csvs", nargs="+", required=True, help="裁判A(选优)csv, 与 cand-dirs 一一对应")
    ap.add_argument("--eval-csvs", nargs="+", default=None,
                    help="裁判B(独立评测)csv, 与 cand-dirs 一一对应; 缺省则退化为A选A评(不能作结论)")
    ap.add_argument("--seeds", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaling", action="store_true", help="额外输出 N=2..Nmax scaling 曲线")
    a = ap.parse_args()
    N = len(a.cand_dirs)
    assert len(a.sel_csvs) == N, "sel-csvs 数量须与 cand-dirs 一致"
    seeds = a.seeds or [str(i) for i in range(N)]

    sel = [load_qwen(c) for c in a.sel_csvs]
    separated = bool(a.eval_csvs)
    ev = [load_qwen(c) for c in a.eval_csvs] if separated else sel
    if separated:
        assert len(a.eval_csvs) == N, "eval-csvs 数量须与 cand-dirs 一致"

    prompts = common_prompts([{**s} for s in sel] + ([{**e} for e in ev] if separated else []))
    if not prompts:
        raise SystemExit("候选间无公共 prompt(文件名不一致?)。检查 generated_only 文件名。")

    res = select_and_eval(sel, ev, prompts, N)
    per_prompt = []
    for p in prompts:
        sel_sev = [sel[i].get(p, (99, 1))[0] for i in range(N)]
        ev_sev = [ev[i].get(p, (99, 1))[0] for i in range(N)]
        best_i = min(range(N), key=lambda i: (sel_sev[i], sel[i].get(p, (99, 1))[1]))
        per_prompt.append({"prompt": p, "best_seed": seeds[best_i],
                           "sel_severity": sel_sev, "eval_severity": ev_sev,
                           "chosen_eval_severity": ev_sev[best_i]})

    summary = {"N": N, "seeds": seeds, "n_prompts": len(prompts), "separated": separated, **res}
    summary["reduction_pct_B"] = (round(100 * res["reduction_B"] / res["random_sev_B"], 1)
                                  if res["random_sev_B"] > 0 else None)

    out = {"summary": summary, "selection": per_prompt}
    if a.scaling:
        out["scaling"] = scaling_curve(sel, ev, prompts, N)
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if separated:
        print(f"\n[主结论·独立裁判B] 随机选(B)={res['random_sev_B']} -> best-of-N(A选/B评)={res['bestofN_sev_evalB']} "
              f"(真实降 {res['reduction_B']:+.3f}, {summary['reduction_pct_B']}%); oracle上界={res['oracle_sev_evalB']}")
        print(f"[去循环] 同量表降幅: A(自报)={res['reduction_A']:+.3f} vs B(独立)={res['reduction_B']:+.3f}; "
              f"循环夸大量={res['circularity_gap']:+.3f} (越小越说明A选不是靠拟合A的噪声)")
        print("[判据] reduction_B 显著>0 (独立裁判也确认降) = best-of-N 真降语义偷懒, 非噪声拟合。")
    else:
        print(f"\n[警告] 未提供 --eval-csvs, 仅 A选A评 自报口径(乐观, 有循环)。论文结论须用分离评测。")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
