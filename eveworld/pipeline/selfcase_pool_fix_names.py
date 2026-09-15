#!/usr/bin/env python3
"""重考池文件名修复: 分片生成的本地序号前缀 -> 全局 request_id 前缀。

generate_eag 的文件名 = f"{枚举序号n}_{slug(prompt)[:80]}.mp4"。分片(quarter) worker
的 n 是分片内本地序号, 与 selfcase_mine_dispatch 的 int(前缀)=全局 idx 假设冲突。
slug 截断到 80 字符存在撞名(gr92 里 8 对), 不能只按 slug 匹配——用 (slug, 本地序号)
联合匹配: 分片 q=d[q::4] 的本地 n 对应全局 g=q+4n, 即 g//4==n。

匹配次序: ①文件名已等于规范名(早期全量 run 残留/首分片 0 号) -> 直接认领;
②其余文件按 slug+g//4==n 认领未被占的规范名 -> 重命名;
③认领撞车(同一规范名两个文件 = 早期残留+分片重生成, 同 seed 确定性采样内容相同)
  -> 保留规范名者删其余。generated_only 与 side_by_side 同步处理。

用法: python selfcase_pool_fix_names.py <待修池根目录> <规范池根目录> [--dry-run]
"""
import argparse
import os
import sys
from glob import glob


def parse(name):
    """'{n}_{slug}.mp4' -> (n, slug)"""
    stem = name[:-4] if name.endswith(".mp4") else name
    head, _, rest = stem.partition("_")
    assert head.isdigit(), f"非法前缀: {name}"
    return int(head), rest


def fix_dir(cur_dir, canon_dir, dry, shards=4, pure_sharded=False):
    canon_names = sorted(os.path.basename(p)
                         for p in glob(os.path.join(canon_dir, "*.mp4")))
    canon_set = set(canon_names)
    # slug -> [(global g, 规范名)]
    canon_by_slug = {}
    for b in canon_names:
        g, s = parse(b)
        canon_by_slug.setdefault(s, []).append((g, b))

    files = sorted(glob(os.path.join(cur_dir, "*.mp4")))
    claimed = {}   # 规范名 -> 认领文件路径
    dupes = []     # 多余文件(已有认领者)
    unknown = []
    # ① 名字已是规范名的直接认领(仅限混有全量 run 残留的池;
    #    纯分片池必须跳过——撞名对的本地序号可能恰好撞上另一规范名, 如 half 池的 {7,14})
    rest = []
    for p in files:
        b = os.path.basename(p)
        if not pure_sharded and b in canon_set:
            claimed[b] = p
        else:
            rest.append(p)
    # ② 其余按 (slug, g//4==n) 认领
    for p in rest:
        n, s = parse(os.path.basename(p))
        cands = [(g, b) for g, b in canon_by_slug.get(s, []) if g // shards == n]
        if len(cands) != 1:
            unknown.append(p)
            continue
        _, b = cands[0]
        if b in claimed:
            dupes.append(p)      # 早期残留已占位, 分片重生成的是复制品
        else:
            claimed[b] = p

    renamed = deleted = kept = 0
    for p in dupes:
        print(f"  DEL {os.path.basename(p)}")
        if not dry:
            os.remove(p)
        deleted += 1
    # 两阶段改名: 先全部挪到临时名再落最终名——撞名组的目标名可能正是
    # 另一个待改文件的当前名(如 half 池 {7,14}), 直接改会静默覆盖(13 号坑)。
    pending = [(b, p) for b, p in claimed.items() if os.path.basename(p) != b]
    kept += sum(1 for b, p in claimed.items() if os.path.basename(p) == b)
    tmps = []
    for b, p in pending:
        print(f"  REN {os.path.basename(p)} -> {b}")
        tmp = os.path.join(cur_dir, ".fixtmp__" + b)
        if not dry:
            os.rename(p, tmp)
        tmps.append((tmp, os.path.join(cur_dir, b)))
        renamed += 1
    if not dry:
        for tmp, dst in tmps:
            assert not os.path.exists(dst), f"目标仍被占用: {dst}"
            os.rename(tmp, dst)

    missing = [b for b in canon_names if b not in claimed]
    n_final = len(glob(os.path.join(cur_dir, "*.mp4")))
    print(f"  [{cur_dir}] canon={len(canon_names)} kept={kept} renamed={renamed} "
          f"deleted={deleted} missing={len(missing)} unknown={len(unknown)} final={n_final}")
    if missing:
        print(f"  !! 缺规范条目: {missing[:5]}")
    if unknown:
        print(f"  !! 无法归属: {[os.path.basename(u) for u in unknown[:5]]}")
    return not missing and not unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_root")
    ap.add_argument("canon_root")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shards", type=int, default=4,
                    help="生成时的题目分片数 (s200/s400 池=4, half 池=2)")
    ap.add_argument("--pure-sharded", action="store_true",
                    help="池完全由分片 worker 生成(无全量 run 残留)时必须开启")
    a = ap.parse_args()
    ok = True
    for seed_dir in sorted(glob(os.path.join(a.pool_root, "seed*_f93"))):
        seed = os.path.basename(seed_dir)
        for sub in ("generated_only", "side_by_side"):
            cur = os.path.join(seed_dir, sub)
            canon = os.path.join(a.canon_root, seed, sub)
            if not os.path.isdir(cur):
                continue
            if not os.path.isdir(canon):
                print(f"  !! 规范目录缺失: {canon}")
                ok = False
                continue
            print(f"== {seed}/{sub} ==")
            ok = fix_dir(cur, canon, a.dry_run, a.shards, a.pure_sharded) and ok
    print("RESULT:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
