#!/usr/bin/env python3
"""SELF-CASE 病例构建器 (70 号 A1, 阶段2 worker)。

输入: exam_v3 合并结果里 dup=True 的 rollout 视频。
产出: 每条病例一个 npz —— (base=复制区时域修补后的干净化 rollout, patch=复制品裁块,
boxes/frames/lat/alpha=固定贴入计划) + 预览 png + 逐视频记录 json。

第一性构造 (70 号 §3-A1 对抗修正版):
  输入视频  = rollout 本体 (模型自己犯的错, 不投人工毒)
  目标视频  = 同 rollout, 复制品区域用"出现前背景帧"逐帧回填 (时域修补)
  经现有 paste 机制实现: base=修补版, patch=复制品裁块, alpha=1 -> images_aug≈原 rollout。

病例过滤 (干净签名, 宁缺毋滥):
  F1 复制品轨迹静止 (中心漂移 < DRIFT_PX)
  F2 持续到片尾 (last >= NF-5) 且 d0 ∈ [D0_MIN, D0_MAX]
  F3 原物滞留: 原物轨迹在复制品生存期内 >= COEXIST_RATIO 帧可见 (排除"合法到达"误判)
  F4 全程与夹爪无重叠 (tier A); 中途接触 -> tier B (仅统计, 不出资产)
  F5 修补源帧 (d0-BACK_OFF) 处该区域无夹爪、无物体
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import t4g_probe as P
from t4g_detect import detect_all
from t4g_exam_v2 import NF, HPIX, WPIX, T_LAT
from t4g_gdino import GDinoLocator, parse_objects

# ---- 阈值 (第一性标定, 全部集中在此; v2 复盘放宽: 52 dup 只出 2 例的三个病根) ----
R_MATCH = 120     # 逐帧轨迹接力匹配半径 (px)
R_CLUSTER = 60    # 额外检出聚类半径 (= exam 去重半径)
R_SEP = 80        # 额外实例与原物轨迹的最小分离距离 (px)
DRIFT_MAX = 160   # 轨迹漂移 sanity 上限 (复制品会变形抖动, E5/E6; 真闸是面积上限)
AREA_CAP = 0.25   # union 框面积上限 (占全帧比例)
DUR_MIN = 12      # 病程最短像素帧数 (≈3 latent 档)
D0_MIN = 1        # 出现帧下界 (帧0 是条件锚, 复制不可能在 0)
COEXIST_RATIO = 0.4      # F3 原物共存比例 (原物用未过滤检出跟踪, 被抓持也算在场)
GRIP_OVL = 0.35   # 夹爪重叠阈 (= exam)
BACK_OFF = 2      # 修补源帧 = d0 - BACK_OFF (再往前回退直到干净)
MARGIN = 12       # 复制框外扩 (px; > 羽化带宽 10, 保证羽化区是纯背景混合)
MIN_PATCH = 16    # 最小 patch 边长


def _overlap(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[2], a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    return inter / area


def _valid_instances(obj_dets, grip_dets):
    """夹爪过滤 + 60px 去重, 返回 [(cx,cy,box,score)] (保框版 count_valid_instances)。"""
    out = []
    gboxes = [g[2] for g in grip_dets]
    for (cx, cy, box, sc) in obj_dets:
        if any(_overlap(box, gb) > GRIP_OVL for gb in gboxes):
            continue
        if all(abs(cx - o[0]) + abs(cy - o[1]) > R_CLUSTER for o in out):
            out.append((cx, cy, box, sc))
    return out


def _detect_video(loc, frames, name):
    """全 93 帧检测: 返回 per-frame (valid_instances, grip_boxes, raw_obj)。"""
    per = []
    for t in range(len(frames)):
        obj = detect_all(loc, frames[t], name, topk=6, box_thr=0.35)
        grip = detect_all(loc, frames[t], 'robot gripper', topk=3, box_thr=0.15)
        per.append((_valid_instances(obj, grip), [g[2] for g in grip], obj))
    return per


def _track_originals(per, anchors):
    """从帧0锚点在**未过滤**检出上逐帧接力追踪原物 (被夹爪抓持也算在场)。
    返回 present(A,T,bool) 与逐帧原物位置 orig_pos[t]=[(cx,cy)...]。"""
    T = len(per)
    present = np.zeros((max(1, len(anchors)), T), bool)
    orig_pos = [[] for _ in range(T)]
    for ai, (ax, ay) in enumerate(anchors):
        px, py = ax, ay
        for t in range(T):
            best, bd = None, R_MATCH
            for (cx, cy, box, sc) in per[t][2]:
                d = abs(cx - px) + abs(cy - py)
                if d < bd:
                    best, bd = (cx, cy), d
            if best is not None:
                px, py = best
                present[ai, t] = True
                orig_pos[t].append(best)
    return present, orig_pos


def _extra_tracks(per, orig_pos):
    """与原物轨迹位置分离 (>R_SEP) 的有效检出 -> 按空间聚类成额外轨迹。"""
    tracks = []   # each: dict(frames=[t...], centers=[(cx,cy)], boxes=[box])
    for t in range(len(per)):
        for (cx, cy, box, sc) in per[t][0]:
            if any(abs(cx - ox) + abs(cy - oy) <= R_SEP for ox, oy in orig_pos[t]):
                continue
            best, bd = None, R_CLUSTER
            for tr in tracks:
                mx, my = tr['centers'][-1]
                d = abs(cx - mx) + abs(cy - my)
                if d < bd:
                    best, bd = tr, d
            if best is None:
                tracks.append(dict(frames=[t], centers=[(cx, cy)], boxes=[box]))
            else:
                best['frames'].append(t)
                best['centers'].append((cx, cy))
                best['boxes'].append(box)
    return tracks


def build_case(loc, rec, it2v, out_dir, preview_dir):
    """单条 dup 视频 -> 病例 npz (或 None + 淘汰原因)。"""
    path, prompt = rec['video'], rec['prompt']
    cond_img = rec['cond_img']
    frames = P.sample_frames_like_training(path, NF, HPIX, WPIX)
    name = parse_objects(prompt)['mover']

    from PIL import Image
    cond = np.array(Image.open(cond_img).convert('RGB').resize((WPIX, HPIX)))
    cobj = detect_all(loc, cond, name, topk=6, box_thr=0.35)
    cgrip = detect_all(loc, cond, 'robot gripper', topk=3, box_thr=0.15)
    anchors = [(c[0], c[1]) for c in _valid_instances(cobj, cgrip)]
    per = _detect_video(loc, frames, name)
    if not anchors:
        anchors = [(c[0], c[1]) for t in range(3) for c in per[t][0]][:2]
    if not anchors:
        return None, 'no_anchor'

    present, orig_pos = _track_originals(per, anchors)
    tracks = _extra_tracks(per, orig_pos)
    tracks = [t for t in tracks if len(t['frames']) >= 6]
    if not tracks:
        return None, 'no_persistent_extra'
    tracks.sort(key=lambda t: -len(t['frames']))

    for tr in tracks:
        fs = tr['frames']
        d0, dend = fs[0], fs[-1] + 1
        if d0 < D0_MIN:
            d0 = D0_MIN
        cs = np.array(tr['centers'])
        drift = float(np.abs(cs - cs[0]).sum(axis=1).max())
        if drift > DRIFT_MAX:                     # sanity (真闸是面积上限)
            reason = 'drift'; continue
        bx = np.array(tr['boxes'])
        ub = (max(0, int(bx[:, 0].min()) - MARGIN), max(0, int(bx[:, 1].min()) - MARGIN),
              min(WPIX, int(bx[:, 2].max()) + MARGIN), min(HPIX, int(bx[:, 3].max()) + MARGIN))
        x0, y0, x1, y1 = ub
        if (x1 - x0) < MIN_PATCH or (y1 - y0) < MIN_PATCH:
            reason = 'tiny'; continue
        if (x1 - x0) * (y1 - y0) > AREA_CAP * HPIX * WPIX:
            reason = 'area_cap'; continue
        # F4 夹爪接触 -> 截断病程 (而非整例否决)
        for t in range(d0, dend):
            if any(_overlap((x0, y0, x1, y1), gb) > 0.05 for gb in per[t][1]):
                dend = t
                break
        if dend - d0 < DUR_MIN:                   # F2 病程下限 (含截断后)
            reason = 'span'; continue
        cover = present[:, d0:dend].any(axis=0).mean() if present.size else 0.0
        if cover < COEXIST_RATIO:                 # F3 原物滞留 (未过滤跟踪)
            reason = 'orig_gone'; continue
        # F6 GT 对照闸: 同帧段内 GT 在该区域也有同类 -> 合法内容 (背景操作员持物/
        #    静止干扰物闪烁), 不是模型幻觉, 否决。限定同帧段, 不误杀目的地复制
        #    (GT 尾段合法到达不落入病例帧段)。
        gt_frames = rec.get('_gt_frames')
        if gt_frames is None:
            gt_frames = P.sample_frames_like_training(rec['gt_video'], NF, HPIX, WPIX)
            rec['_gt_frames'] = gt_frames
        pad = 40
        gx0, gy0 = max(0, x0 - pad), max(0, y0 - pad)
        gx1, gy1 = min(WPIX, x1 + pad), min(HPIX, y1 + pad)
        gt_hit = False
        for gt_t in np.linspace(d0, dend - 1, 10).astype(int):
            for (cx, cy, box, sc) in detect_all(loc, gt_frames[gt_t], name,
                                                topk=6, box_thr=0.25):
                if gx0 <= cx <= gx1 and gy0 <= cy <= gy1:
                    gt_hit = True
                    break
            if gt_hit:
                break
        if gt_hit:
            reason = 'gt_has_obj_there'; continue

        # F5 修补源帧
        src = -1
        for cand in range(max(0, d0 - BACK_OFF), -1, -1):
            g_ok = not any(_overlap((x0, y0, x1, y1), gb) > 0.02 for gb in per[cand][1])
            o_ok = not any(_overlap((x0, y0, x1, y1), c[2]) > 0.02 for c in per[cand][0])
            if g_ok and o_ok:
                src = cand
                break
        if src < 0:
            reason = 'no_clean_src'; continue

        # ---- 构造资产 (修补区边缘 10px 线性羽化, 压接缝; 贴回 alpha=1 会在输入端整块还原) ----
        base = frames.copy()
        bg = frames[src][y0:y1, x0:x1].astype(np.float32)
        fh, fw = y1 - y0, x1 - x0
        ramp = 10.0
        yy = np.minimum(np.arange(fh), np.arange(fh)[::-1]).astype(np.float32)
        xx = np.minimum(np.arange(fw), np.arange(fw)[::-1]).astype(np.float32)
        feather = np.minimum(np.minimum(yy[:, None], xx[None, :]) / ramp, 1.0)[..., None]
        for t in range(d0, dend):
            orig = frames[t][y0:y1, x0:x1].astype(np.float32)
            base[t][y0:y1, x0:x1] = (feather * bg + (1 - feather) * orig).astype(np.uint8)
        mids = [f for f in fs if d0 <= f < dend]
        mid = mids[len(mids) // 2] if mids else d0
        patch = frames[mid][y0:y1, x0:x1].copy()
        boxes = np.zeros((NF, 4), np.int64)
        boxes[d0:dend] = (y0, x0, y1, x1)         # (y0,x0,y1,x1), 与 trainer 贴入约定一致
        rep = np.linspace(0, NF - 1, T_LAT).astype(int)
        lt_lo = int(np.searchsorted(rep, d0))
        lt_hi = max(lt_lo + 1, int(np.searchsorted(rep, dend - 1, side='right')))
        lat = np.array([lt_lo, lt_hi, y0 // 8, min(HPIX // 8, (y1 + 7) // 8),
                        x0 // 8, min(WPIX // 8, (x1 + 7) // 8)], np.int64)

        vid = rec['gt_vid']
        seed = rec['seed']
        case_id = f'{vid}__s{seed}'
        np.savez_compressed(os.path.join(out_dir, case_id + '.npz'),
                            base=base, patch=patch, boxes=boxes,
                            frames=np.array([d0, dend], np.int64), lat=lat,
                            alpha=np.float32(1.0))
        # 预览: 原 rollout | 修补版 | patch (mid 帧)
        import cv2
        h = 240
        a = cv2.resize(frames[mid], (WPIX * h // HPIX, h))
        b = cv2.resize(base[mid], (WPIX * h // HPIX, h))
        pv = np.concatenate([a, b], axis=1)
        cv2.rectangle(pv, (x0 * a.shape[1] // WPIX, y0 * h // HPIX),
                      (x1 * a.shape[1] // WPIX, y1 * h // HPIX), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(preview_dir, case_id + '.png'), pv[:, :, ::-1])
        return dict(case_id=case_id, vid=vid, seed=seed, d0=int(d0), dend=int(dend),
                    src=int(src), box=[int(v) for v in (y0, x0, y1, x1)],
                    n_frames=len(fs), drift=drift, coexist=float(cover), video=path), 'ok'
    return None, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exam-json', required=True)
    ap.add_argument('--it2v', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--preview-dir', required=True)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.preview_dir, exist_ok=True)

    it2v = {int(str(r['request_id']).split('_')[0]): r for r in json.load(open(a.it2v))}
    recs = [r for r in json.load(open(a.exam_json)) if r.get('dup')]
    recs = recs[a.shard_index::a.num_shards]
    loc = GDinoLocator(device='cuda')
    done, results = 0, []
    for r in recs:
        idx = int(os.path.basename(r['video']).split('_')[0])
        meta = it2v[idx]
        r = dict(r)
        r['prompt'] = meta['prompt']
        r['cond_img'] = meta['image']
        r['gt_video'] = meta['source_video']
        r['gt_vid'] = os.path.splitext(meta['source_file_name'])[0]
        r['seed'] = r.get('arm', 'a').split('_s')[-1]
        try:
            case, why = build_case(loc, r, it2v, a.out_dir, a.preview_dir)
        except Exception as e:  # noqa: BLE001 病例级隔离, 单条失败不拖垮分片
            case, why = None, f'error:{type(e).__name__}:{e}'
        results.append(dict(video=r['video'], gt_vid=r['gt_vid'], seed=r['seed'],
                            status=why, case=case))
        done += 1
        if done % 5 == 0:
            print(f'[build shard{a.shard_index}] {done}/{len(recs)}', flush=True)
    json.dump(results, open(a.out, 'w'), indent=1)
    print(f'[build shard{a.shard_index}] DONE {done}', flush=True)


if __name__ == '__main__':
    main()
