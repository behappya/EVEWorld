#!/usr/bin/env python3
"""T4G-WEIGHTMAP-VARIANTS: IGR 空间权重图六变体 (并列可选, 无推荐)。

对应论文表格 tab:weightmap_design 的六行设计:
  legacy_multilevel  Legacy multi-level (0.5/2/3/4/6 五档, 即 t4g_weightmap 现行方案)
  binary_3x          Binary-3            (base 1x, 全部区域组 3x)
  path_loc_2x        Path+Loc 2x         (Path/Loc. 区 2x, 其余 1x)
  path_loc_3x        Path+Loc 3x         (Path/Loc. 区 3x, 其余 1x)
  interaction_2x     Interaction 2x      (Path/Loc. + Arm/Grip. + Paste 窗 2x, 其余 1x)
  interaction_3x     Interaction 3x      (同上, 3x)
六者是并列的设计选项, 本模块不给出任何"更好/默认"的倾向; 选用哪一种由使用者决定。

区域组来自 t4g_weightmap.build_weightmap_regions 的七组分解
(obj/trans/b_dest/grip/empty/distractor/bg)。论文表格三列按【语义】还原为区域组:
  Path/Loc.  = obj + trans + b_dest + empty  (物体轨迹 + 源/目的地抓放窗 + 两处该空)
  Arm/Grip.  = grip                          (夹爪检测框)
  Paste      = 抓取窗/放置窗 (trans) + 训练时才知道的"实际贴入区";
               六行的 Paste 列均为勾选, 故 trans 归入 Path/Loc. (六变体都开),
               真正贴入区的权重由 t4g_joint_trainer 的 t4g_w_paste clamp 另行抬到 3x。
  干扰物 distractor 单独一组 (仅 legacy_multilevel 与 binary_3x 纳入)
按语义还原, 不保证与论文作者内部实现逐位相同 (详见 t4g_weightmap_variants.md)。

契约 (与训练侧对齐, 变更前先看 t4g_aug_trainer / t4g_joint_trainer):
  写盘的 .npy 存【档位原值】(未归一化, normalize=False), 因为 t4g_joint_trainer 在 loss
  内自己做逐样本均值 1 归一化, 且 transform 校验 np.unique(w) ⊆ wmap_values、region 账本
  按档位精确匹配 (|region_map - level| < 1e-3)。normalize=True 只用于离线分析/可视化。

用法:
  python3 eveworld/pipeline/t4g_weightmap_variants.py --list-variants
  python3 eveworld/pipeline/t4g_weightmap_variants.py --self-test
  python3 eveworld/pipeline/t4g_weightmap_variants.py --variant interaction_3x [--out-dir D]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:                                     # 包内导入
    from . import t4g_weightmap as W
except ImportError:                      # 直接以脚本运行
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import t4g_weightmap as W

GAGI = '/data/datasets/gagi'
PROBE = GAGI + '/eve_v2_outputs/track4gen_probe'
PACKED = GAGI + '/gr1_finetune_data/packed_data'
PRETRAIN = GAGI + '/giga_world_0_video_pretrain'
ANNO_DIR = PROBE + '/t4g_anno'
ASSETS_DIR = PROBE + '/aug_assets'
OUTPUT_ROOT = GAGI + '/eve_v2_outputs/t4g_weightmap_variants'

CACHE_PREFIX = 'weightmap_cache_'         # 每变体独立缓存目录: <root>/weightmap_cache_<variant>
ROOT_ENV = 'T4G_WMAP_ROOT'                # 缓存根目录覆盖 (默认 PROBE)
EXPECTED_SAMPLES = 92
W_PASTE = 3.0                             # 六变体统一, 隔离权重图这一个变量

# 论文列 -> 区域组 (语义还原): 表格里 Path/Loc. 与 Paste 两列在六行都是勾选,
# 因此 Path/Loc. 取 {obj, trans, b_dest, empty} (轨迹 + 源/目的地抓放窗 + 两处该空),
# Paste 的"实际贴入区"不属于离线掩码, 由 t4g_w_paste clamp 在训练时补足。
COL_PATH_LOC = ('obj', 'trans', 'b_dest', 'empty')
COL_ARM_GRIP = ('grip',)
COL_PASTE = ('trans',)                       # 抓取窗/放置窗 = 贴入发生的源/目的地窗口
COL_DISTRACTOR = ('distractor',)

# 变体注册表 (并列, 无默认/推荐): region_levels 是权威的"区域组 -> 档位"映射,
# levels 是期望的 np.unique(wmap) 级别集 (由 base_level + region_levels 的值导出)。
VARIANTS = {
    'legacy_multilevel': dict(
        mode='legacy',
        levels=(0.5, 2.0, 3.0, 4.0, 6.0),
        base_level=0.5,
        emphasis_level=None,
        regions=('obj', 'trans', 'empty', 'b_dest', 'grip', 'distractor'),
        region_levels=dict(obj=4.0, trans=6.0, empty=2.0, b_dest=3.0, grip=3.0,
                           distractor=2.0),
        description='多档设计: 背景 0.5x, 各语义区域按高危程度取 2/3/4/6x (现行 t4g_weightmap 方案)',
    ),
    'binary_3x': dict(
        mode='binary',
        levels=(1.0, 3.0),
        base_level=1.0,
        emphasis_level=3.0,
        regions=('obj', 'trans', 'empty', 'b_dest', 'grip', 'distractor'),
        region_levels=dict(obj=3.0, trans=3.0, empty=3.0, b_dest=3.0, grip=3.0,
                           distractor=3.0),
        description='二值设计: 上述六类区域统一 3x, 背景 1x',
    ),
    'path_loc_2x': dict(
        mode='binary',
        levels=(1.0, 2.0),
        base_level=1.0,
        emphasis_level=2.0,
        regions=('obj', 'trans', 'b_dest', 'empty'),
        region_levels=dict(obj=2.0, trans=2.0, b_dest=2.0, empty=2.0),
        description='只强调物体路径/位置区 (Path/Loc. = 轨迹 + 源/目的地抓放窗 + 两处该空), 2x',
    ),
    'path_loc_3x': dict(
        mode='binary',
        levels=(1.0, 3.0),
        base_level=1.0,
        emphasis_level=3.0,
        regions=('obj', 'trans', 'b_dest', 'empty'),
        region_levels=dict(obj=3.0, trans=3.0, b_dest=3.0, empty=3.0),
        description='只强调物体路径/位置区 (Path/Loc. = 轨迹 + 源/目的地抓放窗 + 两处该空), 3x',
    ),
    'interaction_2x': dict(
        mode='binary',
        levels=(1.0, 2.0),
        base_level=1.0,
        emphasis_level=2.0,
        regions=('obj', 'trans', 'b_dest', 'empty', 'grip'),
        region_levels=dict(obj=2.0, trans=2.0, b_dest=2.0, empty=2.0, grip=2.0),
        description='在 Path/Loc. 之上再加 Arm/Grip. (夹爪框), 2x; 其余 1x',
    ),
    'interaction_3x': dict(
        mode='binary',
        levels=(1.0, 3.0),
        base_level=1.0,
        emphasis_level=3.0,
        regions=('obj', 'trans', 'b_dest', 'empty', 'grip'),
        region_levels=dict(obj=3.0, trans=3.0, b_dest=3.0, empty=3.0, grip=3.0),
        description='在 Path/Loc. 之上再加 Arm/Grip. (夹爪框), 3x; 其余 1x',
    ),
}

VARIANT_NAMES = tuple(VARIANTS)


def variant_spec(variant):
    """取变体规格; 未知名字给出可用列表。"""
    if variant not in VARIANTS:
        raise ValueError(f'未知权重图变体 {variant!r}; 可用: {", ".join(VARIANT_NAMES)}')
    return VARIANTS[variant]


def variant_levels(variant):
    """该变体期望出现的权重档位 (升序 tuple), 即 np.unique(wmap) 应等于此集合。"""
    return tuple(variant_spec(variant)['levels'])


def variant_regions(variant):
    """该变体纳入加权的区域组 (其余区域组保持 base_level)。"""
    return tuple(variant_spec(variant)['regions'])


def variant_wmap_dir(variant, root=None):
    """该变体的缓存目录: <root>/weightmap_cache_<variant>; root 默认 $T4G_WMAP_ROOT 或 PROBE。"""
    base = root or os.environ.get(ROOT_ENV) or PROBE
    return os.path.join(base, CACHE_PREFIX + variant)


def variant_region_names(variant):
    """t4g_region_names: 与 variant_levels 等长的档位名 (仅作日志标签)。"""
    spec = variant_spec(variant)
    if spec['mode'] == 'legacy':
        return 'bg0.5x,empty2x,gripB3x,obj4x,trans6x'      # 历史串, 保持既有约定
    base, emph = spec['base_level'], spec['emphasis_level']
    return f'base{base:g}x,emph{emph:g}x'


def _region_union(regions, keep):
    """各区域掩码取并 (keep 为区域组名集合)。"""
    out = np.zeros((W.T_LAT, W.H_LAT, W.W_LAT), bool)
    for k in keep:
        out |= regions[k]
    return out


def build_variant_weightmap(anno, variant, normalize=True, t_grasp=None, t_release=None):
    """按 variant 设计把区域掩码落成权重图, 返回 (T_LAT,30,48) float32。

    normalize=True (默认): 再逐样本除以自身均值 -> 均值恰为 1 (与 t4g_joint_trainer
      loss 内归一化一致, 便于离线对比); normalize=False: 档位原值 (写 cache 用这种)。
    """
    spec = variant_spec(variant)
    _, _, regions = W.build_weightmap_regions(anno, t_grasp=t_grasp, t_release=t_release)
    w = np.full((W.T_LAT, W.H_LAT, W.W_LAT), np.float32(spec['base_level']), np.float32)
    for k in W.REGION_ORDER:               # 级别升序落值 (高档覆盖低档)
        lev = spec['region_levels'].get(k)
        if lev is not None:
            w[regions[k]] = np.float32(lev)
    if normalize:
        w = (w / w.mean()).astype(np.float32)
    return w


def variant_config(variant, root=None):
    """生成该变体的完整训练 config (六份 config 薄壳共用, 避免各写一份导致漂移)。

    权重图相关字段全部由注册表派生: wmap_dir / wmap_values / t4g_wmap_dir /
    t4g_region_levels / t4g_region_names; 其余超参与六变体完全平行 (隔离权重图变量)。
    """
    spec = variant_spec(variant)
    levels = variant_levels(variant)
    levels_csv = ','.join(f'{lev:g}' for lev in levels)
    wmap_dir = variant_wmap_dir(variant, root)
    anno_dir = ANNO_DIR
    idx2vid = anno_dir + '/_packidx2vid.json'

    return dict(
        runners=['eveworld.pipeline.t4g_joint_trainer.T4GJointTrainer'],
        project_dir=f'{OUTPUT_ROOT}/{variant}/experiments',
        launch=dict(
            gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            distributed_type='DEEPSPEED',
            deepspeed_config=dict(
                deepspeed_config_file='accelerate_configs/zero2.json',
            ),
        ),
        dataloaders=dict(
            train=dict(
                data_or_config=[PACKED],
                batch_size_per_gpu=1,
                num_workers=6,
                transform=dict(
                    type='T4GAugTransform',
                    num_frames=93,
                    height=480,
                    width=768,
                    fps=16,
                    image_cfg=dict(
                        mask_generator=dict(max_ref_frames=1, start=1, factor=4),
                    ),
                    idx2vid_path=idx2vid,
                    anno_dir=anno_dir,
                    assets_dir=ASSETS_DIR,
                    wmap_dir=wmap_dir,
                    wmap_values=levels_csv,
                    strict_mapping=True,
                    strict_assets=True,
                    strict_wmap=True,
                    p_aug=0.5,
                    zone_bias='0.4,0.2,0.4',
                    seed=20260721,
                    p_corridor=0.0,
                    p_alien=0.0,
                ),
                sampler=dict(type='DefaultSampler', shuffle=True),
            ),
        ),
        models=dict(
            vae_model_path=PRETRAIN + '/vae',
            transformer_model_path=PRETRAIN + '/transformer',
            t4g_id_block='block22',
            t4g_change_block='block25',
            t4g_sigma_lo=0.2,
            t4g_sigma_hi=0.5,
            t4g_warmup=20,
            t4g_lambdas='0.5,0.0',
            t4g_tau=0.07,
            t4g_tol=0.2,
            t4g_anno_dir=anno_dir,
            t4g_idx2vid=idx2vid,
            t4g_expected_samples=EXPECTED_SAMPLES,
            t4g_wmap_dir=wmap_dir,
            t4g_w_paste=W_PASTE,
            t4g_region_levels=levels_csv,
            t4g_region_names=variant_region_names(variant),
        ),
        optimizers=dict(type='CAME8Bit', lr=2 ** (-14.5)),
        schedulers=dict(type='ConstantScheduler'),
        train=dict(
            resume=True,
            max_steps=300,
            gradient_accumulation_steps=8,
            mixed_precision='bf16',
            checkpoint_interval=50,
            checkpoint_total_limit=8,
            checkpoint_start_step=0,
            checkpoint_strict=False,
            log_with='tensorboard',
            log_interval=1,
            with_ema=True,
            activation_checkpointing=True,
            activation_class_names=['TransformerBlock'],
            seed=6666,
        ),
    )


def list_anno_vids(anno_dir=None):
    """列出 anno 目录下的 vid (与 t4g_weightmap_precompute.py 同款过滤: 数字开头的 json)。"""
    d = anno_dir or W.ANNO_DIR
    return sorted(f[:-5] for f in os.listdir(d)
                  if f.endswith('.json') and not f.startswith('_') and f[0].isdigit())


def precompute_variant(variant, root=None, out_dir=None, vids=None, normalize=False,
                       anno_dir=None, assets_dir=None, gripper_dir=None, quiet=False):
    """把一个变体的权重图逐 vid 写盘, 返回统计 dict (供 CLI 与 self-test 复用)。

    out_dir 优先; 否则 <root>/weightmap_cache_<variant> (root 默认 $T4G_WMAP_ROOT 或 PROBE)。
    normalize=False (默认) 写【档位原值】, 满足训练侧 wmap_values / region 账本契约。
    """
    spec = variant_spec(variant)
    if anno_dir:
        W.ANNO_DIR = anno_dir
    if assets_dir:
        W.ASSETS_DIR = assets_dir
    if gripper_dir:
        W.GRIPPER_DIR = gripper_dir
    out = out_dir or variant_wmap_dir(variant, root)
    os.makedirs(out, exist_ok=True)
    vids = list(vids) if vids else list_anno_vids(W.ANNO_DIR)
    if not vids:
        raise RuntimeError(f'{W.ANNO_DIR} 下没有可用的 anno json')

    levels = variant_levels(variant)
    base = spec['base_level']
    counts = {lev: 0 for lev in levels}
    coverages = []
    for vid in vids:
        anno = W.load_anno(vid)
        w = build_variant_weightmap(anno, variant, normalize=normalize)
        if w.shape != (W.T_LAT, W.H_LAT, W.W_LAT):
            raise AssertionError(f'{vid}: shape {w.shape} != {(W.T_LAT, W.H_LAT, W.W_LAT)}')
        if not np.isfinite(w).all():
            raise AssertionError(f'{vid}: 权重图含非有限值')
        if normalize:
            if abs(float(w.mean()) - 1.0) > 1e-5:
                raise AssertionError(f'{vid}: 归一化后均值 {float(w.mean())} != 1')
        else:
            u = set(float(x) for x in np.unique(w))
            if not u <= set(levels):
                raise AssertionError(f'{vid}: 档位 {sorted(u)} ⊄ {sorted(levels)} (wmap_values 契约)')
            for lev in levels:
                counts[lev] += int((w == np.float32(lev)).sum())
            coverages.append(float((w > np.float32(base)).mean()))
        np.save(os.path.join(out, f'{vid}.npy'), w)

    stats = dict(variant=variant, out_dir=out, n=len(vids), normalize=normalize,
                 levels=levels, base_level=base, counts=counts,
                 coverage_median=(float(np.median(coverages)) if coverages else None),
                 coverage_min=(min(coverages) if coverages else None),
                 coverage_max=(max(coverages) if coverages else None))
    if not quiet:
        print(f'[{variant}] 写出 {len(vids)} 个 .npy -> {out}')
        if normalize:
            print('  (normalize=True: 已逐样本归一化到均值 1, 不满足训练侧 wmap_values 契约)')
        else:
            total = sum(counts.values())
            for lev in levels:
                print(f'  档位 {lev:g}: {counts[lev]:>12d} 格 ({counts[lev] / max(1, total):.4%})')
            cov = stats['coverage_median']
            if cov is not None:
                print(f'  覆盖率 (w > base {base:g}x) 中位 {cov:.4f}  '
                      f'范围 [{stats["coverage_min"]:.4f}, {stats["coverage_max"]:.4f}]  '
                      f'(对照论文 Coverage 列, 中位数为 {cov:.4f})')
    return stats


def _check_registry():
    """注册表自洽性: 区域组/档位/期望级别集三者必须互相自洽。"""
    for name, spec in VARIANTS.items():
        assert set(spec['regions']) == set(spec['region_levels']), name
        assert set(spec['region_levels']) <= set(W.REGION_KEYS), name
        derived = set(spec['region_levels'].values()) | {spec['base_level']}
        assert derived == set(spec['levels']), (name, sorted(derived), spec['levels'])
        assert min(spec['levels']) == spec['base_level'], name
        assert tuple(sorted(spec['levels'])) == tuple(spec['levels']), name
        if spec['mode'] == 'binary':
            assert spec['emphasis_level'] == max(spec['levels']), name
            assert all(v == spec['emphasis_level'] for v in spec['region_levels'].values()), name
        else:                                     # legacy: 逐区域档位 = 现行 t4g_weightmap 常量
            assert spec['emphasis_level'] is None, name
            assert spec['region_levels'] == {k: W.REGION_VALUES[k] for k in spec['regions']}, name
            assert spec['base_level'] == W.W_BG, name


def _fake_anno(vid='9999'):
    """self-test 用假 anno: 轨迹先静后动 -> t_grasp/t_release 可估; 七个区域组全非空。"""
    traj = [(10, 12)] * 8 + [(10, 15), (9, 16), (8, 18), (7, 20)] + [(6, 22)] * 12
    per = [dict(target_cell=list(c), b_cells=[[6, 30], [6, 31]]) for c in traj]
    return dict(vid=vid, per_lat_frame=per, t_arrival=8, gate_enabled=True,
                target_cell_0=[10, 12], distractor_cells=[[2, 2]])


def self_test():
    """内置自检: 区域分解 / 六变体档位集 / 归一化 / legacy 向后兼容 / config 自洽。

    全部基于合成假 anno (无数据集); 打印的覆盖率【不】对应论文 Coverage 列, 只用于验证
    六变体之间的相对关系 (legacy 与 binary_3x 覆盖率必然相等: 两者都覆盖全部六组区域)。
    """
    import tempfile
    _check_registry()

    tmp = tempfile.mkdtemp(prefix='wmap_variants_selftest_')
    grip_dir = os.path.join(tmp, 'gripper_anno')
    os.makedirs(grip_dir)
    anno = _fake_anno()
    json.dump({'per_lat_gripper': [[dict(center=[15, 20], cells=[[15, 20], [15, 21]],
                                         score=1.0)] for _ in range(W.T_LAT)]},
              open(os.path.join(grip_dir, f'{anno["vid"]}.json'), 'w'))

    old_grip, old_assets = W.GRIPPER_DIR, W.ASSETS_DIR
    W.GRIPPER_DIR = grip_dir
    W.ASSETS_DIR = os.path.join(tmp, 'no_such_assets')       # 无框 -> DEFAULT_HALF
    try:
        w0, seg, regions = W.build_weightmap_regions(anno)
        n_cells = w0.size
        # 1) 七个区域组全部非空
        for k in W.REGION_KEYS:
            n = int(regions[k].sum())
            assert n > 0, f'区域组 {k} 为空 (自检假 anno 失效)'
            print(f'  区域 {k:10s}: {n:>6d} 格 ({n / n_cells:.4%})  级别 {W.REGION_VALUES[k]}')
        assert seg['t_grasp'] == 8 and seg['t_release'] == 12 and seg['traj_reliable'], seg
        print(f'  段位: t_grasp={seg["t_grasp"]} t_release={seg["t_release"]} '
              f't_arrival={seg["t_arrival"]} reliable={seg["traj_reliable"]}')

        # 2) 六变体: 档位集 / 区域落值 / base 落值
        base_map = ~_region_union(regions, set(W.REGION_KEYS) - {'bg'})
        for name in VARIANT_NAMES:
            spec = VARIANTS[name]
            base = np.float32(spec['base_level'])
            raw = build_variant_weightmap(anno, name, normalize=False)
            assert raw.dtype == np.float32
            lv = sorted(float(x) for x in np.unique(raw))
            assert lv == list(spec['levels']), (name, lv, spec['levels'])
            emph = _region_union(regions, set(spec['regions']))
            # 独立复算 (逐区域 np.maximum 归约): 区域可能重叠 -> 取各区域档位的最高者
            cand = np.full((W.T_LAT, W.H_LAT, W.W_LAT), base, np.float32)
            for k, lev in spec['region_levels'].items():
                assert np.all(raw[regions[k]] >= np.float32(lev)), (name, k)   # 本组至少本档
                cand = np.maximum(cand, np.where(regions[k], np.float32(lev), base))
            assert np.array_equal(raw, cand), (name, 'max 归约不一致')
            assert np.all(raw[base_map] == base), name                        # 未被覆盖 = base
            # 二值变体: 强调格/普通格 比值恰为 emphasis:1
            ratio = ''
            if spec['mode'] == 'binary':
                r = float(raw[emph].mean() / raw[~emph].mean())
                assert abs(r - spec['emphasis_level']) < 1e-6, (name, r)
                ratio = f'  强调比 {r:g}:1'
            # 归一化: 均值 1, 相对比值保持
            nz = build_variant_weightmap(anno, name, normalize=True)
            assert nz.dtype == np.float32
            assert abs(float(nz.mean()) - 1.0) < 1e-5, (name, float(nz.mean()))
            if spec['mode'] == 'binary':
                rn = float(nz[emph].mean() / nz[~emph].mean())
                assert abs(rn - spec['emphasis_level']) < 1e-4, (name, rn)
            cov = float((raw > np.float32(spec['base_level'])).mean())
            print(f'  {name:19s} levels={lv}  覆盖率 {cov:.4f}'
                  f'  归一化后均值 {float(nz.mean()):.6f}{ratio}')

        # 3) legacy 变体与现行 build_weightmap 逐位一致
        legacy_raw = build_variant_weightmap(anno, 'legacy_multilevel', normalize=False)
        ref, _ = W.build_weightmap(anno)
        assert legacy_raw.dtype == ref.dtype == np.float32
        assert np.array_equal(legacy_raw, ref), 'legacy 变体与 build_weightmap 不一致'
        print('  legacy_multilevel == build_weightmap (逐位一致)')

        # 4) 归一化默认值: 均值 1
        assert abs(float(build_variant_weightmap(anno, 'interaction_3x').mean()) - 1.0) < 1e-5

        # 5) precompute 端到端 (临时 anno/assets/gripper 目录, 只跑两个 vid)
        anno_dir = os.path.join(tmp, 'anno')
        os.makedirs(anno_dir)
        for vid in ('9001', '9002'):
            a = _fake_anno(vid)
            json.dump(a, open(os.path.join(anno_dir, f'{vid}.json'), 'w'))
            json.dump({'per_lat_gripper': [[dict(center=[15, 20], cells=[[15, 20]],
                                                 score=1.0)] for _ in range(W.T_LAT)]},
                      open(os.path.join(grip_dir, f'{vid}.json'), 'w'))
        for name in VARIANT_NAMES:
            st = precompute_variant(name, out_dir=os.path.join(tmp, 'cache', name),
                                    anno_dir=anno_dir, assets_dir=W.ASSETS_DIR,
                                    gripper_dir=grip_dir, quiet=True)
            assert st['n'] == 2, st
            for vid in ('9001', '9002'):
                w = np.load(os.path.join(st['out_dir'], f'{vid}.npy'))
                assert w.shape == (W.T_LAT, W.H_LAT, W.W_LAT) and w.dtype == np.float32
                assert set(float(x) for x in np.unique(w)) <= set(st['levels'])
        print(f'  precompute: 六变体 × 2 vid 写出并通过档位契约校验 -> {os.path.join(tmp, "cache")}')

        # 6) 六份 config 自洽且互相平行
        cfgs = {name: variant_config(name, root=tmp) for name in VARIANT_NAMES}
        for name, cfg in cfgs.items():
            spec = VARIANTS[name]
            tf = cfg['dataloaders']['train']['transform']
            md = cfg['models']
            assert tf['wmap_dir'] == md['t4g_wmap_dir'] == variant_wmap_dir(name, tmp)
            assert name in tf['wmap_dir'], name
            assert set(tf['wmap_values'].split(',')) == {f'{lev:g}' for lev in spec['levels']}
            assert tf['wmap_values'] == md['t4g_region_levels']
            assert len(md['t4g_region_levels'].split(',')) == len(md['t4g_region_names'].split(','))
            assert md['t4g_expected_samples'] == EXPECTED_SAMPLES == 92
            assert tf['strict_wmap'] is True and md['t4g_w_paste'] == W_PASTE
            assert cfg['project_dir'] == f'{OUTPUT_ROOT}/{name}/experiments'
            assert tf['anno_dir'] == md['t4g_anno_dir'] == ANNO_DIR
            assert tf['assets_dir'] == ASSETS_DIR
            assert tf['idx2vid_path'] == md['t4g_idx2vid']
        def _shape(d, drop):
            if not isinstance(d, dict):
                return repr(d)
            return {k: _shape(v, drop) for k, v in sorted(d.items()) if k not in drop}
        ref_shape = _shape(cfgs['legacy_multilevel'],
                           {'wmap_dir', 'wmap_values', 't4g_wmap_dir', 't4g_region_levels',
                            't4g_region_names', 'project_dir'})
        for name, cfg in cfgs.items():
            s = _shape(cfg, {'wmap_dir', 'wmap_values', 't4g_wmap_dir', 't4g_region_levels',
                             't4g_region_names', 'project_dir'})
            assert s == ref_shape, f'{name} 与 legacy 变体结构不平行'
        print(f'  六份 config: 结构互相平行, wmap_dir/级别集/区域名自洽, '
              f'expected_samples={EXPECTED_SAMPLES}')
    finally:
        W.GRIPPER_DIR, W.ASSETS_DIR = old_grip, old_assets

    print('SELF-TEST OK')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='IGR 空间权重图六变体 (并列可选): 预计算 / 自检 / 列表')
    ap.add_argument('--variant', help='变体名, 或 all (全部六个)')
    ap.add_argument('--list-variants', action='store_true', help='列出六变体及其级别/区域')
    ap.add_argument('--self-test', action='store_true', help='内置自检 (假 anno, 无需数据集)')
    ap.add_argument('--out-dir', help='缓存输出目录 (优先级高于 --root)')
    ap.add_argument('--root', help=f'缓存根目录, 默认 ${ROOT_ENV} 或 {PROBE}')
    ap.add_argument('--vids', help='逗号分隔的 vid 子集, 默认 anno 目录全部')
    ap.add_argument('--normalize', action='store_true',
                    help='写归一化到均值 1 的图 (仅离线分析; 不满足训练侧 wmap_values 契约)')
    ap.add_argument('--anno-dir', help='覆盖 anno 目录')
    ap.add_argument('--assets-dir', help='覆盖 aug_assets 目录')
    ap.add_argument('--gripper-dir', help='覆盖 gripper_anno 目录')
    args = ap.parse_args(argv)

    if args.list_variants:
        _check_registry()
        for name in VARIANT_NAMES:
            spec = VARIANTS[name]
            print(f'{name}\n'
                  f'  级别: {tuple(spec["levels"])}  base={spec["base_level"]:g}  '
                  f'强调={spec["emphasis_level"] if spec["emphasis_level"] is not None else "多档"}\n'
                  f'  区域: {", ".join(spec["regions"])}\n'
                  f'  缓存: {variant_wmap_dir(name, args.root)}\n'
                  f'  说明: {spec["description"]}')
        return 0
    if args.self_test:
        return self_test()
    if not args.variant:
        ap.print_help()
        return 2

    _check_registry()
    names = VARIANT_NAMES if args.variant == 'all' else (args.variant,)
    for name in names:
        variant_spec(name)
    vids = [v for v in args.vids.split(',') if v] if args.vids else None
    for name in names:
        precompute_variant(name, root=args.root, out_dir=args.out_dir, vids=vids,
                           normalize=args.normalize, anno_dir=args.anno_dir,
                           assets_dir=args.assets_dir, gripper_dir=args.gripper_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
